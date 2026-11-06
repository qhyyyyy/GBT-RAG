import argparse
import csv
import gzip
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.llm import Embedder


def build_index(source, output, embedder, batch_size=64, limit=None):
    import faiss
    source, output = Path(source), Path(output)
    if not source.is_file():
        raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Index output directory must be empty; use a new directory to rebuild")
    db = sqlite3.connect(output / "passages.sqlite")
    db.execute("CREATE TABLE passages(row_id INTEGER PRIMARY KEY, source_id TEXT, title TEXT, text TEXT)")
    index = faiss.IndexFlatIP(embedder.embed_dim)
    opener = gzip.open if source.suffix == ".gz" else open
    count, batch = 0, []
    corpus_digest = hashlib.sha256()
    def flush(rows):
        nonlocal count
        if not rows:
            return
        vectors = embedder.encode_passages([row["text"] for row in rows], batch_size=batch_size)
        index.add(vectors)
        db.executemany("INSERT INTO passages VALUES(?,?,?,?)",
                       [(count + i, str(row["id"]), row["title"], row["text"]) for i, row in enumerate(rows)])
        db.commit()
        count += len(rows)
    try:
        with opener(source, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not {"id", "title", "text"}.issubset(reader.fieldnames or []):
                raise ValueError("DPR TSV must have id, text and title columns")
            for row in reader:
                if not row["text"].strip():
                    continue
                if limit is not None and count + len(batch) >= limit:
                    break
                corpus_digest.update(json.dumps([row["id"], row["title"], row["text"]], ensure_ascii=False).encode())
                batch.append(row)
                if len(batch) >= batch_size:
                    flush(batch)
                    batch = []
                    if count % (batch_size * 100) == 0:
                        print(f"Indexed {count} passages", flush=True)
            flush(batch)
        if not count:
            raise ValueError("Corpus contains no valid passages")
        faiss.write_index(index, str(output / "index.faiss"))
        metadata = {"num_passages": count, "source": str(source.resolve()),
                    "source_size_bytes": source.stat().st_size, "corpus_rows_sha256": corpus_digest.hexdigest(),
                    "embedding": embedder.signature(), "index_type": "IndexFlatIP",
                    "debug_truncated_corpus": limit is not None}
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return metadata
    finally:
        db.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--embedding-model", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--limit", type=int, help="Explicit small DEBUG corpus only")
    args = p.parse_args()
    embedder = Embedder(args.embedding_model, device=args.device)
    result = build_index(args.corpus, args.output, embedder, args.batch_size, args.limit)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
