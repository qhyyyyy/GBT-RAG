import json
import re
import sqlite3
from pathlib import Path

import numpy as np


def _clean_docs(docs):
    result = []
    for doc in docs:
        if isinstance(doc, dict) and str(doc.get("text", "")).strip():
            result.append({**doc, "title": str(doc.get("title", "")),
                           "text": " ".join(str(doc["text"]).split())})
    return result


def _query_encoding(embedder, texts):
    return embedder.encode_queries(texts) if hasattr(embedder, "encode_queries") else embedder.encode(texts)


def _passage_encoding(embedder, texts):
    return embedder.encode_passages(texts) if hasattr(embedder, "encode_passages") else embedder.encode(texts)


class DenseRetriever:
    def __init__(self, embedder):
        self.embedder = embedder
        self.index = None
        self.docs = []
        self.search_calls = 0

    def build(self, docs):
        import faiss
        self.docs = _clean_docs(docs)
        self.search_calls = 0
        self.index = None
        if self.docs:
            vectors = _passage_encoding(self.embedder, [d["text"] for d in self.docs])
            self.index = faiss.IndexFlatIP(vectors.shape[1])
            self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    def search(self, query, top_k=10):
        self.search_calls += 1
        if not query or top_k <= 0:
            raise ValueError("query and positive top_k are required")
        if self.index is None:
            return []
        vectors = np.ascontiguousarray(_query_encoding(self.embedder, query), dtype=np.float32)
        scores, indices = self.index.search(vectors, min(top_k, len(self.docs)))
        return [{**self.docs[int(i)], "score": float(s)} for s, i in zip(scores[0], indices[0]) if i >= 0]


class BM25Retriever:
    def __init__(self):
        self.docs = []
        self.bm25 = None
        self.search_calls = 0

    def build(self, docs):
        from rank_bm25 import BM25Okapi
        self.docs = _clean_docs(docs)
        self.search_calls = 0
        self.bm25 = BM25Okapi([re.findall(r"\w+", d["text"].lower()) for d in self.docs]) if self.docs else None

    def search(self, query, top_k=10):
        self.search_calls += 1
        if not query or top_k <= 0:
            raise ValueError("query and positive top_k are required")
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(re.findall(r"\w+", query.lower()))
        indices = np.argsort(-scores, kind="stable")[:top_k]
        return [{**self.docs[int(i)], "score": float(scores[i])} for i in indices]


class GlobalDenseRetriever:

    def __init__(self, embedder, index_dir):
        import faiss
        self.embedder = embedder
        root = Path(index_dir)
        required = [root / name for name in ["index.faiss", "passages.sqlite", "metadata.json"]]
        missing = [str(p) for p in required if not p.is_file()]
        if missing:
            raise FileNotFoundError("Global Wikipedia index is missing: " + ", ".join(missing))
        self.metadata = json.loads((root / "metadata.json").read_text())
        expected = self.metadata["embedding"]
        actual = embedder.signature()


        keys = ["dimension", "query_prefix", "passage_prefix", "normalized", "mock", "max_seq_length", "config_hashes", "weight_files_sha256"]
        if any(expected.get(key) != actual.get(key) for key in keys):
            raise ValueError("Index/encoder mismatch; rebuild or use the encoder that created this index")
        self.index = faiss.read_index(str(root / "index.faiss"))
        if self.index.metric_type != faiss.METRIC_INNER_PRODUCT or not isinstance(self.index, faiss.IndexFlatIP):
            raise ValueError("Paper configuration requires an exact FAISS IndexFlatIP")
        if self.index.d != actual["dimension"] or self.index.ntotal != self.metadata["num_passages"]:
            raise ValueError("Index dimensions or passage count disagree with metadata")
        self.db = sqlite3.connect((root / "passages.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
        count = self.db.execute("SELECT count(*) FROM passages").fetchone()[0]
        if count != self.index.ntotal:
            raise ValueError("Passage database and FAISS index are not aligned")
        self.search_calls = 0

    def build(self, docs=None):

        return None

    def search(self, query, top_k=10):
        self.search_calls += 1
        if not query or top_k <= 0:
            raise ValueError("query and positive top_k are required")
        vectors = _query_encoding(self.embedder, query)
        scores, indices = self.index.search(vectors, min(top_k, self.index.ntotal))
        result = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            record = self.db.execute("SELECT source_id,title,text FROM passages WHERE row_id=?", (int(idx),)).fetchone()
            if record is None:
                raise ValueError(f"Missing passage row {idx}")
            result.append({"id": record[0], "title": record[1], "text": record[2], "score": float(score)})
        return result


class Reranker:
    def __init__(self, model_path, device="auto"):
        if not Path(model_path).is_dir():
            raise FileNotFoundError(f"Reranker model not found: {model_path}; disable reranker explicitly if unavailable")
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(str(model_path), device=None if device == "auto" else device,
                                  local_files_only=True, trust_remote_code=False)

    def rerank(self, query, docs, top_k=5):
        if not docs:
            return []
        scores = np.asarray(self.model.predict([(query, d["text"]) for d in docs])).reshape(-1)
        if len(scores) != len(docs) or not np.isfinite(scores).all():
            raise ValueError("Invalid reranker output")
        indices = np.argsort(-scores, kind="stable")[:top_k]
        return [{**docs[int(i)], "rerank_score": float(scores[i])} for i in indices]


_GLOBAL_CACHE = {}


def make_retriever(cfg, embedder):
    method = str(cfg.retrieval.method).lower()
    scope = str(getattr(cfg.retrieval, "scope", "global"))
    if scope == "global":
        if method != "dense":
            raise ValueError("Global mode currently supports dense retrieval; BM25 is available in candidate_debug mode")
        index_dir = str(cfg.retrieval.index_dir)
        signature = json.dumps(embedder.signature(), sort_keys=True)
        key = (str(Path(index_dir).resolve()), signature)
        if key not in _GLOBAL_CACHE:
            _GLOBAL_CACHE[key] = GlobalDenseRetriever(embedder, index_dir)
        retriever = _GLOBAL_CACHE[key]
        if retriever.metadata.get("debug_truncated_corpus") and not bool(getattr(cfg.retrieval, "allow_debug_corpus", False)):
            raise ValueError("A truncated debug corpus cannot be used as the paper's Wikipedia corpus; set retrieval.allow_debug_corpus=true explicitly for testing")
        return retriever
    if scope != "candidate_debug":
        raise ValueError("retrieval.scope must be global or candidate_debug")
    if method == "dense":
        return DenseRetriever(embedder)
    if method == "bm25":
        return BM25Retriever()
    raise ValueError(f"Unknown retrieval method: {method}")
