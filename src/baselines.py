import os
import sys
import re
import time

_LONGFORM_SYSTEM = (
    "Answer the question comprehensively, covering all relevant aspects "
    "and interpretations. Write a detailed paragraph-length answer."
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm import LLM, Embedder
from src.retrieval import DenseRetriever, BM25Retriever, make_retriever
from src.tree import RetrievalDecisionTree
from src.prompts import (
    STANDARD_RAG,
    CRITIQUE,
    REFINE,
    EXTRACT_SHORT_ANSWER,
    MAKE_SEARCH_QUERY,
    fmt,
)


def _safe_doc(d, i: int = 0) -> dict:

    if isinstance(d, dict):
        return d
    elif isinstance(d, (list, tuple)) and len(d) >= 2:
        return {"title": str(d[0]), "text": str(d[1])}
    elif isinstance(d, str):
        return {"title": f"Document {i+1}", "text": d}
    else:
        return {"title": f"Document {i+1}", "text": str(d)}


def _format_docs(docs: list, max_docs: int = 5) -> str:

    if not docs:
        return "(No documents retrieved.)"
    lines = []
    for i, d in enumerate(docs[:max_docs]):
        if isinstance(d, dict):
            title = d.get("title", f"Document {i+1}")
            text  = d.get("text", "").strip()
        elif isinstance(d, (list, tuple)) and len(d) >= 2:
            title = str(d[0])
            text  = str(d[1]).strip()
        else:
            print(f"  [WARN] Skipping malformed doc in baseline (type={type(d).__name__})")
            continue
        if text:
            lines.append(f"[{title}]\n{text}")
    return "\n\n".join(lines) if lines else "(No documents retrieved.)"


def _extract_short(llm: LLM, question: str, detailed: str) -> str:

    if not detailed or not detailed.strip():
        return ""

    short = llm(fmt(EXTRACT_SHORT_ANSWER, question=question,
                    evidence="(No direct evidence available.)",
                    detailed_answer=detailed))

    if short and short.strip():
        short = short.strip().strip('"\'')
        if short.lower().startswith("answer:"):
            short = short[len("answer:"):].strip()
        for prefix in ("the answer is", "based on", "according to"):
            if short.lower().startswith(prefix):
                short = short[len(prefix):].strip().lstrip(":,").strip()
                break
        if short.endswith(".") and "." not in short[:-1]:
            short = short[:-1]
        return short.strip()


    m = re.search(r"[.!?]", detailed)
    return (detailed[: m.start()].strip() if m else detailed[:100].strip())


def _build_retriever(cfg, embedder: Embedder, docs: list):

    retriever = make_retriever(cfg, embedder)
    retriever.build(docs)
    return retriever


def _new_query_from_critique(llm: LLM, question: str, critique: str) -> str:

    synthesis = (
        f"The original question is: {question}\n"
        f"The current answer is missing: {critique}\n"
        f"Generate a search query to find the missing information."
    )
    resp = llm.json(fmt(MAKE_SEARCH_QUERY, question=synthesis))
    if "_error" not in resp and "query" in resp:
        q = str(resp["query"]).strip()
        if q:
            return q

    tokens = re.sub(r"[^\w\s]", " ", critique.lower()).split()
    stop = {"what", "who", "where", "when", "why", "how", "the", "a", "an",
            "is", "are", "was", "were", "does", "did", "has", "have"}
    keywords = [t for t in tokens if t not in stop]
    return " ".join(keywords[:8]) if keywords else question


class StandardRAG:


    name = "standard_rag"

    def __init__(self, llm: LLM, embedder: Embedder, cfg):
        self.llm      = llm
        self.embedder = embedder
        self.cfg      = cfg

    def run(self, question: str, docs: list,
            task_type: str = "shortform") -> dict:

        self.llm.reset_call_count()
        t0 = time.time()


        top_k = int(getattr(self.cfg.retrieval, "top_k", 10))
        retriever = _build_retriever(self.cfg, self.embedder, docs)
        retrieved = retriever.search(question, top_k=top_k)
        context   = _format_docs(retrieved, max_docs=5)


        if task_type == "longform":
            answer = self.llm(
                fmt(STANDARD_RAG, question=question, context=context),
                system=_LONGFORM_SYSTEM,
            )
        else:
            answer = self.llm(fmt(STANDARD_RAG, question=question, context=context))

        if not answer or not answer.strip():
            print(f"[StandardRAG] Warning: LLM returned empty answer.")
            answer = ""


        if task_type == "shortform":
            final_answer = _extract_short(self.llm, question, answer)
        else:
            final_answer = answer

        elapsed = round(time.time() - t0, 2)
        print(f"[StandardRAG] Done in {elapsed}s | "
              f"{len(retrieved)} docs | answer={final_answer[:80]!r}")

        return {
            "answer":          final_answer,
            "detailed_answer": answer,
            "task_type":       task_type,
            "context_docs":    len(retrieved),
            "num_llm_calls":   self.llm.count_calls(),
            "elapsed_seconds": elapsed,
        }


class IterativeRAG:


    name = "iterative_rag"

    def __init__(self, llm: LLM, embedder: Embedder, cfg):
        self.llm      = llm
        self.embedder = embedder
        self.cfg      = cfg

    def run(self, question: str, docs: list, max_rounds: int = 3,
            task_type: str = "shortform") -> dict:

        self.llm.reset_call_count()
        t0 = time.time()
        top_k = int(getattr(self.cfg.retrieval, "top_k", 10))


        cfg_rounds = getattr(getattr(self.cfg, "gbtrag", None), "max_rounds", None)
        if cfg_rounds is not None:
            max_rounds = int(cfg_rounds)

        retriever    = _build_retriever(self.cfg, self.embedder, docs)
        all_retrieved: list = []
        llm_calls    = 0


        initial_docs = retriever.search(question, top_k=top_k)
        all_retrieved.extend(initial_docs)
        context  = _format_docs(initial_docs, max_docs=5)
        if task_type == "longform":
            answer = self.llm(
                fmt(STANDARD_RAG, question=question, context=context),
                system=_LONGFORM_SYSTEM,
            )
        else:
            answer = self.llm(fmt(STANDARD_RAG, question=question, context=context))
        llm_calls += 1
        if not answer or not answer.strip():
            print("[IterativeRAG] Warning: Round-0 answer is empty.")
            answer = ""

        print(f"[IterativeRAG] Round 0 answer ({len(answer)} chars).")


        for round_num in range(1, max_rounds):
            if not answer.strip():
                print(f"[IterativeRAG] Round {round_num}: skipping — no answer to critique.")
                break


            critique = self.llm(fmt(CRITIQUE, question=question, answer=answer))
            llm_calls += 1
            if not critique or not critique.strip():
                print(f"[IterativeRAG] Round {round_num}: empty critique — stopping.")
                break


            if "complete and correct" in critique.lower():
                print(f"[IterativeRAG] Round {round_num}: critique says answer is complete.")
                break

            print(f"[IterativeRAG] Round {round_num} critique: {critique[:80]}...")


            new_query = _new_query_from_critique(self.llm, question, critique)
            llm_calls += 1
            print(f"[IterativeRAG] Round {round_num} new query: {new_query!r}")

            new_docs = retriever.search(new_query, top_k=top_k)
            if not new_docs:
                print(f"[IterativeRAG] Round {round_num}: no new docs — stopping.")
                break

            all_retrieved.extend(new_docs)


            seen_titles: set = set()
            unique_docs: list = []
            for d in all_retrieved:
                t = _safe_doc(d).get("title", "")
                if t not in seen_titles:
                    seen_titles.add(t)
                    unique_docs.append(d)
            all_retrieved = unique_docs


            full_context = _format_docs(all_retrieved, max_docs=8)
            if task_type == "longform":
                answer = self.llm(
                    fmt(STANDARD_RAG, question=question, context=full_context),
                    system=_LONGFORM_SYSTEM,
                )
            else:
                answer = self.llm(
                    fmt(STANDARD_RAG, question=question, context=full_context)
                )
            llm_calls += 1
            if not answer or not answer.strip():
                print(f"[IterativeRAG] Round {round_num}: empty answer — keeping previous.")

                break
            print(f"[IterativeRAG] Round {round_num} answer ({len(answer)} chars).")


        if task_type == "shortform":
            final_answer = _extract_short(self.llm, question, answer)
            llm_calls += 1
        else:
            final_answer = answer

        elapsed = round(time.time() - t0, 2)
        print(f"[IterativeRAG] Done in {elapsed}s | "
              f"{len(all_retrieved)} unique docs | answer={final_answer[:80]!r}")

        return {
            "answer":           final_answer,
            "detailed_answer":  answer,
            "task_type":        task_type,
            "num_rounds":       max_rounds,
            "total_docs":       len(all_retrieved),
            "num_llm_calls":    llm_calls,
            "elapsed_seconds":  elapsed,
        }


class SingleTreeRAG:


    name = "single_tree"

    def __init__(self, llm: LLM, embedder: Embedder, cfg):
        self.llm      = llm
        self.embedder = embedder
        self.cfg      = cfg

    def run(self, question: str, docs: list,
            task_type: str = "shortform") -> dict:

        self.llm.reset_call_count()
        t0 = time.time()

        gbtrag_cfg = self.cfg.gbtrag
        top_k      = int(getattr(self.cfg.retrieval, "top_k", 10))

        retriever = _build_retriever(self.cfg, self.embedder, docs)


        tree = RetrievalDecisionTree(
            tree_id=0,
            llm=self.llm,
            embedder=self.embedder,
            retriever=retriever,
            max_children=int(gbtrag_cfg.max_children),
        )
        tree._objective = question
        tree.build(
            objective=question,
            max_depth=int(gbtrag_cfg.max_tree_depth_round0),
            context="",
        )
        tree.execute(top_k=top_k)
        detailed = tree.aggregate()
        stats    = tree.stats()

        if not detailed or not detailed.strip():
            print("[SingleTreeRAG] Warning: tree aggregation returned empty answer.")
            detailed = ""

        print(f"[SingleTreeRAG] Tree built: {stats}")


        if task_type == "shortform":
            final_answer = _extract_short(self.llm, question, detailed)
        else:
            final_answer = detailed

        elapsed = round(time.time() - t0, 2)
        print(f"[SingleTreeRAG] Done in {elapsed}s | answer={final_answer[:80]!r}")

        return {
            "answer":          final_answer,
            "detailed_answer": detailed,
            "task_type":       task_type,
            "tree_stats":      stats,
            "num_llm_calls":   self.llm.count_calls() if hasattr(self.llm, "_call_count") else -1,
            "elapsed_seconds": elapsed,
        }


class SelfRefineRAG:


    name = "self_refine"

    def __init__(self, llm: LLM, embedder: Embedder, cfg):
        self.llm      = llm
        self.embedder = embedder
        self.cfg      = cfg

    def run(self, question: str, docs: list, max_rounds: int = 3,
            task_type: str = "shortform") -> dict:

        self.llm.reset_call_count()
        t0 = time.time()
        top_k = int(getattr(self.cfg.retrieval, "top_k", 10))


        cfg_rounds = getattr(getattr(self.cfg, "gbtrag", None), "max_rounds", None)
        if cfg_rounds is not None:
            max_rounds = int(cfg_rounds)


        retriever = _build_retriever(self.cfg, self.embedder, docs)
        retrieved = retriever.search(question, top_k=top_k)
        context   = _format_docs(retrieved, max_docs=5)

        if task_type == "longform":
            answer = self.llm(
                fmt(STANDARD_RAG, question=question, context=context),
                system=_LONGFORM_SYSTEM,
            )
        else:
            answer = self.llm(fmt(STANDARD_RAG, question=question, context=context))
        llm_calls = 1
        if not answer or not answer.strip():
            print("[SelfRefineRAG] Warning: Round-0 answer is empty.")
            answer = ""

        print(f"[SelfRefineRAG] Round 0 answer ({len(answer)} chars).")

        rounds_done = 1


        for round_num in range(1, max_rounds):
            if not answer.strip():
                print(f"[SelfRefineRAG] Round {round_num}: no answer to critique — stopping.")
                break


            critique = self.llm(fmt(CRITIQUE, question=question, answer=answer))
            llm_calls += 1
            if not critique or not critique.strip():
                print(f"[SelfRefineRAG] Round {round_num}: empty critique — stopping.")
                break

            if "complete and correct" in critique.lower():
                print(f"[SelfRefineRAG] Round {round_num}: answer judged complete.")
                break

            print(f"[SelfRefineRAG] Round {round_num} critique: {critique[:80]}...")


            refine_sys = _LONGFORM_SYSTEM if task_type == "longform" else None
            refined = self.llm(
                fmt(REFINE, question=question, answer=answer,
                    critique=critique, context=context),
                system=refine_sys,
            )
            llm_calls += 1

            if not refined or not refined.strip():
                print(f"[SelfRefineRAG] Round {round_num}: empty refinement — "
                      "keeping previous answer.")
                break


            if len(refined) < len(answer) * 0.4:
                print(f"[SelfRefineRAG] Round {round_num}: refinement suspiciously "
                      f"short ({len(refined)} vs {len(answer)}) — keeping current.")
                break

            answer = refined.strip()
            rounds_done += 1
            print(f"[SelfRefineRAG] Round {round_num} refined answer ({len(answer)} chars).")


        if task_type == "shortform":
            final_answer = _extract_short(self.llm, question, answer)
            llm_calls += 1
        else:
            final_answer = answer

        elapsed = round(time.time() - t0, 2)
        print(f"[SelfRefineRAG] Done in {elapsed}s | {rounds_done} rounds | "
              f"answer={final_answer[:80]!r}")

        return {
            "answer":          final_answer,
            "detailed_answer": answer,
            "task_type":       task_type,
            "num_rounds":      rounds_done,
            "context_docs":    len(retrieved),
            "num_llm_calls":   llm_calls,
            "elapsed_seconds": elapsed,
        }


def get_method(name: str, llm, embedder, cfg):


    from src.gbtrag import GBTRAG

    methods = {
        "gbtrag":        lambda: GBTRAG(cfg),
        "standard_rag":  lambda: StandardRAG(llm, embedder, cfg),
        "iterative_rag": lambda: IterativeRAG(llm, embedder, cfg),
        "single_tree":   lambda: SingleTreeRAG(llm, embedder, cfg),
        "self_refine":   lambda: SelfRefineRAG(llm, embedder, cfg),
    }

    if name not in methods:
        raise KeyError(
            f"Unknown method '{name}'. "
            f"Available: {sorted(methods.keys())}"
        )

    return methods[name]()


if __name__ == "__main__":
    import json
    from omegaconf import OmegaConf


    class MockLLM:
        _call_count = 0

        def __call__(self, prompt: str, system: str = None) -> str:
            MockLLM._call_count += 1
            p = prompt.lower()
            if "concise" in p or "direct answer" in p or "short answer" in p:
                return "yes"
            if "critique" in p or "missing" in p:
                return "The answer does not mention the founding year."
            if "improve" in p or "refine" in p:
                return "A more detailed answer covering all aspects of the question."
            return "The answer to the question is that Tesla focuses on pricing."

        def json(self, prompt: str, system: str = None) -> dict:
            MockLLM._call_count += 1
            p = prompt.lower()
            if '"query"' in p or "search query" in p:
                return {"query": "Tesla China strategy pricing"}
            return {}

        def count_calls(self) -> int:
            return MockLLM._call_count

    class MockEmbedder:
        embed_dim = 64

        def encode(self, texts, batch_size=64):
            import numpy as np
            if isinstance(texts, str):
                texts = [texts]
            n = len(texts)
            vecs = np.random.randn(n, self.embed_dim).astype("float32")
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs / (norms + 1e-9)

    class MockDenseRetriever:
        def build(self, docs):
            self._docs = docs

        def search(self, query, top_k=10):
            return [
                {"title": "Tesla China", "text": "Tesla cut prices 20% in Q1 2024.", "score": 0.9},
                {"title": "BYD",         "text": "BYD deliveries exceeded Tesla in 2023.", "score": 0.8},
                {"title": "EV Market",   "text": "China EV market grew 40% YoY.", "score": 0.7},
            ]


    import src.baselines as _bmod
    _orig_make = _bmod.make_retriever
    _bmod.make_retriever = lambda cfg, emb: MockDenseRetriever()

    cfg = OmegaConf.create({
        "llm":       {"base_url": "http://localhost:8000/v1", "model": "mock",
                      "temperature": 0.0, "max_tokens": 512, "timeout": 30},
        "embedding": {"model_path": "./models/bge-base-en-v1.5", "device": "cpu"},
        "reranker":  {"enabled": False, "model_path": "./models/bge-reranker-v2-m3",
                      "device": "cpu", "top_k": 5},
        "gbtrag":    {"max_rounds": 2, "coverage_threshold": 0.65,
                      "convergence_epsilon": 0.02,
                      "max_tree_depth_round0": 1, "max_tree_depth_residual": 1,
                      "max_children": 2, "enable_dimension_discovery": False},
        "retrieval": {"top_k": 5, "method": "dense"},
        "experiment":{"cache_dir": None},
    })

    llm = MockLLM()
    emb = MockEmbedder()

    SAMPLE_DOCS = [
        {"title": "Tesla China", "text": "Tesla cut prices 20% in Q1 2024."},
        {"title": "BYD Growth",  "text": "BYD deliveries exceeded Tesla in Q3 2023."},
        {"title": "EV Market",   "text": "China EV market grew 40% year-over-year."},
    ]
    QUESTION = "Were Scott Derrickson and Ed Wood of the same nationality?"

    print("=" * 60)
    print("Testing baselines (mock components)")
    print("=" * 60)


    print("\n--- StandardRAG ---")
    srag = StandardRAG(llm, emb, cfg)
    r1 = srag.run(QUESTION, SAMPLE_DOCS)
    assert isinstance(r1, dict) and "answer" in r1
    assert isinstance(r1["answer"], str)
    print(f"  answer={r1['answer']!r}  calls={r1['num_llm_calls']}")


    print("\n--- IterativeRAG ---")
    irag = IterativeRAG(llm, emb, cfg)
    r2 = irag.run(QUESTION, SAMPLE_DOCS, max_rounds=2)
    assert isinstance(r2, dict) and "answer" in r2
    assert "num_rounds" in r2
    print(f"  answer={r2['answer']!r}  rounds={r2['num_rounds']}  "
          f"calls={r2['num_llm_calls']}")


    class MockTree:
        def __init__(self, *a, **kw):
            self._objective = ""
        def build(self, objective, max_depth, context=""):
            self._objective = objective
        def execute(self, top_k=10):
            pass
        def aggregate(self):
            return "Mock tree answer about Tesla pricing and market share."
        def get_answer(self):
            return "Mock tree answer about Tesla pricing and market share."
        def stats(self):
            return {"tree_id": 0, "num_nodes": 3, "num_leaves": 2,
                    "max_depth": 1, "avg_evidence_length": 50.0}
        def print_tree(self, *a, **kw):
            pass

    import src.baselines as _bmod2
    _orig_tree = _bmod2.RetrievalDecisionTree
    _bmod2.RetrievalDecisionTree = MockTree

    print("\n--- SingleTreeRAG ---")
    strag = SingleTreeRAG(llm, emb, cfg)
    r3 = strag.run(QUESTION, SAMPLE_DOCS)
    assert isinstance(r3, dict) and "answer" in r3
    assert "tree_stats" in r3
    print(f"  answer={r3['answer']!r}  tree={r3['tree_stats']}")

    _bmod2.RetrievalDecisionTree = _orig_tree


    print("\n--- SelfRefineRAG ---")
    srrag = SelfRefineRAG(llm, emb, cfg)
    r4 = srrag.run(QUESTION, SAMPLE_DOCS, max_rounds=2)
    assert isinstance(r4, dict) and "answer" in r4
    assert "num_rounds" in r4
    print(f"  answer={r4['answer']!r}  rounds={r4['num_rounds']}  "
          f"calls={r4['num_llm_calls']}")


    print("\n--- get_method factory ---")
    for mname in ("standard_rag", "iterative_rag", "single_tree", "self_refine"):
        m = get_method(mname, llm, emb, cfg)
        assert hasattr(m, "run"), f"{mname} missing .run()"
        print(f"  {mname}: {type(m).__name__}  OK")

    try:
        get_method("nonexistent", llm, emb, cfg)
        assert False, "Should have raised KeyError"
    except KeyError as e:
        print(f"  Unknown method KeyError: {e}  OK")


    _bmod.make_retriever = _orig_make


    print("\n--- JSON serialisability ---")
    for label, result in [("standard_rag", r1), ("iterative_rag", r2),
                           ("single_tree", r3), ("self_refine", r4)]:
        s = json.dumps(result)
        print(f"  {label}: {len(s)} bytes  OK")

    print("\nAll baselines tests PASSED.")
