import os
import sys
import json
import re
import uuid
from time import perf_counter
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.llm import LLM, Embedder
from src.prompts import (
    SPLIT_QUESTION,
    LEAF_DECISION,
    MAKE_SEARCH_QUERY,
    LEAF_QUERY_FACTOID,
    EXTRACT_EVIDENCE,
    RETRY_LEAF_QUERY,
    AGGREGATE,
    AGGREGATE_ANCHORED,
    AGGREGATE_ANCHORED_LONGFORM,
    fmt,
)


_DIMENSION_ATTRS = ("name", "ideal", "sub_aspects")


@dataclass
class Node:

    id: str
    question: str
    node_type: str
    depth: int

    children: list = field(default_factory=list)


    search_query: str = ""
    retrieved_docs: list = field(default_factory=list)
    evidence: str = ""


    partial_answer: str = ""
    partial_answer_embed: np.ndarray = field(default=None, repr=False)

    def is_leaf(self) -> bool:
        return self.node_type == "LEAF"

    def is_split(self) -> bool:
        return self.node_type == "SPLIT"


class RetrievalDecisionTree:


    def __init__(
        self,
        tree_id: int,
        llm: LLM,
        embedder: Embedder,
        retriever,
        max_children: int = 4,
        force_root_split: bool = True,
        deterministic_tree: bool = True,
        use_anchored_aggregation: bool = False,
        use_longform_anchored_aggregation: bool = False,
        use_factoid_queries: bool = False,
        aggregation_target_words: int = 80,
        query_history=None,
    ):

        self.tree_id = tree_id
        self.llm = llm
        self.embedder = embedder
        self.retriever = retriever
        self.max_children = max_children
        self.force_root_split = force_root_split
        self.deterministic_tree = deterministic_tree
        self.use_anchored_aggregation = use_anchored_aggregation
        self.use_longform_anchored_aggregation = use_longform_anchored_aggregation
        self.use_factoid_queries = use_factoid_queries
        self.aggregation_target_words = aggregation_target_words
        self.query_history = query_history if query_history is not None else {}
        self.timings = dict(query_generation_seconds=0.0, retrieval_seconds=0.0,
                            rerank_seconds=0.0, evidence_seconds=0.0, aggregation_seconds=0.0)

        self.root: Node = None
        self.nodes: dict = {}
        self._counter: int = 0


    def _next_id(self) -> str:
        nid = f"t{self.tree_id}_n{self._counter}"
        self._counter += 1
        return nid


    def build(
        self,
        objective: str,
        max_depth: int,
        context: str = "",
        dimensions: list = None,
        sub_aspects_per_dim: int = 3,
        forced_leaf_questions: list = None,
    ) -> None:

        use_deterministic = (
            self.deterministic_tree
            and dimensions
            and len(dimensions) >= 1
        )
        mode = "deterministic" if use_deterministic else (
            "forced-leaves" if (forced_leaf_questions and len(forced_leaf_questions) >= 2)
            else "recursive"
        )
        print(f"[Tree {self.tree_id}] Building with objective: {objective[:80]}...")
        print(f"[Tree {self.tree_id}] mode={mode}, force_root_split={self.force_root_split}")

        self._counter = 0
        self.nodes = {}

        root_id = self._next_id()
        self.root = Node(id=root_id, question=objective, node_type="SPLIT", depth=0)
        self.nodes[root_id] = self.root

        if use_deterministic:
            self._build_deterministic(dimensions, sub_aspects_per_dim)
        elif forced_leaf_questions and len(forced_leaf_questions) >= 2:
            self._build_from_forced_leaves(forced_leaf_questions)
        else:
            self._build_recursive(self.root, max_depth, context)


        leaves = self.get_leaves()
        s = self.stats()
        n = len(self.nodes)


        if use_deterministic:
            assert s["max_depth"] >= 2, (
                f"Tree 0 deterministic build failed: max_depth={s['max_depth']} < 2. "
                f"Check that dimensions have sub_aspects. "
                f"({len(leaves)} leaves, {n} nodes)"
            )


        print(f"[Tree {self.tree_id}] Build complete: {n} nodes, {len(leaves)} leaves.")
        if self.tree_id == 0 and not use_deterministic and n < 3:
            print(
                f"WARNING: Tree 0 has only {n} nodes — decomposition may have "
                f"failed. This will degrade GBTRAG to single-retrieval baseline."
            )
        self.print_tree()


    def _build_deterministic(self, dimensions: list, sub_aspects_per_dim: int) -> None:

        dim_list = dimensions

        for dim in dim_list:

            if all(hasattr(dim, a) for a in _DIMENSION_ATTRS):

                dim_name     = str(dim.name).strip()
                dim_ideal    = str(dim.ideal).strip()
                sub_aspects  = [str(s).strip() for s in (dim.sub_aspects or []) if str(s).strip()]
            elif isinstance(dim, dict):
                dim_name     = str(dim.get("name", "")).strip()
                dim_ideal    = str(dim.get("ideal", dim_name)).strip()
                sub_aspects  = [str(s).strip() for s in dim.get("sub_aspects", []) if str(s).strip()]
            else:
                dim_name    = str(dim).strip()
                dim_ideal   = dim_name
                sub_aspects = []


            if not sub_aspects:
                sub_aspects = [dim_ideal or dim_name]


            dim_question = f"{dim_ideal}" if dim_ideal else dim_name
            dim_id  = self._next_id()
            dim_node = Node(
                id=dim_id,
                question=dim_question,
                node_type="SPLIT",
                depth=1,
            )
            self.root.children.append(dim_node)
            self.nodes[dim_id] = dim_node


            for aspect_index, aspect in enumerate(sub_aspects):
                coordinate_ids = getattr(dim, "leaf_ids", [])
                leaf_id = coordinate_ids[aspect_index] if coordinate_ids else self._next_id()
                leaf = Node(
                    id=leaf_id,
                    question=str(aspect),
                    node_type="LEAF",
                    depth=2,
                )
                self._make_leaf(leaf, "    ", reason="sub_aspect")
                dim_node.children.append(leaf)
                self.nodes[leaf_id] = leaf

            print(
                f"  [SPLIT d=1] Dimension '{dim_name}' "
                f"→ {len(dim_node.children)} sub-aspect leaf(ves)"
            )

    def _build_from_forced_leaves(self, leaf_questions: list) -> None:

        for q in leaf_questions:
            child_id = self._next_id()
            child = Node(
                id=child_id,
                question=str(q),
                node_type="LEAF",
                depth=1,
            )
            self._make_leaf(child, "  ", reason="forced_sub_aspect")
            self.root.children.append(child)
            self.nodes[child_id] = child

        print(
            f"  [SPLIT d=0] Forced {len(leaf_questions)} sub-aspect leaves "
            f"for single-dimension residual tree."
        )


    def _build_recursive(self, node: Node, max_depth: int, context: str) -> None:

        depth = node.depth
        indent = "  " * depth


        if depth >= max_depth:
            self._make_leaf(node, indent, reason="max_depth")
            return


        if depth == 0 and self.force_root_split:
            is_leaf = False
        else:


            leaf_decision = self.llm.json(
                fmt(LEAF_DECISION, question=node.question,
                    depth=depth, max_depth=max_depth)
            )


            if not isinstance(leaf_decision, dict):
                leaf_decision = {"_error": "Expected object"}
            is_leaf = bool(leaf_decision.get("is_leaf", True))
            if "_error" in leaf_decision:
                print(f"{indent}  [Tree {self.tree_id}] LEAF_DECISION parse failed "
                      f"— defaulting to LEAF.")
                is_leaf = True

        if is_leaf:
            self._make_leaf(node, indent, reason="leaf_decision")
            return


        split_response = self.llm.json(
            fmt(SPLIT_QUESTION, question=node.question, context=context or "None")
        )

        sub_questions = []
        if isinstance(split_response, dict) and "_error" not in split_response:
            raw_subs = split_response.get("sub_questions", [])
            if isinstance(raw_subs, list):
                sub_questions = [
                    str(q).strip() for q in raw_subs if str(q).strip()
                ]


        if depth == 0 and self.force_root_split and len(sub_questions) < 2:
            print(f"{indent}  [Tree {self.tree_id}] Root returned fewer than 2 "
                  f"sub-questions ({len(sub_questions)}) — retrying with stronger prompt.")
            stronger_prompt = (
                fmt(SPLIT_QUESTION, question=node.question, context=context or "None")
                + "\n\nIMPORTANT: You MUST produce at least 2 distinct "
                "sub-questions that together cover the original question. "
                "Do not return a single sub-question."
            )
            retry_response = self.llm.json(stronger_prompt)
            if isinstance(retry_response, dict) and "_error" not in retry_response:
                raw_subs = retry_response.get("sub_questions", [])
                if isinstance(raw_subs, list):
                    sub_questions = [
                        str(q).strip() for q in raw_subs if str(q).strip()
                    ]

            if len(sub_questions) < 2:
                print(f"{indent}  [Tree {self.tree_id}] Retry still < 2 sub-questions "
                      f"— using heuristic entity split.")
                sub_questions = self._heuristic_split(node.question)

        if not sub_questions:

            print(f"{indent}  [Tree {self.tree_id}] SPLIT_QUESTION returned no "
                  f"sub-questions — falling back to LEAF.")
            self._make_leaf(node, indent, reason="split_fallback")
            return


        if len(sub_questions) > self.max_children:
            print(f"{indent}  [Tree {self.tree_id}] Clamping {len(sub_questions)} "
                  f"sub-questions to {self.max_children}.")
            sub_questions = sub_questions[: self.max_children]

        node.node_type = "SPLIT"
        print(f"{indent}[SPLIT d={depth}] {node.question[:70]}")

        for sub_q in sub_questions:
            child_id = self._next_id()
            child = Node(
                id=child_id,
                question=sub_q,
                node_type="LEAF",
                depth=depth + 1,
            )
            node.children.append(child)
            self.nodes[child_id] = child

            self._build_recursive(child, max_depth, context)

    def _heuristic_split(self, question: str) -> list:


        entity_prompt = (
            f"Identify the key entities in the following question and create one "
            f"sub-question per entity:\n\nQuestion: {question}\n\n"
            f'Return JSON only: {{"sub_questions": ["...", "..."]}}'
        )
        entity_response = self.llm.json(entity_prompt)
        if isinstance(entity_response, dict) and "_error" not in entity_response:
            raw_subs = entity_response.get("sub_questions", [])
            if isinstance(raw_subs, list):
                subs = [str(q).strip() for q in raw_subs if str(q).strip()]
                if len(subs) >= 2:
                    return subs


        tokens = [t for t in re.sub(r"[^\w\s]", " ", question).split() if t]
        mid = max(1, len(tokens) // 2)
        first_half = " ".join(tokens[:mid])
        second_half = " ".join(tokens[mid:]) if tokens[mid:] else " ".join(tokens)
        print(f"  [Tree {self.tree_id}] Using keyword-halving last resort for "
              f"root split.")
        return [
            f"What is {first_half}?",
            f"What about {second_half}?",
        ]

    def _make_leaf(self, node: Node, indent: str, reason: str = "") -> None:

        node.node_type = "LEAF"
        node.children = []

        query_started = perf_counter()
        previous = self.query_history.get(node.id)
        if previous:
            query_response = self.llm.json(fmt(RETRY_LEAF_QUERY,
                original_question=self.root.question, gap_description=node.question,
                original_query=previous["query"], failed_passages=previous.get("passages", "(none)")))
        elif self.use_factoid_queries:
            query_response = self.llm.json(
                fmt(LEAF_QUERY_FACTOID, sub_aspect_description=node.question)
            )
        else:
            query_response = self.llm.json(
                fmt(MAKE_SEARCH_QUERY, question=node.question)
            )

        if isinstance(query_response, dict) and "_error" not in query_response and "query" in query_response:
            node.search_query = str(query_response["query"]).strip()
        else:

            node.search_query = _fallback_query(node.question)
            prompt_name = "LEAF_QUERY_FACTOID" if self.use_factoid_queries else "MAKE_SEARCH_QUERY"
            print(f"{indent}  [Tree {self.tree_id}] {prompt_name} failed "
                  f"— using fallback query: {node.search_query!r}")

        if not node.search_query:
            node.search_query = _fallback_query(node.question)
        self.timings["query_generation_seconds"] += perf_counter() - query_started
        reason_tag = f" ({reason})" if reason else ""
        print(f"{indent}[LEAF  d={node.depth}]{reason_tag} "
              f"query={node.search_query!r}")


    def execute(self, top_k: int = 10, reranker=None, rerank_top_k: int = 5) -> None:

        leaves = self.get_leaves()
        print(f"\n[Tree {self.tree_id}] Executing retrieval for {len(leaves)} leaves...")

        for leaf in leaves:

            docs = []
            if leaf.search_query:
                retrieval_started = perf_counter()
                docs = self.retriever.search(leaf.search_query, top_k=top_k)
                self.timings["retrieval_seconds"] += perf_counter() - retrieval_started


            safe_docs = []
            for d in docs:
                if isinstance(d, dict):
                    safe_docs.append(d)
                elif isinstance(d, (list, tuple)) and len(d) >= 2:

                    safe_docs.append({"title": str(d[0]), "text": str(d[1])})
                else:
                    print(f"  [WARN] Skipping malformed doc: {type(d)}")
            docs = safe_docs

            if reranker is not None and docs:
                rerank_started = perf_counter()
                docs = reranker.rerank(leaf.search_query, docs, top_k=rerank_top_k)
                self.timings["rerank_seconds"] += perf_counter() - rerank_started
            leaf.retrieved_docs = docs

            if not docs:
                leaf.evidence = ""
                print(f"  [{leaf.id}] 0 docs retrieved for: {leaf.search_query!r}")
                continue


            doc_texts = "\n---\n".join(
                f"[{d.get('title', 'Unknown')}] {d.get('text', '')}"
                for d in docs
            )


            evidence_started = perf_counter()
            extraction = self.llm.json(
                fmt(EXTRACT_EVIDENCE, question=leaf.question, documents=doc_texts)
            )
            self.timings["evidence_seconds"] += perf_counter() - evidence_started

            if isinstance(extraction, dict) and "_error" not in extraction:
                relevant = extraction.get("relevant", True)
                if isinstance(relevant, str):
                    relevant = relevant.strip().lower() == "true"
                raw_evidence = extraction.get("evidence", "")
                evidence = "\n".join(x for x in raw_evidence if isinstance(x, str)) if isinstance(raw_evidence, list) else str(raw_evidence or "").strip()
                if relevant and evidence:
                    leaf.evidence = evidence
                else:
                    leaf.evidence = ""
            else:

                leaf.evidence = docs[0].get("text", "")[:500]
                print(f"  [Tree {self.tree_id}] EXTRACT_EVIDENCE parse failed for "
                      f"node {leaf.id} — using raw first doc text.")


            if leaf.evidence:
                source = [" ".join(d.get("text", "").split()) for d in docs]
                normalized = " ".join(leaf.evidence.split())
                if not any(normalized in text for text in source):
                    pieces = [re.sub(r"^[-*•]\s+", "", x.strip()).strip('"') for x in re.split(r"\n+|(?<=[.!?])\s+", leaf.evidence) if x.strip()]
                    literal = [x for x in pieces if any(" ".join(x.split()) in text for text in source)]
                    leaf.evidence = "\n".join(literal)
            print(
                f"  [{leaf.id}] {len(docs)} docs retrieved, "
                f"{len(leaf.evidence)} chars evidence | query={leaf.search_query!r}"
            )

        evidence_count = sum(1 for lf in leaves if lf.evidence)
        print(f"[Tree {self.tree_id}] Retrieved evidence for "
              f"{evidence_count}/{len(leaves)} leaves.")


    def aggregate(self) -> str:

        if self.root is None:
            print(f"[Tree {self.tree_id}] Warning: aggregate() called on empty tree.")
            return ""

        print(f"\n[Tree {self.tree_id}] Aggregating answers bottom-up...")
        aggregate_started = perf_counter()
        self._aggregate_node(self.root)
        self.timings["aggregation_seconds"] += perf_counter() - aggregate_started
        answer = self.root.partial_answer
        print(f"[Tree {self.tree_id}] Root answer ({len(answer)} chars): "
              f"{answer[:100]}{'...' if len(answer) > 100 else ''}")
        return answer

    def _aggregate_node(self, node: Node) -> None:

        if node.is_leaf():

            if node.evidence:
                node.partial_answer = node.evidence
            else:
                node.partial_answer = "No relevant information found."

        else:

            for child in node.children:
                self._aggregate_node(child)

            if not node.children:

                node.partial_answer = "No relevant information found."
            else:

                children_text = ""
                for child in node.children:
                    children_text += (
                        f"Q: {child.question}\n"
                        f"A: {child.partial_answer}\n\n"
                    )
                children_text = children_text.strip()

                if self.use_longform_anchored_aggregation:


                    synthesized = self.llm(
                        fmt(
                            AGGREGATE_ANCHORED_LONGFORM,
                            question=node.question,
                            children_qa=children_text,
                            target_words=self.aggregation_target_words,
                        )
                    )
                elif self.use_anchored_aggregation:

                    synthesized = self.llm(
                        fmt(
                            AGGREGATE_ANCHORED,
                            question=node.question,
                            children_qa=children_text,
                            target_words=self.aggregation_target_words,
                        )
                    )
                else:
                    synthesized = self.llm(
                        fmt(
                            AGGREGATE,
                            question=node.question,
                            children_qa=children_text,
                            target_words=self.aggregation_target_words,
                        )
                    )

                if synthesized.strip():
                    node.partial_answer = synthesized.strip()
                else:

                    print(f"  [Tree {self.tree_id}] AGGREGATE returned empty for "
                          f"node {node.id} — using concatenation fallback.")
                    node.partial_answer = children_text


        if self.embedder is not None and node.partial_answer:
            try:
                node.partial_answer_embed = self.embedder.encode(
                    node.partial_answer
                )[0]
            except Exception as e:
                print(f"  [Tree {self.tree_id}] Embedding failed for node "
                      f"{node.id}: {e}")
                node.partial_answer_embed = None


    def get_answer(self) -> str:

        return self.root.partial_answer if self.root else ""

    def get_leaves(self) -> list:

        return [n for n in self.nodes.values() if n.is_leaf()]

    def get_all_nodes(self) -> list:

        return list(self.nodes.values())

    def get_all_evidence(self) -> list:

        return [
            {"query": lf.search_query, "evidence": lf.evidence,
             "docs": lf.retrieved_docs}
            for lf in self.get_leaves()
            if lf.evidence
        ]


    def stats(self) -> dict:

        leaves = self.get_leaves()
        depths = [n.depth for n in self.nodes.values()]
        evidence_lengths = [len(lf.evidence) for lf in leaves if lf.evidence]
        return {
            "tree_id":              self.tree_id,
            "timings": dict(self.timings),
            "num_nodes":            len(self.nodes),
            "num_leaves":           len(leaves),
            "max_depth":            max(depths) if depths else 0,
            "avg_evidence_length":  (
                sum(evidence_lengths) / len(evidence_lengths)
                if evidence_lengths else 0.0
            ),
        }


    def to_dict(self) -> dict:

        def node_to_dict(n: Node) -> dict:
            return {
                "id":              n.id,
                "question":        n.question,
                "node_type":       n.node_type,
                "depth":           n.depth,
                "search_query":    n.search_query,
                "evidence":        n.evidence,
                "partial_answer":  n.partial_answer,
                "retrieved_docs":  [
                    {"title": d.get("title", ""), "score": d.get("score", 0.0)}
                    for d in n.retrieved_docs
                ],
                "children":        [node_to_dict(c) for c in n.children],
            }

        return {
            "tree_id":  self.tree_id,
            "stats":    self.stats(),
            "root":     node_to_dict(self.root) if self.root else None,
        }


    def print_tree(self, node: Node = None, indent: int = 0) -> None:

        if node is None:
            if self.root is None:
                print("[Tree] (empty)")
                return
            print(f"\n[Tree {self.tree_id}] Structure:")
            node = self.root

        prefix = "  " * indent

        if node.is_leaf():
            doc_count = len(node.retrieved_docs)
            ev_len = len(node.evidence)
            print(
                f"{prefix}[LEAF  d={node.depth}] "
                f"query={node.search_query!r} "
                f"→ {doc_count} docs, {ev_len} chars"
            )
        else:
            q_preview = node.question[:70]
            if len(node.question) > 70:
                q_preview += "..."
            print(f"{prefix}[SPLIT d={node.depth}] {q_preview}")
            for child in node.children:
                self.print_tree(child, indent + 1)


def _fallback_query(question: str) -> str:

    stop_words = {
        "what", "who", "where", "when", "why", "how", "which",
        "is", "are", "was", "were", "does", "did", "do", "has",
        "have", "had", "will", "would", "could", "should",
        "the", "a", "an", "in", "of", "to", "and", "or",
    }
    tokens = re.sub(r"[^\w\s]", " ", question.lower()).split()
    filtered = [t for t in tokens if t not in stop_words]

    query = " ".join(filtered[:8])
    return query if query else question


if __name__ == "__main__":


    class MockLLM:


        def __call__(self, prompt: str, system: str = None) -> str:
            p = prompt.lower()
            if "synthesize" in p or "children_qa" in p or "combine" in p:
                return "This is a synthesized answer combining all sub-answers."
            return "A factual answer to the question."

        def json(self, prompt: str, system: str = None) -> dict:
            p = prompt.lower()
            if "is_leaf" in p:

                if "current depth: 0" in p:
                    return {"is_leaf": False}
                return {"is_leaf": True}
            if "sub_questions" in p:
                return {
                    "sub_questions": [
                        "What is the first sub-aspect?",
                        "What is the second sub-aspect?",
                    ]
                }
            if '"query"' in p or "search query" in p:
                return {"query": "mock search query terms"}
            if "evidence" in p:
                return {
                    "evidence": "The answer is found in the documents.",
                    "relevant": True,
                }
            return {"_error": "unrecognised prompt in MockLLM"}

        def count_calls(self) -> int:
            return 0

    class MockRetriever:
        def search(self, query: str, top_k: int = 10) -> list:
            return [
                {"title": "Doc A", "text": "This is document A about the topic.", "score": 0.9},
                {"title": "Doc B", "text": "Document B provides supporting context.", "score": 0.8},
                {"title": "Doc C", "text": "Document C has additional information.", "score": 0.7},
            ]

    class MockEmbedder:
        def encode(self, texts, batch_size: int = 64) -> np.ndarray:
            if isinstance(texts, str):
                texts = [texts]
            n = len(texts)
            vecs = np.random.randn(n, 768).astype(np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs / np.where(norms == 0, 1.0, norms)


    print("=" * 60)
    print("Test 1: Full build → execute → aggregate pipeline")
    print("=" * 60)

    mock_llm = MockLLM()
    mock_ret = MockRetriever()
    mock_emb = MockEmbedder()

    tree = RetrievalDecisionTree(
        tree_id=0,
        llm=mock_llm,
        embedder=mock_emb,
        retriever=mock_ret,
        max_children=4,
    )

    tree.build(
        objective="What factors determine Tesla's competitiveness in the Chinese EV market?",
        max_depth=2,
        context="",
    )

    print("\n--- Stats after build ---")
    s = tree.stats()
    print(f"  nodes={s['num_nodes']}, leaves={s['num_leaves']}, "
          f"max_depth={s['max_depth']}")
    assert s["num_leaves"] > 0, "Tree must have at least one leaf"

    tree.execute(top_k=5)

    answer = tree.aggregate()
    print(f"\nFinal answer: {answer[:120]}")
    assert answer, "Aggregate must return a non-empty answer"


    print("\n" + "=" * 60)
    print("Test 2: max_depth=0 — root must be a leaf")
    print("=" * 60)

    tree0 = RetrievalDecisionTree(0, mock_llm, mock_emb, mock_ret)
    tree0.build("Simple factual question?", max_depth=0)
    assert len(tree0.get_leaves()) == 1, "max_depth=0 must yield exactly 1 leaf"
    assert tree0.root.is_leaf(), "Root must be a leaf when max_depth=0"
    print("  PASS: single leaf at max_depth=0")


    print("\n" + "=" * 60)
    print("Test 3: Embeddings populated after aggregate()")
    print("=" * 60)

    tree.aggregate()
    for node in tree.get_all_nodes():
        assert node.partial_answer_embed is not None, (
            f"Node {node.id} has no embedding after aggregate()"
        )
        assert node.partial_answer_embed.shape == (768,), (
            f"Embedding shape mismatch for {node.id}: "
            f"{node.partial_answer_embed.shape}"
        )
    print(f"  PASS: all {len(tree.get_all_nodes())} nodes have (768,) embeddings")


    print("\n" + "=" * 60)
    print("Test 4: to_dict() produces JSON-serializable output")
    print("=" * 60)

    d = tree.to_dict()
    serialized = json.dumps(d)
    assert len(serialized) > 0
    print(f"  PASS: serialized {len(serialized)} bytes of JSON")


    print("\n" + "=" * 60)
    print("Test 5: get_all_evidence()")
    print("=" * 60)

    evidence_list = tree.get_all_evidence()
    print(f"  Evidence entries: {len(evidence_list)}")
    for e in evidence_list:
        assert "query" in e and "evidence" in e and "docs" in e
    print("  PASS: all evidence entries have required keys")


    print("\n" + "=" * 60)
    print("Test 6: _fallback_query()")
    print("=" * 60)

    fq_cases = [
        ("What is the capital of France?",  "capital france"),
        ("Who founded Apple Inc.?",          "founded apple inc"),
        ("When was the Eiffel Tower built?", "eiffel tower built"),
        ("",                                 ""),
    ]
    for q, expected in fq_cases:
        got = _fallback_query(q)
        print(f"  {q!r:45s} → {got!r}  (expected {expected!r})")

    print("\nAll tests passed.")
