import re
from dataclasses import dataclass, field, replace

import numpy as np

from src.prompts import DECOMPOSE_QUERY, COVERAGE_AUDIT, fmt


@dataclass
class Dimension:
    name: str
    ideal: str
    sub_aspects: list = field(default_factory=list)
    sub_aspect_embeds: object = field(default=None, repr=False)
    coverage: float = 0.0
    sub_aspect_sims: list = field(default_factory=list)
    grounding_weight: float = 1.0
    is_grounded: bool = True
    leaf_ids: list = field(default_factory=list)
    leaf_weights: list = field(default_factory=list)
    grounding_scores: list = field(default_factory=list)


@dataclass
class Gap:
    dim_name: str
    severity: float
    failed_node_id: str = ""
    failed_query: str = ""
    leaf_id: str = ""
    sub_aspect: str = ""
    aspect_index: int = 0


@dataclass
class Residual:
    gaps: list = field(default_factory=list)
    loss: float = 0.0
    new_dimensions: list = field(default_factory=list)


def select_aspects(dim, indices):

    if not indices:
        return None
    return replace(dim,
                   sub_aspects=[dim.sub_aspects[i] for i in indices],
                   sub_aspect_embeds=dim.sub_aspect_embeds[indices].copy(),
                   leaf_ids=[dim.leaf_ids[i] for i in indices],
                   leaf_weights=[dim.leaf_weights[i] for i in indices],
                   grounding_scores=[dim.grounding_scores[i] for i in indices] if dim.grounding_scores else [],
                   sub_aspect_sims=[dim.sub_aspect_sims[i] for i in indices] if dim.sub_aspect_sims else [])


class GBTRAGResidual:
    def __init__(self, embedder, tau=0.65, dedup_threshold=0.85,
                 max_total_leaves=24, coverage_fn="cosine", **legacy):
        self.embedder = embedder
        self.tau = float(tau)
        self.dedup_threshold = float(dedup_threshold)
        self.max_total_leaves = int(max_total_leaves)
        self.coverage_fn = coverage_fn
        self.reset()

    def reset(self):
        self._next_leaf = 0
        self._current_query = ""

    def init_dimensions(self, dim_dicts):
        dimensions = []
        for raw in dim_dicts:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name", "")).strip()
            ideal = str(raw.get("ideal", raw.get("explanation", name))).strip()
            aspects = raw.get("sub_aspects", [])
            if not isinstance(aspects, list):
                continue
            aspects = list(dict.fromkeys(s.strip() for s in aspects if isinstance(s, str) and s.strip()))
            aspects = aspects or ([ideal or name] if ideal or name else [])
            if not aspects:
                continue
            ids = [f"s{self._next_leaf + i}" for i in range(len(aspects))]
            self._next_leaf += len(aspects)
            dimensions.append(Dimension(name or ideal, ideal or name, aspects,
                                        self.embedder.encode(aspects), leaf_ids=ids,
                                        leaf_weights=[1.0] * len(aspects)))
        return dimensions

    def compute_coverage(self, answer, dims):
        chunks = _chunk_answer(answer)
        embeddings = self.embedder.encode(chunks) if chunks else None
        result = []
        for dim in dims:
            if not chunks:
                scores = [0.0] * len(dim.sub_aspects)
            elif self.coverage_fn == "word_overlap":
                scores = [_word_overlap_coverage(s, answer) for s in dim.sub_aspects]
            else:
                scores = np.clip(embeddings @ dim.sub_aspect_embeds.T, -1, 1).max(axis=0).tolist()
            result.append(replace(dim, sub_aspect_sims=scores,
                                  coverage=float(np.mean(scores)) if scores else 0.0))
        return result

    def compute_loss(self, dims):
        total = 0.0
        for dim in dims:
            scores = dim.sub_aspect_sims or [dim.coverage] * len(dim.sub_aspects)
            weights = dim.leaf_weights or [dim.grounding_weight] * len(scores)
            if len(scores) != len(weights):
                raise ValueError("Coordinate/weight length mismatch")
            total += sum(w * max(0.0, self.tau - s) for w, s in zip(weights, scores))
        return float(total)

    def backpropagate(self, dims, tree=None):
        nodes = {leaf.id: leaf for leaf in tree.get_leaves()} if tree else {}
        gaps = []
        for dim in dims:
            scores = dim.sub_aspect_sims or [dim.coverage] * len(dim.sub_aspects)
            for i, (leaf_id, aspect, score) in enumerate(zip(dim.leaf_ids, dim.sub_aspects, scores)):
                if score < self.tau:
                    node = nodes.get(leaf_id)
                    gaps.append(Gap(dim.name, self.tau - score,
                                    node.id if node else "", node.search_query if node else "",
                                    leaf_id, aspect, i))
        return sorted(gaps, key=lambda x: (-x.severity, x.leaf_id))

    def discover(self, answer, dims, llm, task_type="shortform"):
        if sum(len(d.sub_aspects) for d in dims) >= self.max_total_leaves:
            return []
        existing = "\n".join(f"{lid}: {s}" for d in dims for lid, s in zip(d.leaf_ids, d.sub_aspects))
        scope = ("Cover the valid interpretations of this ambiguous long-form question."
                 if task_type == "longform" else
                 "Find only missing facts or reasoning steps needed for the SAME short-answer question. Do not expand to alternative meanings, unrelated people, or general background.")
        response = llm.json(fmt(COVERAGE_AUDIT, query=self._current_query,
                               existing_leaves=existing, current_answer=answer,
                               task_scope=scope, max_new_dims=2))
        raw = response if isinstance(response, list) else response.get("new_dimensions", []) if isinstance(response, dict) else []
        return [x for x in raw if isinstance(x, dict)] if isinstance(raw, list) else []

    def compute_residual(self, answer, dims, tree=None, llm=None,
                         discover_dims=False, task_type="shortform"):
        covered = self.compute_coverage(answer, dims)
        proposals = self.discover(answer, covered, llm, task_type=task_type) if discover_dims and llm is not None else []
        return Residual(self.backpropagate(covered, tree), self.compute_loss(covered), proposals)

    def active_dimensions(self, dims):
        result = []
        for dim in dims:
            scores = dim.sub_aspect_sims or [dim.coverage] * len(dim.sub_aspects)
            selected = select_aspects(dim, [i for i, score in enumerate(scores) if score < self.tau])
            if selected is not None:
                result.append(selected)
        return result

    def prune_dead_leaves(self, dims, evidence, coverage_cutoff=0.15, evidence_cutoff=0.30):
        evidence = [text for text in evidence if text and text.strip()]
        encoded = self.embedder.encode(evidence) if evidence else None
        result, removed = [], []
        for dim in dims:
            keep = []
            for i, score in enumerate(dim.sub_aspect_sims):
                support = float(np.max(encoded @ dim.sub_aspect_embeds[i])) if encoded is not None else -1.0
                if score < coverage_cutoff and support < evidence_cutoff:
                    removed.append(dim.leaf_ids[i])
                else:
                    keep.append(i)
            selected = select_aspects(dim, keep)
            if selected is not None:
                result.append(selected)
        return result, removed

    def should_stop(self, residual, prev_loss=float("inf"), round_num=0,
                    max_rounds=3, epsilon=0.01, accepted=False, delta=0.001):

        if not residual.gaps:
            return True, "empty_active_set"
        if residual.loss < epsilon:
            return True, "absolute_convergence"
        if accepted and prev_loss - residual.loss < delta:
            return True, "accepted_stagnation"
        if round_num >= max_rounds:
            return True, "budget_exhausted"
        return False, ""


def _chunk_answer(answer):
    if not answer or not answer.strip():
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", answer) if p.strip()]
    if len(paragraphs) > 1:
        return paragraphs
    sentences = _split_sentences(answer)
    if len(sentences) <= 3:
        return [answer.strip()]
    return [" ".join(sentences[i:i + 3]) for i in range(0, len(sentences), 2)]


def _split_sentences(text):
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if p.strip()]


def _word_overlap_coverage(aspect, answer):
    stop = {"the", "a", "an", "is", "of", "in", "to", "and", "for"}
    a = set(re.findall(r"\w+", aspect.lower())) - stop
    b = set(re.findall(r"\w+", answer.lower()))
    return len(a & b) / len(a) if a else 0.0


def build_dimensions_from_llm(query, llm, embedder, tau=0.65):
    engine = GBTRAGResidual(embedder, tau=tau)
    response = llm.json(fmt(DECOMPOSE_QUERY, query=query))
    raw = response.get("dimensions", []) if isinstance(response, dict) else []
    if not raw:
        raw = [{"name": query, "ideal": query, "sub_aspects": [query]}]
    return engine, engine.init_dimensions(raw)
