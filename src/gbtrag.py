import time
import re
from dataclasses import replace

import numpy as np

from src.llm import LLM, Embedder
from src.tree import RetrievalDecisionTree
from src.residual import GBTRAGResidual, Residual, select_aspects
from src.retrieval import make_retriever, Reranker
from src.prompts import (DECOMPOSE_QUERY, DRAFT_ANSWER, AMBIGUITY_CRITIQUE,
                         REWRITE, COMPOSE, COMPRESSION_PROMPT, GROUNDED_DECOMPOSE,
                         EXTRACT_SHORT_ANSWER, EXTRACT_SHORT_ANSWER_ALT, fmt)


class GBTRAG:
    def __init__(self, cfg, llm=None, embedder=None, retriever=None, reranker=None):
        self.cfg = cfg
        self.llm = llm or LLM(**{
            key: getattr(cfg.llm, key) for key in
            ["base_url", "model", "temperature", "max_tokens", "timeout", "seed", "max_retries"]},
            cache_dir=cfg.experiment.cache_dir)
        self.retrieval_embedder = embedder or Embedder(**dict(cfg.embedding))
        residual_cfg = getattr(cfg, "residual_embedding", None)
        self.embedder = Embedder(**dict(residual_cfg)) if residual_cfg else self.retrieval_embedder
        self._retriever = retriever
        self.reranker = reranker
        if self.reranker is None and cfg.reranker.enabled:
            self.reranker = Reranker(cfg.reranker.model_path, cfg.reranker.device)
        g = cfg.gbtrag
        if float(getattr(g, "rollback_tolerance", 0)) != 0:
            raise ValueError("The paper's gate requires rollback_tolerance=0")
        self.residual_engine = GBTRAGResidual(
            self.embedder, tau=g.coverage_threshold,
            dedup_threshold=g.dimension_dedup_threshold,
            max_total_leaves=g.max_total_leaves,
            coverage_fn=getattr(g, "coverage_fn", "cosine"))

    def run(self, question, docs=None, task_type="shortform", collect_checkpoints=False):
        if task_type not in {"shortform", "longform"}:
            raise ValueError(f"Unknown task_type: {task_type}")
        previous_context = getattr(self.llm, "context", "")
        self.llm.context = ""
        try:
            return self._run(question, docs or [], task_type, collect_checkpoints)
        finally:
            self.llm.context = previous_context

    def _run(self, question, docs, task_type, collect_checkpoints):
        started = time.perf_counter()
        cfg, g = self.cfg, self.cfg.gbtrag
        engine = self.residual_engine
        engine.reset()
        engine._current_query = question
        self.llm.reset_call_count()
        retriever = self._retriever or make_retriever(cfg, self.retrieval_embedder)
        retriever.build(docs)
        total_start = getattr(retriever, "search_calls", 0)
        initial_docs = retriever.search(question, top_k=int(cfg.retrieval.top_k))
        entities = self._detect_named_entities(question, initial_docs)
        entity_lines = "\n".join(f'- "{entity}"' for entity in entities)
        self.llm.context = (f"Original question: {question}\n"
                            f"Verified named entities (preserve their surface forms):\n{entity_lines or '(none)'}")
        self._initial_documents = initial_docs
        self._query_history = {}
        phase0 = str(g.phase0_mode)
        if phase0 == "auto":
            phase0 = "draft_critique" if task_type == "longform" else "grounded_decompose"
        if phase0 == "draft_critique":
            raw, draft, fallback, draft_len = self._phase0_draft_critique(
                question, retriever, entity_lines, task_type, initial_docs=initial_docs)
        elif phase0 == "grounded_decompose":
            passages = self._format_passages(initial_docs)
            draft = self.llm(fmt(DRAFT_ANSWER, question=question, documents=passages)).strip()
            raw = self._phase0_decompose(question, entity_lines, draft, initial_docs)
            fallback, draft_len = False, len(draft)
        elif phase0 == "decompose":
            raw = self._phase0_decompose(question, entity_lines, grounded=False)
            draft, fallback, draft_len = "", False, 0
        else:
            raise ValueError(f"Unknown phase0_mode: {phase0}")
        dimensions = self._cap_initial(engine.init_dimensions(raw))
        if not dimensions:
            raise RuntimeError("Phase 0 returned no usable coordinates")
        dimensions = self._ground_dimensions(dimensions, retriever, initial=True)
        tree0 = self._make_tree(0, retriever, question, dimensions, task_type,
                                deterministic=(g.tree_mode == "deterministic"))
        leaf_calls = len(tree0.get_leaves())
        current = tree0.aggregate()
        if not current.strip():
            raise RuntimeError("Initial tree produced no answer")
        trees = [tree0]
        evidence = [lf.evidence for lf in tree0.get_leaves() if lf.evidence]
        dimensions = engine.compute_coverage(current, dimensions)
        trace = [self._trace_row(0, current, dimensions, leaf_calls)]
        logs, dead_counts, rejected = [], [], []
        stop_reason = "budget_exhausted"


        all_leaf_updates = bool(getattr(g, "no_residual", False))
        max_rounds = int(g.max_rounds)
        if getattr(g, "single_round", False):
            max_rounds = 0
        for m in range(1, max_rounds + 1):

            added_ids = []
            if g.enable_dimension_discovery:
                proposals = engine.discover(current, dimensions, self.llm, task_type=task_type)
                dimensions, added_ids = self._expand(dimensions, proposals, retriever)
            dimensions = engine.compute_coverage(current, dimensions)
            dead = []
            if g.enable_dead_leaf_pruning:
                dimensions, dead = engine.prune_dead_leaves(
                    dimensions, evidence, g.dead_leaf_coverage_cutoff, g.dead_leaf_evidence_cutoff)
            dead_counts.append(len(dead))

            dimensions = engine.compute_coverage(current, dimensions)
            residual = Residual(engine.backpropagate(dimensions, trees[-1]),
                                engine.compute_loss(dimensions))
            if residual.loss < g.convergence_epsilon or not residual.gaps:
                stop_reason = "absolute_convergence" if residual.loss < g.convergence_epsilon else "empty_active_set"
                logs.append({"round": m, "decision": "stop_before_update", "reason": stop_reason,
                             "added_leaf_ids": added_ids, "pruned_leaf_ids": dead,
                             "coverage_loss_before": residual.loss})
                break
            active = dimensions if all_leaf_updates else engine.active_dimensions(dimensions)
            active_ids = {gap.leaf_id for gap in residual.gaps}
            target_ids = ({lid for d in dimensions for lid in d.leaf_ids}
                          if all_leaf_updates else active_ids)
            tree = self._make_tree(m, retriever, question, active, task_type)
            if {lf.id for lf in tree.get_leaves()} != target_ids:
                raise AssertionError("Update tree leaf IDs must equal its declared support")
            leaf_calls += len(tree.get_leaves())
            increment_summary = tree.aggregate()
            trees.append(tree)
            leaf_evidence = [lf.evidence for lf in tree.get_leaves() if lf.evidence]
            evidence.extend(leaf_evidence)
            before = current
            candidate = self._compose(current, leaf_evidence, residual, question) if leaf_evidence else current
            candidate_dims = engine.compute_coverage(candidate, dimensions)
            candidate_loss = engine.compute_loss(candidate_dims)
            gate_enabled = bool(g.enable_monotone_gate)
            accepted = bool(leaf_evidence) and (not gate_enabled or candidate_loss <= residual.loss)
            if accepted:
                current, dimensions = candidate, candidate_dims
            else:
                rejected.append(m)
            after_loss = engine.compute_loss(dimensions)
            log = {"round": m, "coverage_loss_before": residual.loss,
                   "candidate_loss": candidate_loss, "coverage_loss_after": after_loss,
                   "delta_loss": residual.loss - after_loss,
                   "decision": "accept" if accepted else "rollback",
                   "gate_enabled": gate_enabled, "increment_summary": increment_summary,
                   "residual_leaves": len(active_ids),
                   "active_leaf_ids": sorted(active_ids), "added_leaf_ids": added_ids,
                   "pruned_leaf_ids": dead, "residual_evidence_empty": not bool(leaf_evidence),
                   "coordinate_ids": [lid for d in dimensions for lid in d.leaf_ids]}
            if all_leaf_updates:
                log.update(residual_restricted=False, tree_leaf_ids=sorted(target_ids))
            logs.append(log)
            trace.append(self._trace_row(m, current, dimensions, leaf_calls))
            if accepted and after_loss < g.convergence_epsilon:
                stop_reason = "absolute_convergence"
                break
            if accepted and residual.loss - after_loss < g.stagnation_delta:
                stop_reason = "accepted_stagnation"
                break

        final_raw = current
        inference_seconds = time.perf_counter() - started
        final_started = time.perf_counter()
        compressed = False
        if task_type == "longform" and g.compress_output:
            answer, compressed = self._phase7_compress(question, final_raw)
        elif task_type == "shortform":
            answer = self._extract_short_answer(question, final_raw, trees)
            answer = " ".join(answer.split()[:10])
        else:
            answer = final_raw
        final_seconds = time.perf_counter() - final_started
        decisions = [x for x in logs if x["decision"] in {"accept", "rollback"}]
        result = {"question": question, "answer": answer, "detailed_answer": final_raw,
                  "a_final_raw": final_raw, "a_compressed": answer if compressed else None,
                  "answer_stage": "post_compression" if compressed else ("short_span" if task_type == "shortform" else "uncompressed"),
                  "task_type": task_type, "phase7_applied": compressed,
                  "num_rounds": len(trees), "num_boosting_rounds": len(trees) - 1,
                  "losses": [row["coverage_loss"] for row in trace],
                  "final_loss": engine.compute_loss(dimensions), "round_trace": trace,
                  "round_logs": logs, "stop_reason": stop_reason,
                  "composition_rejected_rounds": rejected,
                  "num_rounds_rejected_by_rollback": len(rejected),
                  "num_dead_leaves_removed_per_round": dead_counts,
                  "num_retrieval_calls": getattr(retriever, "search_calls", total_start) - total_start,
                  "boosting_leaf_retrieval_calls": leaf_calls,
                  "retrieval_scope": str(cfg.retrieval.scope),
                  "trees": [tree.stats() for tree in trees],
                  "dimensions": [{"name": d.name, "coverage": d.coverage, "leaf_ids": d.leaf_ids,
                                  "sub_aspects": d.sub_aspects, "leaf_weights": d.leaf_weights,
                                  "sub_aspect_sims": d.sub_aspect_sims} for d in dimensions],
                  "embedding_signature": self.embedder.signature(), "retrieval_embedding_signature": self.retrieval_embedder.signature(),
                  "call_stats": self.llm.get_call_stats(), "num_llm_calls": self.llm.count_calls(),
                  "boosting_seconds": inference_seconds, "final_postprocess_seconds": final_seconds,
                  "elapsed_seconds": time.perf_counter() - started,
                  "verified_entities": entities, "phase0_mode_used": phase0, "phase0_num_interpretations": len(raw), "phase0_draft_length": draft_len,
                  "phase0_critique_used_retrieval_evidence": phase0 == "draft_critique",
                  "phase0_fallback_triggered": fallback, "composition_operator_used": str(g.compose_op),
                  "rollback_summary": {"total_rounds_attempted": len(decisions),
                                       "rounds_accepted": sum(x["decision"] == "accept" for x in decisions),
                                       "rounds_rolled_back": len(rejected),
                                       "rollback_rate": len(rejected) / max(len(decisions), 1),
                                       "final_round_used": max((x["round"] for x in decisions if x["decision"] == "accept"), default=0)}}
        if collect_checkpoints:
            result["_checkpoints"] = {"initial_draft": draft, "verified_entities": entities,
                                      "trees": [tree.to_dict() for tree in trees]}
        return result

    def _cap_initial(self, dimensions):
        left = int(self.cfg.gbtrag.max_total_leaves)
        result = []
        for dim in dimensions:
            if left <= 0:
                break
            n = min(left, len(dim.sub_aspects))
            result.append(select_aspects(dim, list(range(n))))
            left -= n
        return result

    def _ground_dimensions(self, dimensions, retriever, initial):
        result = []
        threshold = float(self.cfg.gbtrag.grounding_threshold)
        for dim in dimensions:
            scores = []
            for i, aspect in enumerate(dim.sub_aspects):
                docs = retriever.search(aspect, top_k=int(self.cfg.retrieval.top_k))
                vectors = self.embedder.encode([d["text"] for d in docs]) if docs else None
                score = float(np.max(vectors @ dim.sub_aspect_embeds[i])) if vectors is not None else -1.0
                scores.append(score)
            weights = [1.0 if score >= threshold else 0.5 for score in scores]
            dim = replace(dim, leaf_weights=weights, grounding_scores=scores,
                          is_grounded=all(s >= threshold for s in scores))
            if not initial:
                dim = select_aspects(dim, [i for i, s in enumerate(scores) if s >= threshold])
            if dim is not None:
                result.append(dim)
        return result

    def _expand(self, dimensions, proposals, retriever):
        result = list(dimensions)
        added = []
        cap = int(self.cfg.gbtrag.max_total_leaves)
        for candidate in self.residual_engine.init_dimensions(proposals):

            for i in range(len(candidate.sub_aspects)):
                if sum(len(d.sub_aspects) for d in result) >= cap:
                    return result, added
                vectors = [d.sub_aspect_embeds for d in result]
                if vectors and float(np.max(np.concatenate(vectors) @ candidate.sub_aspect_embeds[i])) >= self.cfg.gbtrag.dimension_dedup_threshold:
                    continue
                selected = select_aspects(candidate, [i])
                grounded = self._ground_dimensions([selected], retriever, initial=False)
                if grounded:
                    result.extend(grounded)
                    added.append(selected.leaf_ids[0])
        return result, added

    def _make_tree(self, m, retriever, question, dimensions, task_type, deterministic=True):
        g = self.cfg.gbtrag
        tree = RetrievalDecisionTree(
            m, self.llm, self.embedder, retriever, max_children=g.max_children,
            force_root_split=g.force_root_split, deterministic_tree=deterministic,
            use_anchored_aggregation=task_type == "shortform",
            use_longform_anchored_aggregation=task_type == "longform",
            use_factoid_queries=True,
            aggregation_target_words=g.longform_agg_target_words if task_type == "longform" else g.shortform_agg_target_words,
            query_history=self._query_history)
        tree.build(question, max_depth=2, dimensions=dimensions,
                   sub_aspects_per_dim=g.sub_aspects_per_dim)
        tree.execute(int(self.cfg.retrieval.top_k), reranker=self.reranker,
                     rerank_top_k=int(self.cfg.reranker.top_k))
        for leaf in tree.get_leaves():
            self._query_history[leaf.id] = {
                "query": leaf.search_query,
                "passages": self._format_passages(leaf.retrieved_docs[:3])}
        return tree

    def _trace_row(self, m, answer, dimensions, calls):
        return {"m": m, "answer_raw": answer, "answer_stage": "uncompressed",
                "coverage_loss": self.residual_engine.compute_loss(dimensions),
                "active_set_size": len(self.residual_engine.backpropagate(dimensions)),
                "cumulative_leaf_retrievals": calls,
                "coordinate_ids": [lid for d in dimensions for lid in d.leaf_ids]}

    @staticmethod
    def _format_passages(documents):
        return "\n---\n".join(f"[{d.get('title', '')}] {d.get('text', '')}" for d in documents) or "(none)"

    def _phase0_decompose(self, question, entity_lines, draft_answer="", documents=None, grounded=True):
        g = self.cfg.gbtrag
        if grounded:
            prompt = fmt(GROUNDED_DECOMPOSE, query=question,
                         num_dimensions=g.generic_dimensions, num_sub_aspects=g.sub_aspects_per_dim,
                         verified_entities=entity_lines or "(none)", draft_answer=draft_answer or "(not available)",
                         retrieved_passages=self._format_passages(documents if documents is not None else getattr(self, "_initial_documents", [])))
        else:
            prompt = fmt(DECOMPOSE_QUERY, query=question)
        response = self.llm.json(prompt)
        dims = response.get("dimensions", []) if isinstance(response, dict) else []
        if not dims:
            dims = [{"name": question, "ideal": question, "sub_aspects": [question]}]
        return dims

    @staticmethod
    def _detect_named_entities(question, documents):
        import re
        candidates = re.findall(r'"([^\"]+)"|\b([A-Z][a-z]+(?:\s+(?:[A-Z][a-z]+|of|the|in|and))+)\b', question)
        text = "\n".join(d.get("title", "") + " " + d.get("text", "") for d in documents)
        return list(dict.fromkeys((a or b).strip() for a, b in candidates if (a or b).strip() in text))

    def _compose(self, existing, evidence, residual, question):
        gaps = "\n".join(f"- {g.sub_aspect} (residual {g.severity:.3f})" for g in residual.gaps)
        if self.cfg.gbtrag.compose_op == "additive":
            prompt = fmt(COMPOSE, question=question, existing=existing,
                         increment="\n\n".join(evidence), gaps=gaps)
        else:
            prompt = fmt(REWRITE, question=question, current_answer=existing,
                         new_evidence="\n---\n".join(evidence), target_dimensions=gaps)
        return self.llm(prompt, max_tokens=int(self.cfg.llm.rewrite_max_tokens)).strip()

    def _phase7_compress(self, question, a_final):

        g = self.cfg.gbtrag
        prompt = fmt(COMPRESSION_PROMPT, A_final=a_final,
                     target_words=g.compress_target_words,
                     structure_hint=g.compress_structure_hint)
        text = self.llm(prompt, max_tokens=int(self.cfg.llm.final_max_tokens)).strip()
        for prefix in ["compressed answer:", "answer:"]:
            if text.lower().startswith(prefix):
                text = text[len(prefix):].strip()
        if not text:
            raise RuntimeError("Compression returned an empty answer")
        return text, True
    def _phase0_draft_critique(
        self,
        question: str,
        retriever,
        entity_lines: str,
        task_type: str,
        initial_docs=None,
    ) -> tuple:


        try:
            raw_docs = initial_docs if initial_docs is not None else retriever.search(question, top_k=10)
        except Exception as e:
            print(f"  [Phase0] retrieval error: {e} — falling back to decompose")
            return self._phase0_decompose(question, entity_lines), "", True, 0


        safe_docs = []
        for d in raw_docs:
            if isinstance(d, dict):
                safe_docs.append(d)
            elif isinstance(d, (list, tuple)) and len(d) >= 2:
                safe_docs.append({"title": str(d[0]), "text": str(d[1])})
        doc_texts_str = "\n---\n".join(
            f"[{d.get('title', 'Doc')}] {d.get('text', '')[:400]}"
            for d in safe_docs[:10]
        )
        if not doc_texts_str:
            doc_texts_str = "(No documents retrieved.)"


        draft_answer = ""
        try:
            draft_raw = self.llm(
                fmt(DRAFT_ANSWER, question=question, documents=doc_texts_str)
            )
            draft_answer = draft_raw.strip() if draft_raw else ""
        except Exception as e:
            print(f"  [Phase0] draft LLM error: {e}")


        critique_prompt = fmt(
            AMBIGUITY_CRITIQUE,
            question=question,
            draft_answer=draft_answer or "(no draft available)",
            documents=doc_texts_str,
        )
        if entity_lines:
            critique_prompt += (
                "\n\nIMPORTANT — named entities detected in this question:\n"
                + entity_lines
                + "\nTreat each entity as a PROPER NOUN referring to a specific "
                "real-world thing, not as a literal description."
            )

        critique_result = self.llm.json(critique_prompt)


        raw_interps = []
        if isinstance(critique_result, list):
            raw_interps = critique_result
        elif isinstance(critique_result, dict) and "_error" not in critique_result:

            for key in ("interpretations", "dimensions", "items", "results"):
                val = critique_result.get(key)
                if isinstance(val, list):
                    raw_interps = val
                    break


        raw_interps = [r for r in raw_interps if isinstance(r, dict)]


        MIN_DIMS_LONG = 2
        needs_retry = (
            task_type == "longform"
            and len(raw_interps) < MIN_DIMS_LONG
            and not (raw_interps and raw_interps[0].get("justification"))
        )

        if needs_retry:
            print(
                f"  [Phase0] long-form returned {len(raw_interps)} interpretation(s) "
                f"without justification — retrying once."
            )
            retry_prompt = critique_prompt + (
                "\n\nIMPORTANT: This is a LONG-FORM question that MUST have "
                "multiple distinct interpretations. Your previous response "
                f"returned only {len(raw_interps)} interpretation(s). "
                "You MUST identify at least 2 DISTINCT interpretations. "
                "Look more carefully at the documents for evidence of "
                "different entities, time periods, or contexts the draft missed."
            )
            retry_result = self.llm.json(retry_prompt)
            retry_interps = []
            if isinstance(retry_result, list):
                retry_interps = retry_result
            elif isinstance(retry_result, dict) and "_error" not in retry_result:
                for key in ("interpretations", "dimensions", "items"):
                    val = retry_result.get(key)
                    if isinstance(val, list):
                        retry_interps = val
                        break
            retry_interps = [r for r in retry_interps if isinstance(r, dict)]
            if len(retry_interps) > len(raw_interps):
                raw_interps = retry_interps


        if not raw_interps:
            print(
                "  [Phase0] Critique parse failed — falling back to "
                "DECOMPOSE_QUERY. Logging phase0_fallback."
            )
            fallback_dicts = self._phase0_decompose(question, entity_lines)
            return fallback_dicts, draft_answer, True, len(draft_answer)


        dim_dicts = []
        for interp in raw_interps:
            name        = str(interp.get("name", "")).strip()
            explanation = str(interp.get("explanation", interp.get("ideal", name))).strip()
            sub_aspects = [
                str(s).strip()
                for s in interp.get("sub_aspects", [])
                if str(s).strip()
            ]
            if not name:
                continue
            dim_dicts.append({
                "name":        name,
                "ideal":       explanation,
                "sub_aspects": sub_aspects,
            })

        if not dim_dicts:
            print("  [Phase0] No valid interpretations after normalisation "
                  "— falling back to DECOMPOSE_QUERY.")
            fallback_dicts = self._phase0_decompose(question, entity_lines)
            return fallback_dicts, draft_answer, True, len(draft_answer)

        return dim_dicts, draft_answer, False, len(draft_answer)


    def _extract_short_answer(
        self,
        question: str,
        detailed_answer: str,
        trees: list,
    ) -> str:

        if not detailed_answer or not detailed_answer.strip():
            return ""


        all_evidence = []
        for tree in trees:
            for leaf in tree.get_leaves():
                if leaf.evidence and len(leaf.evidence) > 0:
                    all_evidence.append(leaf.evidence)

        seen: set = set()
        unique_evidence = []
        for e in all_evidence:
            key = e
            if key not in seen:
                seen.add(key)
                unique_evidence.append(e)

        evidence_text = "\n---\n".join(unique_evidence)
        if not evidence_text:
            evidence_text = "(No direct evidence available.)"


        response = self.llm(
            fmt(
                EXTRACT_SHORT_ANSWER,
                question=question,
                evidence=evidence_text,
                detailed_answer=detailed_answer,
            ), max_tokens=int(self.cfg.llm.final_max_tokens)
        ).strip()
        response = _clean_short_answer(response)

        if not response:
            print("[GBTRAG] Warning: EXTRACT_SHORT_ANSWER returned empty — "
                  "using first sentence fallback.")
            return _first_sentence(detailed_answer)


        if response.lower() in ("yes", "no") and not _is_yes_no_question(question):
            print(f"  [WARN] Extractor returned {response!r} for non-yes/no "
                  f"question. Retrying with stricter prompt (evidence only).")
            response = self.llm(
                fmt(
                    EXTRACT_SHORT_ANSWER,
                    question=question,
                    evidence=evidence_text,
                    detailed_answer="(use only the evidence above)",
                )
            ).strip().strip('"\'.,;:')
            response = _clean_short_answer(response)


        if response.lower() in ("yes", "no") and not _is_yes_no_question(question):
            print(f"  [WARN] Retry 1 still returned {response!r}. "
                  f"Falling back to Tree 0 answer.")
            tree0_answer = trees[0].get_answer()
            response = self.llm(
                fmt(
                    EXTRACT_SHORT_ANSWER,
                    question=question,
                    evidence=evidence_text,
                    detailed_answer=tree0_answer,
                )
            ).strip().strip('"\'.,;:')
            response = _clean_short_answer(response)
            if response.lower() in ("yes", "no") and not _is_yes_no_question(question):
                print(f"  [WARN] All retries failed — using first sentence of "
                      f"Tree 0 answer as last resort.")
                response = _first_sentence(tree0_answer)


        do_consistency = bool(
            getattr(getattr(self.cfg, "gbtrag", None),
                    "self_consistency_check", False)
        )
        if do_consistency:
            alt_response = self.llm(
                fmt(
                    EXTRACT_SHORT_ANSWER_ALT,
                    question=question,
                    evidence=evidence_text,
                    detailed_answer=detailed_answer,
                )
            ).strip()
            alt_response = _clean_short_answer(alt_response)

            if alt_response and alt_response.lower() == response.lower():
                print(f"[GBTRAG] Self-consistency: both calls agree → {response!r}")
            elif alt_response:
                print(
                    f"[GBTRAG] Self-consistency: calls disagree "
                    f"({response!r} vs {alt_response!r}) — "
                    f"using evidence-grounded answer."
                )

            else:
                print("[GBTRAG] Self-consistency: alt call returned empty — "
                      "keeping primary answer.")

        return response


def _is_yes_no_question(question: str) -> bool:

    q = question.lower().strip()
    yes_no_starters = (
        "did ", "do ", "does ", "is ", "was ", "were ", "are ",
        "has ", "have ", "can ", "could ", "will ", "would ",


        "did?", "do?", "does?", "is?", "was?", "were?", "are?",
        "has?", "have?", "can?", "could?", "will?", "would?",
    )
    return q.startswith(yes_no_starters)


def _first_sentence(text: str) -> str:

    text = text.strip()

    m = re.search(r"[.!?]", text)
    if m:
        return text[: m.start()].strip()

    return text[:100].strip()


def _clean_short_answer(text: str) -> str:

    text = text.strip().strip('"\'')


    if text.lower().startswith("answer:"):
        text = text[len("answer:"):].strip()


    for prefix in (
        "the answer is",
        "based on",
        "according to",
        "from the evidence,",
        "from the analysis,",
    ):
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip().lstrip(":,").strip()
            break


    if text.endswith(".") and "." not in text[:-1]:
        text = text[:-1]

    return text.strip()
