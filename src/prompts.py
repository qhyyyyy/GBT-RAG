def fmt(template: str, **kwargs) -> str:

    return template.format(**kwargs)


DECOMPOSE_QUERY = """\
You are decomposing a question into the distinct DIMENSIONS that a \
complete answer must cover.

Question: {query}

WHAT IS A DIMENSION?
A dimension is one of:
  (a) A distinct INTERPRETATION of an ambiguous question. Ambiguous \
questions have multiple valid readings — each reading is a separate dimension.
  (b) A distinct ENTITY or INSTANCE the question could refer to. If \
the entity has multiple categories (men's vs women's, NFL vs CFL, \
original vs cover, biological vs adoptive, etc.), each category is a \
separate dimension.
  (c) A distinct ANALYTICAL ASPECT the question requires (cause, \
effect, timeline, comparison, mechanism, impact, examples).
  (d) A distinct TIME PERIOD or VERSION (historical vs current, \
original vs remake, first vs last vs most recent).

RULES:
1. You MUST produce at least 2 dimensions for ANY non-trivial question.
2. If the question is ambiguous (and most interesting questions are), \
enumerate every reasonable interpretation as a separate dimension.
3. When in doubt, OVER-decompose. Too many dimensions is much better \
than too few — extras can be merged later, but missing dimensions \
are invisible.
4. Do not collapse different interpretations into a single "general" \
dimension. "Song identification" is WRONG when there are multiple \
versions; "Original artist" + "Cover artists" is RIGHT.
5. For each dimension provide: name (short), ideal (1 sentence \
describing thorough coverage), and 2-3 sub_aspects (specific facts \
to look for).

WORKED EXAMPLES:

Example A — ambiguous person question:
Q: "Who's Henry's father in Once Upon a Time?"
{{"dimensions": [
  {{"name": "Biological father",
    "ideal": "Identify Henry Mills's biological father including the actor",
    "sub_aspects": ["biological father name", "actor who plays him", "character backstory"]}},
  {{"name": "Step-father or adoptive father",
    "ideal": "Identify any step-father or adoptive father figure for Henry",
    "sub_aspects": ["step-father name", "actor who plays him", "relationship to Henry"]}}
]}}

Example B — multi-category sports question:
Q: "Who is the best rushing quarterback of all time?"
{{"dimensions": [
  {{"name": "Best NFL rushing QB", "ideal": "Top rushing QB in National Football League history", "sub_aspects": ["name", "career rushing yards", "era"]}},
  {{"name": "Best CFL rushing QB", "ideal": "Top rushing QB in Canadian Football League history", "sub_aspects": ["name", "career rushing yards", "era"]}},
  {{"name": "Best college rushing QB (FBS)", "ideal": "Top rushing QB in FBS college football", "sub_aspects": ["name", "school", "career stats"]}},
  {{"name": "Best rushing QB in lower divisions", "ideal": "Top rushing QB in FCS or non-varsity football", "sub_aspects": ["name", "division", "stats"]}}
]}}

Example C — temporal ambiguity:
Q: "When was the last perfect game thrown in baseball?"
{{"dimensions": [
  {{"name": "Last perfect game by a right-handed pitcher", "ideal": "Most recent MLB perfect game by a RHP", "sub_aspects": ["pitcher name", "date", "opponent"]}},
  {{"name": "Last perfect game by a left-handed pitcher", "ideal": "Most recent MLB perfect game by a LHP", "sub_aspects": ["pitcher name", "date", "opponent"]}}
]}}

Example D — multi-interpretation song question:
Q: "Who sang I'm Coming Out?"
{{"dimensions": [
  {{"name": "Original artist", "ideal": "Who first recorded and released the song",
    "sub_aspects": ["original singer", "release year", "album"]}},
  {{"name": "Cover versions", "ideal": "Notable cover artists who recorded the song",
    "sub_aspects": ["cover artists", "cover year", "context of cover"]}}
]}}

Example E — analytical question:
Q: "How does Tesla's pricing strategy affect its competitiveness in China?"
{{"dimensions": [
  {{"name": "Pricing actions and timeline", "ideal": "What price changes Tesla made and when", "sub_aspects": ["price cut amounts", "timing", "affected models"]}},
  {{"name": "Competitor responses", "ideal": "How BYD, NIO, Xpeng reacted to Tesla pricing", "sub_aspects": ["competitor price changes", "market share shifts", "strategic responses"]}},
  {{"name": "Margin and profitability impact", "ideal": "Effect on Tesla's profit margins in China", "sub_aspects": ["margin trends", "unit economics", "profitability"]}},
  {{"name": "Market share and sales volume", "ideal": "How pricing affected Tesla's China deliveries", "sub_aspects": ["delivery numbers", "market share percent", "trend direction"]}}
]}}

OUTPUT FORMAT:
Return ONLY this JSON, no markdown fences, no preamble:
{{"dimensions": [
  {{"name": "...", "ideal": "...",
    "sub_aspects": ["...", "...", "..."]}}
]}}

Now decompose the question. Remember: at least 2 dimensions, \
over-decompose when in doubt."""


DRAFT_ANSWER = """\
Question: {question}

Retrieved documents:
{documents}

Write a concise 2-3 sentence draft answer based ONLY on these documents. \
Do NOT speculate beyond what the documents explicitly say. \
If the documents do not contain enough information to answer fully, state \
what they do contain and note the gap.

Draft answer:"""


AMBIGUITY_CRITIQUE = """\
Question: {question}

Draft answer (may only cover ONE interpretation):
{draft_answer}

Retrieved documents:
{documents}

The draft answer may address only the most obvious reading of the question. \
Your task: identify ALL DISTINCT interpretations a truly complete answer \
must cover.

WHAT COUNTS AS A DISTINCT INTERPRETATION?
A genuinely separate reading of the question — different entities, time \
periods, domains, contexts, or categories — NOT sub-topics of the same \
reading.

EXAMPLES OF DISTINCT VS NON-DISTINCT:
  - "Best rushing QB?" → NFL / CFL / college divisions = DISTINCT.
    "NFL QBs by era" = NOT distinct (same league, same category).
  - "When did the Kim family come to power?" → Silla dynasty / Gaya / \
North Korea DPRK = DISTINCT.
    "Decades of DPRK rule" = NOT distinct.
  - "Who plays Sirius Black?" → film / audiobook / stage adaptation = \
DISTINCT.
    "Different Harry Potter films" = NOT distinct (same medium).

HOW TO FIND MISSED INTERPRETATIONS:
1. Look for entities, time periods, or contexts the draft ignored.
2. Look for document passages the draft answer did not use — they may \
support a different interpretation.
3. Think about ambiguity in the question itself: multiple referents, \
multiple time-frames, multiple domains.

RULES:
- Output 2-6 interpretations. If the question is GENUINELY unambiguous, \
output exactly 1 with a "justification" field explaining why.
- Do NOT invent interpretations unsupported by the documents or by the \
question's plain meaning.
- Each interpretation must be substantively different from every other.
- sub_aspects: 2-3 specific facts or sub-questions that would confirm \
coverage of this interpretation.

Return ONLY valid JSON, no markdown fences, no preamble:
[
  {{
    "name": "short label",
    "explanation": "one sentence describing what this interpretation covers",
    "sub_aspects": ["specific fact 1", "specific fact 2", "specific fact 3"]
  }}
]

If genuinely unambiguous (output 1 item with justification):
[
  {{
    "name": "...",
    "explanation": "...",
    "sub_aspects": ["...", "...", "..."],
    "justification": "This question is unambiguous because ..."
  }}
]"""


SPLIT_QUESTION = """\
You are an expert at breaking down complex multi-hop questions into simpler parts.

Question to decompose: {question}

What we already know:
{context}

Your task: decompose the question into 2-4 sub-questions that, when answered \
together, fully answer the original question.

Rules:
- Each sub-question should be specific enough that a single document search \
  could answer it.
- Aim for 2-4 sub-questions — no more, no fewer.
- Sub-questions should be independent where possible.
- Do NOT repeat information already covered in "What we already know".
- Sub-questions should together cover the full original question.

Return ONLY valid JSON. No markdown code fences. No explanation before or \
after the JSON.

{{
  "sub_questions": [
    "first sub-question",
    "second sub-question"
  ]
}}"""


LEAF_DECISION = """\
You are deciding whether a question should be answered directly via document \
retrieval, or whether it needs to be broken down further.

Question: {question}
Current depth: {depth}
Maximum allowed depth: {max_depth}

Decision rules (apply in order):
1. If depth >= max_depth, the node MUST be a leaf. Set is_leaf to true.
2. If the question asks about a single specific fact, entity, date, number, \
   or data point, it should be a leaf (direct retrieval will suffice).
3. If the question requires comparing multiple entities, tracing a chain of \
   reasoning, or combining information from logically distinct sources, it \
   should NOT be a leaf — split it.

Return ONLY valid JSON. No markdown code fences. No explanation before or \
after the JSON.

{{
  "is_leaf": true
}}

or

{{
  "is_leaf": false
}}"""


MAKE_SEARCH_QUERY = """\
Convert the following question into a concise keyword search query suitable \
for a document retrieval system.

Question: {question}

Instructions:
- Remove question words (what, how, why, when, where, who, which, is, are, \
  was, were, does, did).
- Keep key entities, proper nouns, and specific technical terms.
- Keep dates, numbers, and units if present.
- The query should be 3-10 words.
- Example: "What was Tesla's revenue in China in 2024?" → "Tesla revenue \
  China 2024"
- Example: "Who founded the company that makes the iPhone?" → "Apple iPhone \
  founder"

Return ONLY valid JSON. No markdown code fences. No explanation before or \
after the JSON.

{{
  "query": "concise keyword search query here"
}}"""


EXTRACT_EVIDENCE = """\
You are extracting relevant evidence from retrieved documents to answer a \
specific question.

Question: {question}

Retrieved documents:
{documents}

Instructions:
- Extract ONLY information directly relevant to answering the question.
- Use the exact wording from the documents where possible (do not paraphrase \
  facts — paraphrasing can introduce errors).
- If multiple documents contain relevant information, combine them.
- If NO document contains information relevant to the question, set \
  "relevant" to false and "evidence" to an empty string.
- Keep the extracted evidence concise — do not copy entire paragraphs if \
  only a sentence is relevant.

Return ONLY valid JSON. No markdown code fences. No explanation before or \
after the JSON.

{{
  "evidence": "the extracted relevant text from the documents",
  "relevant": true
}}"""


AGGREGATE = """\
You are synthesizing answers to sub-questions into a single coherent answer \
to the parent question.

Parent question: {question}

Answers to sub-questions:
{children_qa}

Instructions:
- Combine the information from ALL sub-answers into one coherent, \
  well-organized response.
- Do NOT just concatenate the sub-answers — synthesize them into prose that \
  directly answers the parent question.
- Resolve any contradictions between sub-answers (prefer the more specific \
  or more recently mentioned information).
- If a sub-answer is marked as "not found" or empty, acknowledge the gap but \
  still synthesize what is available.
- Write in full sentences. Be factual and concise.
- Target length: {target_words} words. One concise sentence per distinct fact \
  or interpretation. Do not pad or over-explain.

Provide ONLY the synthesized answer as plain text. No JSON. No preamble like \
"Based on the sub-answers..." — just the answer itself."""


COMPOSE = '''You are merging new information into an existing answer.

Original question: {question}

Existing answer:
{existing}

New information to integrate:
{increment}

Information gaps this new content addresses:
{gaps}

RULES:
1. PRESERVE all factually correct content from the existing answer.
2. INTEGRATE the new information where it directly addresses the listed gaps.
3. Do NOT fabricate any claim not present in either the existing answer or the new information.
4. If a gap cannot be filled from the provided materials, omit it.
5. Each distinct interpretation or answer should be clearly stated with its key facts (names, dates, numbers).
6. Remove redundant repetition but keep all distinct facts.

Output the merged answer (plain text only):'''


REWRITE = '''\
Rewrite the current answer to additionally cover the listed dimensions \
using the new evidence.

Original question: {question}

Current answer:
{current_answer}

New evidence from retrieval:
{new_evidence}

Residual dimensions being targeted:
{target_dimensions}

RULES:
1. Preserve ALL factually correct information from the current answer.
2. Use the new evidence to cover the listed target dimensions.
3. Do NOT introduce information not supported by the new evidence or the \
current answer.
4. Do NOT fabricate any claim absent from both inputs.
5. Remove redundant repetition but keep all distinct facts.
6. Output the COMPLETE rewritten answer — not a diff, not a summary, \
not a commentary. Plain text only.

Complete rewritten answer:'''


DISCOVER_DIMENSIONS = """\
You are auditing whether the current answer covers all the important \
interpretations of the question.

Question: {query}

Current dimensions being tracked: {dimensions_list}

Current answer:
{current_answer}

Your task: identify any IMPORTANT INTERPRETATIONS, ENTITIES, or CATEGORIES \
that the question could be asking about but which are NOT covered by the \
current dimensions.

Think carefully:
1. Does the question have any ambiguity the current dimensions miss?
2. Are there entity categories (men's/women's, original/cover, \
biological/step, NFL/CFL, etc.) that aren't represented?
3. Are there time periods or versions that aren't represented?
4. Does the answer mention concepts that hint at uncovered dimensions?

ONLY add a new dimension if:
- It is genuinely DIFFERENT from all current dimensions (not a paraphrase \
or sub-topic of an existing one).
- It addresses a valid interpretation a knowledgeable reader would expect \
to see covered.
- It is supported by context in the question itself.

Return ONLY valid JSON, no markdown fences, no preamble. \
If nothing is missing, return an empty list.

{{"new_dimensions": [
  {{"name": "...", "ideal": "...",
    "sub_aspects": ["...", "...", "..."]}}
]}}

If no genuinely new dimensions are needed: {{"new_dimensions": []}}"""


AGGREGATE_ANCHORED = """\
You are synthesizing answers to sub-questions into a factual answer.

Parent question: {question}

Answers to sub-questions (treat as source material):
{children_qa}

CRITICAL RULES — follow these before anything else:
1. Preserve ALL named entities, proper nouns, dates, and numbers VERBATIM.
   BAD: "a peace treaty in the late 18th century"
   GOOD: "the Treaty of Paris (September 3, 1783)"
   BAD: "North Korea's founding leader came to power"
   GOOD: "Kim Il-sung established the DPRK on September 9, 1948"
2. Target {target_words} words total. One concise sentence per distinct fact or interpretation.
3. Use the EXACT surface form of all key facts as they appear in the sub-answers above.
4. If a sub-answer is "No relevant information found", skip that sub-question.
5. Do NOT merge different interpretations into a single vague statement — keep each distinct.

Factual answer (verbatim key phrases preserved, ~{target_words} words):"""


AGGREGATE_ANCHORED_LONGFORM = """\
You are synthesizing answers to sub-questions into a comprehensive factual answer \
for a disambiguation question.

Parent question: {question}

Answers to sub-questions (treat as source material):
{children_qa}

CRITICAL RULES — follow these before anything else:
1. Preserve ALL named entities, proper nouns, dates, and numbers VERBATIM from the sub-answers.
   BAD: "a peace treaty in the late 18th century"
   GOOD: "the Treaty of Paris (September 3, 1783)"
   BAD: "North Korea's founding leader came to power"
   GOOD: "Kim Il-sung established the DPRK on September 9, 1948"
Keep the synthesized answer within approximately {target_words} words.\

2. For EACH distinct interpretation or disambiguation target, write 1-3 complete sentences \
that fully cover that interpretation. Do NOT compress into one vague sentence.
3. Use the EXACT surface form of all key facts as they appear in the sub-answers above.
4. If a sub-answer is "No relevant information found", skip that sub-question.
5. Do NOT merge different interpretations into a single vague statement — each must be \
explicitly and completely addressed.

Comprehensive factual answer (verbatim key phrases, every disambiguation target fully covered):"""


REWRITE_ANCHORED = """\
You are extending a factual answer with new evidence to cover information gaps.

EXISTING ANSWER — copy this VERBATIM as the start of your output, then append:
{current_answer}

NEW EVIDENCE for these gaps: {target_dimensions}
{new_evidence}

TASK: Append 1–3 new sentences after the existing answer to cover the gaps above.
RULES:
1. Output the EXISTING ANSWER exactly as written above, then add new sentences.
2. Do NOT rewrite, rephrase, or restructure ANY part of the existing answer.
3. Do NOT merge existing sentences together.
4. Preserve ALL named entities, dates, and numbers VERBATIM from the new evidence.
5. New content should be at most {gap_words} words.
6. If the gap is already covered in the existing answer, add nothing.
7. Total answer must not exceed {max_words} words.

Complete answer (existing answer verbatim, then new sentences appended):"""


LEAF_QUERY_FACTOID = """\
Convert this sub-aspect into a SHORT factoid search query (5-10 words) that \
would retrieve a Wikipedia passage directly answering it.

Sub-aspect: {sub_aspect_description}

Rules:
- Include specific named entities, years, and domain keywords.
- Mimic the vocabulary of an encyclopedia passage — avoid question phrasing.
- Output ONLY the query string, never a question.

Examples:
  sub-aspect: "year when the Kim family gained political control in North Korea"
  query: "Kim Il-sung DPRK founded 1948 North Korea"

  sub-aspect: "the treaty that ended the American Revolutionary War"
  query: "Treaty of Paris 1783 American Revolution ended signed"

  sub-aspect: "who played Sirius Black in the Harry Potter films"
  query: "Gary Oldman Sirius Black Harry Potter actor"

Return ONLY valid JSON. No markdown, no explanation.
{{"query": "5-10 word factoid search query here"}}

Sub-aspect: {sub_aspect_description}"""


SUGGEST_BETTER_QUERY = """\
A previous document search failed to find useful information. Help improve \
the search query.

Dimension we are trying to cover: {dimension}

Failed search query (retrieved nothing useful): {failed_query}

Context gathered so far: {context}

Instructions:
- Think about WHY the original query might have failed (too specific, wrong \
  terminology, missing key entity, etc.).
- Suggest a reformulated query that is more likely to find relevant documents.
- Try a different angle: use synonyms, broader/narrower terms, or rephrase \
  around a related entity.
- Keep the query to 3-10 keywords.

Return ONLY valid JSON. No markdown code fences. No explanation before or \
after the JSON.

{{
  "query": "improved search query here"
}}"""


EXTRACT_SHORT_ANSWER = """\
You are extracting the final short answer from a detailed analysis. \
Be very careful.

Question: {question}

Supporting evidence (TRUST THIS MOST):
{evidence}

Detailed analysis (may contain errors or contradictions):
{detailed_answer}

RULES:
1. The answer must be a direct response to the exact question asked.
2. The answer MUST appear in the supporting evidence. Do not invent \
answers not grounded in evidence.
3. If the question asks "what X" (e.g. "what heritage", "what year", \
"what nationality"), return the X — NOT yes/no.
4. Only return "yes" or "no" if the question literally starts with \
"Did", "Do", "Does", "Is", "Was", "Were", "Are", "Has", "Have", \
"Can", "Could", "Will", or "Would".
5. If the detailed analysis contains contradictions, choose the answer \
best supported by the evidence.
6. Return ONLY the shortest possible answer — a name, date, number, or \
short phrase. No explanation. No preamble.

Examples of correct output format:
  Question: "Who directed Inception?" → Christopher Nolan
  Question: "Is water a compound?" → yes
  Question: "What year was the Eiffel Tower built?" → 1889
  Question: "What heritage does the Black Book band have?" → Dutch heritage
  Question: "How many legs does a spider have?" → eight

Question again: {question}
Answer:"""


EXTRACT_SHORT_ANSWER_ALT = """\
Given the question, the raw evidence, and a detailed analysis, state the \
precise answer as briefly as possible.

Question: {question}

Raw evidence retrieved:
{evidence}

Analysis:
{detailed_answer}

Rules:
- Give only the answer itself — no explanation, no preamble.
- The answer must appear in or be inferable directly from the raw evidence.
- If the evidence and the analysis disagree, the evidence wins.
- yes/no for yes/no questions; a single name, year, or phrase otherwise.

Answer:"""


COMPRESSION_PROMPT = """\
You are rewriting a comprehensive answer into a concise factual answer that matches \
the style of reference encyclopedia entries.

ORIGINAL COMPREHENSIVE ANSWER:
{A_final}

TASK: Rewrite this as a single cohesive answer of approximately {target_words} words that:
1. Covers ALL distinct interpretations or disambiguation targets in the original
2. Uses VERBATIM names, dates, and key factual phrases — do NOT paraphrase proper nouns or numbers
3. Writes in natural flowing prose, NOT as bullet points or "[label]: [fact]" clauses
4. Omits background explanation, hedging, and meta-commentary about interpretations
5. Each distinct answer gets one clear sentence; do not pad or elaborate

Structure hint: {structure_hint}

BAD format (do not use):
"X has multiple answers depending on interpretation. Interpretation 1: [fact]. \
Interpretation 2: [fact]."

GOOD format (use this style):
"[Direct answer sentence covering first target]. [Direct answer sentence covering \
second target]. [Third if applicable]."

Real example of the target style:
Q: "Who sang I'm Coming Out?"
BAD:  "Multiple interpretations exist. Original: Diana Ross, 1980. Cover: Amerie, 2003."
GOOD: "\"I'm Coming Out\" was originally recorded by Diana Ross in 1980. \
In 2003, Amerie recorded an official cover for the film Maid in Manhattan."

Compressed answer (approximately {target_words} words, natural prose):"""


STANDARD_RAG = """\
Answer the following question based on the provided context. Be concise and \
factual. If the context does not contain enough information to answer \
definitively, say what you can infer and note the limitation.

Question: {question}

Context:
{context}

Provide only the answer as plain text. No preamble, no "Based on the context" \
— just the answer."""


CRITIQUE = """\
You are critically evaluating an answer to a question.

Question: {question}

Answer to evaluate:
{answer}

Identify what is MISSING, INCORRECT, or INCOMPLETE in this answer. Be \
specific:
- Point to exact facts that are wrong or uncertain.
- Identify important aspects of the question that were not addressed.
- Note if the answer is too vague, contradictory, or makes unsupported claims.

If the answer is fully correct and complete, say "The answer is complete and \
correct." and nothing else.

Provide your critique as plain text. Be concise and specific."""


REFINE = """\
You are improving an answer based on a critique and additional context.

Question: {question}

Original answer:
{answer}

Critique of the original answer:
{critique}

Additional context that may help:
{context}

Instructions:
- Fix the specific issues raised in the critique.
- Incorporate relevant information from the additional context.
- Keep all correct information from the original answer.
- Be concise — do not add unnecessary padding.

Provide only the improved answer as plain text. No preamble."""


if __name__ == "__main__":
    print("Testing fmt() helper...")


    result = fmt(MAKE_SEARCH_QUERY, question="Who is the CEO of OpenAI?")
    print("\nMAKE_SEARCH_QUERY formatted:")
    print(result)


    print("\nTesting missing key handling:")
    result_bad = fmt(SPLIT_QUESTION, question="What is X?")
    print(f"Template returned (first 80 chars): {result_bad[:80]}...")


    import re

    templates = {
        "DECOMPOSE_QUERY": (DECOMPOSE_QUERY, ["query"]),
        "SPLIT_QUESTION": (SPLIT_QUESTION, ["question", "context"]),
        "LEAF_DECISION": (LEAF_DECISION, ["question", "depth", "max_depth"]),
        "MAKE_SEARCH_QUERY": (MAKE_SEARCH_QUERY, ["question"]),
        "EXTRACT_EVIDENCE": (EXTRACT_EVIDENCE, ["question", "documents"]),
        "AGGREGATE": (AGGREGATE, ["question", "children_qa"]),
        "DRAFT_ANSWER":      (DRAFT_ANSWER,      ["question", "documents"]),
        "AMBIGUITY_CRITIQUE":(AMBIGUITY_CRITIQUE,["question", "draft_answer", "documents"]),
        "COMPOSE":  (COMPOSE,  ["question", "existing", "increment", "gaps"]),
        "REWRITE":  (REWRITE,  ["question", "current_answer", "new_evidence", "target_dimensions"]),
        "DISCOVER_DIMENSIONS": (DISCOVER_DIMENSIONS, ["query", "dimensions_list", "current_answer"]),
        "SUGGEST_BETTER_QUERY": (SUGGEST_BETTER_QUERY, ["dimension", "failed_query", "context"]),
        "EXTRACT_SHORT_ANSWER":     (EXTRACT_SHORT_ANSWER,     ["question", "detailed_answer", "evidence"]),
        "EXTRACT_SHORT_ANSWER_ALT": (EXTRACT_SHORT_ANSWER_ALT, ["question", "detailed_answer", "evidence"]),
        "STANDARD_RAG": (STANDARD_RAG, ["question", "context"]),
        "CRITIQUE": (CRITIQUE, ["question", "answer"]),
        "REFINE": (REFINE, ["question", "answer", "critique", "context"]),
    }

    print("\nChecking all templates for required placeholders:")
    all_ok = True
    for name, (template, required_keys) in templates.items():
        found = set(re.findall(r"\{(\w+)\}", template))
        missing = [k for k in required_keys if k not in found]
        status = "OK" if not missing else f"MISSING: {missing}"
        print(f"  {name:<28} {status}")
        if missing:
            all_ok = False

    print(f"\nAll checks passed: {all_ok}")


GROUNDED_DECOMPOSE = """You are a meticulous research assistant that decomposes complex questions into exhaustive, non-overlapping dimensions.
Rules:
1. Return ONLY valid JSON — no prose, no markdown fences.
2. Produce exactly {num_dimensions} dimensions.
3. Each dimension must have exactly {num_sub_aspects} sub_aspects.
4. sub_aspects must be atomic: one retrievable fact per entry.
5. Do NOT repeat information across dimensions.
6. Preserve verified named entities: {verified_entities}.

Query: {query}
Draft answer (for context only — do not copy it):
{draft_answer}
Retrieved passages (for grounding):
{retrieved_passages}

This is a short-answer question. Decompose the facts and reasoning needed to answer THIS question. Do not invent alternative referents, unrelated people, or background topics.
Return {{"dimensions": [{{"name": "short label", "ideal": "one-sentence target", "sub_aspects": ["atomic fact", "atomic fact", "atomic fact"]}}]}}.
"""

COVERAGE_AUDIT = """You are a coverage auditor. Propose facets missing from the answer AND not represented by any existing leaf.
Rules:
1. Return ONLY valid JSON.
2. Propose at most {max_new_dims} new facets.
3. Each must be semantically distinct from all existing leaves.
4. Only propose query-relevant facets that keyword retrieval can support.
5. Return {{"new_dimensions": []}} when nothing is missing.

Task: {task_scope}
Original question: {query}
Current answer:
{current_answer}
Existing leaf descriptions (already tracked):
{existing_leaves}

Return {{"new_dimensions": [{{"name": "facet", "explanation": "why needed for the original question", "sub_aspects": ["atomic fact"]}}]}}.
"""

RETRY_LEAF_QUERY = """Suggest an alternative retrieval query for a gap whose earlier query did not resolve coverage.
Return ONLY JSON. The query must contain 3-10 keywords, preserve the question's entities, and use different wording where useful. Do not introduce unsupported named entities.
Original question: {original_question}
Gap facet: {gap_description}
Previous query: {original_query}
Previous retrieved passages:
{failed_passages}
Return {{"query": "alternative keywords"}}.
"""
