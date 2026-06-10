# The Unofficial Guide — Project 1

> A Retrieval-Augmented Generation (RAG) system over unofficial NOVA student knowledge.
> This README documents the final system, its evaluation against the **real ingested corpus**,
> and the main tradeoffs made along the way.

---

## Demo Video

📹 **[Watch the 3–5 minute demo](https://www.loom.com/share/4c921576ec8f4eed94e65e40ec1d3b23)**

The walkthrough shows three queries answered with visible source citations, one query where retrieval and generation work well, the physics question where the system correctly refuses (the documented failure case), and a tour of the evaluation report.

---

## Domain

This system covers unofficial student knowledge about the Computer Science and Engineering path at Northern Virginia Community College (NOVA). The corpus focuses on transfer paths, course difficulty, workload advice, programming-language expectations, and sequencing decisions that are rarely captured in official catalogs. That information is valuable because the official course descriptions explain requirements, but they do not explain what students actually experience in class, how transfer pathways behave in practice, or which courses are risky to take out of sequence.

---

## Document Sources

The project uses 10 sources from `r/nvcc` (nine Reddit threads plus the community wiki index). Together they cover programming-sequence advice, data-structures difficulty, ZyBooks, the CS-major requirements, discrete-math study strategies, engineering design, physics choices, and transfer planning.

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | CSC 221 course advice | Reddit thread | https://www.reddit.com/r/nvcc/comments/xcujjw/please_advise_me_regarding_csc221_course/ |
| 2 | CSC 223 conceptual discussion | Reddit thread | https://www.reddit.com/r/nvcc/comments/1d3qzwx/csc_223/ |
| 3 | CSC 223 summer-semester details | Reddit thread | https://www.reddit.com/r/nvcc/comments/1txt0tq/csc_223/ |
| 4 | The ZyBooks situation | Reddit thread | https://www.reddit.com/r/nvcc/comments/ymwqzt/the_zybooks_situation/ |
| 5 | Computer Science major question | Reddit thread | https://www.reddit.com/r/nvcc/comments/18jfmba/computer_science_major_question/ |
| 6 | MTH 288 studying advice | Reddit thread | https://www.reddit.com/r/nvcc/comments/syzpm2/discrete_math_mth_288_studying_advice/ |
| 7 | EGR 122 evaluation | Reddit thread | https://www.reddit.com/r/nvcc/comments/qf9b29/egr_122/ |
| 8 | PHYS 232 with Prof Medvar | Reddit thread | https://www.reddit.com/r/nvcc/comments/uzgmh7/phys_232_with_prof_medvar/ |
| 9 | Transfer from A.S. Engineering to CS | Reddit thread | https://www.reddit.com/r/nvcc/comments/1t92mmm/transfer_from_as_engineering_to_cs_degree_at_va/ |
| 10 | NOVA wiki index | Community wiki | https://www.reddit.com/r/nvcc/wiki/index/ |

The raw exports live in [documents/](documents/) as `thread1.json` … `thread10.json`.

---

## Pipeline at a Glance

```
ingest.py            build_index.py              retrieval_engine.py        app.py
documents/*.json  →  clean + recursive chunk  →  embed (MiniLM) + upsert →  retrieve top-k →  rerank →  Groq grounded answer
                     (243 chunks)                 (ChromaDB, cosine)         (k=50 pool)      (k=5)     (+ source list)
```

- **`ingest.py`** — parses the Reddit JSON exports, cleans HTML/boilerplate, prepends thread title + original post into each chunk, and recursively chunks the text.
- **`build_index.py`** — the glue between ingestion and the vector store: it runs ingestion, then embeds and upserts the chunks into a freshly rebuilt ChromaDB collection. Run this once before querying.
- **`retrieval_engine.py`** — ChromaDB persistence, embedding, and top-k retrieval with structured, attributable results.
- **`app.py`** — retrieval → lexical reranking → grounded generation through Groq, wrapped in a Gradio UI.

**To reproduce the index and run the app:**

```bash
python build_index.py     # builds chroma_db/ from the 243 real chunks
python app.py             # launches the Gradio interface at http://127.0.0.1:7860
```

---

## Chunking Strategy

**Chunk size:** 1200 characters &nbsp;&nbsp; **Overlap:** 250 characters &nbsp;&nbsp; **Final chunk count: 243**

**Why these choices fit these documents:** Reddit comments are short, but they are context-dependent and full of pronouns like "that class," "his professor," or "this sequence." A recursive character splitter (implemented in [ingest.py](ingest.py#L348) as `recursive_split_text`) is a good fit because it preserves natural breaks first (`\n\n`, then `\n`, then sentence boundaries) and only falls back to a hard cutoff when needed. The 1200-character target keeps chunks small enough to stay topic-focused while still preserving enough of the parent post or comment chain to make a standalone chunk meaningful.

The 250-character overlap preserves local context across boundaries so a course code, professor name, or transfer note does not get separated from the comment that refers to it. Before chunking, the ingestion script strips HTML markup, decodes entities (`&amp;`, `&#39;`, `&nbsp;`), removes `[deleted]`/`[removed]` markers and bot boilerplate, and prepends the thread title (and, for replies, the ancestor comment chain) into each chunk.

The 243-chunk total sits comfortably inside the "50–2,000" healthy range from the spec: large enough that specific queries match precisely, small enough that each embedding carries real meaning.

---

## Sample Chunks

Five representative chunks pulled from the live collection (`build_index.py` reports 243). Each is shown with its source document.

**Sample 1 — `thread1.json`** (chunk_type: post)
> THREAD TITLE: Please advise me regarding CSC-221 course
> ORIGINAL POST: Hi everyone! … - 35 years old, a husband and a dad of 2 little kids, taking my first semester this fall at NOVA after 17 years of working. - No prior programming experience. - I haven't touched Math since 2005. 3 months ago I reviewed Pre-Algebra, Algebra 1, and Algebra 2 on Udemy. … I plan to take CSC-221 (Introduction to Problem Solving and Programming) in the Spring 2023 semester.

**Sample 2 — `thread10.json`** (community wiki, chunk_type: post)
> THREAD TITLE: Common Questions: ORIGINAL POST: #Common Questions: ##Transferring: - George Mason University (GMU): - Use the transfer equivalency chart to know what the NVCC to GMU course equivalencies are: [GMU Guide] - Are you deadset at GMU? If your major has the ADVANCE program, DO IT! You can take GMU courses that cannot be offered at NVCC at NVCC tuition costs …

**Sample 3 — `thread2.json`** (chunk_type: post)
> THREAD TITLE: Csc 223? ORIGINAL POST: Hey, I'm trying to get a masters in CS at VT and required to get a B or better to be admitted. I took intro to Java, intro python and some other basic coding courses in and out of school. I haven't taken Csc 222 or 221. How difficult is Csc 223? How are the proctored exams? …

**Sample 4 — `thread3.json`** (chunk_type: post)
> THREAD TITLE: CSC 223 ORIGINAL POST: Hey, currently taking CSC 223 online over the summer with professor Jiang Li. I don't really have much experience with Java or OOP in general but I think I'm getting along fine in the beginning with Youtube and the Zybooks. … how does the workload look like for the course as it progresses?

**Sample 5 — `thread4.json`** (chunk_type: post)
> THREAD TITLE: The zyBooks situation ORIGINAL POST: I think that most CS students right now agree that the zyBooks textbooks are really frustrating and annoying to use … The new CSC 221-223 classes were redesigned and first implemented this year to conform more closely with the computer science curriculum at universities throughout Virginia so that it'll be easier to transfer after Nova.

Each chunk is self-contained: the thread title is bound to the body, so a chunk is answerable on its own without reading the rest of the thread.

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers`, with `normalize_embeddings=True` and cosine space in ChromaDB (`{"hnsw:space": "cosine"}`).

**Production tradeoff reflection:** `all-MiniLM-L6-v2` is a good local baseline because it is fast, compact, and good enough for short forum text — and it runs offline with no API key or rate limits. If cost were not a constraint, I would weigh a stronger model (better handling of domain-specific phrasing like course abbreviations, better robustness to thread-local slang, longer context) against latency, memory footprint, and local-deployment simplicity. For this project the lightweight model was the better fit: the corpus is small, the deployment is local, and the main challenge is conversational context rather than long-document reasoning.

---

## Retrieval Approach & Test Results

**Top-k:** 5 chunks reach the LLM. To improve precision, retrieval pulls a wider candidate pool (`RETRIEVAL_CANDIDATE_POOL = 50`) and then reranks down to the top 5 (see Reranking below).

**Distance scores are cosine distance** (lower = closer); `similarity = 1 − distance`. Below are the top-3 retrieved chunks for three evaluation queries, taken directly from the live index.

**Query: "Should a complete beginner take CSC 221, or skip straight to CSC 222?"**

| Rank | Distance | Source | Snippet |
|------|----------|--------|---------|
| 1 | 0.219 | thread1.json | "You really don't need to prepare for CSC221, it's a basic introductory class … aimed at people who have no background in programming…" |
| 2 | 0.270 | thread9.json | "Regarding the AS in engineering vs computer science, you can sign yourself up for whatever courses you want…" |
| 3 | 0.285 | thread1.json | "No. I did CSC 221 and am doing CSC 222 now. CSC 221 is recommended for a complete beginner…" |

*Why these are relevant:* ranks 1 and 3 come straight from the CSC-221 advice thread and directly address whether a beginner should take 221 or skip it — rank 1 is a strong match at distance 0.219, well under the 0.5 "good retrieval" bar.

**Query: "What do students say about studying for MTH 288 discrete math?"**

| Rank | Distance | Source | Snippet |
|------|----------|--------|---------|
| 1 | 0.464 | thread6.json | "Just look over your homework. 288 is not as complex as you're brain is making you think it is." |
| 2 | 0.482 | thread6.json | "…Can't recommend Professor Leonard enough…" |
| 3 | 0.526 | thread6.json | "Chegg" |

*Why these are relevant:* all three top hits come from the MTH 288 thread (`thread6.json`) and contain the actual study strategies students gave.

**Query: "Do the CS classes at NOVA use C, or do they use another language?"**

| Rank | Distance | Source | Snippet |
|------|----------|--------|---------|
| 1 | 0.475 | thread1.json | "Hi I know this was a long time ago but what language did your class use in CSC-221? Was it python?" |
| 2 | 0.519 | thread2.json | "…trying to get a masters in CS at VT… I took intro to Java, intro python…" |
| 3 | 0.528 | thread5.json | "Computer Science major question … Is Physics a requirement for George Mason?" |

*Note:* here the **highest-similarity chunk is the *question* ("was it python?"), not the answer.* The actual answer ("CSC 221: Python; CSC 222/223: Java") is a very short comment with low semantic density. This is exactly the case the reranker exists to fix (see below).

### Reranking

Pure semantic similarity is not always enough for short, keyword-heavy answers. [`app.py`](app.py#L206) adds a lightweight reranking pass over the 50-candidate pool with three signals:

1. **Semantic similarity** (the base score).
2. **A bounded lexical nudge** — the fraction of query terms present in the chunk, weighted `0.1`. It is normalized to `[0, 1]` and capped deliberately so a chunk that merely repeats many query tokens **cannot** overpower a chunk that is semantically more relevant.
3. **A question-gated language bonus** — if (and only if) the *question* is about programming languages, chunks that name Python/Java get a small boost, with extra weight when they also mention a CSC 22X course code.

> ⚠️ **Tuning note (an honest one):** the first version of this reranker weighted raw lexical hits at `0.15` and applied the language bonus based only on the *chunk's* text. That backfired: for "should a beginner take CSC 221 or 222," every chunk in the CSC-223 transfer thread (`thread2.json`) was dense with "CSC 221/222/223" tokens and mentioned Java, so all five top slots were captured by the *wrong* thread and the system fell back with "I don't have enough information." The fix was to (a) normalize the lexical signal and shrink its weight, and (b) gate the language bonus on the **question**, not the chunk. After that change the beginner question retrieves the correct thread-1 chunk and answers correctly, while the language question still surfaces its short answer.

---

## Grounded Generation

**Model:** Groq `llama-3.3-70b-versatile`, `temperature=0.1`, `max_tokens=700`.

**How grounding is enforced (system prompt):**

> You are the Unofficial NOVA Student Guide assistant. Answer the user's question only using the provided context blocks. Do not use outside knowledge, training data, assumptions, or guesses. Cite every factual claim inline using the source labels exactly as written, such as [Source A]. If the provided context does not contain enough specific facts to answer the question, respond exactly with: *I don't have enough information on that.*

**How source attribution is surfaced:** retrieved chunks are formatted into labeled blocks (`[Source A]`, `[Source B]`, …) by `_build_context`, and a **deduplicated source list is built programmatically** from the retrieved chunks' metadata — it is not left to the LLM to invent. The UI shows the generated answer in one field and the source list in a second field, so every answer is traceable to the documents that supported it.

---

## Example Responses

**Example 1 — grounded answer with citation** (Q: "Should a complete beginner take CSC 221, or skip straight to CSC 222?")
> A complete beginner should take CSC 221, as it is a basic introductory class starting from the very beginning and aimed at people who have no background in programming **[Source A]**. It gives a solid foundation to transition to CSC 222 **[Source A]**. Skipping straight to CSC 222 is not recommended for complete beginners, but it might be possible for those with prior programming knowledge **[Source B]**. Additionally, the CS associate's degree requires taking all three CSC 221, 222, & 223 **[Source A]**.
>
> **Retrieved Sources:**
> Source A: …/please_advise_me_regarding_csc221_course/ / thread1.json
> Source B: …/transfer_from_as_engineering_to_cs_degree_at_va/ / thread9.json

**Example 2 — grounded answer with citation** (Q: "Do the CS classes at NOVA use C, or do they use another language?")
> The CS classes at NOVA use Python in the context of CSC-221 **[Source A]**, and Java for the later courses — one user mentions "intro to Java" and CSC 223 using Java and OOP concepts **[Source B]**. There is no mention of the language C being used in the provided context.
>
> **Retrieved Sources:**
> Source A: …/please_advise_me_regarding_csc221_course/ / thread1.json
> Source B: …/comments/1d3qzwx/csc_223/ / thread2.json

**Example 3 — out-of-scope query correctly refuses** (Q: "Which dorm at NOVA has the best food?")
> I don't have enough information on that.

NOVA is a commuter community college with no dorms, and nothing in the corpus covers dining — so the system declines rather than fabricating an answer.

---

## Query Interface

A minimal Gradio web UI ([app.py](app.py#L308)), launched with `python app.py` at `http://127.0.0.1:7860`.

- **Input** — a single textbox, *"Your question"* (submits on Enter or via the **Ask** button).
- **Outputs** — two read-only textboxes: *"Answer"* (the grounded response with inline `[Source X]` citations) and *"Retrieved Sources"* (the deduplicated source list, one labeled line per source).

**Sample interaction transcript:**

```
Your question:  Do the CS classes at NOVA use C, or do they use another language?

Answer:         The CS classes at NOVA use Python in the context of CSC-221 [Source A],
                and Java for the later courses (intro to Java, CSC 223 using Java/OOP)
                [Source B]. There is no mention of the language C being used.

Retrieved Sources:
                Source A: https://www.reddit.com/r/nvcc/comments/xcujjw/... / thread1.json
                Source B: https://www.reddit.com/r/nvcc/comments/1d3qzwx/csc_223/ / thread2.json
```

---

## Evaluation Report

All five questions were run end-to-end against the **real 243-chunk corpus** (no mock seed data) through Groq.

| # | Question | Expected answer | System response (summarized) | Retrieval | Accuracy |
|---|----------|-----------------|------------------------------|-----------|----------|
| 1 | Should a complete beginner take CSC 221, or skip straight to CSC 222? | Beginners should take CSC 221 for a solid foundation; students with prior programming experience can request to skip to CSC 222 (CSC 222's first weeks review CSC 221). | Said a complete beginner should take CSC 221 for a solid foundation and that skipping to 222 is for those with prior experience, with inline citations to thread1/thread9. | Relevant (top dist 0.219) | **Accurate** |
| 2 | What advice is given for transferring from NOVA to a four-year CS/engineering program? | Don't chase the "easiest" classes; build a real foundation; a course-waiver form with documentation can substitute for a class; expect a difficulty jump after transfer. | Advised against taking the easiest class, stressed work ethic and understanding the material, and mentioned the course-waiver form, citing thread2/thread9. | Relevant (top dist 0.298) | **Accurate** |
| 3 | Which physics sequence do students recommend for engineering-oriented transfer, and when should each option be used? | University Physics (PHYS 231/232) is calculus-based and required for engineering/rigorous CS; College Physics (PHYS 201/202) is algebra-based and won't satisfy engineering transfer. | Returned the fallback: *"I don't have enough information on that."* | Partially relevant | **Inaccurate (fallback)** |
| 4 | What do students say about studying for MTH 288 discrete math? | Use Professor Leonard videos, focus on understanding over memorization, and practice consistently/weekly. | Recommended looking over homework, using Professor Leonard on YouTube, repeating homework until second nature, and Chegg — cited thread6. | Relevant (top dist 0.464) | **Accurate** |
| 5 | Do the CS classes at NOVA use C, or do they use another language? | Not C: CSC 221 uses Python; CSC 222/223 use Java (via ZyBooks). | Said the classes use Python (CSC-221) and Java (later courses) and explicitly noted C is not mentioned — cited thread1/thread2. The per-course split was slightly hedged. | Relevant (top dist 0.475) | **Partially accurate** |

**Retrieval quality scale:** Relevant / Partially relevant / Off-target
**Response accuracy scale:** Accurate / Partially accurate / Inaccurate

**Summary:** 4 of 5 questions were answered from the corpus with inline citations (Q5 captured the key facts but hedged the per-course breakdown); Q3 failed and is analyzed below.

---

## Failure Case Analysis

**Question that failed:** Which physics sequence do students recommend for engineering-oriented transfer, and when should each option be used?

**What the system returned:** `I don't have enough information on that.`

**Root cause (tied to a specific pipeline stage — retrieval coverage, not generation):** This question requires a *comparison* — University Physics (PHYS 231/232, calculus-based, required for engineering) **versus** College Physics (PHYS 201/202, algebra-based, not accepted). The corpus has a PHYS 232 thread (`thread8.json`) that describes University Physics II as calculus-based, but **no single chunk, and no single thread, contains both halves of the contrast.** At query time the top results were dominated by the engineering→CS transfer thread (`thread9.json`, all three top hits at distance ~0.465) and the PHYS chunk that did surface only covered one side. Because the system prompt strictly forbids inferring or combining outside knowledge, the generator correctly refused rather than fabricating the missing half. This is **incomplete retrieval coverage for a multi-fact comparison**, not hallucination — and a strict, honest fallback is the desired behavior here.

**What I would change to fix it:** (1) raise top-k for comparison-style questions so both PHYS chunks can co-occur in the prompt; (2) add the College Physics (PHYS 201/202) perspective to the corpus, since it is currently underrepresented; (3) add a query-decomposition step that splits comparison questions ("X vs Y, when to use each") into sub-queries and retrieves for each side separately before assembling one grounded context.

---

## Spec Reflection

**One way the spec helped:** The planning document committed me to a strict fallback response, inline citations, and top-k retrieval *before* I wrote code. That kept every implementation choice in `ingest.py`, `retrieval_engine.py`, and `app.py` aligned around context preservation and answer traceability rather than around producing the longest possible answer. When Q3 had no complete evidence, the pre-committed fallback rule made "refuse" the obvious, correct behavior instead of a special case.

**Three ways the implementation diverged from the spec, and why:**
1. **A missing ingestion→index step.** The plan implied chunks would flow straight from ingestion into the vector store, but no glue existed and the store had been seeded with mock chunks. I added [build_index.py](build_index.py) so the store is rebuilt deterministically from the real 243 chunks only — this removed planted "answer" chunks that were inflating evaluation scores.
2. **The reranking rule was re-scoped.** The plan called for "a lightweight ranking rule that boosts direct answers with matching course numbers." In practice that exact rule backfired — it promoted lexically dense but off-topic chunks and broke the beginner question. I re-scoped it to a *bounded, normalized* lexical nudge plus a *question-gated* language bonus (see Reranking). The lesson: keyword boosts must be capped and conditioned on the query, or they override semantic relevance.
3. **One evaluation question was swapped.** The original plan listed "CSC 205 vs CSC 215," but the collected corpus contains no thread on those courses, so any "correct" answer could only have come from outside the documents. I replaced it with "Should a complete beginner take CSC 221 or skip to CSC 222?", which the corpus genuinely answers, and kept the physics question as the documented failure.

---

## AI Usage

**Instance 1 — Ingestion and chunking**
- *What I gave the AI:* my `planning.md` chunking and retrieval spec plus `requirements.txt`.
- *What it produced:* a first-pass `ingest.py` that parses Reddit Listing exports, recurses through nested comments, preserves parent-thread context, cleans HTML, and prints validation chunks.
- *What I changed or overrode:* I handled the wiki-page edge case, adjusted metadata normalization, and made the chunker robust to the actual export format in `documents/`.

**Instance 2 — Retrieval engine**
- *What I gave the AI:* the Milestone 4 requirements (persistent ChromaDB, `all-MiniLM-L6-v2`, structured attributable results).
- *What it produced:* a Chroma-backed retrieval engine with upsert/query functions and a smoke test.
- *What I changed or overrode:* I removed reliance on the built-in mock corpus for evaluation and wired retrieval to the real index via `build_index.py`.

**Instance 3 — Generation and interface**
- *What I gave the AI:* the Milestone 5 grounding and interface requirements (strict fallback, inline source labels, answer + source list).
- *What it produced:* a Gradio app connecting the retriever to Groq with graceful error handling.
- *What I changed or overrode:* I kept the interface intentionally minimal so the grounding behavior stayed obvious, and made error handling return user-facing messages instead of crashing the server thread.

**Instance 4 — Debugging the reranker and honest evaluation**
- *What I gave the AI:* the real end-to-end outputs for all five questions and the observation that the beginner question was falling back despite strong retrieval.
- *What it produced:* a diagnosis that the lexical weight was too high and the language bonus was firing on non-language questions, plus the gated/normalized rewrite.
- *What I changed or overrode:* I verified the fix didn't regress the language question, rebuilt the index without mock data, and re-ran the full evaluation so this README reflects the *actual* system behavior.
