# 2026-08-09 — Data strategy research record (law_v1 dataset build)

Verified sourcing + creation strategy for the law_v1 dataset. Governing spec (unchanged by this record): 15–20k examples, 60/16/24 source mix, ≥80% reasoning-trace hard floor, single 8192 packed bucket, drop-never-truncate, synthesis T=0.6–0.8, ~37–50M tokens ≈ one quota-week/epoch. This record is the build-phase reference; training-lane fixes are tracked separately in code.

All external claims below were verified by live fetch/search on 2026-08-08 unless marked otherwise.

---

## 1. Corrections and confirmations against the existing spec

1. **The 60/16/24 mix is source-based** (60% grounded synthesis / 16% curated legal sets / 24% general replay), not task-based. The task split lives inside the 60%. (Design spec `2026-08-04-indian-law-adapter-design.md:36-40`.)
2. **Closed-API distillation stays excluded** by spec line 14 (OpenAI/Gemini ToS bar training use). Only open-weight teachers on free serving tiers are in scope.
3. **OpenRouter `:free` correction:** at $0 balance the cap is **50 requests/day** (20 RPM), which makes the spec's Nemotron-`:free` primary-teacher route near-useless as planned. A one-time $10 top-up raises it to 1,000 req/day permanently (credits never expire) — the highest-ROI spend in the project, but it is a spend, so it needs an operator decision.
4. **Proposed charter amendment (operator decision pending):** make the blind pairwise judge the primary success gate and demote BhashaBench-Legal/MMLU/IFEval to forgetting guards — see §4 Aalap finding. The charter's ≥+3 MCQ delta may be unable to detect a genuinely good adapter.

## 2. Sources

### 2.1 Ranked table (quality × license-safety × reasoning-potential)

| # | Source | License | Role |
|---|---|---|---|
| 1 | AWS Open Data: `s3://indian-high-court-judgments` (17.8M judgments, 25 HCs, ~1.25 TiB) + `s3://indian-supreme-court-judgments` (~35k, 1950–2025, 52 GB) | CC-BY-4.0 | **Primary grounding corpus** |
| 2 | `L-NLProc/PredEx_Instruction-Tuning_Pred-Exp` (12,178 rows, real court reasoning in Output) | Apache-2.0 | **Reasoning seed — highest value/row on the Hub** |
| 3 | `GSMS-B/indian-legal-sections-bns-bnss-bsa-2023` (1,059 §§) + `GSMS-B/Indian-Legal-QA-BNS-BNSS-BSA` (6,354 grounded QA) | Apache-2.0 | **New-code grounding — nothing else covers this** |
| 4 | `sinhal/Indian_Supreme_Court_Judgments` (41,839 with `full_text`, `sections_cited`, `articles_cited`) | OpenRAIL | Dev shortcut only; release build re-extracts from AWS PDFs |
| 5 | `L-NLProc/TathyaNyaya-and-FactLegalLlama-NyayaFacts-Datasets` (25.4k) | Apache-2.0 | Second reasoning seed |
| 6 | `KanoonGPT/indian-case-laws` (17.1M rows, `neutral_citation` column) | Apache-2.0 | **Citation-verification index** (strip `headnote_text`, see §2.4) |
| 7 | Official gazette PDFs: BNS `egazette.gov.in/WriteReadData/2023/250883.pdf`, BNSS + BSA via mha.gov.in | §52(1)(q) Copyright Act | Authoritative statute text |
| 8 | `169Pi/indian_law` (47,789 CoT rows) | Apache-2.0 | **Style anchor only** — old-code content, ~600–800-token shallow traces |
| 9 | Zenodo 5088102: 858 Central Acts as JSON (≤2020) | CC-BY-4.0 | Non-criminal statute coverage |
| 10 | `opennyaiorg/InLegalNER` (46,545 entities, 14 types) | MIT | Entity extraction for the verifier |
| 11 | Indian Kanoon API | metered | Gap-fill only (see §2.3) |
| 12 | `opennyaiorg/aalap_instruction_dataset` (22,278) | mixed per-task | Drafting templates, CC-BY-NC tasks stripped |
| — | IL-TUR / IL-PCSR / ILDC / HLDC / BhashaBench-Legal / AIBE / `adalat-ai/indian-legal-exam-benchmark` | NC / ND | **EVAL + DECONTAMINATION ONLY** |
| — | SCC Online / Manupatra / LiveLaw / Bar & Bench headnotes+summaries | — | **UNSAFE** (see §2.4) |

### 2.2 AWS Open Data buckets — the headline find

- Managed by Dattam Labs; eCourts-sourced (their scraper solves the CAPTCHA). Registry: `registry.opendata.aws/indian-high-court-judgments/`, code `github.com/vanga/indian-high-court-judgments` (+ sibling SC repo).
- Access: `aws s3 sync --no-sign-request`, region `ap-south-1`. Prefixes `data/tar/`, `metadata/parquet/` (18 metadata fields incl. `citation`, `cnr`, `decision_date`, `disposal_nature`).
- **Caveat: judgment text is PDF-only** — no OCR/extraction provided. Budget real extraction engineering. OCR quality of pre-1990 PDFs unverified.
- Build-time check: whether the scrapers still resolve against the post-May-2025 SCR portal (`scr.sci.gov.in` replaced eSCR/DigiSCR).

### 2.3 Indian Kanoon API (terms fetched verbatim 2026-08-08)

- Metered: search/original doc ₹0.50, doc ₹0.20, fragment ₹0.05, metainfo ₹0.02. ₹500 signup credit.
- Non-commercial tier: "free Rs. 10,000 every month but will require use-case verification by the site administrator" ≈ 50k doc calls/month — **qualification process completely undocumented; do not plan around approval.**
- ToS explicitly permits fine-tuning use with "powered by IKanoon" attribution; redistribution of retrieved docs as a dataset is NOT covered.
- Scraping the free site is unsafe: robots.txt name-blocks CCBot/Semrush/Yandex/Ahrefs and lists ~9,000 court-ordered-redaction doc IDs (independent contempt/privacy risk).
- Verdict: AWS buckets cover the same documents cleaner; Kanoon is gap-fill only.

### 2.4 Copyright rules (controlling: *EBC v. D.B. Modak*, (2008) 1 SCC 1)

- Raw judgment text: public domain (also §52(1)(q)(iv) Copyright Act 1957 for judgments/orders).
- **Headnotes, footnotes, long notes: copyrighted** (genuine skill-and-judgment synthesis). Copy-editing/paragraph numbering creates no new copyright.
- Therefore: strip `headnote_text` from KanoonGPT (provenance undocumented); never use SCC Online/Manupatra/LiveLaw/Bar & Bench summaries (all four ToS also forbid extraction; EBC is the *Modak* plaintiff).
- SCR-portal judgments carry editorial headnotes — strip on ingest.
- India Code is **not** GODL — no bulk download, no API; the legal basis for statute text is §52(1)(q), and the machine-readable gazette PDFs (§2.1 row 7) are the source of truth. Derive our own IPC↔BNS mapping from the two statute texts; NCRB SANKALAN comparison charts are grey (ordinary government copyright).
- Grey/excluded: PRS (self-contradictory CC-BY vs non-commercial disclaimer), Law Commission reports (permission-by-email; format unverified; existence of reports 290+ unverified), parliamentary debates (NC), OpenNyAI Rhetorical Roles (Apache vs CC-BY-SA conflict unresolved).

### 2.5 BNS §358 — the transition finding that shapes the dataset

Offences committed **before 1 July 2024** continue under IPC/CrPC/IEA as if the new codes had not been enacted. **Both code families are live law simultaneously for a decade.** Consequences:
- Pre-2024 corpora are period-specific, not stale — the model must learn the temporal rule, not just new numbering.
- Nearly every public Indian-law dataset is silently frozen pre-July-2024 (incl. `169Pi/indian_law`, which enumerates IPC/CrPC/Evidence Act). A dedicated old↔new-code transition stream (~1,100 examples: "which code applies given the offence date", §358 savings reasoning) is simultaneously the biggest risk mitigation and the dataset's only genuine moat.

### 2.6 Hub inventory notes

- **No high-quality Indian-law reasoning-trace dataset exists** — that gap is this project's justification.
- `169Pi/indian_law`: only purpose-built Indian CoT set (prompt/complex_cot/response); 169Pi = Alpie-Core team (DeepSeek-32B lineage, MIT upstream — clean), but avg example ~1,046 tokens → traces 600–800 tokens; old-code coverage. Style/format anchor only; audit before any content use.
- Confirmed NOT on HF: IndicLegalQA (Mendeley/Kaggle, CC-BY-4.0), official MILDSum (email-gated), raw ILDC/HLDC, InLegalBERT pretraining corpus (never released).
- Tier C slop cluster (`viber1/indian-law-dataset` descendants, `AISimplyExplained/LegalReasoningIndianLaw` — no reasoning field despite the name): do not use. `joyboseroy/inIRAC` (511 rows, TinyLlama/Mistral-extracted): worthless as data; its IRAC schema + Verifier Agent design (Falkor-IRAC, arXiv:2605.14665) worth copying.

## 3. Reasoning-trace creation (zero API budget)

### 3.1 The binding constraint is the token budget

37–50M ÷ 15–20k = **~2,500 tokens/example**: ~700 prompt + ~1,400 `<think>` + ~400 answer.
- LIMO's best tier averages 5,290 reasoning tokens; R1 averages 12k (R1-0528: 23k) — naïve R1-style distillation blows the 8192 cap or halves the example count. Target 1,200–1,800 thinking tokens; get depth from structure + self-verification, not length.
- **Never paste full judgments.** IL-TUR prompt-length stats: ILDC ≈4.4k tokens, In-Abs summarization ≈5.8k, IL-PCR ≈10.7k (always dropped), L-NER ≈8.1k (always dropped). Segment with rhetorical roles (`opennyaiorg` role sets, `L-NLProc/LegalSeg_*`) to Facts+Issues+operative-provision prompts of 800–1,500 tokens.
- With prompts ≤1,500 and teacher `max_tokens` ≈4,000: expected drop rate <5% (vs ~17% for unfiltered math-R1 traces at 8k, ~100% for raw R1-on-AIME).
- Under `packing=True`+bfd, short examples cost nothing — zero incentive to pad traces.

### 3.2 Teachers — verified free tiers (open-weight only)

| Provider | Verified limits | Models | Realistic tok/day | Verdict |
|---|---|---|---|---|
| Groq | 30 RPM, 6,000 TPM (binding), 14,400 RPD | Qwen3-32B, gpt-oss, R1-Distill, Llama 4 | ~3–5M | Best volume. ToS clean: customer owns outputs; the no-training clause binds Groq, not us |
| Cerebras | 5 RPM, 30k TPM, 1M tok/day, **ctx capped 8,192** | gpt-oss-120b (Apache-2.0), GLM-4.7 | 1M | Best quality/token; 8k ctx matches the bucket |
| OpenRouter | $0: 50 req/day; after one-time $10: 1,000/day | rotating `:free` (R1, Qwen3, GLM) | — | Needs the $10 decision (§1.3) |

- Distillation licensing: DeepSeek-R1 MIT card explicitly permits "distillation for training other LLMs"; gpt-oss-120b Apache-2.0; Qwen3 Apache-2.0; GLM-4.6 MIT.
- Throughput: ~14k traces × 2,500 tok ≈ 35M tok → **7–10 days** saturated Groq+Cerebras, resumable. Entirely feasible without self-hosting.
- **Self-hosted teacher on Kaggle: rejected.** T4 sm_75 (no bf16, no Marlin, AWQ fp16-only) caps a 14B-int4 teacher at a few hundred tok/s; 35M tokens ≈ 20 GPU-h = two-thirds of the weekly quota the training run needs. Fallback only if API tiers collapse.

### 3.3 Qwen3 format specifics

- Thinking-mode sampling per model card: T=0.6, top_p 0.95, top_k 20, min_p 0; never greedy (repetition risk). Spec's T=0.6–0.8 is consistent.
- **Non-reasoning examples (≤20%) must emit an empty `<think>\n\n</think>` block** — Qwen3's mode-switching was trained that way; omitting the tags damages it.

## 4. Quality methodology — what measurably moves the needle

- **LIMO (arXiv:2502.03387):** 817 examples beat 100×-data baselines; the L1→L5 chain-quality ablation is ~15 AIME points. The single most-cited L5 property: **explicit self-verification steps** in the trace. → gate on verification presence.
- **s1 (arXiv:2501.19393):** difficulty, diversity, quality as validated selection axes.
- **"A Llama walks into the Bar" (arXiv:2504.04945):** legal MBE, LoRA. 20 samples/domain moved Llama-3 35.8%→52.5% (saturates fast); IRAC-structured explanations scale at **R²=0.839 vs 0.248 unstructured**. → domain breadth beats depth-per-domain; structure the outputs.
- **Aalap negative result (arXiv:2402.01758):** 22k Indian legal instructions on Mistral-7B improved task-shaped generation but **zero movement on AIBE/knowledge MCQ**. → the ≥+3 BhashaBench gate may not detect success; judge gate should be primary (§1.4).
- **MSLR negative result (arXiv:2511.07979):** human-designed CoT scaffolds cost QwQ-32B −33.8% LLM score / −10.2% IRAC recall vs self-initiated reasoning. → **never script the `<think>` block; enforce IRAC only in the final answer.** Probably the highest-value single rule in this record.
- **Citation reality (Magesh et al., JELS 2025 / arXiv:2405.20362):** Lexis+ AI 17%, Westlaw 33%, GPT-4 43% hallucination *with* RAG; general LLMs 58–82% on legal queries. → citation-existence must be a hard rejection gate, never repair.
- **Trace compression:** dense-factual domains (law-analogous: medicine −4.7–8.5 pts) suffer most from over-compression; difficulty-based length mixing (short for easy, long for hard) gave +1.2% at 8,192 with 2× fewer tokens — exactly this regime.
- **Decontamination:** 13-gram + Jaccard>0.8, plus embedding top-k (BGE-large) + LLM paraphrase judge (n-grams are trivially defeated by paraphrase). **Also dedup at case-identifier level (CNR/neutral citation)** — PredEx and IL-TUR CJPE share appellate pools; text-level screens miss case-level leakage. Screen against: IL-TUR (all 8 tasks), IL-PCSR, BhashaBench-Legal, adalat-ai benchmark, AIBE. Shared statute quotations alone are not contamination (spec rule stands).
- **Diversity axes:** statute-application vs precedent-reasoning (essential); civil/criminal/constitutional/commercial breadth (essential); old-code vs new-code (essential, §2.5); court hierarchy (cheap, moderate value); **Hindi: cut for v1** (curse of multilinguality at 8B/15–20k; all 29+ BhashaBench models score worse on Hindi; report BhashaBench-HI read-only to seed v2). `Wasserstoff-AI/legalTransEn_Indic` (89,951 En–Hi pairs, MIT) stays on the shelf.

## 5. Verifier design (cheap, concrete)

1. **Citation existence = O(1):** hash set of 17.1M `neutral_citation` values from KanoonGPT (SC format `2023 INSC <n>` adopted 06.07.2023, HC e.g. `2023:DHC:2720`; `neutralcitation.in` maps legacy citations for the pre-2023 tail). Statute checks are set-membership: GSMS-B 1,059 §§ + Zenodo 858 Acts + own IPC↔BNS mapping. Any unverifiable citation → reject the whole example.
2. **Temporal-validity gate (Indian-specific, nobody else does it):** post-01.07.2024 facts citing IPC → flag; pre-01.07.2024 citing BNS without §358 savings → flag.
3. **Self-verification presence** (LIMO L5 discriminator): traces with zero verification moves → drop.
4. Format floors: `<think>` present+closed, IRAC headings in answer, length band.
5. Two blind judges from different families than the generator, pointwise 1–5, pass = both ≥4 on every criterion (per design spec §1.4).
6. Falkor-IRAC metrics for the eval side: citation grounding accuracy, path validity, hallucinated-precedent rate, conflict detection.

## 6. Pipeline

- **Step 0 — grounding (CPU-only, ~1 week):** sync AWS SC metadata+PDFs (HC selectively by year); build the neutral-citation hash set (drop `headnote_text`); statutes from gazette PDFs + GSMS-B + Zenodo; segment judgments by rhetorical role into 800–1,500-token units. `sinhal/*` full_text for development only.
- **Step 1 — seeds:** PredEx (12,178) + TathyaNyaya (25.4k) + NyayaAnumana-Explanation. **Core trick: the teacher REWRITES authentic court reasoning into first-person `<think>` traces + IRAC answers — never synthesizes legal reasoning from nothing.** Grounding by construction; Aalap's failure mode is structurally impossible.
- **Step 2 — generate (7–10 days, resumable):** Cerebras gpt-oss-120b/GLM-4.7 + Groq Qwen3-32B/R1-Distill; prompts specify the output IRAC contract, never the thinking procedure; max_tokens ~4,000, prompts ≤1,500; randomize area/persona/difficulty; difficulty labeled at generation. SQLite state store, exponential backoff on 429, per-provider daily budgets, append-only raw outputs, idempotent resume.
- **Step 3 — verify:** rule-based floors first (saves quota), then judges (§5). Expect 50–75% survival → generate ~20–24k raw for ~14k accepted.
- **Step 4 — dedup + decontaminate** (§4).
- **Step 5 — mix and package (~18,000 examples, ~45M tokens, ≈93% traces):**

| Stream | Share | Count | Composition |
|---|---|---|---|
| Grounded synthesis | 60% | 10,800 | IRAC case analysis from seeds ~4,300; statute-application QA ~2,700; **old↔new-code transition ~1,100**; drafting with plan-then-draft traces ~1,900; judgment summarization + outcome reasoning ~800 |
| Curated legal | 16% | 2,900 | GSMS-B QA rewritten into think format ~1,200; PredEx-Prediction ~800; Aalap NC-stripped ~600; audited 169Pi subset ~300 |
| General replay | 24% | 4,300 | OpenThoughts-114k filtered <6k tokens ~2,500; Nemotron chat/STEM ~1,200; non-reasoning chat with empty `<think></think>` ~600 |

Held-out: 10% stratified by task type, chronologically-later judgments preferred.

## 7. Top risks

1. **Teaching repealed law** — mitigated by the temporal gate + transition stream + GSMS-B anchor + date-stamping every case-derived example.
2. **Citation hallucination baked in at generation** — mitigated by the hard existence gate (reject, never repair).
3. **Eval that can't detect success** (Aalap precedent) — mitigated by the §1.4 charter amendment.
Runners-up: NC-license leakage (per-example provenance metadata, NC stripped at ingest); packing/attention misconfiguration silently cross-contaminating packed sequences (tracked on the training-lane side).

## 8. If only five things

1. Pull the two AWS buckets + build the 17.1M neutral-citation index.
2. Build the 60% stream by rewriting PredEx/TathyaNyaya real reasoning — not synthesizing from scratch.
3. Enforce IRAC in the answer; leave `<think>` unscripted.
4. Citation existence + temporal validity as hard gates; ship the transition stream.
5. Budget 2,500 tok/example; segment don't paste; cut Hindi from v1.

## 9. Not verified / open items

- Law Commission report format + whether reports 290+ exist; AWS scraper vs post-May-2025 SCR portal; NyayaAnumana upstream license behind its Apache-2.0 HF tag; OCR quality of pre-1990 PDFs; legal accuracy of any Tier-C synthetic set (no manual fact-checking of legal content performed anywhere in this research).
- Operator decisions pending: $10 OpenRouter top-up; Indian Kanoon NC-tier application; §1.4 charter amendment.
