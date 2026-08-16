# Meeting Report — Layer-2 Tag Expansion (v3)

**Date:** 2026-08-05 · **For:** Babak / Isaac · **By:** Naveen

Follows up the last meeting's directive to **refactor the data and retrain Layer 2 to allow for
multiple tags**, add more label categories, and weight by frequency. This report: what was done, the
current numbers, the decisions made, and what's next.

---

## 1. What the directive actually needed

Two of the asks were already in place and confirmed:
- **Multi-label is already how it works** — one shared transformer with independent sigmoid heads
  (`BCEWithLogitsLoss`, `problem_type="multi_label_classification"`), so a comment already receives 0,
  1, or several tags. Comorbidity ("if you have X you're likelier to have Y") is captured by the
  shared encoder.
- **Frequency weighting is already there** — per-tag `pos_weight = #neg/#pos`, capped.

So the real change was **adding tags** so off-taxonomy themes aren't forced into a wrong head.

## 2. What was done (10 → 17 tags)

Added 7 tags: `substance_use`, `loneliness_isolation`, `sleep`, `trauma_ptsd`,
`adhd_neurodivergence`, `identity_retirement`, `exercise_dependence`.

Full pipeline, same method as the original 10:
1. **Rules** — high-precision + high-recall + wrong-sense guards for each new tag, in the same
   abstention design. Guards target the fitness-corpus traps: *drink **water*** ≠ substance, *train
   **alone*** ≠ lonely, *rest-day/**recovery** sleep* ≠ sleep problem, joke *"ptsd from a **free
   throw**"*, *"**add** weight"* ≠ ADHD. Behaviourally unit-tested before training.
2. **Prevalence probe** — measured how often each new tag actually appears, to size labeling and flag
   the rare ones up front.
3. **Gold labeling** — the 7 new tags labeled on the 976 existing gold comments (kept their 10-tag
   labels + splits), **plus 382 enriched comments** labeled for all 17, so the new tags have test
   coverage. Two blinded multi-agent passes, same protocol as before.
4. **Silver** — regenerated 128k weakly-labeled rows for 17 tags. **Caught and fixed a leak** the
   merge introduced (the enriched rows lived only in the split files, not the source rated files —
   the exclusion now covers the split files too). **Verified 0 / 424 test-row leakage.**
5. **Retrained** — two-stage (silver → gold fine-tune), 17 heads, prevalence-matched thresholds.

## 3. Current measured statistics

All figures are **measured**, not estimated: Layer 1 on its held-out relevance test set, Layer 2 on
the 16-tag gold-test set (n=424, cue-enriched), prevalence-matched thresholds tuned on gold-dev.

### 3a. Layer 1 — relevance gate (athlete × mental health), one binary class

| Metric | Value |
|---|---|
| Precision | **0.96** |
| Recall | **0.83** |
| F1 | **0.89** |
| Accuracy | **0.90** |

Corpus application: 129,834 of 669,110 keyword-matched comments flagged relevant (19.4%) vs the
control arm at 0.49% — a **39× separation**. Second-rater agreement κ 0.81–0.86.

### 3b. Layer 2 — per-tag statistics (16 classes, gold-test n=424)

Precision / Recall / F1 / Accuracy for **every** classification label. `n+` = positive test cases.
Accuracy is per-tag (correct decisions ÷ 424); note it runs high on rare tags because most decisions
are true negatives — **F1 is the honest metric for these imbalanced labels, accuracy is not.**

| tag | P | R | F1 | Acc | n+ | ≥0.8 all |
|---|---|---|---|---|---|---|
| anxiety | 0.94 | 0.89 | **0.92** | 0.96 | 92 | ✅ |
| depression | 0.89 | 0.91 | **0.90** | 0.96 | 74 | ✅ |
| self_harm_suicide | 0.82 | 0.95 | **0.88** | 0.99 | 19 | ✅ |
| burnout_motivation | 0.84 | 0.90 | **0.87** | 0.98 | 29 | ✅ |
| help_seeking | 0.82 | 0.89 | **0.85** | 0.97 | 46 | ✅ |
| loneliness_isolation | 0.78 | 0.86 | 0.82 | 0.98 | 21 | ✗ (P .78) |
| body_image_eating | 0.79 | 0.83 | 0.81 | 0.95 | 58 | ✗ (P .79) |
| sleep | 0.85 | 0.74 | 0.79 | 0.98 | 23 | ✗ (near) |
| stress_pressure | 0.72 | 0.85 | 0.78 | 0.96 | 33 | ✗ |
| adhd_neurodivergence | 0.81 | 0.65 | 0.72 | 0.98 | 20 | ✗ |
| performance_psych | 0.69 | 0.67 | 0.68 | 0.95 | 33 | ✗ |
| injury_distress | 0.57 | 0.77 | 0.66 | 0.95 | 26 | ✗ |
| exercise_coping | 0.68 | 0.59 | 0.63 | 0.85 | 93 | ✗ |
| substance_use | 0.81 | 0.52 | 0.63 | 0.96 | 25 | ✗ (recall) |
| trauma_ptsd | 0.73 | 0.36 | 0.48 | 0.96 | 22 | ✗ (recall) |
| exercise_dependence | 0.17 | 0.23 | 0.20 | 0.90 | 22 | ✗ (rules under-fire) |

| Layer 2 aggregate | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| **Micro (pooled, 6,784 decisions)** | **0.77** | **0.76** | **0.76** | 0.955 |
| Macro (mean of 16 tags) | 0.74 | 0.72 | 0.73 | 0.96 |

**Production-ready now (all of P, R, F1 ≥ 0.8):** anxiety, depression, self_harm, burnout,
help_seeking → **5 solid tags**; loneliness/body_image/sleep sit right at the line.

### 3c. Overall pipeline (Layer 1 → Layer 2, compositional)

The two layers run as a cascade, so end-to-end performance on a raw comment is roughly the **product**
of the two stages: relevance is caught at F1 0.89, and a comment that passes is tagged at micro
F1 0.76 — i.e. a correct *relevant-and-tagged* decision lands near **0.89 × 0.76 ≈ 0.68** for a
typical tag, and higher (≈0.82) for the five production tags. This is a composition of the two
measured numbers, **not** a separately measured joint score — a true end-to-end eval on raw
(ungated) comments is listed under limitations.

> These are the **round-1** numbers for the currently committed model (`models/layer2_tags_v3`,
> pre-rule-broadening). `substance_use`, `trauma_ptsd`, and `exercise_dependence` under-fire; the
> broadened-rule retrain (§5) is in progress and expected to lift them.

Adding the new (mostly hard) tags pulled micro from the 10-tag model's 0.81 → 0.76 — expected: more
heads, several genuinely hard, shared encoder capacity.

**Why the weak ones are weak (diagnosed, not guessed):** the same cheap diagnostic that drove the
original build — *rule prevalence ÷ gold prevalence* — flags substance (0.25), trauma (0.36),
exercise_dependence (0.07) as **under-firing rules**. The model faithfully learns the narrow concept
the rules describe. This is the exact pattern the original 6 weak tags had, and it was fixed by
broadening the rules (its single biggest lever).

### 3d. Inter-rater reliability (κ) — the label quality, not the model

Previously Layer 2 had **no** second-rater agreement (the standing gap that blocked calling the
labels "gold"). We ran a **second independent blinded pass** over a 160-record stratified sample of
gold-test (every tag represented), applying the same rubric with no access to the first rater's
labels, then scored per-tag Cohen's κ (`code/metrics_interrater_l2.py`).

| | κ |
|---|---|
| **Pooled (all tag decisions)** | **0.879** |
| Mean of per-tag κ (16 tags) | **0.862** |
| Tags with κ ≥ 0.8 | **13 / 16** |

Per-tag κ ranges from 0.58 (stress_pressure) and 0.65 (burnout) — the two most semantically fuzzy —
up to 0.96 (substance/adhd/exercise_dependence). This is **on par with Layer 1's 0.81 / 0.86** and
says the codebook is applied consistently. **Honest scope:** the second rater is another blinded
*model* pass, so this measures **rubric reliability / test-retest consistency**, not human
validation — a human κ on the same committed blinded files is still the final step before print.
But it closes the "single-rater, no agreement at all" gap with a real number.

## 4. Decisions

- **Dropping `identity_retirement`** → **16 tags.** At 0.1% prevalence (only 7 gold-test positives, ~0
  in silver) it cannot be honestly evaluated at the 0.8 bar; the original taxonomy cut it for exactly
  this reason. Chasing it wastes rounds on something unmeasurable.
- **Keeping the other weak new tags** (substance, trauma, adhd, exercise_dependence) as **provisional**
  — they're improvable with the known lever.

## 5. What's next (immediate)

1. **Broaden the under-firing new-tag rules** into co-occurrence patterns (the proven r1→r2 lever) →
   regenerate silver → retrain → re-evaluate. Expect substance/trauma/exercise_dependence/adhd recall
   to rise toward 0.8.
2. **Apply the model to the full 130k dataset** (per the directive) — refresh `final_dataset_tagged.csv`
   with the 16 tag columns.
3. Ship the validated tags (~7–9 after the rule round), mark the rest provisional — same honest
   framing as Layer 1 and the original Layer 2.

## 5b. FINAL OUTCOME (2026-08-10) — supersedes §5

The rule-broadening in §5.1 was tried three ways (v3b batch-32, v3c batch-16, v3d selective) and **all
three net-regressed** vs round-1: the broadened silver is noisier and drags even untouched tags down
(body_image 0.81→0.63, injury 0.66→0.47). Recorded as a negative result. **Round-1 `v3` is the
deliverable.**

Applying an **F1 ≥ 0.65 release bar** to `v3` (Naveen's call) cut four tags — exercise_coping (0.63),
substance_use (0.63), trauma_ptsd (0.48), exercise_dependence (0.20) — leaving **12 released tags**:

| | Precision | Recall | F1 | κ |
|---|---|---|---|---|
| Layer 1 gate | 0.96 | 0.83 | **0.89** | 0.81 / 0.86 |
| Layer 2 (12 tags, micro) | 0.81 | 0.84 | **0.83** | 0.88 |
| Cascade end-to-end (measured) | 0.77 | 0.84 | **0.80** | — |

- **5 fully validated tags** (anxiety, depression, self_harm, burnout, help_seeking ≥0.8 on P/R/F1);
  7 provisional (0.66–0.79).
- **Joint eval now measured, not composed**: gate rejects 97% of true negatives (3% false-pass).
- **Final corpus: `final_dataset_tagged.csv` = 105,206 comments** (matched arm, filtered to ≥1 of the
  12 tags, multi-label; the 24,628 relevant-but-untagged are held out). Control arm written separately
  as `control_baseline.csv` for base-rate comparison. Built by `code/tag_corpus_final.py`.
- Full stats now in the paper draft (`docs/AMHC_paper_draft_v2.{md,docx}`), Tables 1–3.

## 6. Standing limitations (unchanged, worth restating)

- **Model-only second rater so far** — the κ 0.879 above is Claude-vs-Claude on the same rubric
  (rubric reliability). A **human** pass on the committed blinded files is still needed before the
  paper calls these labels human-validated gold. (Was: "no second rater at all" — now closed with a
  real reliability number, human validation pending.)
- **Enriched-sample metrics ≠ corpus metrics** — per-tag P/R/F1 are on a cue-enriched test set;
  natural prevalence comes from the proportional sample.
- **Eleven of sixteen tags are below 0.8** on the round-1 model and must not be presented as if they
  clear it. (The broadened-rule retrain, §5, targets three of them.)
- **No end-to-end joint eval yet** — the 0.89 × 0.76 ≈ 0.68 pipeline figure (§3c) is a composition of
  two separately measured numbers. A real joint measurement on raw ungated comments is a to-do.
- **Recreational, not elite, corpus** (numbers corrected 2026-08-10 from the actual 129,834-row
  dataset) — **77.5% recreational/individual-training** communities (top-4 fitness subs r/xxfitness,
  r/running, r/Fitness, r/bodybuilding = 66%); **~19% team/pro-sport** communities (r/nba 15%,
  r/tennis 4%) that are mostly *fan/spectator* discussion, not elite-athlete self-disclosure. (Earlier
  "~74% fitness / ~4% pro" was wrong — the 4-sub figure is 66% and pro-sport communities are ~19%.)
  The title, abstract, and any generalization claim have to honor that scope.
- **DAPT separation caveat** — DAPT improved every held-out Layer-1 metric but moved the
  matched-vs-baseline separation the wrong way (48× → 39×); the ~1,000 extra baseline-arm flags need
  an audit before any separation figure is quoted.

*Full technical detail + reasoning: `docs/layer2_tags_process_2026-07-25.md` §10, `docs/layer2_tag_rubric.md`
(v3 tag definitions), `experiments/trials_log.md` (row L2v3-1).*
