# Athlete Mental-Health Classifier — Meeting Report

**Date:** 2026-07-26 · **Prepared for:** Babak / Isaac · **Prepared by:** Naveen

A two-layer classifier that (1) finds athlete-mental-health comments in a 1.6 M-comment Reddit corpus,
then (2) tags each with the specific mental-health themes it contains. Both layers are built, applied
to the full corpus, and committed. This is the current status, the numbers, and where each layer
still needs work.

---

## The pipeline

```
raw corpus (1.63 M comments)
   → Layer 1 gate:  is this athlete mental health?           → 129,834 relevant comments
   → Layer 2 tags:  which of 10 MH themes does it contain?    → tagged study dataset
```

Output: `final_dataset_tagged.csv` — every relevant comment with a probability + binary flag per tag.

---

## Layer 1 — relevance gate (athlete × mental health)

**Method:** two RoBERTa models (`mh` and `sport`) combined by an AND rule. Base model is
`twitter-roberta`, further domain-adaptive-pretrained (DAPT) on our own corpus.

**Held-out performance** (n = 292, 47% relevant, balanced by construction):

| metric | value |
|---|---|
| Precision | **0.96** |
| Recall | **0.83** |
| F1 | **0.89** |
| Accuracy | **0.90** |

**Corpus application:** 129,834 of 669,110 keyword-matched comments flagged relevant (19.4%), vs the
baseline control arm at 0.49% — a **39× separation** that validates the gate isolates signal from
control.

**How it got here (for context):** roberta-base baseline F1 0.68 → domain-matched twitter-roberta →
over-flagging fixed via active learning → DAPT, F1 0.89. Domain match, not model size or more data,
was the breakthrough.

**Areas to improve:**
- **Second-rater agreement is measured but thin.** Human-vs-Claude κ was 0.81 (mh) / 0.86 (sport) on
  the underlying yes/no questions — good, but on a small set. A larger human audit strengthens the
  paper's ground-truth claim.
- **DAPT slightly hurt matched-vs-baseline separation** (48× → 39×) even while improving held-out
  metrics — it flags ~1,000 more control-arm comments. Worth a targeted audit before we publish a
  separation figure.
- **Residual false positives** cluster in physical-injury / training-logistics / meta-reply text
  (~the known floor). Further reduction has diminishing returns.

---

## Layer 2 — theme tags (10, multi-label)

**Method:** one transformer with 10 sigmoid heads, trained by **weak supervision** (rule-based silver
labels over 129k comments, with *abstention* on uncertain cases) then fine-tuned on **610 hand-labeled
gold comments**. Thresholds set by prevalence matching. Evaluated on a fixed 304-comment gold test set.

### Overall

| | Precision | Recall | F1 |
|---|---|---|---|
| **Micro (pooled across all tag decisions)** | **0.84** | **0.78** | **0.81** ✅ |
| Macro (mean across the 10 tags) | 0.82 | 0.76 | 0.78 |

Pooled, the layer clears the 0.8 bar. The macro average is dragged down by a few weak tags.

### What each tag means (plain-language)

**Condition tags (the problem):**
- **depression** — low mood as a state: hopeless, empty, "in a dark place," diagnosed depression.
- **anxiety** — general worry/panic/nerves/fear (social anxiety, gym-timidity) — *not* competition nerves.
- **stress_pressure** — feeling overwhelmed / under pressure from life/work/family. *Not* physical "stress" (stress fracture).
- **burnout_motivation** — *mental* burnout, no motivation, dreading training. *Not* sore muscles / physical fatigue.
- **performance_psych** — sports-*performance* psychology: choking, the yips, self-doubt tied to performing, pressure affecting play.
- **body_image_eating** — disordered eating or real body-image distress. *Not* ordinary calorie/macro/cutting talk.
- **injury_distress** — an injury causing *emotional* fallout (miserable, lost, fear of re-injury). Requires injury **and** distress.
- **self_harm_suicide** — suicidal thoughts/attempts, self-harm. *Not* hyperbole ("this workout is killing me").

**Response tags (what they do about it):**
- **help_seeking** — therapy, counseling, meds, diagnosis, "get some therapy." *Not* physical therapy/physio.
- **exercise_coping** — using exercise to manage mental health ("keeps me sane," "my therapy"), or it failing to. The corpus's most distinctive signal (~26%).

*Underlying rule for all: keyword ≠ theme — the word alone never earns the tag; negation flips it to no; jokes/figurative don't count.*

### Per-tag (gold test, n = 304)

| tag | P | R | F1 | test pos. | ≥0.8? |
|---|---|---|---|---|---|
| **anxiety** | 0.99 | 0.87 | **0.92** | 77 | ✅ |
| **depression** | 0.93 | 0.88 | **0.91** | 60 | ✅ |
| **burnout_motivation** | 0.83 | 0.86 | **0.84** | 28 | ✅ |
| **self_harm_suicide** | 0.84 | 0.84 | **0.84** | 19 | ✅ |
| help_seeking | 0.82 | 0.77 | 0.79 | 35 | ✗ (near) |
| body_image_eating | 0.75 | 0.75 | 0.75 | 44 | ✗ |
| exercise_coping | 0.76 | 0.74 | 0.75 | 78 | ✗ |
| stress_pressure | 0.83 | 0.65 | 0.73 | 31 | ✗ |
| performance_psych | 0.74 | 0.72 | 0.73 | 32 | ✗ |
| injury_distress | 0.71 | 0.48 | 0.57 | 21 | ✗ |

**4 of 10 tags are production-ready** (all of P/R/F1 ≥ 0.8): anxiety, depression, burnout, self-harm.

### Prevalence in the corpus (independent calibration check — tracks gold estimates closely)

exercise_coping 26% · anxiety 26% · depression 16% · body_image_eating 15% · help_seeking 12% ·
performance_psych 11% · burnout 10% · stress_pressure 8% · injury_distress 5% · self_harm_suicide 1%

**Areas to improve (per tag):**
- **`injury_distress` (F1 0.57 — the real weak point).** It's the only *conjunction* tag (injury AND
  distress); the model misses implicitly-linked cases. Needs its own rubric pass + a dedicated
  labeled set, not more of the same.
- **`stress_pressure` (recall 0.65).** "Stress" is the most overloaded word in a fitness corpus
  (stress fracture, joint stress); precision is protected at recall's cost.
- **`performance_psych`, `body_image_eating`, `exercise_coping` (0.73–0.75).** Fuzzy boundaries the
  rubric itself doesn't draw sharply — sharper sub-definitions (a v2 rubric) would help more than more
  data.
- **`help_seeking` (0.79)** — a near miss; likely crosses 0.8 with one more active-learning round.

---

## Cross-cutting: what's solid vs. what's provisional

**Solid:** Layer 1 (all metrics ≥ 0.8); Layer 2 pooled (micro F1 0.81) and the 4 production tags;
full corpus tagged; every step documented and reproducible; failures recorded, not buried.

**Provisional / needs a decision:**
1. **Six Layer-2 tags below 0.8** — recommend shipping the 4 validated tags now and marking the other
   6 as provisional (same as Layer 1 shipped a narrowed scope). *Decision for Babak.*
2. **No second-rater κ on Layer 2 yet** — all 610 gold labels are single-rater (Claude, from the
   written rubric). A human pass over a subset is required before the paper reports these as gold.
3. **`self_harm_suicide` rests on 19 test positives** — usable point estimate, wide confidence
   interval.

**Recommended next steps (priority order):**
1. Human second-rater pass on a gold subset → inter-rater κ for both layers.
2. Dedicated rubric + labeled set for `injury_distress`; one more AL round for `help_seeking`.
3. v2 sub-definitions for the fuzzy tags (`performance_psych`, `body_image_eating`, `exercise_coping`).
4. Audit the ~1,000 extra baseline-arm flags from DAPT before quoting a separation figure.

---

*Full technical detail: `docs/methods/relevance_layer1_process.md` (Layer 1), `docs/methods/layer2_tags_process_2026-07-25.md`
(Layer 2), `experiments/trials_log.md` (every run). Rubrics: `docs/methods/relevance_layer1_methods.md`,
`docs/methods/layer2_tag_rubric.md`.*
