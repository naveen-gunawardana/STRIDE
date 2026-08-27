# Layer-2 Tag Classifier — Process & Diagnostic Log

**Started 2026-07-25.** Companion to `docs/methods/relevance_layer1_process.md`. Same BRM-style intent:
record the **reasoning and the failures**, not just the final number, so the methods section can be
written from this file.

Companions: `docs/methods/layer2_tag_rubric.md` (labeling rules), `experiments/trials_log.md` (one row per
run), `data/data_layer2_ratings/` (the hand labels behind every metric).

---

## 0. Where this sits

Layer 1 answers *is this comment about athlete mental health?* (`mh AND sport`, held-out
P 0.96 / R 0.83 / F1 0.89 / acc 0.90 after DAPT). Layer 2 answers the next question:
***which* mental-health themes does it contain?** — multi-label, run only on the comments Layer 1
kept. The cascade is:

```
raw corpus (1.63 M) → Layer 1 gate → 129,834 athlete-MH comments → Layer 2 tags → study dataset
```

---

## 1. Rebuilding the corpus on the DAPT gate (prerequisite)

The Layer-1 DAPT model improved every held-out metric, so the corpus was re-pruned with it before
any Layer-2 work, so the tags are learned from the dataset we will actually ship.
`code/driver_classify.py` over all three arms, 1,625,377 comments, 29.7 min on the RTX 2060.
Dataset assembly, previously ad hoc, is now scripted (`code/build_final_dataset.py`).

| arm | processed | relevant | % |
|---|---|---|---|
| matched 2018–2022 | 574,639 | 113,398 | 19.73% |
| matched 2023 | 94,471 | 16,436 | 17.40% |
| **matched total (the dataset)** | **669,110** | **129,834** | **19.40%** |
| baseline (control) | 956,267 | 4,700 | 0.49% |

### ⚠️ An honest negative result: separation got *worse*

| gate | held-out F1 | matched % | baseline % | separation |
|---|---|---|---|---|
| balanced (generic twitter-roberta) | 0.85 | 18.9% | 0.39% | **48×** |
| **DAPT** | **0.89** | 19.4% | 0.49% | **39×** |

The DAPT model wins on every held-out metric but **flags more of both arms**, and the control arm
grows proportionally more, so matched-vs-baseline separation falls 48× → 39×. Both cannot be
"better" — they measure different things. The held-out set is 47% relevant by construction; the
baseline arm is ~0.4% relevant. A model can be better calibrated on the former and slightly more
permissive in the far tail of the latter, which is what appears to have happened.

**We keep DAPT as the deliverable** because the target Babak set is held-out P/R/F1 ≥ 0.8 and DAPT
is strictly better there, but this tension is recorded rather than buried: *separation is a
secondary validation statistic and it moved the wrong way.* Worth a targeted audit of the ~1,000
extra baseline-arm flags before the paper claims a separation figure.

---

## 2. Designing the tag set from the data, not from a textbook

Rather than importing a clinical taxonomy, prevalence was measured first (`code/explore_tags.py`,
high-recall regex probes over the 110k relevant comments) and 50 comments were read raw to see
what the corpus actually contains.

Two facts shaped everything:

1. **The corpus is recreational, not elite.** 74% of relevant comments are from r/xxfitness,
   r/running, r/Fitness, r/bodybuilding. Only ~4% is pro-sport discussion (r/nba, r/tennis).
   A taxonomy built around elite-athlete constructs (selection pressure, retirement, media) would
   mostly not fire.
2. **"Exercise as coping" is everywhere** — *"running is the only thing that keeps me sane"* — and
   is arguably the corpus's most distinctive signal. It is a *response*, not a disorder, so a
   pure symptom taxonomy would have thrown it away.

### The v1 tag set (10, multi-label)

**Condition tags** — what the person is experiencing:
`depression`, `anxiety`, `stress_pressure`, `burnout_motivation`, `performance_psych`,
`body_image_eating`, `injury_distress`, `self_harm_suicide`

**Response tags** — what is being done about it: `help_seeking`, `exercise_coping`

Rubric: `docs/methods/layer2_tag_rubric.md`, written in the same style as the Layer-1 rubric (decided
rules first, anchor examples, explicit ✗ cases, negation flips to NO, figurative usage is NO).

**Considered and cut from v1** (recorded, not forgotten): `substance_use`, `loneliness_isolation`,
`sleep`, `trauma_ptsd`, `adhd_neurodivergence`, `identity_retirement`. Each is real in the corpus
(probe prevalence 1.9–5.6%) but is either dominated by wrong-sense text in a fitness corpus
("drinking" = water, "alone" = training solo, "sleep" = recovery) or too rare to *honestly
evaluate* at the 0.8 bar. Labeling exposed one more gap worth adding in v2: **exercise
dependence / addiction** ("my climbing addiction is crippling my life") has no home in v1 — it was
forced into `exercise_coping` or dropped.

---

## 3. Weak supervision with abstention (the core design decision)

100 hand labels cannot train a 10-head transformer. The training signal comes from labeling
functions over the 129k relevant comments (`code/layer2_lexicon.py`), but with a **three-valued**
output rather than the obvious binary one:

| rule outcome | silver label |
|---|---|
| high-precision cue fires, un-negated | **1** |
| high-precision cue fires but is **negated** | **0** (a hard negative) |
| only the broad high-recall cue fires | **abstain** — masked out of the loss |
| no cue at all | **0** |

**Why abstention is the whole trick.** If every comment without the precise cue were labeled 0,
then paraphrases the rules cannot express (*"i just don't want to get out of bed anymore"* for
depression) would be trained as **negatives**, teaching the model not to generalize past the
lexicon — which defeats the point of using a transformer at all. Abstaining leaves the model free
to learn those from context. The scale of this matters: for `exercise_coping` the rules fire on
3.4% of comments while the true prevalence is 27%, so **85,756 comments abstain** on that tag.
Without masking, nearly all of them would have been wrong negatives.

**Negated cues become explicit 0s, not abstains** — these are the highest-value negatives in the
set, teaching *"my anxiety was bad"* vs *"i don't get anxious"* instead of keying on the keyword.
Layer 1 needed exactly this hardening; Layer 2 inherits it.

Loss is `BCEWithLogits` with a per-element mask plus per-tag `pos_weight` (capped at 20) — tags run
1.1%–24% positive and without reweighting the rare heads collapse to all-zero.

### Two bugs found while unit-testing the labeling functions

Both were caught by a 25-case behavioural test before any training, not after.

1. **The negation guard fired on exhortations.** *"**don't let** a knee injury… get you depressed"*
   was killed as negated. "don't let X" asserts the theme; it does not deny it. Fixed with a
   `NEG_EXEMPT` list (`let/want/mean/saying/sure`) and by disabling the guard for the
   co-occurrence patterns (injury+distress, exercise+mood), where a negator near the first element
   says nothing about the construct.
2. **`\bdepress\b` cannot match "depress*ed*.”** A trailing word-boundary in an alternation of
   prefixes silently broke the entire `injury_distress` pattern. Classic regex own-goal; it would
   have shipped a tag with near-zero recall.

Wrong-sense guards were needed per tag, and are mostly fitness-specific: *"my arms **burnout** for
the rest of the set"* (muscular, not psychological), *"box squats put more **stress** on my back"*,
*"**stress** fracture"*, *"this leg workout is **killing me**"*, *"high lat insertions give
**suicidal ideation**"* (a gym joke), *"**binge** drinking"* (not eating). A corpus audit of rule
positives also demoted *"saved my life"* from high-precision to high-recall — it leaked hyperbole
(*"basketball saved my life"* about a rough patch).

---

## 4. Gold labeling — 610 comments, four samples, and why four

All labeling followed `docs/methods/layer2_tag_rubric.md` on **blinded text only** (no subreddit, author,
or date), matching the Layer-1 protocol. Provenance keys are written to `*_key.csv` and gitignored.

| sample | n | how drawn | purpose |
|---|---|---|---|
| `layer2_prop100` | 100 | year × sport-family, **proportional** | natural-prevalence estimate; **test only** |
| `layer2_tagstrat` | 330 | 30/tag via **high-recall** cues + 30 cue-free | per-tag support |
| `layer2_focus` | 120 | mid-precision cues, 3 thinnest tags | per-tag support |
| `layer2_focus2` | 60 | mid-precision cue, self-harm only | per-tag support |

**Why it took four passes.** The proportional 100 is the honest prevalence estimate but it yielded
**one** `self_harm_suicide` positive — recall on that tag was simply unmeasurable. The
theme-stratified 330 was supposed to fix it and only reached 6, because self-harm's high-recall net
("kill", "die", "dead", "cut") is overwhelmingly *sports slang* on this corpus. Two further focused
draws using mid-precision cues brought it to 45. This is a methods lesson worth reporting:
**for a rare label in a domain with heavy wrong-sense vocabulary, cue-based enrichment has to be
tuned per tag; a single high-recall net does not enrich at all.**

The focused draws deliberately used cues *between* the high-recall and high-precision patterns, so
the rater — not the rule — still decides, and recall is not measured on the rules' own hits. The
`focus2` file is full of *"suicide cut"*, *"suicide grip"* and fan hyperbole (*"7pm game 7? really
tryna make me suicidal"*), which is precisely the false-positive distribution the tag must survive.

### Measured prevalence (from the proportional sample only, n=100)

| tag | % | | tag | % |
|---|---|---|---|---|
| anxiety | 32% | | performance_psych | 13% |
| exercise_coping | 29% | | stress_pressure | 12% |
| depression | 17% | | burnout_motivation | 12% |
| body_image_eating | 17% | | injury_distress | 6% |
| help_seeking | 14% | | self_harm_suicide | 1% |

Mean 1.53 tags per comment; **13/100 carry no tag** — Layer-1 residual false positives plus generic
"mental health is important" talk with no specific theme.

### Splits

`code/split_gold.py`. The proportional 100 goes **entirely to test** — it is the only set at
natural prevalence and must never be trained or tuned on. The other 510 are split by **iterative
stratification** (each row placed into whichever split is furthest from quota on that row's
*rarest* label): with 10 correlated labels a random split easily strands a rare tag at 4 test
positives.

Result: train 204 / dev 102 / test 304, with 19–78 test positives per tag.

### ⚠️ A leak, caught and fixed

The silver set was first built **before** the two focus samples were drawn, so 182 gold rows —
including **70 test rows** — were sitting in the training data. Every metric would have been
inflated. Fixed by rebuilding with all 610 gold ids excluded; a second pass found **3 more** test
rows still present, because the corpus contains **duplicate comment bodies under different ids**
(reposts, automod copies of a post), which id-based exclusion cannot catch. `layer2_silver.py` now
excludes by **id and by text**. Verified 0/304 leakage before training.

**Lesson (same shape as the Layer-1 split-cache bug):** the ordering of pipeline steps is itself a
correctness property. Draw *all* evaluation data before building *any* training data, and verify
the leak count rather than assuming it.

---

## 5. Round 1 — and the diagnostic that explained it

**Setup.** `twitter-roberta-dapt` + 10 sigmoid heads, masked BCE with per-tag `pos_weight`,
2 epochs over 127,655 silver rows + 204 gold-train rows at loss weight 8. Thresholds tuned per tag
on gold-dev, scored on the 304-row gold-test.

**Result: macro F1 0.65, and only 3 of 10 tags cleared 0.8.**

| | P | R | F1 |
|---|---|---|---|
| micro (pooled) | 0.64 | 0.71 | 0.67 |
| macro (mean of tags) | 0.63 | 0.68 | 0.65 |

### First false lead: the thresholds

8 of 10 tuned thresholds came back as **exactly 0.95** — the top of the tuning grid. That looks
like a smoking gun, so it was checked before anything else: the heads are strongly **bimodal**
(median p ≈ 0.01, p90 ≈ 0.999), so a fixed 0.05–0.95 grid places almost no candidates near the
actual boundary. Threshold search was replaced with **data-driven candidates** (the observed dev
probabilities and their midpoints).

**It bought +0.00 macro F1** (0.65 → 0.64). Worth recording as a negative result: the pegged
thresholds were a *symptom* of saturation, not the cause of the low score. The problem was the
model, not the decision rule — the same lesson the Layer-1 `P(max)` diagnostic taught.

### The real diagnostic: silver/gold prevalence mismatch

Comparing how often the *rules* fire against how often the *rubric* says the tag is present:

| tag | silver % | gold % | ratio | test F1 |
|---|---|---|---|---|
| exercise_coping | 3.4 | 29 | **0.12** | 0.58 |
| injury_distress | 1.1 | 6 | **0.19** | 0.29 |
| performance_psych | 2.8 | 13 | **0.21** | 0.46 |
| stress_pressure | 2.8 | 12 | **0.24** | 0.55 |
| body_image_eating | 7.8 | 17 | 0.46 | 0.55 |
| burnout_motivation | 7.2 | 12 | 0.60 | 0.65 |
| anxiety | 23.9 | 32 | 0.75 | **0.87** |
| help_seeking | 11.6 | 14 | 0.83 | 0.72 |
| depression | 15.1 | 17 | 0.89 | **0.83** |
| self_harm_suicide | 2.0 | 1 | 1.97 | **0.85** |

The relationship is nearly monotonic. **Every tag whose rules under-fired scored badly, and every
tag whose rules matched the rubric's prevalence scored well.** The model was not failing — it was
faithfully learning a *narrower concept than the rubric defines*, because that is what the silver
labels described. `exercise_coping` is the extreme case: the rules fired on 3.4% of comments for a
theme that is genuinely present in 27% of them.

**This is the Layer-2 analogue of the Layer-1 domain-match finding: the cheap diagnostic (compare
rule prevalence to rubric prevalence, per tag) pointed straight at the cause, and no amount of
threshold tuning or extra training would have fixed it.**

### Acting on it: making the rules measurable

The rules were promoted from "written once and trusted" to a **measured component**
(`code/tune_lexicon.py`), scored as a classifier against gold train+dev (test never touched):

| | rule macro P | rule macro R | rule macro F1 |
|---|---|---|---|
| v1 rules | — | — | **0.513** |
| v2 rules | — | — | **0.650** |

v2 rewrites the six under-firing tags as **co-occurrence patterns over shared word classes**
(`_EX`, `_MOOD`, `_LINK`, `_COMP`, `_PSYCH`, `_EMO`) with a window allowed to cross sentence
boundaries — v1 required both halves of the construct inside one clause, which is simply not how
people write. Worst-case recall went from 0.11 to 0.44. The window is a swept hyperparameter:

| window (chars) | 40 | 60 | **80** | 100 | 140 | 180 |
|---|---|---|---|---|---|---|
| rule macro F1 | 0.632 | 0.635 | **0.650** | 0.647 | 0.647 | 0.652 |

80 was chosen: joint best F1 with better precision than the wider settings (weak supervision is
hurt more by noisy positives than by missing ones).

Silver prevalence after v2 — the mismatch that predicted failure is largely closed:

| tag | silver v1 | silver v2 | gold |
|---|---|---|---|
| exercise_coping | 3.4% | **15.6%** | 29% |
| injury_distress | 1.1% | **2.6%** | 6% |
| performance_psych | 2.8% | **9.5%** | 13% |
| stress_pressure | 2.8% | **9.4%** | 12% |
| body_image_eating | 7.8% | **9.1%** | 17% |
| burnout_motivation | 7.2% | **7.8%** | 12% |

## 6. Rounds 2–5: what moved the number

Every score below is on the **same 304-row gold-test set**, which never changed across rounds, so
the comparisons are honest. Thresholds always tuned on gold-dev, never on test.

| round | change | macro P | macro R | **macro F1** | micro F1 |
|---|---|---|---|---|---|
| 1 | silver v1 + 204 gold, argmax-F1 thresholds | 0.63 | 0.68 | **0.65** | 0.67 |
| 1b | + data-driven threshold candidates | 0.67 | 0.62 | 0.64 | 0.67 |
| 2 | **silver v2 rules** (co-occurrence, gold-tuned) | 0.70 | 0.67 | **0.68** | 0.70 |
| 3 | **+ stage-2 gold fine-tune** | 0.74 | 0.70 | **0.71** | 0.73 |
| 4 | + AL round 1 (210 labels) → 414 gold | 0.82 | 0.70 | **0.74** | 0.77 |
| 4b | **+ prevalence-matched thresholds** | 0.79 | 0.75 | **0.77** | 0.79 |
| 5 | + AL round 2 (156 labels) → 570 gold | 0.82 | 0.76 | **0.78** | **0.81** |

Four levers mattered, in order of size:

1. **Fixing the rules (r1→r2, +0.03).** Covered in §5 — the silver labels defined the task, so a
   narrow rule taught a narrow concept.
2. **Two-stage gold fine-tuning (r2→r3, +0.03).** 204 gold rows mixed into 121k silver rows are
   invisible however high the loss weight; initialising from the silver model and then fine-tuning
   on gold alone is what actually re-draws the boundary. A variant anchored with 8k silver rows to
   prevent forgetting scored *worse* on dev (0.74 vs 0.77) — the forgetting was the point.
3. **Active learning (r3→r5, +0.07).** Two rounds, 366 labels, drawn from the model's own
   uncertainty rather than at random. Same lever that fixed Layer-1 over-flagging. AL rows go
   wholly into train — they are model-selected and would flatter any test set they entered.
4. **Prevalence-matched thresholds (+0.03).** Argmax-F1 on ~100 dev rows systematically picks
   thresholds that are too HIGH: with few positives, raising the bar sheds false positives faster
   than true ones, so it looked optimal on dev and cost ~0.12 macro recall on test (macro P 0.83 vs
   R 0.69). Setting each threshold so the predicted rate matches the gold prevalence uses the one
   quantity a small sample estimates well. `performance_psych` went 0.50 → 0.68 on this alone.
   A bootstrap-median of the argmax was tried first and did **nothing** (0.74 → 0.74): the heads
   are so saturated that every resample picks ~0.99. Recorded as a negative result.

### Final result (round 5, gold-test n=304)

| tag | thr | P | R | F1 | test n+ | clears 0.8? |
|---|---|---|---|---|---|---|
| anxiety | 0.65 | 0.99 | 0.87 | **0.92** | 77 | ✅ |
| depression | 0.96 | 0.93 | 0.88 | **0.91** | 60 | ✅ |
| burnout_motivation | 0.95 | 0.83 | 0.86 | **0.84** | 28 | ✅ |
| self_harm_suicide | 0.98 | 0.84 | 0.84 | **0.84** | 19 | ✅ |
| help_seeking | 0.67 | 0.82 | 0.77 | 0.79 | 35 | ✗ (R 0.77) |
| body_image_eating | 0.72 | 0.75 | 0.75 | 0.75 | 44 | ✗ |
| exercise_coping | 0.67 | 0.76 | 0.74 | 0.75 | 78 | ✗ |
| stress_pressure | 0.92 | 0.83 | 0.65 | 0.73 | 31 | ✗ (R 0.65) |
| performance_psych | 0.66 | 0.74 | 0.72 | 0.73 | 32 | ✗ |
| injury_distress | 0.85 | 0.71 | 0.48 | 0.57 | 21 | ✗ |
| **MICRO (pooled)** | | **0.84** | **0.78** | **0.81** | | ✅ |
| **MACRO** | | **0.82** | **0.76** | **0.78** | | ✗ |

**Where this lands against the 0.8 target.** Pooled across all tag decisions the layer clears the
bar (micro P 0.84 / R 0.78 / F1 0.81). **Four of ten tags clear 0.8 on precision, recall and F1
individually** — those four are production-ready. Six do not, and it would be dishonest to present
a macro average as if it did.

### Why the six fall short, and what would fix them

- **`injury_distress` (0.57) is the real failure.** It is the only tag defined as a *conjunction*
  — injury AND psychological consequence — and the corpus is full of both halves separately
  (injury/rehab mechanics everywhere; distress everywhere). Recall 0.48 says the model misses
  cases where the two are linked implicitly (*"haven't been able to play, so that doesn't help with
  my depression"*). This probably needs its own rubric pass and a dedicated labeled set, not more
  of the same.
- **`stress_pressure` recall 0.65**: "stress" is the single most overloaded word in a fitness
  corpus (stress fracture, stress on the joint, time under tension). The kill patterns protect
  precision (0.83) at recall's expense.
- **`performance_psych`, `body_image_eating`, `exercise_coping` (0.73–0.75)** are broad, fuzzy
  constructs whose boundary the rubric itself does not draw sharply — ordinary cutting talk shades
  into body-image distress; "the gym is my happy place" shades into coping. Labeling produced 14–19
  `x` (unclear) cells per 330 on these, which is the honest signature of a fuzzy boundary. Sharper
  sub-definitions in a v2 rubric would help more than more data.
- **`help_seeking` is a near miss** (R 0.77) and would likely cross with one more AL round.

**Recommended deliverable:** ship the four ≥0.8 tags as the validated Layer-2 output and mark the
other six as provisional, exactly as Layer 1 shipped a narrowed scope and documented the rest.
That is a decision for Naveen and Babak, not one to make silently — the full 10-tag model and its
per-tag numbers are committed either way.

---

### Round 6 — a negative result

Re-ran stage 1 from scratch with all 570 gold rows mixed into the silver at loss weight 25, then
stage-2 fine-tuned as before. **Dev macro F1 0.78 vs round 5's 0.81** — worse. Round 5 (whose
stage-1 saw only the earlier 204 gold rows) stays the deliverable. Consistent with the round-3
finding: gold's value comes from the *separate* fine-tuning stage, not from being mixed into the
silver, and a ~50-minute retrain bought nothing. Recorded so nobody repeats it.

## 7. Applying Layer 2 to the corpus

`code/tag_corpus.py` over all 134,534 Layer-1-relevant comments, 6.0 min on the RTX 2060 →
`data/classified/final_dataset_tagged.csv` (one probability column and one binary column per tag).

**The corpus tag rates are an independent calibration check** — they were never fitted to, and they
track the gold prevalence estimates closely:

| tag | corpus % | gold estimate % |
|---|---|---|
| exercise_coping | 26.2 | 29 |
| anxiety | 25.9 | 32 |
| depression | 16.0 | 17 |
| body_image_eating | 14.9 | 17 |
| help_seeking | 12.2 | 14 |
| performance_psych | 11.3 | 13 |
| burnout_motivation | 9.7 | 12 |
| stress_pressure | 7.8 | 12 |
| injury_distress | 4.5 | 6 |
| self_harm_suicide | 1.3 | 1 |

Every tag runs slightly *below* its gold estimate, which is what a precision-favouring operating
point should do; nothing is wildly mis-calibrated. The two largest gaps (`stress_pressure`,
`anxiety`) are the two tags with the lowest recall, as expected.

## 8. Honest limitations

- **Single-rater gold.** All 610 labels are Claude's, applied from the written rubric. Layer 1
  established this protocol and measured it (human-vs-Claude κ = 0.81 mh / 0.86 sport on the split
  questions). **Layer 2 has no second-rater agreement yet** — a human pass over a subset, scored
  with `code/metrics_interrater.py`, is required before the paper reports these numbers as gold.
  The blinded rating files are committed so that pass can be run directly.
- **`self_harm_suicide` rests on 19 test positives.** The point estimate is usable; the confidence
  interval is wide. Reported with support so the reader can judge.
- **Enriched-sample metrics are not corpus metrics.** Per-tag P/R/F1 are measured on a
  cue-enriched test set. Corpus-level prevalence comes from the proportional 100 only.
- **Unclear (`x`) cells are excluded per tag**, as in Layer 1. They are frequent on
  `body_image_eating` (19/330) — ordinary cutting/bulking talk shades into body-image distress with
  no clean boundary.
- **Six of ten tags do not meet the 0.8 bar** and should not be reported as if they do. See §6.

## 9. Reproducing this

```bash
# Layer 1 (already done) -> data/classified/final_dataset.csv
.venv/Scripts/python.exe code/driver_classify.py          # 1.63M comments, ~30 min
.venv/Scripts/python.exe code/build_final_dataset.py

# Layer 2
.venv/Scripts/python.exe code/sample_layer2.py --n 100 --out .../layer2_prop100   # gold samples
.venv/Scripts/python.exe code/apply_labels.py <rated.csv> <spec.txt>              # record labels
.venv/Scripts/python.exe code/tune_lexicon.py                    # score rules vs gold train+dev
.venv/Scripts/python.exe code/layer2_silver.py                   # weak-supervision training set
.venv/Scripts/python.exe code/split_gold.py                      # train/dev/test, leak-checked
.venv/Scripts/python.exe code/train_layer2.py --epochs 2 --gold data/layer2/gold_train.csv \
    --gold-weight 20 --out models/layer2_tags_s1                 # stage 1 (silver), ~55 min
.venv/Scripts/python.exe code/train_layer2.py --init models/layer2_tags_s1 --no-silver \
    --gold data/layer2/gold_train.csv --epochs 18 --lr 1e-5 --out models/layer2_tags
.venv/Scripts/python.exe code/eval_layer2.py --model models/layer2_tags \
    --gold data/layer2/gold_test.csv --dev data/layer2/gold_dev.csv
.venv/Scripts/python.exe code/tag_corpus.py                      # apply to the corpus, ~6 min
```

Knobs: `L2_WINDOW` (lexicon co-occurrence window, default 80), `L2_THR_MODE`
(`rate` = prevalence matching, default; `f1` = argmax-F1), `L2_BOOTSTRAP`.
Seeds are fixed throughout (`set_seed(1)`, sampler seed 20260725, split seed 20260726).

---

## 10. v3 — expanding to 17 tags (2026-07-26, per Babak)

**Meeting directive.** Babak asked to *"refactor the data and retrain Layer 2 to allow for multiple
tags,"* add more label categories (*"could reduce forcing labels on it"*), weight by frequency, and
keep it multi-label. Two of those were already in place — the model is `BCEWithLogits` with 10
independent sigmoid heads (`problem_type="multi_label_classification"`, so a comment already gets 0/1/
many tags) and `pos_weight` per tag is already `#neg/#pos` capped. **The substantive change is adding
tags** so off-taxonomy themes aren't forced into a wrong head or dropped.

### The 7 new tags (10 → 17)

Added from the v1 cut-list plus one gap labeling had exposed: `substance_use`,
`loneliness_isolation`, `sleep`, `trauma_ptsd`, `adhd_neurodivergence`, `identity_retirement`,
`exercise_dependence`. Rubric definitions written in the same style (construct, ✓/✗, negation→NO) —
see `docs/methods/layer2_tag_rubric.md`.

**Prevalence probe first** (silver HP-fire rate over 20k matched comments), to size the labeling and
flag the rare ones — the same discipline as the rest of Layer 2:

| tag | HP-fire % | verdict |
|---|---|---|
| adhd_neurodivergence | 3.2 | trainable |
| sleep | 2.9 | trainable |
| loneliness_isolation | 2.3 | trainable |
| trauma_ptsd | 1.9 | trainable |
| substance_use | 1.7 | trainable |
| exercise_dependence | 0.4 | ⚠ ultra-rare — enrich, low confidence |
| identity_retirement | 0.1 | ⚠ ultra-rare — original cut for exactly this |

### Rules — same abstention design, new fitness wrong-sense guards

Each new tag gets HP / HR / KILLS in `layer2_lexicon.py`, and each new KILL targets the dominant
wrong sense a fitness corpus throws at it — the single most important part:
`substance_use` ✗ *drink **water** / pre-workout / caffeine*; `loneliness_isolation` ✗ *training
**alone** / solo*; `sleep` ✗ *sleep for **recovery** / rest day*; `trauma_ptsd` ✗ *"ptsd from missing
a **free throw**"*; `adhd_neurodivergence` ✗ *"**add** weight / a set"*; `exercise_dependence` ✗
*"addicted to the **endorphins**"*. Behaviourally unit-tested before any training: all 7 anchor
positives fire, all wrong-senses correctly abstain (killed cues abstain — only *negated* cues become
hard 0s, by design).

### Gold labeling — merge + enrich (two blinded workflows)

- **A:** the 7 new tags labeled on the **976 existing gold** comments (17-agent workflow) → merged
  with their existing 10-tag labels, keeping their existing train/dev/test split. Preserves all prior
  labels; no re-labeling of the 10.
- **B:** **382 enriched candidates** (HP-cue hits for the new tags, drawn disjoint from gold) labeled
  for **all 17 tags** (7-agent workflow), then iterative-stratified into the splits so the new tags
  have test positives.

Result — new 17-tag gold: **793 train / 141 dev / 424 test** (976 + 382). New-tag **test** positives:
substance 25, loneliness 21, sleep 23, trauma 22, adhd 20, exercise_dep 22 — and **identity_retirement
just 7** (enriched draw found only 27 in 79k comments; its estimate will have a wide CI, as flagged).

### Silver regen — a leak the merge introduced, and the fix

The enriched gold rows live only in the split CSVs, not the `*_rated.csv` files `layer2_silver.py`
read to exclude gold — so they would have leaked into silver. Fixed `gold_texts()` to also exclude
every text in `gold_{train,dev,test}.csv` (excluded 1,360 gold texts vs 976 ids). *Same lesson as the
original Layer-2 leak and the Layer-1 split-cache bug: whenever the gold set changes, re-verify the
exclusion covers the new rows, by text not just id.*

*(Retrain metrics + corpus re-application appended after training — see trials L2-v3.)*

## 11. v3b — round 2: broadening the weak rules, and a regex-hang that had to be fixed first (2026-08-10)

Round 1 (§10) left `substance_use`, `trauma_ptsd`, `exercise_dependence` under-firing (prevalence
ratio 0.25 / 0.36 / 0.07). The known lever is to broaden their rules into **co-occurrence patterns** —
a term class near a context class within a window — exactly what lifted the original six weak tags in
v2. Also, per directive, `identity_retirement` was dropped (0.1% prevalence, unmeasurable) → **16 tags**.

### The co-occurrence rules, and the catastrophic-backtracking hang

First attempt wired the broadening straight into the high-precision regex, the same shape v2 used:

```
A[\s\S]{0,W}B | B[\s\S]{0,W}A          # term-class A within W chars of context-class B
```

Silver regeneration then **stalled dead at ~25k comments** — one core pegged at 100% for minutes on a
single comment, no progress. This is classic **catastrophic regex backtracking**: a greedy
variable-length gap between two large alternations, scanned by `finditer` over a long *unpunctuated*
Reddit comment, explores exponentially many ways to fail. The v2 tags never triggered it on this
corpus, but the new, larger word-classes did.

Two fixes, in order:
1. **Lazy quantifier** (`[\s\S]{0,W}?`) — helped (got past 25k) but a worse comment still pegged a
   core. Necessary, not sufficient.
2. **Moved the proximity test out of regex entirely** (`_cooc()` in `code/layer2_lexicon.py`). Each
   word-class is matched independently (linear, no gap → no backtracking), then a plain Python check
   asks whether any A-span and B-span lie within `W` chars. This *cannot* backtrack. Verified: a full
   130k-comment scan runs steadily with **zero** comments over 0.1 s, and silver regeneration completes
   end-to-end. The boolean semantics are identical to the old regex (co-occurrence exists ⇔ same
   answer); wrong-sense KILLS still apply to the matched span.

*Lesson for the paper's methods notes: proximity/co-occurrence over user text belongs in code, not in a
variable-gap regex — the gap is where the engine backtracks. Keep each regex class anchored and simple.*

### Silver, retrain, results

- **Silver regenerated**, 16 tags, broadened rules: 128,195 rows, **0 leakage** across train/dev/test.
  The three target tags' silver prevalence rose to substance 3.7 % (was 1.5), exercise_dependence
  3.9 % (was 0.4), trauma 2.1 % (implicit trauma stays hard) — closing most of the gap to gold.
- **Retrained** two-stage (silver+gold → gold-only fine-tune), batch 32 (per Babak: larger batch,
  within GPU memory), model `models/layer2_tags_v3b`.
- **Inter-rater κ** (the standing gap): a second independent blinded rater pass over 160 stratified
  gold-test records, per-tag Cohen's κ via `code/metrics_interrater_l2.py` → **pooled 0.879, mean
  0.862, 13/16 tags ≥ 0.8** — on par with Layer-1's 0.81/0.86. This is model-vs-model rubric
  reliability; a human pass on the committed blinded files is the remaining step.
- **Joint eval** added (`code/eval_joint.py`): runs the real cascade (gate → tags) on relevant gold +
  control-arm negatives, so the pipeline number is measured, not composed.

### Round-2 result and the final release (2026-08-10)

The broadening was tried three ways — v3b (batch 32), v3c (batch 16), v3d (broaden exercise_dependence
only) — and **all three net-regressed** against round-1 `v3`. The extra silver positives are noisier,
and the shared encoder passes that noise to tags that were never broadened (body_image 0.81→0.63,
injury 0.66→0.47). This is the mirror image of the v2 story: broadening helped the *original* six tags
because their rules were high-precision-but-narrow; the v3 new-tag rules were already broad, so
widening them only added false positives. **Negative result — round-1 `v3` ships.**

Applying an **F1 ≥ 0.65 release bar** to `v3` dropped four tags (exercise_coping 0.63, substance_use
0.63, trauma_ptsd 0.48, exercise_dependence 0.20), leaving **12 released tags**: micro P/R/F1
0.81/0.84/**0.83**, macro 0.79/0.83/0.81, five fully validated. End-to-end cascade (measured, not
composed) micro-F1 **0.80** with the gate rejecting 97% of labelled-irrelevant comments. The released
corpus (`code/tag_corpus_final.py`) is `final_dataset_tagged.csv` = **105,206** matched comments
carrying ≥1 of the 12 tags, with the control arm split out to `control_baseline.csv`. Note `v3` is a
17-head model (identity_retirement at index 15); the tagger maps heads via `thresholds.json` so the
12 kept tags (all at indices ≤14) align exactly. Full numbers in `docs/paper/AMHC_paper_draft_v2.md`.
