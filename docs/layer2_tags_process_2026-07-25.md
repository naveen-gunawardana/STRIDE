# Layer-2 Tag Classifier — Process & Diagnostic Log

**Started 2026-07-25.** Companion to `docs/relevance_layer1_process.md`. Same BRM-style intent:
record the **reasoning and the failures**, not just the final number, so the methods section can be
written from this file.

Companions: `docs/layer2_tag_rubric.md` (labeling rules), `experiments/trials_log.md` (one row per
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

Rubric: `docs/layer2_tag_rubric.md`, written in the same style as the Layer-1 rubric (decided
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

All labeling followed `docs/layer2_tag_rubric.md` on **blinded text only** (no subreddit, author,
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

## 5. Results

*(filled in below as runs complete — see `experiments/trials_log.md` for the per-run table)*

---

## 6. Honest limitations

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
