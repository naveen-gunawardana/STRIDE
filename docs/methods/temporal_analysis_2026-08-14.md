# Temporal Analysis — Process & Results

**Run 2026-08-14.** Companion to `docs/methods/relevance_layer1_process.md` and
`docs/methods/layer2_tags_process_2026-07-25.md`, same intent: record the reasoning and the failures so
the results section can be written from this file.

Scripts: `code/temporal_analysis.py`, `code/rate_adjusted.py`, `code/subreddit_composition.py`,
`code/tag_trends.py`, `code/standardize_rate.py`. Outputs in `results_temporal/`.

---

## 0. What was asked for

Per the analysis brief: plot documents per month across years, look for spikes, prefer
proportions over raw frequencies, and **always compare against a baseline**, because Reddit
volume changes over time and a raw count of mental-health talk moves with the platform whether
or not anything about mental health changed.

That last instruction turned out to be the whole analysis. Every headline number here changes
sign or significance depending on what it is divided by.

---

## 1. Two denominator bugs, found before any result was believed

**Bug 1 — structural zeros in the denominator.** `code/classify_corpus.py` drops dedicated
mental-health subreddits (`DROP_SUBS`: r/Anxiety, r/depression, r/mentalhealth,
r/getting_over_it, …) *before* running the models, on the reasoning that being in r/depression
does not make a comment about sport. But the per-month row count it logs is taken **before** that
drop. Those comments can never be flagged relevant, so they enter every rate as guaranteed
zeros.

| arm | rows read | dropped pre-classification | actually classified |
|---|---|---|---|
| matched | 669,110 | 85,452 | **583,658** |
| baseline (control) | 956,267 | 65,486 | **890,781** |

This is not a rounding issue. r/Anxiety alone contributes 34,351 comments with 0 relevant by
construction.

| statistic | as previously published | corrected |
|---|---|---|
| matched relevance rate | 19.40% | **22.24%** |
| control relevance rate | 0.49% | **0.53%** |
| separation | 39.5× | **42.2×** |

*Every prior document in this repo quotes the 19.4% / 39× pair. They are the rate among comments
**read**, not among comments **classified**. Both are defensible if defined, but the corrected
pair is the one that answers "how often does the gate fire when it is allowed to fire?" and it is
what the paper should report.*

**Bug 2 — a control arm that does not span the corpus.** The baseline arm covers 2018–2022 only;
the matched arm runs to 2023. A first pass computed "matched share of all activity" as
`matched / (matched + baseline)` across all 72 months, so for the twelve 2023 months the control
volume was 0 and the share was forced to 1.0. That manufactured a **+19.96%/yr** growth trend out
of nothing. Restricted to months where both arms exist, the real figure is **+0.95%/yr**.

**Lesson, same shape as the Layer-1 split-cache bug and the Layer-2 gold leak:** a denominator is
a correctness property. Check what is in it before believing any ratio built on it.

---

## 2. The corpus is 19 communities, and the mix moves

`code/subreddit_composition.py`. The relevant corpus is not "Reddit"; it is a fixed set of 19
subreddits.

| subreddit | n | % | | subreddit | n | % |
|---|---|---|---|---|---|---|
| xxfitness | 29,532 | 22.7 | | tennis | 5,304 | 4.1 |
| running | 24,805 | 19.1 | | weightroom | 3,333 | 2.6 |
| Fitness | 23,314 | 18.0 | | crossfit | 2,866 | 2.2 |
| nba | 19,391 | 14.9 | | Swimming | 2,520 | 1.9 |
| bodybuilding | 8,306 | 6.4 | | (11 more) | | |

Top four = **74.7%**. Two facts matter for every claim downstream:

1. **r/nba is 14.9% of the relevant corpus** — the fourth-largest contributor. It is a spectator
   community: comments there are mostly fans discussing professional players, not athletes
   writing about themselves. Its relevance rate is **7.7%** against **45.7%** in r/xxfitness and
   **48.7%** in r/running. Grouping the four spectator communities (nba, CollegeBasketball,
   Basketball, tennis) against the rest: **8.68%** vs **37.61%**.
2. **The mix drifts hard.** First 12 months vs last 12: xxfitness −11.1pp, Fitness −11.5pp,
   tennis +5.1pp, crossfit +2.9pp, running +3.1pp. The effective number of communities
   (1/Herfindahl) goes **2.6 → 6.7**.

A corpus whose composition moves that much cannot support an uncontrolled temporal claim.

---

## 3. The headline: the crude trend and the real trend have opposite signs

`code/rate_adjusted.py` builds (month × subreddit) denominators from the raw matched monthly
files and fits a binomial GLM on the cells.

| model | change per year | p |
|---|---|---|
| crude — `logit(relevant) ~ time` | **+0.18%** | 0.32 (ns) |
| adjusted — `+ subreddit fixed effects` | **+4.36%** | 3.2e-94 |
| adjusted, participant communities only | **+5.36%** | 1.6e-98 |
| adjusted, spectator communities only | **+2.08%** | 4.5e-08 |

Direct standardisation makes the same point without a coefficient (`code/standardize_rate.py`):
hold the community mix at the pooled six-year composition and recompute each month from each
community's own rate.

| series | first 12 months | last 12 months | slope |
|---|---|---|---|
| crude | 21.87% | 20.66% | −0.03 pp/yr |
| standardised | 21.08% | **22.76%** | **+0.60 pp/yr** |

**Within every community, athlete mental-health discussion is rising. Pooled across communities
it looks flat, because the corpus drifted toward communities that discuss it less.** A textbook
composition effect, and the single most important methodological result in this analysis: the
naive plot is the one a reader would draw, and it is wrong.

The control arm is flat throughout (+0.33%/yr, p = 0.74), which is what a control should do.

---

## 4. Seasonality

Month-of-year fixed effects on top of the linear trend: **LR = 563.3, df = 11, p = 9.8e-114**.

Relative to January, the relevance odds are highest in **August (1.17)**, September (1.09) and
October (1.07), lowest in **May (0.89)** and February (0.92). The January reference point is
itself a high-volume month (new-year training), so the seasonal shape is better read as
*late-summer elevated, late-spring depressed*. Any month-level comparison in a downstream study
has to carry this term.

---

## 5. Spikes, and one honest negative result

Deviation from a 13-month centred rolling median, scaled by MAD (a mean baseline would absorb a
real spike and hide it).

| month | rate | robust z | reading |
|---|---|---|---|
| 2020-04 | 31.4% | **+3.14** | first full month of pandemic lockdown |
| 2022-08 | 28.1% | **+2.70** | composition (r/running 28.8%, r/nba down to 7.3%) |

**COVID interrupted time series** at 2020-03: level shift **OR 1.209**, p = 4.8e-45. It survives
the composition control (**OR 1.178**, p = 2.6e-29), so this is a real jump in how often people
discussed mental health, not just a change in who was posting. The mechanism is visible in the
composition series: r/nba's share collapses 15.0% → 5.8% between January and April 2020, matching
the NBA's suspension of its season on 11 March 2020, while r/xxfitness rises 23.1% → 36.7%.

### The negative result

Two events were nominated *a priori* as the athlete-mental-health moments of the period:

| event | robust z | elevated? |
|---|---|---|
| Naomi Osaka withdraws from the French Open, May 2021 | −0.02 | no |
| Simone Biles withdraws from Olympic finals, July 2021 | +0.74 | no |

**Neither produced a detectable spike.** This is worth reporting rather than burying. The most
likely reason is scope: the corpus is dominated by recreational and amateur athletes talking
about their own training, and only ~4% of it is professional-sport discussion. Elite-athlete news
events move fan communities, and fan communities are the part of this corpus that discusses
mental health least (8.68% vs 37.61%). A corpus built to catch people talking about themselves
should not be expected to spike on somebody else's press conference — but that had to be checked,
not assumed.

---

## 6. Per-tag trends, naive and adjusted

`code/tag_trends.py`, on the 16-tag corpus re-tagged with the shipped model
`models/layer2_tags_v3b` (`data/classified/final_dataset_tagged_v3b.csv`, 134,534 comments).
Change per year in the share of relevant comments carrying each tag.

| tag | prevalence | naive %/yr | adjusted %/yr | no-spectator %/yr | verdict |
|---|---|---|---|---|---|
| anxiety | 29.3% | −4.29*** | −6.38*** | −4.98*** | robust |
| exercise_coping | 26.2% | −4.42*** | −2.21*** | −1.74*** | attenuated |
| depression | 17.9% | −13.38*** | −12.65*** | −11.40*** | **robust** |
| body_image_eating | 14.1% | −11.30*** | −2.62*** | −3.05*** | **mostly composition** |
| help_seeking | 13.7% | +6.13*** | +5.82*** | +1.64** | robust, weaker without spectators |
| burnout_motivation | 10.0% | +2.85*** | +2.99*** | +3.08*** | robust |
| performance_psych | 9.9% | +8.52*** | −4.02*** | −1.12 (ns) | **sign flips — composition** |
| stress_pressure | 7.7% | +0.43 (ns) | +0.57 (ns) | +1.03 (ns) | flat |
| exercise_dependence | 5.7% | −7.11*** | −3.61*** | −4.50*** | attenuated |
| injury_distress | 5.0% | −1.11 (ns) | −0.80 (ns) | −0.77 (ns) | flat |
| adhd_neurodivergence | 2.8% | +20.93*** | **+24.57***** | +28.03*** | **robust, largest effect** |
| sleep | 2.2% | −6.10*** | −1.28 (ns) | −0.52 (ns) | **mostly composition** |
| loneliness_isolation | 1.7% | −8.85*** | −8.02*** | −7.28*** | robust |
| substance_use | 1.7% | −4.48*** | −3.31* | −5.19*** | robust |
| self_harm_suicide | 1.6% | −6.50*** | −8.75*** | −9.14*** | robust |
| trauma_ptsd | 1.3% | +9.01*** | +4.14** | +1.89 (ns) | **mostly composition** |

`*** p<.001, ** p<.01, * p<.05`

**Four tags change materially once composition is controlled**, and one (`performance_psych`)
**reverses sign**: it looks like one of the fastest-growing themes in the raw series and is
actually declining within communities. The mechanism is visible in the by-community prevalences —
performance psychology is 29.8% of relevant comments in r/tennis and 18.4% in r/nba against 3.3%
in r/xxfitness, so the drift toward tennis manufactures the rise.

*(These figures are stable across tagger versions. Re-running the whole analysis on the earlier
`layer2_tags_v3` tagging gives the same four composition-driven tags, the same sign flip, and the
same ranking. That the qualitative result survives a change of classifier is worth more than any
individual coefficient here.)*

### Tag prevalence by community (validity check)

| tag | xxfitness | running | Fitness | nba | bodybuilding | tennis |
|---|---|---|---|---|---|---|
| anxiety | 31.7% | 35.1% | 25.9% | 21.6% | 20.1% | 26.1% |
| exercise_coping | 31.9% | **49.2%** | 28.3% | 3.6% | 20.9% | 2.7% |
| depression | 14.7% | 21.1% | 20.1% | 19.9% | 24.0% | 13.9% |
| body_image_eating | **25.1%** | 6.8% | 22.8% | 1.4% | **25.5%** | 1.3% |
| performance_psych | 3.3% | 8.0% | 3.8% | 18.4% | 3.4% | **29.8%** |

The tags behave the way the constructs predict: body image concentrates in the physique-oriented
communities and is near zero in spectator ones; exercise-as-coping peaks in r/running;
performance psychology peaks in the competitive-sport communities. Nothing was fitted to produce
this, so it is usable as convergent evidence that the tags measure what they claim.

---

## 7. What can and cannot be claimed

**Can be claimed**
- Within-community, athlete mental-health discussion rose ~4–5% per year, 2018–2023.
- The pandemic produced a real level shift (OR ≈ 1.18–1.21) that survives composition control.
- Strong, stable seasonality with an August peak.
- ADHD/neurodivergence discussion grew ~25%/yr, the largest effect in the tag set, robust to
  every control tried (and larger still, ~28%/yr, once spectator communities are excluded).
- Depression, self-harm and loneliness talk all declined within communities.
- Tag prevalences differ across communities in exactly the directions the constructs predict.

**Cannot be claimed**
- Anything about the *raw* trend: pooled, the rate is flat (p = 0.32) and the raw series is
  dominated by composition.
- Any elite-athlete event effect. Osaka and Biles do not register.
- `performance_psych`, `body_image_eating`, `sleep` and `trauma_ptsd` trends at face value.
- Prevalence generalisation beyond these 19 communities.
- Anything resting on the weaker tags: 11 of 16 do not meet the 0.8 bar, and four sit below F1
  0.70 (`performance_psych`, `substance_use`, `trauma_ptsd`, `injury_distress`). Their trends are
  significant at p < .001 and should still be treated as provisional. **Statistical significance
  is a statement about the classifier's output, not about athletes**, and the two coincide only as
  far as the classifier is good.

---

## 8. Reproducing

```bash
PY=.venv/Scripts/python.exe
$PY code/subreddit_composition.py                     # composition + drift
$PY code/rate_adjusted.py                             # (month x subreddit) GLM, ~4 min
$PY code/standardize_rate.py                          # direct standardisation + fig3
$PY code/tag_corpus.py --model models/layer2_tags_v3b \
     --thr models/layer2_tags_v3b/thresholds.json \
     --out data/classified/final_dataset_tagged_v3b.csv  # ~6 min on the RTX 2060
$PY code/tag_trends.py --in data/classified/final_dataset_tagged_v3b.csv
$PY code/temporal_analysis.py --tagged data/classified/final_dataset_tagged_v3b.csv
```

Figures: `fig1_volume_and_rate.png`, `fig2_tag_trends.png`, `fig3_composition.png`.
Seeds are fixed upstream; nothing here is stochastic.
