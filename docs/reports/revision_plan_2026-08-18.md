# Revision plan — Babak's review of "ai paper version 2"

**Compiled 2026-08-18.** Source: Babak's comment thread on the shared draft (Aug 17 and Aug 18),
plus Naveen's accepted copyedits. Working doc: `docs/paper/AMHC_paper_draft_v2.docx` / `.md`.

---

## 0. Headline: the stated blocker is already resolved

Babak, on Layer-2 second-rater agreement:

> "Let's get this done ASAP, as it is the key thing currently blocking this work from getting
> published somewhere very good."

**It was done on 2026-08-10 and never made it into the draft.** From
`run_logs/kappa_layer2_2026-08-10.log`:

| | value |
|---|---|
| records double-rated | **160** (37.7% of the 424-row gold test set) |
| sampling | stratified so every tag has positives present |
| **pooled kappa** (all tag decisions) | **0.879** (raw agreement 0.976) |
| **mean per-tag kappa** | **0.862** |
| tags at kappa >= 0.8 | 13 / 16 |
| tags 0.6–0.8 | 1 (burnout_motivation 0.647) |
| below 0.6 | 1 (stress_pressure 0.582) |

This exceeds what Babak asked for ("stratified subsamples of the gold labeled set, say 20 percent,
that contain all the possible tags"). It is the highest-value edit in this plan: it converts the
paper's biggest stated weakness into a strength.

**Also already done and missing from the draft:** an end-to-end joint evaluation
(`run_logs/eval_joint_v3.log`), cascade micro P 0.77 / R 0.84 / **F1 0.80** on n=580. Limitation
bullet 6 currently states this does not exist.

⚠️ Caveat before either goes in: both were run against `models/layer2_tags_v3`, and the shipped
model is now `layer2_tags_v3b`. Kappa is annotation-level and therefore model-independent, so it
stands as-is. **The joint eval must be re-run on v3b** (~15 min).

---

## 1. Venue: unresolved and now urgent

Babak refers twice to "a regular CS conference audience" and asks for the writing to be retargeted
accordingly. That conflicts with the JMIR Mental Health restructure in
`docs/paper/AMHC_paper_JMIRMH_v3.docx`, written on the reasoning that the paper contains no CS methods
advance.

Both cannot hold at once. This needs deciding at the next meeting, because it determines whether
§5.3 is rewritten product-first (CS conference) or the whole paper moves to IMRaD (JMIR).
Everything in §2 below is venue-neutral; §3.3 is not.

---

## 2. Edits applicable immediately (no new analysis)

| # | Change | Where | Source |
|---|---|---|---|
| 2.1 | Delete the two-leaks paragraph entirely | §4.3, last para | Babak, marked resolved |
| 2.2 | Delete the "All >= 0.8" column from Table 4 | Table 4 | "that was just a development target for us" |
| 2.3 | Delete "Eleven of 16 tags fall below 0.8 on at least one metric." | §8 | Babak, explicit |
| 2.4 | Rewrite the F1<0.70 bullet. Babak: "For a highly subjective domain like this, even .6 something F1 can be okay." Reframe as expected for subjective constructs rather than as a disclaimer | §8 | Babak disagrees with current text |
| 2.5 | Replace "close to useless"; soften judgmental phrasing throughout. Babak: "make less judgment and just describe what you do and why, in this case, accounting for platform volume in evaluating the rates" | §6.1 and passim | Babak |
| 2.6 | Rewrite the 2023 control-arm limitation — Babak: "Not sure I follow this one." State plainly that the control arm stops at 2022, so any control-referenced comparison is restricted to 2018–2022 | §8 | Babak |
| 2.7 | Add the ISAAC citation (repo "program" citation for now; preprint expected within weeks) | References | Babak + Naveen |
| 2.8 | Add a sample-size limitation: 424-row test set, with future work to confirm robustness on larger labeled subsets | §8 | Babak |
| 2.9 | Insert the kappa result (§0) into §4 or §5; delete the "no second-rater agreement yet" bullet | §4/§5, §8 | see §0 |
| 2.10 | Insert the joint eval; delete the "no end-to-end joint evaluation" bullet — after re-running on v3b | §5, §8 | see §0 |

Naveen's Aug 17 copyedits are already in the pasted text and carry into the working doc.

**Effort: about half a day**, all writing except 2.10.

---

## 3. New work required

### 3.1 Confidence intervals on the per-tag monthly shares — required

> "For that to be realized, we'd want not only the raw shares, but also their confidence
> intervals." — Babak, on §6.5

Figure 2 currently plots point estimates only. Add Wilson intervals per tag per month, and add a CI
column to Table 6. The GLM confidence intervals already exist in
`results_temporal/tag_trends.json`; they are simply not printed in the paper.

*Effort: ~2 hours, in the `code/temporal_analysis.py` figure block plus a table column.*

### 3.2 Deepen the per-tag findings — required for the paper to be interesting

> "There are some interesting patterns here that are a bit buried among the many panels. One could
> dig deeper into these to 1) confirm validity of the dataset (i.e., that it tracks real world
> events and their effects), and 2) give unique empirical findings that could not be achieved
> without your corpus." — Babak

He separately flagged three results as interesting and worth promoting: "within every community,
athlete mental-health discussion is rising" (§6.2), seasonality and the pandemic (§6.3), and the
composition reversals in §6.5.

**Proposal:** cut Figure 2 from 16 panels to the 4–6 themes that carry a real finding, promote each
to its own short paragraph with the mechanism named, and move the full grid to an appendix.
Candidate headline findings:

1. **ADHD +24.6%/yr adjusted** (95% CI 22.0–27.2), the largest effect in the corpus and robust to
   every control applied. Needs a general-population Reddit comparison to separate an athlete
   effect from the broader 2018–2023 rise in ADHD discourse.
2. **The COVID level shift survives composition control** (OR 1.18, P<.001) — real-world event
   validity, which is exactly point 1 of Babak's comment.
3. **The performance_psych sign reversal** — a methodological finding about multi-community corpora
   that generalises beyond this dataset.
4. **Community-level tag prevalence tracks construct expectations** (body image 25.1% in
   r/xxfitness against 1.4% in r/nba; exercise-coping 49.2% in r/running against 2.7% in r/tennis).
   Nothing was fitted to produce this, so it is free convergent validity.

*Effort: 1–2 days, mostly analysis and writing. Item 1 needs a new data pull.*

### 3.3 Rewrite §5.3 product-first — required

> "A lot of this is still development-focused, showing the trajectory of what you actually did
> during the creation of the dataset and the model. For a regular CS conference audience, you want
> to tell them simply what the model you are sharing with them is like and why." — Babak

This applies beyond §5.3. The same development-narrative register runs through §5.1 (the four
failed levers), §4.1 (the kappa 0.38 first attempt) and §5.2. Those are the most distinctive
passages in the draft, and they are also exactly what Babak is asking to cut.

**Recommendation:** keep one short paragraph on the domain-mismatch diagnosis, because it justifies
a model choice the reader has to trust, and move the rest of the trajectory to an appendix or leave
it in the already-committed `experiments/trials_log.md`. Do not delete it from the repo.

*Effort: half a day.*

### 3.4 Re-run the joint evaluation on v3b

`code/eval_joint.py` against `models/layer2_tags_v3b`. Mechanical. *~15 minutes.*

---

## 4. Compliance and admin

### 4.1 ISAAC Data Use Agreement — blocking for release

> "The work uses ISAAC scripts, so it inherits ISAAC's Data Use Agreement. Read it carefully to
> ensure we stay compliant." — Babak

Read the DUA in the Illinois_Social_Attitudes repo and check every release claim in §7 against it,
particularly redistribution of raw comment text. §7 currently promises HuggingFace Datasets
distribution; the DUA may permit only ids plus derived labels.

### 4.2 Sensitive-content controls — blocking for release

> "consider carefully sensitive content controls, e.g., anonymizing comments given topics being
> discussed like self-harm and suicide." — Babak

The corpus carries 2,115 comments flagged `self_harm_suicide`. Options for the meeting: gated
access for the sensitive subset; releasing ids plus labels rather than text for those rows; or
paraphrase-level anonymisation. This interacts with 4.1 and should be decided alongside it.

### 4.3 Corpus name — pronounceable and meaningful

> "It helps the corpus' usage if the abbreviated title can be pronounced rather than read letter by
> letter. Even better if it has a relevant meaning. Also, you need to lay out the abbreviation in
> the first use."

AMHC fails both tests. Candidates, all pronounceable with athletic meaning:

| name | expansion | note |
|---|---|---|
| **STRIDE** | **S**port and **T**raining **R**eddit corpus for **I**dentifying **D**istress and **E**motion | recommended; running term, reads naturally |
| **TEMPO** | **T**raining and **E**xercise **M**ental-health **P**osts **O**nline | pacing term, clean expansion |
| **PACE** | **P**eer **A**ccounts of **C**oncerns in **E**xercise | shortest, expansion is a slight stretch |

Whichever is chosen, expand it on first use in both the abstract and the introduction.

---

## 5. For the meeting

- **§8 "Enriched-sample metrics are not corpus metrics"** — Babak: "Let's discuss this one." The
  distinction is real and load-bearing: per-tag precision and recall come from a cue-enriched set,
  and only the proportional 100 estimates natural prevalence. Worth keeping, probably worth
  rewording.
- **Venue** (§1). Determines the shape of the rest of the rewrite.
- **Whether the literature comparison becomes the paper's spine.**
  `docs/methods/literature_comparison_draft.md` shows sleep and substance use are roughly 12x quieter in
  peer discourse than in screening meta-analyses, and exercise-as-coping has no clinical
  counterpart at all. That is the strongest available candidate for Babak's "unique empirical
  findings that could not be achieved without your corpus." Caveat: the two largest gaps sit on the
  two weakest classifiers, so it needs the validation in §6 first.
- **Layer 1 kappa.** Currently 0.81 / 0.86. Babak asked for stratified subsamples "at both levels",
  so confirm the Layer-1 sample matches the design he specified for Layer 2, and re-run if not.

---

## 6. Deferred, but flagged

Targeted validation of `sleep` (recall 0.70) and `substance_use` (recall 0.52), drawn from comments
the wrong-sense kill rules suppressed. Not needed for the current draft, but required before the
literature-comparison finding can be published, because those rules bias exactly that comparison.
About a day.

---

## 7. Suggested order

1. Re-run the joint eval on v3b (15 min) — unblocks 2.10.
2. Apply all of §2 (half day). The kappa insertion alone changes how the paper reads.
3. Add confidence intervals, §3.1 (2 hours).
4. Rewrite §5.3 and trim the development narrative, §3.3 (half day).
5. Meeting: venue, corpus name, enriched-sample wording, ISAAC DUA, sensitive-content policy.
6. Deepen the per-tag findings, §3.2 (1–2 days) — the largest remaining piece of real work.
7. Implement the DUA and anonymisation decisions in §7 of the paper.

Steps 1–4 are roughly two days and produce a draft that answers every actionable comment. Step 6 is
what moves it from a competent resource paper to one with findings.


---

## 8. Round-2 status (2026-08-18, later)

Completed since this plan was written:

- **§3.1 confidence intervals** — done. `code/fig_tag_ci.py` adds 95% Wilson bands to every
  per-tag monthly series, Table 6 now carries adjusted CIs, and Figure 4 promotes four themes.
- **§3.4 joint eval on v3b** — done. micro P 0.76 / R 0.83 / F1 0.79 (n=580); new §5.3.
- **§5 Layer 1 kappa** — done to the Layer-2 design. `code/metrics_interrater_l1.py`. Full
  overlap n=2,502: mh 0.809, sport 0.861, **relevance 0.819**. Stratified 20% (n=500): 0.809,
  0.815, 0.849. The AND-ed relevance kappa is new; the paper previously reported only the two
  dimensions.
- **§4.3 corpus name** — STRIDE adopted. A search of the mental-health and sports NLP dataset
  literature found no existing corpus of that name (Microsoft's STRIDE threat model and several
  clinical trials use the acronym in unrelated fields).
- **#7 tag_corpus_final.py** — reconciled to `models/layer2_tags_v3b` and to all 16 tags, with
  the F1>=0.65 filter removed. Filtering by a global cutoff conflicts with the reviewer position
  that ~0.6 F1 is acceptable for subjective constructs, and with releasing per-tag figures.
- **§3.2 athlete-specificity** — done, and it changed a headline. See below.

### The ADHD finding needed the baseline, and survived it

`code/vocab_baseline.py` measures each theme's surface vocabulary across all 1,393,421 comments
in the same 19 communities (both arms, 2018-2022, the window where the control arm exists), then
indexes it against the tagged series.

| theme | community vocab %/yr | vocab idx | tagged idx | excess |
|---|---|---|---|---|
| adhd_neurodivergence | +11.67 | 143 | 196 | **1.37** |
| self_harm_suicide | -12.16 | 64 | 79 | 1.23 |
| anxiety | -3.44 | 78 | 88 | 1.13 |
| help_seeking | +4.37 | 111 | 122 | 1.09 |
| depression | -6.31 | 73 | 66 | 0.89 |
| loneliness_isolation | -2.02 | 88 | 72 | 0.81 |
| performance_psych | -10.41 | 49 | 117 | 2.40 (artefact, see paper §6.6) |

ADHD is the only theme that clearly outpaces its own community. Everything else tracks community
vocabulary, so those tagged trends describe the corpus rather than athletes. This is now §6.6.

**A first pass got this badly wrong and is worth recording.** Pooling both arms across all 72
months put the 2023 numerator over a denominator missing the control arm, inflating every 2023
vocabulary rate roughly threefold and making every theme look "inherited, attenuated" with excess
ratios near 0.4. Restricting to the 60 months where both arms exist reversed the conclusion for
ADHD. Third instance of the same bug class in this project: check what is in the denominator.

## 9. Still open

- **Venue** (§1) — unresolved and still the gating decision.
- **ISAAC DUA** — §7 now describes a full release per the author decision, with a TODO flagging
  that the agreement as written restricts redistribution to third parties and needs sign-off from
  the corpus owners.
- **§3.2 deepening** — partly done via §6.6; the "unique empirical findings" ask is now narrower
  and more honest, but the paper has one athlete-specific temporal finding rather than several.
- **Related work comparator table** (§2 TODO in the paper).
- **Sleep and substance_use validation** (§6 above) — still required before the
  literature-comparison analysis can be published.
