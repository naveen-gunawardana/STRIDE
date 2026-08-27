# Draft analysis — how AMHC compares to the sports-psychology prevalence literature

**Drafted 2026-08-14.** Intended as a candidate section for the paper (or, if the contrast holds
up, as the paper's central framing). Companion to `docs/temporal_analysis_2026-08-14.md`.

**Bottom line up front:** the contrast is large, and more usefully, it is *structured*. Peer
discourse and clinical screening do not disagree at random. Two themes that screening instruments
rank near the top (sleep, alcohol) are close to absent from what athletes say to each other, and
one theme that dominates peer discourse (exercise as a coping mechanism) has essentially no
counterpart in the prevalence literature at all. That pattern is a finding in its own right, and
it is the finding this corpus is uniquely able to produce.

**UPDATE 2026-08-26: the recall audit in §5 has been run.** Both gaps shrink but survive. Sleep
2.23% -> 3.50% corrected (gap 11.8x -> 7.5x); substance use 1.67% -> 3.46% (11.3x -> 5.4x). The
wrong-sense kill rules were not the cause: only 14 and 12 comments corpus-wide had a cue vetoed.
The misses sat in the abstention stratum. Full numbers in `results_temporal/audit_corrected.json`
and in the paper (STRIDE_JMIRMH_v5, Results/Recall Audit). This section is superseded by the paper.

---

## 1. The comparison is not apples to apples, and saying so is half the analysis

Three mismatches have to be stated before any number is put next to another.

**Unit.** The literature reports *person-level* prevalence: the share of athletes who screen above
a symptom threshold. We report *utterance-level* prevalence: the share of mental-health-relevant
comments that carry a theme, conditional on the comment already having passed the relevance gate.
These are different quantities and no direct equality should ever be asserted between them. A
theme can be common in people and rare in text, or the reverse.

**Population.** The prevalence literature is overwhelmingly about *elite* athletes. AMHC is
overwhelmingly *recreational*: 60% of the relevant corpus is r/xxfitness, r/running and r/Fitness,
and only about 4% is professional-sport discussion. Elite athletes have competition travel,
selection pressure, and scheduled training loads that recreational gym-goers do not, and several
of the constructs below are sensitive to exactly that.

**Instrument.** Screening questionnaires *ask* about sleep and alcohol. Nobody asks a Reddit
commenter anything. A questionnaire measures a construct on demand; a corpus measures what people
volunteer. The gap between the two is not error. It is the object of study, and it is precisely
the premise the paper opens with — that disclosure-dependent estimates measure willingness to
disclose alongside the condition.

**So what is comparable?** Two things. The *rank ordering* of themes, and the *direction and
magnitude of departure* from that ordering. Those are what §3 reports.

---

## 2. The numbers

AMHC figures are the share of the 129,834 relevant comments carrying each tag (model
`layer2_tags_v3b`). Literature figures are pooled prevalence estimates from the reviews cited in
§7.

| Theme | AMHC discourse % | Literature prevalence % | Population in lit. | Reading |
|---|---|---|---|---|
| anxiety | 28.4 | 33.6 (anxiety/depression combined) | current elite | — |
| depression | 17.3 | (included in the 33.6 above) | current elite | — |
| **anxiety or depression** | **43.7** | **33.6** | current elite | discourse higher |
| exercise as coping | 25.5 | **no comparable estimate exists** | — | **corpus-only signal** |
| body image / eating | 13.8 | 19.2 (6–45 F, 0–19 M; gymnastics 41.5) | athletes, all levels | roughly comparable |
| help-seeking | 13.2 | 22.4 (proportion who actually seek help) | athletes | discourse lower |
| burnout / motivation | 9.8 | 19.6 ("distress", nearest construct) | current elite | discourse lower |
| performance psychology | 9.8 | **no prevalence category** | — | corpus-only signal |
| stress / pressure | 7.6 | 19.6 ("distress", same nearest construct) | current elite | discourse lower |
| exercise dependence | 5.5 | 3–9 overall; 8.2 gym, 14.2 endurance | exercisers / athletes | **clean match** |
| injury distress | 4.9 | no standard prevalence category | — | corpus-only signal |
| ADHD / neurodivergence | 2.7 | 7–8 | elite athletes | ~3x lower |
| **sleep** | **2.2** | **26.4** | current elite | **~12x lower** |
| loneliness / isolation | 1.7 | no standard estimate | — | — |
| **substance use** | **1.6** | **18.8 (alcohol misuse)** | current elite | **~12x lower** |
| self-harm / suicide | 1.6 | varies widely by definition | — | — |
| trauma / PTSD | 1.2 | varies widely by definition | — | — |

Supporting corpus figures: 77.2% of relevant comments carry at least one condition tag, 10.5%
carry only a response tag (help-seeking or exercise-coping with no named condition), and 12.3%
carry no tag at all. Mean 1.51 tags per comment.

---

## 3. Three findings

### 3.1 Sleep and alcohol are loud in screening and near-silent in peer discourse

This is the largest contrast in the table and the two gaps are roughly the same size, about 12x in
each case. Sleep disturbance is the *second* most prevalent symptom in the Gouttebarge meta-analysis
at 26.4%, and independent sleep reviews put insomnia symptoms around 26% of athletes, yet sleep is
the fourth *rarest* tag in our corpus at 2.2%. Alcohol misuse is 18.8% in the same meta-analysis
and substance use is 1.6% here.

The most plausible explanation is framing rather than incidence. In a training community, not
sleeping is a *recovery* topic and drinking is a *training-compliance* topic. Neither gets posted
under the heading of mental health, so neither appears in a corpus gated on athlete mental health.
A questionnaire that asks "how has your sleep been?" inside a mental-health battery converts a
recovery complaint into a mental-health symptom by the act of asking.

If that reading is right, it is a substantive claim about how athletes construe their own
difficulties, and it has an obvious practical consequence: a screening instrument and a
peer-support service will surface different problems from the same population, and a service
designed off screening data will over-provision for sleep and alcohol relative to what its users
actually raise.

**It is also, unfortunately, where our measurement is weakest.** See §5.

### 3.2 Exercise-as-coping is the corpus's signal and has no clinical counterpart

25.5% of relevant comments describe exercise being used to manage mental health, or failing to.
It is the second most common theme in the corpus. There is no prevalence estimate to compare it
to, because prevalence instruments enumerate disorders and this is an adaptive behaviour, not a
disorder. The clinical literature has a large *effects* literature on exercise and mood, but not a
prevalence literature on people using exercise this way.

That asymmetry is the strongest argument for the corpus existing. A resource built from what
athletes volunteer surfaces a category that a resource built from screening instruments cannot,
because the instrument has no item for it. The same holds, more weakly, for performance psychology
(9.8%) and injury distress (4.9%): both are recognised constructs in sports psychology, neither
has a population prevalence figure to compare against.

### 3.3 Exercise dependence is the one clean numeric match, and it validates the pipeline

AMHC puts exercise dependence at 5.5% of relevant comments. The systematic-review estimate is 3–9%
of regular exercisers, 8.2% among gym attendees and 14.2% among endurance athletes. Given that our
corpus is roughly 60% gym-and-running communities, 5.5% sits inside the expected band.

This matters more than it looks. It is the only theme where the unit mismatch happens to be small
(exercise dependence is defined behaviourally and gets discussed in the same terms it is screened
for), and on that theme the corpus and the literature agree. That is a point of convergent
validity that costs nothing to report.

### 3.4 A tension worth noting: ADHD

ADHD discourse is 2.7% of the corpus against a 7–8% elite-athlete prevalence, so on the face of it
it is under-represented. But it is also the fastest-growing tag in the corpus by a wide margin,
rising about 25% per year and roughly tripling across the six years. Either peer discourse is
converging on the clinical rate, or both are riding the broader rise in ADHD discussion and
diagnosis over 2018–2023. The corpus cannot distinguish those on its own; a comparison against
general-population Reddit over the same window would.

---

## 4. What this means for the paper's framing

The user question behind this analysis was whether the contrast is big enough to gear the paper
toward sports psychology. My reading: **yes, and it upgrades the contribution.**

The current draft is a resource paper whose analysis section demonstrates that the corpus supports
temporal work. That is publishable but modest, and it competes with every other dataset paper on
scale and validation quality.

The contrast above supports a stronger claim: *peer discourse and clinical screening disagree
systematically about which athlete mental-health problems are prominent, and the disagreement has
a structure that a corpus can measure and an instrument cannot.* That is a psychology contribution
with an NLP resource attached, rather than an NLP resource with a psychology application. It
engages the theory, it gives the dataset a reason to exist that is not just size, and it lands
better at a psychology-leaning venue.

It also fits what is already the paper's opening argument. The introduction claims that
disclosure-dependent prevalence estimates conflate the condition with the willingness to disclose.
Section 3.1 above is the first actual evidence for that claim rather than an assertion of it.

**Recommended restructure if this is adopted:** keep the corpus and pipeline sections as they are,
demote the temporal analysis to a shorter subsection, and promote this comparison to the main
results section. The temporal composition finding stays, because it is the thing that stops a
reader misusing the resource, but it becomes a methods caution rather than the headline.

---

## 5. Why not to build on this yet

The two gaps carrying the argument are on the two tags our classifier handles worst.

| tag | precision | recall | F1 |
|---|---|---|---|
| sleep | 0.85 | **0.70** | 0.76 |
| substance_use | 0.87 | **0.52** | 0.65 |

Recall of 0.52 on substance use means we are missing roughly half the true positives. That alone
turns a measured 1.6% into something closer to 3%, which is still far from 18.8% but is a factor
of two of the gap explained by measurement rather than by athletes.

Worse, the error is not random with respect to the claim. Both tags carry aggressive wrong-sense
kill rules written specifically to suppress the fitness senses: `sleep` kills the
recovery/rest-day sense, `substance_use` kills hydration and pre-workout. Those rules were correct
for building a clean classifier and they systematically bias exactly the comparison being made
here. We built a tool to ignore sleep-as-recovery and are now reporting that athletes do not
discuss sleep.

**Before this becomes a paper claim, it needs:**

1. A targeted annotation pass on `sleep` and `substance_use`, drawn from comments the kill rules
   suppressed rather than from the usual enrichment, to estimate how much of the gap is our rules.
2. A recall-corrected prevalence estimate for both tags, with an interval, rather than the raw
   flagged rate.
3. Ideally, a general-population Reddit comparison over the same window, which would separate
   "athletes do not frame sleep as mental health" from "nobody on Reddit does."

Item 1 is a day's work and would settle whether §3.1 is a finding or an artefact. Nothing else in
this document depends on it: §3.2 and §3.3 stand on their own.

---

## 6. Honest scorecard

**Solid.** Exercise-as-coping has no clinical counterpart (§3.2) — this rests on the absence of a
literature, not on our recall. Exercise dependence matches the literature band (§3.3) — this rests
on a tag with F1 0.73 and a real numeric agreement. Body image is roughly comparable. The
anxiety-or-depression composite at 43.7% against 33.6% is *not* evidence of anything, because the
units differ; it is reported to show the comparison is being made honestly, not to claim agreement.

**Promising but unproven.** The sleep and alcohol gaps (§3.1). Large enough to be interesting,
measured with the two weakest tags, confounded by our own kill rules.

**Not claimable.** Anything about elite athletes, since the corpus is recreational. Any statement
that AMHC prevalence *corrects* a literature estimate, since the units are not the same quantity.

---

## 7. Sources

Figures in §2 are drawn from these. All need checking against the primary text before they go into
a submitted paper; the numbers here come from review abstracts and secondary summaries.

- Gouttebarge et al. (2019), *Occurrence of mental health symptoms and disorders in current and former elite athletes: a systematic review and meta-analysis*, BJSM — current elite athletes: anxiety/depression 33.6%, sleep disturbance 26.4%, distress 19.6%, alcohol misuse 18.8%; former athletes 16–26%. [University of Portsmouth record](https://researchportal.port.ac.uk/en/publications/occurrence-of-mental-health-symptoms-and-disorders-in-current-and/) · [full text PDF](https://observatorio.fm.usp.br/bitstream/OPI/33194/1/art_GOUTTEBARGE_Occurrence_of_mental_health_symptoms_and_disorders_in_2019.PDF)
- [Does Elite Sport Degrade Sleep Quality? A Systematic Review](https://link.springer.com/article/10.1007/s40279-016-0650-6) and [The Variability of Sleep Among Elite Athletes](https://link.springer.com/article/10.1186/s40798-018-0151-2) — sleep disturbance 13–70% depending on measure; ~26% score for insomnia symptoms.
- [Prevalence of self-reported disordered eating among athletes worldwide: systematic review, meta-analysis and meta-regression](https://link.springer.com/article/10.1186/s40337-024-00982-5) — mean 19.23% across 177 studies and 70,957 athletes; gymnastics highest at 41.5%.
- [Prevalence of Risk for Exercise Dependence: A Systematic Review](https://link.springer.com/article/10.1007/s40279-018-1011-4) — 3–7% of regular exercisers, 6–9% of athletes; [Exercise Addiction in Athletes: a Systematic Review](https://link.springer.com/article/10.1007/s11469-021-00568-1) — endurance 14.2%, gym attendees 8.2%.
- [Athlete mental health help-seeking: a systematic review and meta-analysis of rates, barriers and facilitators](https://pubmed.ncbi.nlm.nih.gov/38128709/) — pooled help-seeking proportion 22.4%.
- ADHD in athletes 7–8% against a general-population figure variously given as 0.8–2.4% or ~5%: [Do Athletes Have More of a Cognitive Profile with ADHD Criteria than Non-Athletes?](https://pmc.ncbi.nlm.nih.gov/articles/PMC8151350/)
- AMHC figures: `results_temporal/tag_trends.json`, `run_logs/tag_corpus_v3b.log`, `run_logs/eval_l2v3b.log`.
