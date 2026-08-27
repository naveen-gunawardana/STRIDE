<!-- Mirror of docs/STRIDE_JMIRMH_v5.docx for git diffing. -->

# What Athletes Tell Each Other Versus What Screening Instruments Find: The STRIDE Corpus of 129,834 Athlete Mental Health Comments

Naveen Gunawardana, Babak Hemmatian

## Abstract

**Background:** Prevalence estimates for mental health problems in athletes come almost entirely from screening instruments, clinical intake, and surveys, all of which require the athlete to disclose. Athletes are known to under-disclose, so these estimates measure willingness to disclose alongside the condition. Peer conversation on public forums offers an unprompted comparison, but no corpus has existed to support one.

**Objective:** To construct and validate a large corpus of athlete mental health discussion from Reddit, and to test whether the themes athletes raise unprompted with each other match the themes screening instruments identify as most prevalent.

**Methods:** A disorder-anchored keyword taxonomy was applied to 1,474,439 comments posted to 19 sport, training, and fitness subreddits between January 2018 and December 2023, followed by a two-stage classifier cascade: a binary relevance gate for athlete mental health, and a multi-label classifier assigning 16 themes. Both stages used a social-media-pretrained transformer with further domain-adaptive pretraining. The theme classifier was trained by weak supervision with abstention over 128,195 rule-labelled comments and fine-tuned on 1,358 hand-labelled comments. Agreement was measured against a blinded second rater at both stages. Theme prevalences were compared with pooled estimates from published meta-analyses, and the two themes with the largest divergence were subjected to a stratified recall audit to separate measurement error from genuine difference. Temporal trends were estimated by binomial regression with community fixed effects.

**Results:** The gate achieved held-out precision 0.96, recall 0.83, and F1 0.89, flagging 22.24% of matched comments against 0.53% of control comments, a 42-fold separation. The theme classifier achieved pooled precision 0.79 and recall 0.79; the full cascade evaluated end to end reached F1 0.79. Second-rater agreement was kappa 0.819 for relevance and 0.879 pooled across theme decisions. Theme prevalence diverged from the clinical literature in a structured way. Sleep disturbance (26.4% in meta-analysis) and alcohol misuse (18.8%) appeared in 2.2% and 1.7% of relevant comments; after a recall audit correcting for missed positives these rose to 3.5% each, leaving gaps of 7.5-fold and 5.4-fold. Exercise used as a coping strategy appeared in 25.5% of comments and has no counterpart in the prevalence literature. Exercise dependence, where units align most closely, matched published estimates (5.5% vs 3%-9%).

**Conclusions:** Peer discourse and clinical screening disagree systematically about which athlete mental health problems are prominent. Themes that instruments explicitly ask about, particularly sleep and alcohol, remain several-fold quieter in unprompted peer conversation even after correcting for classifier recall, while exercise as a coping strategy dominates peer discourse and is invisible to instruments. The corpus, models, pipeline, and a coding-free web interface are released publicly.

Keywords: athlete mental health; sports psychology; social media; Reddit; natural language processing; corpus; prevalence; help-seeking

## Introduction

Mental health problems in athletes are common enough to be a clinical priority and hard enough to measure that the prevalence figures remain uncertain. A systematic review and meta-analysis of current and former elite athletes found symptoms of anxiety or depression in 33.6% of current athletes, sleep disturbance in 26.4%, distress in 19.6%, and alcohol misuse in 18.8% [1]. The International Olympic Committee published a consensus statement the same year treating athlete mental health as a distinct area of sports medicine [2].

Nearly all of these figures come from screening instruments, clinical intake, or surveys administered by someone connected to the athlete's sport, and every one of them requires the athlete to speak first. Athletes are known to hold back. A meta-analysis of help-seeking found that only 22.4% of athletes seek professional support, with stigma the most frequently reported barrier [3,4]. A prevalence estimate built on disclosure therefore measures the willingness to disclose as well as the condition, and once the number exists the two cannot be separated.

One way to cross-validate those rates is to observe what athletes say to each other rather than what they say to a professional. Peer conversation on public forums is unprompted, written for other athletes rather than for a clinician, and available in large volume. Reddit hosts long-running communities organised around individual sports where people post about their own slumps, injuries, and loss of motivation. Natural language processing has made evaluation of such conversation practical at scale, and mental health has been an active application area, including work tracking support-community language through the COVID-19 pandemic [5,6].

Two research literatures meet at this problem and neither covers it. Sports psychology supplies the constructs, separating performance anxiety from generalised anxiety and treating loss of athletic identity after injury as a distinct phenomenon, but works mostly from small hand-coded samples [7]. Computational work on mental health and social media supplies scale and models but flattens the constructs, typically treating anxiety as a single label where DSM-5-TR describes a family of disorders with distinct subtypes and distinct treatments [8].

This study had two objectives. The first was to construct and validate a corpus of athlete mental health discussion large enough to train on and labelled finely enough to be clinically meaningful. The second was to use it to test a question that follows directly from the disclosure problem: do the themes athletes raise unprompted with each other match the themes screening instruments identify as most prevalent? If disclosure shapes prevalence estimates, the two should diverge, and the shape of the divergence should be informative.

## Methods

### Data Source and Keyword Filtering

The source was monthly Reddit comment archives from January 2018 to December 2023 covering 19 sport, training, and fitness communities, in two arms. The matched arm contained comments matching a mental health or sport keyword list. The control arm contained comments from the same communities over the same months matching no mental health keyword, so that any prevalence figure could be interpreted against a baseline.

The keyword taxonomy was disorder-anchored. Disorder terms followed the IOC consensus statement [2] and the Gouttebarge et al meta-analysis [1]; symptom vocabulary was mapped from DSM-5-TR [8] and the nine SCL-90-R dimensions [9], then translated into lay phrasing. Lists were built for recall, with precision restored downstream by the classifiers. Comments posted in dedicated mental health communities were removed before classification, on the grounds that posting in r/depression does not make a comment about sport; this removed 85,452 comments from the matched arm and 65,486 from the control arm, and all rates use the post-exclusion denominator.

### Relevance Classification

Pilot annotation asking a single compound question produced agreement of kappa 0.38. We therefore split it into two questions that are individually unambiguous. The mental health dimension asked whether the comment referred to anyone's mental health, including in passing or about a third person. The sport dimension asked whether the comment content concerned sport, training, competition, or athletic life, judged on the comment rather than the author. Relevance was the conjunction. A total of 2,516 comments were annotated on both dimensions.

Each dimension was modelled as a binary classifier initialised from a RoBERTa variant pretrained on social media text [10], with further domain-adaptive pretraining on 200,000 comments from this corpus [11]. The operating point was P(mental health) >= 0.50 and P(sport) >= 0.40, selected by grid search on a held-out set of 292 comments never used in training.

### Theme Classification

Sixteen themes were derived from the corpus rather than imported from a diagnostic manual. Prevalence was probed with high-recall regular expressions before any theme was defined, and 50 comments were read in full. The corpus is recreational rather than elite, so constructs built around selection pressure or media scrutiny would rarely fire; and exercise used as a coping strategy is pervasive, so the taxonomy carries response themes alongside condition themes rather than symptoms alone. The full rubric is in Multimedia Appendix 1.

Because 1,358 hand-labelled comments cannot train 16 classification heads, training signal came from labelling functions applied to the relevant corpus with a three-valued output. A high-precision cue firing without negation produced a positive label. A cue firing under negation produced an explicit negative, which distinguishes "my anxiety was bad" from "I do not get anxious". When only a broad high-recall cue fired the rule abstained and that cell was masked from the loss. When nothing fired, the label was negative. Abstention is essential: labelling every comment without a precise cue as negative would train the model against every paraphrase the rules cannot express. Wrong-sense guards were written per theme, targeting the fitness senses that dominate this corpus, where muscular burnout is not psychological burnout and "drink more water" is not substance use.

The classifier is a single shared encoder with 16 independent sigmoid outputs, trained with masked binary cross-entropy and per-theme positive weighting capped at 20, in two stages: two epochs over the weakly labelled rows, then fine-tuning on the hand-labelled training split alone. Per-theme decision thresholds were set by prevalence matching on a development set rather than by maximising F1, because with fewer than 150 development rows F1 maximisation selects thresholds that are systematically too high.

### Annotation, Splits, and Agreement

Gold annotation used four draws: a proportional sample of 100 stratified by year and sport family, assigned entirely to test as the only sample at natural prevalence; a theme-stratified draw of 330; and two focused draws of 120 and 60 targeting rare themes. Splits used iterative stratification, giving 793 training, 141 development, and 424 test comments. Gold rows were excluded from the weak-supervision set by text as well as identifier, with leakage verified at zero before each training run.

Agreement was measured against a blinded second rater at both stages. For relevance, 2,502 double-rated comments gave Cohen's kappa 0.809 (95% CI 0.786-0.832) on the mental health dimension, 0.861 (0.839-0.882) on sport, and 0.819 (0.796-0.842) on the conjunction. For themes, a 160-comment stratified subset of the test set drawn so that every theme had positive instances gave pooled kappa 0.879 with raw agreement 0.976 across 2,560 decisions, and a mean per-theme kappa of 0.862, with 14 of 16 themes at or above 0.80.

### Comparison With Published Prevalence Estimates

Theme prevalences were compared against pooled estimates from published systematic reviews and meta-analyses [1,3,12-15]. Three mismatches separate the two quantities and constrain the comparison. The literature reports person-level prevalence, the share of athletes screening above a threshold, whereas we report utterance-level prevalence, the share of relevant comments carrying a theme. The literature concerns elite athletes, whereas this corpus is predominantly recreational. And a questionnaire elicits a construct on demand, whereas a corpus records what is volunteered. We therefore compare rank ordering and the direction and magnitude of departure, and make no claim of numerical equivalence.

### Recall Audit of the Divergent Themes

The two themes that diverged most from the literature, sleep and substance use, are also among the classifier's weakest, and both carry wrong-sense rules that suppress the fitness senses. Attributing their low rates to athletes rather than to our own rules required a separate audit. Three recovery strata were sampled from the relevant corpus: comments where a kill rule vetoed a fired cue, comments where the rule abstained, and comments the model scored below but near threshold. Each stratum was sampled and hand-labelled against the published rubric, and the corrected prevalence is the tagged count plus the stratum size multiplied by its observed positive rate, with Wilson intervals carried through.

### Statistical Analysis

Monthly proportions are reported with Wilson score intervals. Trends were estimated with binomial generalised linear models of the form logit(p) ~ time on (month x community) cells. Because community composition changed substantially over the period, all trend models were refitted with community fixed effects and both estimates are reported, with direct standardisation to the pooled composition as a non-parametric check. Seasonality was tested by adding month-of-year fixed effects and comparing models by likelihood ratio. The COVID-19 period was examined with an interrupted time series at March 2020. To distinguish athlete-specific change from community-wide language shift, each theme's surface vocabulary rate was computed across all comments in the same communities and indexed against the tagged series. Analyses used Python 3.11 with statsmodels; all code and intermediate outputs are released.

### Ethical Considerations

The corpus consists of public Reddit comments. Author identifiers are removed from the released dataset, and comments are distributed with their platform identifiers so that upstream deletions can be honoured. No attempt was made to identify individuals, link accounts across communities, or infer clinical status for any named person. The theme labels are markers of discourse, not diagnoses; the self-harm theme flags language rather than risk and must not be used to target individuals for intervention.

*TODO before submission: IRB determination or exemption, JMIR conflicts-of-interest statement, and confirmation that the release is compatible with the ISAAC Data Use Agreement.*

## Results

### Corpus Characteristics

Table 1 summarises corpus construction.

| Arm | Comments read | Excluded pre-classification | Classified | Relevant | % of classified |
|---|---|---|---|---|---|
| Matched, 2018-2023 | 669,110 | 85,452 | 583,658 | 129,834 | 22.24 |
| Control, 2018-2022 | 956,267 | 65,486 | 890,781 | 4,700 | 0.53 |
| Total | 1,625,377 | 150,938 | 1,474,439 | 134,534 | 9.12 |

*Table 1. Corpus construction. The matched arm was flagged relevant 42 times more often than the control arm drawn from the same communities.*

The corpus is concentrated. The four largest communities (r/xxfitness 22.7%, r/running 19.1%, r/Fitness 18.0%, r/nba 14.9%) account for 74.7% of relevant comments, and relevance rates vary from 48.7% in r/running to 7.7% in r/nba. Grouping the four spectator communities, where comments are largely fans discussing professional players rather than athletes describing themselves, against the remainder gives relevance rates of 8.68% versus 37.61%. Approximately 4% of the relevant corpus concerns professional sport. Of relevant comments, 77.2% carried at least one condition theme, 10.5% carried only a response theme, and 12.3% carried none, with a mean of 1.51 themes per comment.

### Classifier Performance

The relevance gate achieved precision 0.96, recall 0.83, F1 0.89, and accuracy 0.90 on the 292-comment held-out set (47% relevant by construction; an always-positive baseline scores F1 0.61). Evaluated end to end on 580 raw comments combining the gold test set with control-arm comments, the full cascade reached pooled precision 0.76, recall 0.83, and F1 0.79; gate recall on truly relevant comments was 1.00 and its false-pass rate on irrelevant comments was 3.2%. Table 2 gives per-theme performance with second-rater agreement.

| Theme | Precision | Recall | F1 | Test positives | Corpus % | Rater kappa |
|---|---|---|---|---|---|---|
| Depression | 0.89 | 0.92 | 0.91 | 74 | 17.3 | 0.90 |
| Anxiety | 0.89 | 0.90 | 0.90 | 92 | 28.4 | 0.91 |
| Self-harm/suicide | 0.86 | 0.95 | 0.90 | 19 | 1.6 | 0.88 |
| Loneliness/isolation | 0.86 | 0.86 | 0.86 | 21 | 1.7 | 0.81 |
| Help-seeking | 0.80 | 0.89 | 0.85 | 46 | 13.2 | 0.93 |
| ADHD/neurodivergence | 0.73 | 0.95 | 0.83 | 20 | 2.7 | 0.96 |
| Burnout/motivation | 0.72 | 0.90 | 0.80 | 29 | 9.8 | 0.65 |
| Body image/eating | 0.80 | 0.76 | 0.78 | 58 | 13.8 | 0.95 |
| Sleep | 0.84 | 0.70 | 0.76 | 23 | 2.2 | 0.92 |
| Stress/pressure | 0.75 | 0.73 | 0.74 | 33 | 7.6 | 0.58 |
| Exercise dependence | 0.63 | 0.86 | 0.73 | 22 | 5.5 | 0.96 |
| Exercise coping | 0.70 | 0.73 | 0.72 | 93 | 25.5 | 0.82 |
| Performance psychology | 0.68 | 0.64 | 0.66 | 33 | 9.8 | 0.81 |
| Substance use | 0.87 | 0.52 | 0.65 | 25 | 1.6 | 0.96 |
| Trauma/PTSD | 0.85 | 0.50 | 0.63 | 22 | 1.2 | 0.87 |
| Injury distress | 0.62 | 0.58 | 0.60 | 26 | 4.9 | 0.87 |
| Micro-average (6,784 decisions) | 0.79 | 0.79 | 0.79 |  |  | 0.88 |
| Macro-average | 0.78 | 0.77 | 0.77 |  |  | 0.86 |

*Table 2. Theme classifier performance on the 424-comment gold test set, ordered by F1, with the share of the tagged corpus each theme covers and the second-rater kappa. Per-theme accuracy is omitted because it exceeds 0.95 on rare themes purely from true negatives.*

Per-theme F1 and per-theme rater agreement move together. The two themes where independent raters agree least, stress/pressure (kappa 0.58) and burnout/motivation (kappa 0.65), are also where the classifier scores lowest, which locates the remaining headroom on those constructs in the rubric rather than the model. Corpus tag rates were never fitted to and track the independent gold prevalence estimates closely, each running slightly below its gold estimate as a precision-favouring operating point should.

### Peer Discourse Versus Published Prevalence

Table 3 sets theme prevalence against pooled estimates from the prevalence literature.

| Theme | STRIDE discourse % | Published prevalence % | Population | Ratio |
|---|---|---|---|---|
| Anxiety or depression | 43.7 | 33.6 | Current elite [1] | 0.8 |
| Exercise as coping | 25.5 | No estimate exists | - | n/a |
| Body image/eating | 13.8 | 19.2 | Athletes, all levels [12] | 1.4 |
| Help-seeking | 13.2 | 22.4 (sought help) | Athletes [4] | 1.7 |
| Performance psychology | 9.8 | No prevalence category | - | n/a |
| Exercise dependence | 5.5 | 3-9 (8.2 gym attendees) | Exercisers [13,14] | 1.1 |
| Sleep (corrected) | 3.5 | 26.4 | Current elite [1] | 7.5 |
| Substance use (corrected) | 3.5 | 18.8 (alcohol misuse) | Current elite [1] | 5.4 |

*Table 3. Theme prevalence in peer discourse against published prevalence estimates. Units differ (utterance-level versus person-level) and populations differ (recreational versus elite), so the comparison is of rank and direction rather than magnitude. Sleep and substance use are shown after the recall correction in Table 4.*

The divergence is large and structured. Sleep disturbance is the second most prevalent symptom in the reference meta-analysis at 26.4%, and independent sleep reviews place insomnia symptoms near 26% of athletes, yet sleep appears in 2.2% of relevant comments. Alcohol misuse is 18.8% in the same meta-analysis; substance use appears in 1.7%. Both gaps are approximately twelvefold before correction and run in the same direction.

In the opposite direction, exercise used as a coping strategy is the second most common theme at 25.5% and has no counterpart in the prevalence literature, because prevalence instruments enumerate disorders and this is an adaptive behaviour. The same asymmetry applies to performance psychology and injury distress, both recognised constructs in sports psychology without population prevalence estimates. Exercise dependence provides a point of convergent validity: it is the theme where the unit mismatch is smallest, and its discourse prevalence of 5.5% falls inside the published band of 3% to 9% for regular exercisers.

### Recall Audit of the Divergent Themes

Table 4 gives the corrected prevalences.

| Audited theme | Model-tagged % | Corrected % (95% CI) | Gap before | Gap after |
|---|---|---|---|---|
| Sleep | 2.23 | 3.50 (2.78-5.02) | 11.8x | 7.5x |
| Substance use | 1.67 | 3.46 (2.64-4.79) | 11.3x | 5.4x |
| Substance use, excluding tobacco | 1.67 | 3.22 | 11.3x | 5.8x |

*Table 4. Stratified recall audit. Corrected prevalence is the tagged count plus each recovery stratum's size multiplied by its hand-labelled positive rate, with Wilson intervals carried through.*

Roughly a third to a half of each gap is measurement, and the majority survives. Sleep moves from 2.23% to 3.50% (95% CI 2.78-5.02), reducing the gap from 11.8-fold to 7.5-fold; substance use moves from 1.67% to 3.46% (2.64-4.79), from 11.3-fold to 5.4-fold. Notably, the wrong-sense kill rules we expected to be responsible were not: only 14 comments corpus-wide had a sleep cue vetoed by a kill rule, and 12 for substance use. The missed positives sit almost entirely in the abstention stratum, where the rules declined to commit.

One boundary issue emerged. Fifteen of the 33 hand-labelled substance-use positives rest on smoking or nicotine, which the rubric names neither as included nor excluded. Excluding them gives a corrected prevalence of 3.22% and a 5.8-fold gap. Smoking cessation is a common topic in running communities, and a rubric for this domain should decide explicitly whether tobacco counts.

### Temporal Patterns

![figure](results_temporal/fig1_volume_and_rate.png)

*Figure 1. Monthly comment volume and crude relevance rate with 95% Wilson intervals. The control arm covers 2018-2022.*

The crude relevance rate was flat across the study period (+0.18% per year, 95% CI -0.18 to +0.54, P=.32), as was the control arm (+0.33% per year, P=.74). Community composition shifted substantially over the same period: r/xxfitness and r/Fitness each lost approximately 11 percentage points of volume share while r/tennis gained 5.1 and r/crossfit 2.9, and the effective number of communities rose from 2.6 to 6.7. Because community relevance rates range from 7.7% to 48.7%, this shift moves the pooled rate independently of any behavioural change.

| Model | Change per year | 95% CI | P |
|---|---|---|---|
| Crude: logit(relevant) ~ time | +0.18% | -0.18 to +0.54 | .32 |
| Adjusted: + community fixed effects | +4.36% | +3.93 to +4.78 | <.001 |
| Adjusted, participant communities | +5.36% | +4.85 to +5.87 | <.001 |
| Adjusted, spectator communities | +2.08% | +1.33 to +2.83 | <.001 |

*Table 5. Binomial GLM on (month x community) cells, 583,658 comments.*

![figure](results_temporal/fig3_composition.png)

*Figure 2. Community composition by month (top) and the crude rate against the same corpus standardised to a fixed community mix (bottom).*

After adjustment for community, the relevance rate rose 4.36% per year (95% CI 3.93 to 4.78, P<.001). Direct standardisation gives the same result without a model, moving from 21.08% in the first year to 22.76% in the last against a crude slope of -0.03. Within communities, athlete mental health discussion increased; pooled across them it appeared flat because the corpus drifted toward communities that discuss it less.

Seasonality was strong and stable (likelihood ratio 563.3, df=11, P<.001), with relevance odds peaking in August (OR 1.17 versus January) and lowest in May (OR 0.89). An interrupted time series at March 2020 showed a level shift of OR 1.21 (P<.001) that persisted after adjustment for composition (OR 1.18, P<.001). The mechanism is visible in the composition data: r/nba's share fell from 15.0% in January 2020 to 5.8% in April, matching the suspension of the NBA season on 11 March, while r/xxfitness rose from 23.1% to 36.7%. Two events nominated a priori as landmark athlete mental health moments, Naomi Osaka's withdrawal from the French Open in May 2021 and Simone Biles's withdrawal from Olympic finals in July 2021, produced no detectable change (robust z=-0.02 and +0.74).

### Athlete-Specific Versus Community-Wide Change

A theme's share can rise among mental health comments because athletes increasingly discuss it in that frame, or because the vocabulary rose across the whole community. Table 6 compares each theme's tagged series against the rate of its own surface vocabulary across all 1,393,421 comments in the same communities over the 60 months where both arms exist.

| Theme | Community vocabulary %/yr | Vocabulary index | Tagged index | Excess |
|---|---|---|---|---|
| ADHD/neurodivergence | +11.67 | 143 | 196 | 1.37 |
| Self-harm/suicide | -12.16 | 64 | 79 | 1.23 |
| Anxiety | -3.44 | 78 | 88 | 1.13 |
| Help-seeking | +4.37 | 111 | 122 | 1.09 |
| Depression | -6.31 | 73 | 66 | 0.89 |
| Loneliness/isolation | -2.02 | 88 | 72 | 0.81 |

*Table 6. Community vocabulary trend against the tagged series, indexed to 2018 = 100. Excess is the tagged index divided by the vocabulary index.*

![figure](results_temporal/fig5_vocab_baseline.png)

*Figure 3. Four themes indexed to 2018, tagged series against community-wide vocabulary.*

Only ADHD and neurodivergence clearly exceeds its own community baseline, with an excess of 1.37: its vocabulary rose 11.7% per year across these communities and the tagged series rose faster still. Part of that effect is a genuine shift in how athletes frame neurodivergence and part is inherited from a broader change in usage; reporting the tagged trend alone would have credited all of it to athletes. Help-seeking, anxiety, depression, self-harm and loneliness track their community vocabulary within the resolution of this test, so their tagged trends describe this corpus rather than something particular to athletes.

## Discussion

### Principal Findings

Peer discourse and clinical screening disagree systematically about which athlete mental health problems are prominent, and the disagreement has a structure. Sleep and alcohol, ranked first and fourth among symptoms in the reference meta-analysis, are 7.5-fold and 5.4-fold quieter in unprompted peer conversation even after a recall audit correcting for classifier misses. Exercise used to manage mental health, which dominates a quarter of peer discourse, has no counterpart in the prevalence literature at all.

The most plausible account is framing rather than incidence. Within a training community, sleeplessness is a recovery topic and drinking is a training-compliance topic; neither is posted under the heading of mental health, so neither appears in a corpus gated on athlete mental health. A questionnaire that asks about sleep inside a mental health battery converts a recovery complaint into a mental health symptom through the act of asking. If that reading is correct it has a practical consequence: a screening programme and a peer-support service will surface different problems from the same population, and a service designed from screening data will over-provision for sleep and alcohol relative to what its users raise.

A second finding is methodological and applies to any multi-community corpus. The crude and composition-adjusted temporal trends have opposite signs, and the uncorrected series is the one a reader would plot by default. A third, from the vocabulary comparison, is that most per-theme trends in this corpus are inherited from community-wide language shifts rather than specific to athletes.

### Comparison With Prior Work

Prior social media mental health corpora have generally defined populations by the support community a person posts in and labelled a single condition per resource [5,6]. This corpus defines its population by activity rather than diagnosis and carries a 16-theme multi-label taxonomy, which is what makes the comparison in Table 3 possible: a single-label resource cannot show that one theme is over-represented relative to another.

The prevalence literature compared against is almost entirely elite [1,2,7], whereas this corpus is predominantly recreational and amateur competitive. That limits generalisation but arguably increases applied relevance, since recreational and amateur athletes are both far more numerous and more likely to present to a practitioner without a sports medicine team around them. The failure to detect change around the Osaka and Biles withdrawals is consistent with the same scope argument: those events moved spectator communities, and spectator communities are the part of this corpus that discusses mental health least.

### Limitations

- The corpus covers recreational and amateur competitive athletes across 19 communities, not elite athletes and not Reddit as a whole. Prevalence figures do not transfer to elite populations.
- Units differ between the corpus and the literature, utterance-level versus person-level, so no numerical equivalence is claimed and none should be inferred. The comparison rests on the direction and magnitude of departure.
- The recall audit was labelled by the same rater who produced the gold set, so it bounds measurement error rather than validating it independently. It also covers only the two audited themes; other themes may carry similar corrections.
- Per-theme metrics rest on a 424-comment test set, and the rarest themes on fewer than 25 positive instances. Future work will confirm robustness on substantially larger labelled subsets.
- Per-theme performance varies with construct difficulty, and the themes where the model scores lowest are the themes where independent human raters also agree least. For subjective constructs of this kind, F1 near 0.6 remains workable for aggregate analysis, but users should consult the per-theme figures before relying on any single theme.
- Enriched-sample metrics are not corpus metrics. Per-theme precision and recall are measured on a cue-enriched test set, which is necessary to obtain enough positives for the rare themes; natural prevalence is estimated only from the proportional sample.
- The control arm covers 2018 through 2022, so comparisons against the control and the vocabulary analysis are restricted to that window.
- The rubric does not state whether tobacco counts as substance use, which materially affects that theme's prevalence in a corpus where smoking cessation is a common topic.
### Conclusions

Athletes talking to each other raise a different set of mental health problems than screening instruments find in them. Sleep and alcohol, prominent in every prevalence meta-analysis, remain several-fold quieter in peer discourse after correcting for measurement; exercise used as a coping strategy, absent from prevalence instruments, is one of the most common things athletes discuss. The size and direction of that divergence is measurable, and measuring it requires a corpus rather than an instrument.

We release the corpus of 129,834 labelled comments, the trained relevance and theme classifiers, the full pipeline with every reported result reproducible from a command-line argument, and a web interface that applies the models to uploaded text without requiring code. The immediate priority for further work is a second independent rater on the recall audit, and extending the audit to the remaining themes.

## References

- Gouttebarge V, Castaldelli-Maia JM, Gorczynski P, et al. Occurrence of mental health symptoms and disorders in current and former elite athletes: a systematic review and meta-analysis. Br J Sports Med. 2019;53(11):700-706.
- Reardon CL, Hainline B, Aron CM, et al. Mental health in elite athletes: International Olympic Committee consensus statement (2019). Br J Sports Med. 2019;53(11):667-699.
- Castaldelli-Maia JM, Gallinaro JGME, Falcao RS, et al. Mental health symptoms and disorders in elite athletes: a systematic review on cultural influencers and barriers to athletes seeking treatment. Br J Sports Med. 2019;53(11):707-721.
- Athlete mental health help-seeking: a systematic review and meta-analysis of rates, barriers and facilitators. 2024.
- De Choudhury M, Gamon M, Counts S, Horvitz E. Predicting depression via social media. Proc Int AAAI Conf Web Soc Media. 2013.
- Low DM, Rumker L, Talkar T, Torous J, Cecchi G, Ghosh SS. Natural language processing reveals vulnerable mental health support groups and heightened health anxiety on Reddit during COVID-19. J Med Internet Res. 2020;22(10):e22635.
- Rice SM, Purcell R, De Silva S, Mawren D, McGorry PD, Parker AG. The mental health of elite athletes: a narrative systematic review. Sports Med. 2016;46(9):1333-1353.
- American Psychiatric Association. Diagnostic and Statistical Manual of Mental Disorders. 5th ed, text rev. Washington DC: American Psychiatric Publishing; 2022.
- Derogatis LR. SCL-90-R: Administration, Scoring and Procedures Manual. Minneapolis: National Computer Systems; 1994.
- Barbieri F, Camacho-Collados J, Espinosa-Anke L, Neves L. TweetEval: unified benchmark and comparative evaluation for tweet classification. Findings of EMNLP. 2020.
- Gururangan S, Marasovic A, Swayamdipta S, et al. Don't stop pretraining: adapt language models to domains and tasks. Proc ACL. 2020.
- Prevalence of self-reported disordered eating and associated factors among athletes worldwide: a systematic review, meta-analysis and meta-regression. J Eat Disord. 2024;12:1.
- Prevalence of risk for exercise dependence: a systematic review. Sports Med. 2019;49(2):191-205.
- Exercise addiction in athletes: a systematic review of the literature. Int J Ment Health Addiction. 2021.
- Sleep quality in elite athletes: systematic review evidence on insomnia symptom prevalence. Sports Med. 2017.
- Hemmatian B, Kurdi B. The Illinois Social Attitudes Aggregate Corpus [Computer software]. GitHub; 2025. https://github.com/BabakHemmatian/Illinois_Social_Attitudes
- Ratner A, Bach SH, Ehrenberg H, Fries J, Wu S, Re C. Snorkel: rapid training data creation with weak supervision. Proc VLDB Endow. 2017;11(3):269-282.
*References recalled and partially web-verified; every entry requires checking against the primary text before submission, and JMIR requires full author lists and DOIs.*

## Multimedia Appendices

- Appendix 1: Full annotation rubric for the 16 themes, with inclusion and exclusion criteria and anchor examples.
- Appendix 2: Weak-supervision rule definitions, including wrong-sense guards.
- Appendix 3: Dataset datasheet, community list, and release licence.
- Appendix 4: All 16 themes over time with 95% Wilson intervals.
![figure](results_temporal/fig2_tag_trends_ci.png)

*Appendix Figure 4-1. Share of relevant comments carrying each of the 16 themes, by month, with 95% Wilson intervals.*
