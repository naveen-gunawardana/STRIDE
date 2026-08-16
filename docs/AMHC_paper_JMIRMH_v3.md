<!-- Mirror of docs/AMHC_paper_JMIRMH_v3.docx for git diffing.
     Edit the generator and regenerate; do not hand-edit. -->

# What Athletes Say to Each Other Versus What Screening Instruments Find:
A 130,000-Comment Reddit Corpus of Athlete Mental Health

Naveen Gunawardana, Babak Hemmatian

*DRAFT v3 — 2026-08-14. Formatted for JMIR Mental Health (structured abstract, IMRaD, Discussion with Principal Findings / Comparison With Prior Work / Limitations / Conclusions). Second-choice venue: Behavior Research Methods, which would need the resource framing led instead of the finding. Corpus name AMHC is a placeholder. All figures are measured and reproducible; citations need verification against primary sources before submission.*

## Abstract

**Background:** Prevalence estimates for mental health problems in athletes come almost entirely from screening instruments, clinical intake, and surveys, all of which require the athlete to disclose. Athletes are known to under-disclose, so these estimates measure willingness to disclose alongside the condition itself. Peer conversation on public forums offers an unprompted comparison, but no corpus exists to support it.

**Objective:** To construct and validate a large corpus of athlete mental health discussion from Reddit, and to test whether the themes athletes raise with each other match the themes that screening instruments identify as most prevalent.

**Methods:** We applied a disorder-anchored keyword taxonomy to 1,474,439 comments posted to 19 sport, training, and fitness subreddits between January 2018 and December 2023, then a two-stage classifier cascade: a binary relevance gate for athlete mental health, and a multi-label classifier assigning 16 mental health themes. Both stages used a social-media-pretrained transformer with further domain-adaptive pretraining. The theme classifier was trained by weak supervision with abstention over 128,195 rule-labelled comments and fine-tuned on 1,358 hand-labelled comments. A control arm of comments from the same communities without mental health keywords provided a baseline. Theme prevalences were compared against pooled estimates from published prevalence meta-analyses. Temporal trends were estimated with binomial regression, with subreddit fixed effects to control for changing community composition.

**Results:** The gate achieved held-out precision 0.96, recall 0.83, and F1 0.89, flagging 22.24% of matched comments as relevant against 0.53% of control comments, a 42-fold separation. The theme classifier achieved pooled precision 0.79 and recall 0.79 across 6,784 decisions, with 5 of 16 themes meeting 0.80 on precision, recall, and F1. Theme prevalence diverged sharply from the clinical literature in a structured way. Sleep disturbance (26.4% in meta-analysis) and alcohol misuse (18.8%) appeared in only 2.2% and 1.6% of relevant comments respectively. Exercise used as a coping strategy appeared in 25.5% of comments and has no counterpart in the prevalence literature. Exercise dependence, the one theme where units align closely, matched published estimates (5.5% versus 3%-9%). Crude six-year trends in relevance were flat (+0.18%/year, P=.32), but rose after adjustment for community composition (+4.36%/year, P<.001).

**Conclusions:** Peer discourse and clinical screening disagree systematically about which athlete mental health problems are prominent. Themes that instruments explicitly ask about, particularly sleep and alcohol, are close to absent from unprompted peer conversation, while exercise as a coping strategy dominates peer discourse and is invisible to instruments. The corpus, trained models, pipeline, and a coding-free web interface are released publicly.

*Keywords: athlete mental health; sports psychology; social media; Reddit; natural language processing; corpus; prevalence; help-seeking*

## Introduction

Mental health problems in athletes are common enough to be a clinical priority and hard enough to measure that the prevalence figures remain uncertain. A systematic review and meta-analysis of current and former elite athletes found symptoms of anxiety or depression in 33.6% of current athletes, sleep disturbance in 26.4%, distress in 19.6%, and alcohol misuse in 18.8% [1]. The International Olympic Committee published a consensus statement the same year treating athlete mental health as a distinct area of sports medicine [2].

Nearly all of these figures come from screening instruments, clinical intake, or surveys administered by someone connected to the athlete's sport, and every one of them requires the athlete to speak first. Athletes are known to hold back. A meta-analysis of help-seeking found that only 22.4% of athletes seek professional support, with stigma the most frequently reported barrier [3,4]. A prevalence estimate built on disclosure therefore measures the willingness to disclose as well as the condition, and once the number exists the two cannot be separated.

One way to cross-validate those rates is to observe what athletes say to each other rather than what they say to a professional. Peer conversation on public forums is unprompted, written for other athletes rather than for a clinician, and available in large volume. Reddit hosts long-running communities organised around individual sports where people post about their own slumps, injuries, and loss of motivation. Natural language processing has made evaluation of such conversation practical at scale, and mental health has been an active application area, including work tracking support-community language through the COVID-19 pandemic [5,6].

Two research literatures meet at this problem and neither covers it. Sports psychology supplies the constructs, separating performance anxiety from generalised anxiety and treating loss of athletic identity after injury as a distinct phenomenon, but works mostly from small hand-coded samples [7]. Computational work on mental health and social media supplies scale and models but flattens the constructs, typically treating anxiety as a single label where DSM-5-TR describes a family of disorders with distinct subtypes and distinct treatments [8].

This study had two objectives. The first was to construct and validate a corpus of athlete mental health discussion large enough to train on and labelled finely enough to be clinically meaningful. The second was to use it to test a specific question that follows from the disclosure problem above: do the themes athletes raise unprompted with each other match the themes that screening instruments identify as most prevalent? If disclosure shapes prevalence estimates, the two should diverge, and the shape of the divergence should be informative.

## Methods

### Data Source and Keyword Filtering

The source was monthly Reddit comment archives from January 2018 to December 2023 covering 19 sport, training, and fitness communities, in two arms. The matched arm contained comments matching a mental health or sport keyword list. The control arm contained comments from the same communities over the same months matching no mental health keyword, so that any prevalence figure could be interpreted against a baseline.

The keyword taxonomy was disorder-anchored rather than assembled ad hoc. Disorder terms followed the IOC consensus statement [2] and the Gouttebarge et al. meta-analysis [1]; symptom vocabulary was mapped from DSM-5-TR [8] and the nine SCL-90-R dimensions [9], then translated into lay phrasing. Lists were built for recall, with precision restored downstream by the classifiers; keyword filtering alone plateaued at approximately 66% precision in pilot testing because residual false positives contain genuine mental health vocabulary used incidentally.

Comments posted in dedicated mental health communities (r/depression, r/Anxiety, r/mentalhealth, and similar) were removed before classification, on the grounds that posting in r/depression does not make a comment about sport. This removed 85,452 comments from the matched arm and 65,486 from the control arm. All rates reported here use the post-exclusion denominator.

### Relevance Classification

Pilot annotation asking a single question, "is this about sports mental health?", produced human-model agreement of kappa=0.38. The judgment is compound, and raters differed on which half of the construct to weight. We therefore split it into two questions that are individually unambiguous and combined them. The mental health dimension asked whether the comment referred to anyone's mental health, including in passing or about a third person. The sport dimension asked whether the comment content concerned sport, training, competition, or athletic life, judged on the comment rather than the author. Relevance was the conjunction. Agreement on the split questions rose to kappa=0.81 and kappa=0.86.

A total of 2,516 comments were annotated on both dimensions. Each dimension was modelled as a binary classifier initialised from a RoBERTa variant pretrained on social media text [10], with further domain-adaptive pretraining on 200,000 comments from this corpus [11]. The operating point was P(mental health) >= 0.50 and P(sport) >= 0.40, selected by grid search on a held-out set of 292 comments never used in training.

### Theme Classification

Sixteen themes were derived from the corpus rather than imported from a diagnostic manual. Prevalence was probed with high-recall regular expressions before any theme was defined, and 50 comments were read in full. Two observations shaped the taxonomy: the corpus is recreational rather than elite, so constructs built around selection pressure or media scrutiny would rarely fire; and exercise used as a coping strategy is pervasive, so the taxonomy carries response themes alongside condition themes rather than symptoms alone. The full rubric is provided in Multimedia Appendix 1.

Because 1,358 hand-labelled comments cannot train 16 classification heads, training signal came from labelling functions applied to the relevant corpus, using a three-valued output. A high-precision cue firing without negation produced a positive label. A cue firing under negation produced an explicit negative, which distinguishes "my anxiety was bad" from "I do not get anxious". When only a broad high-recall cue fired, the rule abstained and that cell was masked from the loss. When nothing fired, the label was negative. Abstention is essential here: labelling every comment without a precise cue as negative would train the model against every paraphrase the rules cannot express. For exercise-as-coping the rules fire on 3.4% of comments for a theme present in approximately 27%, so 85,756 comments abstain on that theme alone.

Wrong-sense guards were written per theme, targeting the fitness senses that dominate this corpus: muscular burnout is not psychological burnout, a stress fracture is not stress, "this workout is killing me" is not suicidality, and "drink more water" is not substance use. This point is revisited in the Limitations.

The classifier is a single shared encoder with 16 independent sigmoid outputs, trained with masked binary cross-entropy and per-theme positive weighting capped at 20. Training ran in two stages: two epochs over the 128,195 weakly labelled comments, then fine-tuning on the hand-labelled training split alone. Per-theme decision thresholds were set by prevalence matching on a development set rather than by maximising F1, because with fewer than 150 development rows F1 maximisation systematically selects thresholds that are too high.

### Annotation, Sampling, and Splits

Gold annotation used four draws: a proportional sample of 100 stratified by year and sport family, which is the only sample at natural prevalence and was assigned entirely to test; a theme-stratified draw of 330; and two focused draws of 120 and 60 targeting rare themes. Four draws were necessary because rare themes do not enrich under a single net. The proportional sample yielded one self-harm positive and the theme-stratified draw only six, because that theme's high-recall vocabulary is overwhelmingly sports slang in this corpus; two further draws using intermediate-precision cues brought it to 45.

Splits used iterative stratification, assigning each row to whichever split was furthest from quota on that row's rarest label, because with 16 correlated labels random assignment readily leaves a rare theme with too few test positives. Final splits were 793 training, 141 development, and 424 test. Two data leaks were identified and corrected during construction, the second caused by duplicate comment bodies under different identifiers; exclusion is now performed by text as well as by identifier, and leakage is verified at zero before each training run.

### Statistical Analysis

Monthly proportions were reported with Wilson score intervals. Trends were estimated with binomial generalised linear models of the form logit(p) ~ time, fitted on (month x community) cells, with time in years. Because community composition changed substantially over the study period, all trend models were refitted with subreddit fixed effects; both crude and adjusted estimates are reported. Direct standardisation to the pooled six-year community composition was used as a non-parametric check. Seasonality was tested by adding month-of-year fixed effects and comparing models by likelihood ratio. The COVID-19 period was examined with an interrupted time series specified at March 2020, estimating both a level shift and a slope change. Analyses used Python 3.11 with statsmodels; all code and intermediate outputs are released.

### Comparison With Published Prevalence Estimates

Theme prevalences were compared against pooled estimates from published systematic reviews and meta-analyses covering anxiety and depression, sleep, alcohol, disordered eating, exercise dependence, ADHD, and help-seeking [1,3,12-15]. This comparison is interpreted with care, because three mismatches separate the two quantities. The literature reports person-level prevalence, the share of athletes screening above a threshold, whereas we report utterance-level prevalence, the share of relevant comments carrying a theme. The literature concerns elite athletes, whereas this corpus is predominantly recreational. And a questionnaire elicits a construct on demand, whereas a corpus records what is volunteered. We therefore compare rank ordering and the direction and magnitude of departure, and make no claim of numerical equivalence.

### Ethical Considerations

The corpus consists of public Reddit comments. Author identifiers are removed from the released dataset, and comments are distributed with their platform identifiers so that upstream deletions can be honoured. No attempt was made to identify individuals, link accounts across communities, or infer clinical status for any named person. The theme labels are markers of discourse, not diagnoses; the self-harm theme in particular flags language rather than risk and must not be used to target individuals for intervention.

*TODO before submission: IRB determination or exemption letter, and JMIR's required Conflicts of Interest statement.*

## Results

### Corpus Characteristics

Table 1 summarises corpus construction.

| Arm | Comments read | Excluded pre-classification | Classified | Relevant | % relevant |
|---|---|---|---|---|---|
| Matched, 2018-2023 | 669,110 | 85,452 | 583,658 | 129,834 | 22.24 |
| Control, 2018-2022 | 956,267 | 65,486 | 890,781 | 4,700 | 0.53 |
| Total | 1,625,377 | 150,938 | 1,474,439 | 134,534 | 9.12 |

*Table 1. Corpus construction. The matched arm was flagged relevant 42 times more often than the control arm drawn from the same communities.*

The corpus is concentrated. The four largest communities (r/xxfitness 22.7%, r/running 19.1%, r/Fitness 18.0%, r/nba 14.9%) account for 74.7% of relevant comments. Relevance rates vary widely by community, from 48.7% in r/running to 7.7% in r/nba. Grouping the four spectator communities, where comments are largely fans discussing professional players rather than athletes describing themselves, against the remainder gives relevance rates of 8.68% versus 37.61%. Approximately 4% of the relevant corpus concerns professional sport.

Of relevant comments, 77.2% carried at least one condition theme, 10.5% carried only a response theme, and 12.3% carried no theme, with a mean of 1.51 themes per comment.

### Classifier Performance

The relevance gate achieved precision 0.96, recall 0.83, F1 0.89, and accuracy 0.90 on the 292-comment held-out set (47% relevant by construction; an always-positive baseline scores F1 0.61). Table 2 gives per-theme performance on the 424-comment test set.

| Theme | Precision | Recall | F1 | Test positives | Corpus % |
|---|---|---|---|---|---|
| Depression | 0.89 | 0.92 | 0.91 | 74 | 17.3 |
| Anxiety | 0.89 | 0.90 | 0.90 | 92 | 28.4 |
| Self-harm/suicide | 0.86 | 0.95 | 0.90 | 19 | 1.6 |
| Loneliness/isolation | 0.86 | 0.86 | 0.86 | 21 | 1.7 |
| Help-seeking | 0.80 | 0.89 | 0.85 | 46 | 13.2 |
| ADHD/neurodivergence | 0.73 | 0.95 | 0.83 | 20 | 2.7 |
| Burnout/motivation | 0.72 | 0.90 | 0.80 | 29 | 9.8 |
| Body image/eating | 0.80 | 0.76 | 0.78 | 58 | 13.8 |
| Sleep | 0.84 | 0.70 | 0.76 | 23 | 2.2 |
| Stress/pressure | 0.75 | 0.73 | 0.74 | 33 | 7.6 |
| Exercise dependence | 0.63 | 0.86 | 0.73 | 22 | 5.5 |
| Exercise coping | 0.70 | 0.73 | 0.72 | 93 | 25.5 |
| Performance psychology | 0.68 | 0.64 | 0.66 | 33 | 9.8 |
| Substance use | 0.87 | 0.52 | 0.65 | 25 | 1.6 |
| Trauma/PTSD | 0.85 | 0.50 | 0.63 | 22 | 1.2 |
| Injury distress | 0.62 | 0.58 | 0.60 | 26 | 4.9 |
| Micro-average (6,784 decisions) | 0.79 | 0.79 | 0.79 |  |  |
| Macro-average | 0.78 | 0.77 | 0.77 |  |  |

*Table 2. Theme classifier performance, ordered by F1. Per-theme accuracy is omitted because it exceeds 0.95 on rare themes purely from true negatives. Corpus % is the share of the 134,534 tagged comments carrying the theme; these were never fitted to and track the independent gold prevalence estimates closely.*

### Peer Discourse Versus Published Prevalence

Table 3 sets theme prevalence against pooled estimates from the prevalence literature.

| Theme | AMHC discourse % | Published prevalence % | Source population | Direction |
|---|---|---|---|---|
| Anxiety or depression | 43.7 | 33.6 | Current elite [1] | Higher |
| Exercise as coping | 25.5 | No estimate exists | - | Corpus only |
| Body image/eating | 13.8 | 19.2 | Athletes, all levels [12] | Comparable |
| Help-seeking | 13.2 | 22.4 (actually sought help) | Athletes [3] | Lower |
| Performance psychology | 9.8 | No prevalence category | - | Corpus only |
| Exercise dependence | 5.5 | 3-9 (8.2 gym attendees) | Exercisers [13,14] | Match |
| ADHD | 2.7 | 7-8 | Elite athletes [15] | Lower |
| Sleep | 2.2 | 26.4 | Current elite [1] | ~12x lower |
| Substance use | 1.6 | 18.8 (alcohol misuse) | Current elite [1] | ~12x lower |

*Table 3. Theme prevalence in peer discourse against published prevalence estimates. Units differ (utterance-level versus person-level) and populations differ (recreational versus elite), so the comparison is of rank and direction rather than magnitude.*

The divergence is large and structured. Sleep disturbance is the second most prevalent symptom in the reference meta-analysis at 26.4%, and independent sleep reviews place insomnia symptoms around 26% of athletes, yet sleep is among the rarest themes in this corpus at 2.2%. Alcohol misuse is 18.8% in the same meta-analysis; substance use appears in 1.6% of comments here. Both gaps are approximately twelvefold and run in the same direction.

In the opposite direction, exercise used as a coping strategy is the second most common theme in the corpus at 25.5% and has no counterpart in the prevalence literature, because prevalence instruments enumerate disorders and this is an adaptive behaviour. The same asymmetry applies more weakly to performance psychology and injury distress, both recognised constructs in sports psychology without population prevalence estimates.

Exercise dependence provides a point of convergent validity. It is the theme where the unit mismatch is smallest, because the construct is defined behaviourally and discussed in the terms it is screened for. Discourse prevalence of 5.5% falls inside the published band of 3%-9% for regular exercisers and is consistent with the 8.2% reported for gym attendees, given that this corpus is approximately 60% gym and running communities.

### Temporal Patterns

![figure](results_temporal/fig1_volume_and_rate.png)

*Figure 1. Monthly comment volume and crude relevance rate with 95% Wilson intervals. The control arm covers 2018-2022. Circles mark months deviating from a 13-month centred median by more than 2.5 robust SDs.*

The crude relevance rate was flat across the study period (+0.18% per year, 95% CI -0.18 to +0.54, P=.32), as was the control arm (+0.33% per year, P=.74). This flat result is misleading. Community composition shifted substantially: r/xxfitness and r/Fitness each lost approximately 11 percentage points of volume share between the first and last twelve months, while r/tennis gained 5.1 and r/crossfit 2.9, and the effective number of communities rose from 2.6 to 6.7. Because community relevance rates range from 7.7% to 48.7%, this drift moves the pooled rate independently of any behavioural change.

![figure](results_temporal/fig3_composition.png)

*Figure 2. Community composition by month (top) and the crude rate against the same corpus standardised to a fixed community mix (bottom), with linear fits. Both series derive from identical data.*

After adjustment for community, the relevance rate rose 4.36% per year (95% CI 3.93 to 4.78, P<.001), and 5.36% per year within participant communities. Direct standardisation to the pooled composition gives the same result without a model, moving from 21.08% in the first year to 22.76% in the last (+0.60 percentage points per year) against a crude slope of -0.03. Within communities, athlete mental health discussion increased over the six years; pooled across them it appeared flat because the corpus drifted toward communities that discuss it less.

Seasonality was strong and stable (likelihood ratio 563.3, df=11, P<.001), with relevance odds peaking in August (OR 1.17 versus January) and lowest in May (OR 0.89). An interrupted time series at March 2020 showed a level shift of OR 1.21 (P<.001) that persisted after adjustment for composition (OR 1.18, P<.001). April 2020 was the largest single deviation in the series. Two events nominated a priori as landmark athlete mental health moments, Naomi Osaka's withdrawal from the French Open in May 2021 and Simone Biles's withdrawal from Olympic finals in July 2021, produced no detectable change (robust z=-0.02 and +0.74).

![figure](results_temporal/fig2_tag_trends.png)

*Figure 3. Monthly share of relevant comments carrying each theme.*

Table 4 gives per-theme trends before and after adjustment for community.

| Theme | Crude %/year | Adjusted %/year (95% CI) | P | Interpretation |
|---|---|---|---|---|
| ADHD/neurodivergence | +20.9 | +24.6 (22.0 to 27.2) | <.001 | Robust, largest |
| Help-seeking | +6.1 | +5.8 (4.8 to 6.9) | <.001 | Robust |
| Burnout/motivation | +2.9 | +3.0 (1.8 to 4.2) | <.001 | Robust |
| Trauma/PTSD | +9.0 | +4.1 (1.0 to 7.4) | .009 | Mostly composition |
| Performance psychology | +8.5 | -4.0 (-5.1 to -2.9) | <.001 | Sign reverses |
| Stress/pressure | +0.4 | +0.6 (-0.7 to 1.9) | .38 | Flat |
| Injury distress | -1.1 | -0.8 (-2.3 to 0.8) | .31 | Flat |
| Sleep | -6.1 | -1.3 (-3.5 to 1.0) | .27 | Mostly composition |
| Exercise coping | -4.4 | -2.2 (-3.0 to -1.4) | <.001 | Attenuated |
| Body image/eating | -11.3 | -2.6 (-3.6 to -1.6) | <.001 | Mostly composition |
| Substance use | -4.5 | -3.3 (-5.8 to -0.7) | .01 | Robust |
| Anxiety | -4.3 | -6.4 (-7.1 to -5.7) | <.001 | Robust |
| Exercise dependence | -7.1 | -3.6 (-5.0 to -2.2) | <.001 | Attenuated |
| Loneliness/isolation | -8.9 | -8.0 (-10.4 to -5.6) | <.001 | Robust |
| Self-harm/suicide | -6.5 | -8.8 (-11.2 to -6.3) | <.001 | Robust |
| Depression | -13.4 | -12.7 (-13.4 to -11.9) | <.001 | Robust |

*Table 4. Annual change in the share of relevant comments carrying each theme, before and after adjustment for community composition.*

Four themes changed materially under adjustment and performance psychology reversed sign, appearing to grow in the crude series while declining within communities. The mechanism is visible in community-level prevalence: performance psychology accounts for 29.8% of relevant comments in r/tennis and 18.4% in r/nba against 3.3% in r/xxfitness, so drift toward tennis manufactures the apparent rise.

Community-level prevalences also serve as a validity check. Body image concerns concentrate in physique-oriented communities (25.1% in r/xxfitness, 25.5% in r/bodybuilding) and are near zero in spectator communities (1.4% in r/nba, 1.3% in r/tennis). Exercise-as-coping peaks in r/running at 49.2% and is lowest in r/tennis at 2.7%. Nothing was fitted to produce this pattern.

## Discussion

### Principal Findings

The central finding is that peer discourse and clinical screening disagree systematically about which athlete mental health problems are prominent, and that the disagreement has a structure. Two themes that screening instruments rank near the top, sleep and alcohol, are close to absent from unprompted peer conversation, each by roughly a factor of twelve. One theme that dominates peer discourse, exercise used to manage mental health, has no counterpart in the prevalence literature at all.

The most plausible account is framing rather than incidence. Within a training community, sleeplessness is a recovery topic and drinking is a training-compliance topic. Neither is posted under the heading of mental health, so neither appears in a corpus gated on athlete mental health. A questionnaire that asks about sleep inside a mental health battery converts a recovery complaint into a mental health symptom through the act of asking. If that reading is correct, it has a practical consequence: a screening programme and a peer-support service will surface different problems from the same population, and a service designed from screening data will over-provision for sleep and alcohol relative to what its users raise.

A second finding is methodological and applies to anyone using this or a similar corpus. The crude temporal trend and the composition-adjusted trend have opposite signs. Community composition in a multi-community corpus is not a nuisance parameter but a first-order confound, and the uncorrected series is the one a reader would plot by default.

### Comparison With Prior Work

Prior social media mental health corpora have generally defined populations by the support community a person posts in and labelled a single condition per resource [5,6]. This corpus defines its population by activity rather than by diagnosis and carries a 16-theme multi-label taxonomy, which is what makes the comparison in Table 3 possible: a single-label resource cannot show that one theme is over-represented relative to another.

The prevalence literature this work compares against is almost entirely elite [1,2,7], whereas this corpus is predominantly recreational and amateur competitive. That is a limitation for generalisation but arguably an advantage for applied relevance, since recreational and amateur athletes are both far more numerous and more likely to present to a practitioner without a sports-medicine team around them.

The failure to detect any change around the Osaka and Biles withdrawals is consistent with the same scope argument. Those events moved spectator communities, and spectator communities are the part of this corpus that discusses mental health least. A corpus built to capture people writing about themselves should not be expected to respond to another person's press conference, but the null result is reported because a resource claiming temporal validity should state which events it does not detect.

### Limitations

- The two largest divergences in Table 3 rest on the two weakest classifiers. Sleep has recall 0.70 and substance use has recall 0.52, so roughly half of true substance-use mentions are missed. Recall correction alone would move the observed 1.6% to approximately 3%, still far from 18.8% but accounting for a factor of two of the gap.
- The error on those themes is not random with respect to the claim. Both carry wrong-sense rules written specifically to suppress the fitness senses, removing recovery-related sleep and hydration-related drinking. Those rules were appropriate for building a precise classifier and they bias exactly this comparison. A targeted annotation pass drawn from comments the rules suppressed is required before the sleep and alcohol result should be treated as established.
- Units differ between the corpus and the literature (utterance-level versus person-level), so no numerical equivalence is claimed and none should be inferred.
- The corpus covers 19 communities and is predominantly recreational; prevalence figures do not transfer to elite populations, and Reddit users are not a representative sample of athletes.
- All theme annotations are single-rater applications of the written rubric. The relevance layer has human agreement of kappa=0.81 and kappa=0.86; the theme layer has no second-rater agreement yet.
- Eleven of 16 themes fall below 0.80 on at least one metric, and four fall below F1 0.70. Trends for those themes are statistically significant but should be treated as provisional; significance describes the classifier output, not the population.
- Per-theme metrics are measured on a cue-enriched test set and are not corpus-level estimates; only the proportional sample estimates natural prevalence.
- The control arm covers 2018-2022, so no control-referenced statement can be made about 2023.
### Conclusions

Athletes talking to each other raise a different set of mental health problems than screening instruments find in them. Sleep and alcohol, prominent in every prevalence meta-analysis, are nearly absent from peer discourse; exercise used as a coping strategy, absent from prevalence instruments, is one of the most common things athletes discuss. The size and direction of that divergence is measurable, and measuring it requires a corpus rather than an instrument.

We release the corpus of 129,834 labelled comments, the trained relevance and theme classifiers, the full pipeline with every reported result reproducible from a command-line argument, and a web interface that applies the models to uploaded text without requiring code, so that practitioners without computational training can use it directly. The immediate priority for further work is the targeted validation of the sleep and substance use classifiers described in the Limitations, which would determine whether the principal finding is a fact about athletes or an artefact of our own rules.

## References

*Recalled and web-verified summaries; every entry requires checking against the primary text before submission, and JMIR requires full citation detail with DOIs.*

1. Gouttebarge V, Castaldelli-Maia JM, Gorczynski P, et al. Occurrence of mental health symptoms and disorders in current and former elite athletes: a systematic review and meta-analysis. Br J Sports Med. 2019;53(11):700-706.

2. Reardon CL, Hainline B, Aron CM, et al. Mental health in elite athletes: International Olympic Committee consensus statement (2019). Br J Sports Med. 2019;53(11):667-699.

3. Castaldelli-Maia JM, Gallinaro JGME, Falcao RS, et al. Mental health symptoms and disorders in elite athletes: a systematic review on cultural influencers and barriers to athletes seeking treatment. Br J Sports Med. 2019;53(11):707-721.

4. Athlete mental health help-seeking: a systematic review and meta-analysis of rates, barriers and facilitators. Psychol Sport Exerc. 2024.

5. De Choudhury M, Gamon M, Counts S, Horvitz E. Predicting depression via social media. Proc Int AAAI Conf Web Soc Media. 2013.

6. Low DM, Rumker L, Talkar T, Torous J, Cecchi G, Ghosh SS. Natural language processing reveals vulnerable mental health support groups and heightened health anxiety on Reddit during COVID-19. J Med Internet Res. 2020;22(10):e22635.

7. Rice SM, Purcell R, De Silva S, Mawren D, McGorry PD, Parker AG. The mental health of elite athletes: a narrative systematic review. Sports Med. 2016;46(9):1333-1353.

8. American Psychiatric Association. Diagnostic and Statistical Manual of Mental Disorders. 5th ed, text rev. Washington DC: American Psychiatric Publishing; 2022.

9. Derogatis LR. SCL-90-R: Administration, Scoring and Procedures Manual. Minneapolis: National Computer Systems; 1994.

10. Barbieri F, Camacho-Collados J, Espinosa-Anke L, Neves L. TweetEval: unified benchmark and comparative evaluation for tweet classification. Findings of EMNLP. 2020.

11. Gururangan S, Marasovic A, Swayamdipta S, et al. Don't stop pretraining: adapt language models to domains and tasks. Proc ACL. 2020.

12. Prevalence of self-reported disordered eating and associated factors among athletes worldwide: a systematic review, meta-analysis and meta-regression. J Eat Disord. 2024;12:1.

13. Prevalence of risk for exercise dependence: a systematic review. Sports Med. 2019;49(2):191-205.

14. Exercise addiction in athletes: a systematic review of the literature. Int J Ment Health Addiction. 2021.

15. ADHD prevalence in elite and student athletes versus the general adult population. (Verify primary source; figures cited from review summaries.)

16. Ratner A, Bach SH, Ehrenberg H, Fries J, Wu S, Re C. Snorkel: rapid training data creation with weak supervision. Proc VLDB Endow. 2017;11(3):269-282.

## Multimedia Appendices

- Appendix 1: Full annotation rubric for the 16 themes, with inclusion and exclusion criteria and anchor examples.
- Appendix 2: Weak-supervision rule definitions, including wrong-sense guards.
- Appendix 3: Dataset datasheet, community list, and release licence.
- Appendix 4: Complete trial log, including negative results and abandoned approaches.