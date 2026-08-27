<!-- Mirror of docs/STRIDE_paper_v4.docx for git diffing. -->

# STRIDE: A Reddit Corpus for Text-Based Identification of Mental Health Concerns in Athletes

Naveen Gunawardana, Babak Hemmatian

## Abstract

Prevalence estimates for mental health problems in athletes come almost entirely from instruments that require the athlete to disclose, and athletes are known to under-disclose. We present STRIDE (Sport and Training Reddit corpus for Identifying Distress and Emotion), a corpus of 129,834 Reddit comments from 19 sports, training, and fitness communities, posted between 2018 and 2023 and labeled for the mental health themes they contain. The corpus is built by a two-stage cascade: a binary relevance gate that decides whether a comment concerns athlete mental health (held-out precision 0.96, recall 0.83), and a multi-label tagger over 16 themes (pooled precision 0.79, recall 0.79), trained by weak supervision with abstention over 128,195 rule-labeled comments and fine-tuned on 1,358 hand-labeled ones. A blinded second-rater pass over 160 test comments gives a pooled Cohen's kappa of 0.88. Evaluated end to end on raw comments, the full cascade reaches a pooled F1 of 0.79. A control arm of 890,781 comments from the same communities without mental health keywords is flagged at 0.53%, compared with the matched arm's 22.24%, a separation of 42x. We use the corpus to show that the crude monthly rate of athlete mental health discussion is flat over six years, and that accounting for the changing mix of communities reveals a within-community rise of 4.4% per year. We release the models, the pipeline, and a coding-free web application, with corpus access governed by the ISAAC Data Use Agreement.

## 1. Introduction

Mental health problems in athletes are common enough to be a clinical priority and hard enough to measure that we do not know how common they are. A meta-analysis of current and former elite athletes found symptoms of anxiety or depression in roughly a third of the samples it reviewed (Gouttebarge et al., 2019), and the International Olympic Committee published a consensus statement the same year treating athlete mental health as its own area of sports medicine (Reardon et al., 2019). Nearly all of those figures come from screening instruments, clinical intake, or surveys administered by someone affiliated with the athlete's sport, and each one requires the athlete to speak first. Athletes are known to hold back. Stigma, concern about selection, and a culture that frames psychological difficulty as weakness all push in the same direction (Castaldelli-Maia et al., 2019). A prevalence estimate built on disclosure measures both the willingness to disclose and the condition, and once the number exists, the two cannot be disentangled.

One way to cross-validate those rates is to look at what athletes say to each other rather than to a professional. Peer conversation on public forums is unprompted, written for other athletes rather than for a clinician, and there is a great deal of it. Reddit hosts long-running communities built around individual sports, where people post about their own slumps, injuries, and loss of motivation, often in more detail than they would give a coach. Natural language processing has reached the point where conversation of this kind can be evaluated at scale, and mental health has been one of its more active application areas (De Choudhury et al., 2013; Coppersmith et al., 2014; Low et al., 2020).

Two literatures meet at this problem, and neither one covers it. Sports psychology supplies the constructs. It separates performance anxiety from generalized anxiety and treats the loss of athletic identity after injury as a phenomenon in its own right, but it works mostly from small hand-coded samples (Rice et al., 2016). Computational work on mental health and social media provides the scale and models, and flattens constructs along the way. Anxiety is usually a single label. In DSM-5-TR terms, it is a family of disorders with distinct subtypes and distinct treatments (American Psychiatric Association, 2022), and that distinction is the part a practitioner would act on.

We present the STRIDE corpus to support the text-based identification of mental health concerns among athletes. It contains 129,834 Reddit comments posted in sport, training, and fitness communities between 2018 and 2023, each tagged with the mental health themes they contain. The corpus was built by passing 1,625,377 comments through a disorder-anchored keyword taxonomy and then two stages of trained classifiers: a binary gate that decides whether a comment concerns athlete mental health, and a multi-label tagger that assigns which of 16 themes are present. On held-out data, the gate reaches a precision of 0.96 and a recall of 0.83. Across all tag decisions, the tagger achieves a precision of 0.79 and a recall of 0.79. Every comment carries a probability and a binary flag, so users can set their own operating point rather than inherit ours.

The labels are multi-label rather than exclusive because a comment describing a sleepless night before a race and then a decision to see a counselor describes more than one thing, and forcing it into a single category throws the second one away. The six-year span makes temporal analysis possible. The corpus also ships with a control arm of 890,781 comments drawn from the same subreddits without mental health keywords, so the prevalence figure can be read against a baseline rather than on its own. That matters more than it sounds, and Section 6 is largely about how much it matters: the crude six-year trend in this corpus is flat, and the within-community trend is a 4.4% annual rise. The two differ because the mix of communities posting changed underneath the series.

Some scope conditions should be clear from the start. The corpus is dominated by recreational and amateur competitive athletes rather than elite ones: about three-quarters of the tagged comments come from r/xxfitness, r/running, r/Fitness and r/nba, and roughly four percent concern professional sport. That narrows what can be claimed about elite populations, although it covers the population a sports psychologist is most likely to meet in practice. Reddit users are also not a representative sample of athletes. Per-tag performance varies across the taxonomy, and every tag is released with its own precision, recall, F1 and inter-rater agreement so that users can judge which are fit for their purpose.

Several groups can use this. Clinicians and sports psychologists can check which concerns appear in peer discussion and how athletes describe them in their own words, which is not the vocabulary of an intake form. Epidemiologists get a disclosure-independent comparison for survey-based prevalence estimates. NLP researchers get a labelled multi-label dataset in a domain where informal register and wrong-sense vocabulary make the task genuinely hard: in a fitness corpus, "stress" is usually a stress fracture, "burnout" is usually muscular, and "drinking" is usually water. The classifiers are released through HuggingFace Models and the pipeline on GitHub, with every result reproducible from a command-line argument, and a HuggingFace Space runs the classifiers over uploaded text for users who do not write code. Corpus access is governed by the ISAAC Data Use Agreement (Section 7).

This paper makes four contributions.

1. STRIDE, a corpus of 129,834 Reddit comments from athlete and training communities tagged for 16 mental health themes, with a matched control arm and six years of coverage.
1. A two-stage classification pipeline, a relevance gate followed by a multi-label tagger, with per-layer, per-tag and end-to-end performance, and a blinded second-rater agreement study.
1. A temporal analysis showing that the crude trend in this corpus reflects changing community composition, and that the within-community trend has the opposite sign.
1. Public release of the model weights, the pipeline, and a coding-free web application.
## 2. Related work

Work on mental health in social text has a long line of Reddit and Twitter resources. De Choudhury et al. (2013) predicted depression from Twitter behaviour; the CLPsych shared tasks established the evaluation conventions the area still uses (Coppersmith et al., 2014); Low et al. (2020) tracked support-group language across the onset of COVID-19. What these share is a construct granularity of one label per condition, and a population defined by the support community a person posts in. STRIDE differs on both counts: the population is defined by activity rather than by diagnosis, and the labels are a 16-way multi-label taxonomy rather than a binary.

On the clinical side, the IOC consensus statement (Reardon et al., 2019) and the accompanying prevalence meta-analysis (Gouttebarge et al., 2019) define the constructs the keyword taxonomy is anchored to, and Rice et al. (2016) survey the elite-athlete literature. None of that work operates at corpus scale.

STRIDE is built with the ISAAC pipeline (Hemmatian & Kurdi, 2025), which supplies the keyword filtering, language filtering, sampling and anonymisation stages as swappable components; the relevance and theme classifiers described here are the additions specific to this corpus.

*TODO: this section needs a comparator table (corpus, size, span, labels, validation, access), and a paragraph on weak supervision (Ratner et al., 2017) positioning the abstention design of Section 4.2.*

## 3. The corpus

### 3.1 Source data and keyword filtering

The source is monthly Reddit comment dumps from January 2018 to December 2023 across 19 sport, training, and fitness communities, in two arms. The matched arm holds comments that hit a mental-health or sport keyword list. The control arm holds comments from the same communities over the same months that hit no mental-health keyword, and exists so that any prevalence figure can be read against something.

The keyword taxonomy is disorder-anchored rather than assembled ad hoc. Disorder terms follow the IOC consensus statement (Reardon et al., 2019) and the Gouttebarge et al. (2019) meta-analysis; symptom vocabulary is mapped from DSM-5-TR (American Psychiatric Association, 2022) and the nine SCL-90-R dimensions (Derogatis, 1994), then translated into the phrasing people actually use on Reddit. The lists are built for recall on the understanding that precision is restored downstream by the learned classifiers. Keyword filtering alone reaches about 66% precision on this task, because the remaining false positives contain genuine mental-health vocabulary used incidentally, and separating those requires a semantic model.

One empirical finding shaped the sampling. Across all 574,639 matched comments from 2018-2022, a mental-health keyword and an athlete-identity keyword co-occur in the same comment zero times. Comments are short, and people rarely say both in one breath. Athlete context therefore has to be established by the community and by the content of the comment, not by a conjunction of keywords.

### 3.2 Why classifiers, and which ones

We wanted the language ability of a modern large model and also needed to process 1.6 million comments. Those pull in opposite directions on cost. The resolution used throughout is to spend the large model on annotation and a compact encoder on inference: a large language model applies a written rubric to produce labels, and a fine-tuned RoBERTa-family encoder learns from those labels and does the corpus pass. Annotation is where judgment is needed and volume is small; inference is where volume is large and the decision is narrow.

### 3.3 What the finished dataset looks like

Table 1 gives the arm-level counts. Two denominators are reported because they answer different questions. Comments in dedicated mental-health subreddits (r/depression, r/Anxiety, r/mentalhealth and similar) are removed before the models run, on the reasoning that being in r/depression does not make a comment about sport. Those comments cannot be flagged relevant, so including them in the denominator lowers every rate by a fixed amount that carries no information.

| Arm | Comments read | Excluded pre-classification | Classified | Relevant | % of classified |
|---|---|---|---|---|---|
| Matched, 2018-2023 | 669,110 | 85,452 | 583,658 | 129,834 | 22.24% |
| Control, 2018-2022 | 956,267 | 65,486 | 890,781 | 4,700 | 0.53% |
| Total | 1,625,377 | 150,938 | 1,474,439 | 134,534 | 9.12% |

*Table 1. Corpus construction. The matched arm is flagged relevant 42x more often than the control arm drawn from the same communities, which is the primary evidence that the gate isolates signal rather than responding to sports vocabulary in general.*

The corpus is 19 communities, and it is concentrated. Table 2 gives the composition of the relevant set.

| Community | n | % of relevant | Relevance rate | Community | n | % of relevant | Relevance rate |
|---|---|---|---|---|---|---|---|
| xxfitness | 29,532 | 22.7% | 45.7% | tennis | 5,304 | 4.1% | 15.7% |
| running | 24,805 | 19.1% | 48.7% | weightroom | 3,333 | 2.6% | 32.8% |
| Fitness | 23,314 | 18.0% | 30.8% | crossfit | 2,866 | 2.2% | 31.8% |
| nba | 19,391 | 14.9% | 7.7% | Swimming | 2,520 | 1.9% | 46.0% |
| bodybuilding | 8,306 | 6.4% | 25.9% | 11 others | 10,463 | 8.1% | - |

*Table 2. Composition of the relevant corpus. The relevance rate is the share of that community's classified comments the gate flags.*

Two features of this table shape the rest of the paper. The corpus is recreational: r/xxfitness, r/running and r/Fitness are people writing about their own training, and they are 60% of it. And r/nba, the fourth-largest contributor at 14.9%, is a spectator community whose relevance rate is 7.7% against 45.7% in r/xxfitness. Grouping the four spectator communities (nba, CollegeBasketball, Basketball, tennis) against the rest gives 8.68% versus 37.61%. These are different populations using the same vocabulary for different purposes, and Section 6 shows that separating them changes the direction of several trends.

## 4. Labelling

### 4.1 Relevance: decomposing a compound judgment

Relevance is judged as two independent questions rather than one. Dimension mh asks whether the comment refers to anyone's mental health, including in passing, as advice, or about a third person. Dimension sport asks whether the comment content is about sport, training, competition, or athletic life, judged on the comment rather than the author, so that an athlete describing a bad day at work scores zero. Relevance is the conjunction of the two.

The decomposition is not cosmetic. Asking annotators the compound question directly, "is this sports mental health?", produced human-model agreement of kappa 0.38; the two component questions reach kappa 0.81 and 0.86 on the same material. Keeping the dimensions separate also leaves the pipeline reconfigurable: fans by mental health, or athletes by physical injury, can be assembled from the same labels without annotating again. 2,516 comments were labelled on both dimensions. The mh dimension includes sports performance psychology, on the research judgment that for an athlete population, confidence problems, competition nerves and choking are often how mental health difficulty first appears.

### 4.2 Themes: weak supervision with abstention

The 16 themes are listed in Table 4 with their per-tag performance. They were designed from the corpus rather than imported from a clinical manual: prevalence was probed with high-recall regular expressions before any tag was written, and 50 comments were read raw. Two facts shaped the taxonomy. The corpus is recreational, so a taxonomy built around elite constructs such as selection pressure or media scrutiny would rarely fire. And exercise used as a coping mechanism is pervasive in this corpus and is arguably its most distinctive signal, so the taxonomy carries response tags alongside condition tags rather than symptoms alone.

1,358 hand-labelled comments cannot train 16 heads. Training signal comes from labelling functions over the relevant corpus, but with a three-valued output rather than a binary one. A high-precision cue that fires un-negated gives a positive. A cue that fires but is negated gives an explicit negative, which are the most valuable rows in the set because they teach the difference between "my anxiety was bad" and "I don't get anxious" instead of letting the model key on the keyword. When only the broad high-recall cue fires, the rule abstains and that cell is masked out of the loss entirely. When nothing fires, the label is negative.

Abstention is the design decision the layer rests on. If every comment without the precise cue were labelled negative, then every paraphrase the rules cannot express would be trained as a negative, which teaches the model not to generalise past the lexicon and defeats the point of using a transformer at all. The scale is what makes it matter: for exercise_coping the rules fire on 3.4% of comments for a theme genuinely present in about 27%, so 85,756 comments abstain on that tag alone.

Wrong-sense guards are written per tag and are mostly fitness-specific, because that is where a naive lexicon fails on this corpus: muscular burnout is not psychological burnout, a stress fracture is not stress, "this workout is killing me" is not suicidality, binge drinking is not disordered eating, and "drink more water" is not substance use.

### 4.3 Sampling and splits

Gold labelling used four draws. A proportional sample of 100, stratified by year and sport family, is the only set at natural prevalence and goes entirely to test. A theme-stratified draw of 330 and two focused draws of 120 and 60 provide per-tag support. Four draws were needed because rare tags do not enrich with a single net: the proportional 100 yielded one self-harm positive, and the theme-stratified 330 reached only six, because that tag's high-recall vocabulary ("kill", "die", "dead", "cut") is overwhelmingly sports slang here. Two further draws using mid-precision cues brought it to 45. For a rare label in a domain with heavy wrong-sense vocabulary, cue-based enrichment has to be tuned per tag.

Splits use iterative stratification rather than random assignment, placing each row into whichever split is furthest from quota on that row's rarest label. With 16 correlated labels a random split easily strands a rare tag at four test positives. Active-learning rows are model-selected and go wholly into train, since scoring on them would measure the model against the errors it was chosen to have. The result is 793 train, 141 dev, 424 test. Gold rows are excluded from the weak-supervision training set by text as well as by identifier, and leakage is verified at zero before each training run.

### 4.4 Inter-rater agreement

A blinded second rater independently labelled a 160-comment subset of the gold test set, drawn so that every tag has positive instances present, with no access to the first rater's labels. Pooled across all 2,560 tag decisions, Cohen's kappa is 0.879 with raw agreement 0.976; the mean of the 16 per-tag kappas is 0.862. Fourteen of the 16 tags reach kappa 0.8 or above. Per-tag values appear in the last column of Table 4.

The two tags that fall below that level are informative rather than alarming. burnout_motivation (kappa 0.647) and stress_pressure (kappa 0.582) are the two constructs whose boundary the rubric itself draws least sharply, since ordinary training fatigue shades into psychological burnout and ordinary life pressure shades into clinical stress. Their agreement is a property of the construct, not of the annotation procedure, and it places a ceiling on what any classifier can be expected to achieve on them.

## 5. Models and performance

### 5.1 Layer 1: the relevance gate

The gate is two binary classifiers, one per dimension, combined by a hard AND. Both are initialised from cardiffnlp/twitter-roberta-base with further domain-adaptive pretraining on 200,000 comments from this corpus (Barbieri et al., 2020; Gururangan et al., 2020). The operating point is P(mh) >= 0.50 and P(sport) >= 0.40.

The choice of a social-media-pretrained base is the single decision that determines this layer's performance, and it is worth one paragraph because a reader reusing the models needs to know it does not transfer from a general-purpose encoder. With a standard RoBERTa base the gate plateaus at F1 0.68, and the diagnostic is visible in one number: across 292 held-out comments the highest probability the mh model ever emits is 0.57. A classifier whose most confident prediction sits barely above the threshold cannot be repaired by tuning the decision rule, and neither more labelled data, a learned combiner in place of the AND, nor tripling capacity to roberta-large moved it. Substituting a same-sized encoder pretrained on social media raised the maximum emitted probability to 1.00 and the gate to F1 0.92 under the original rubric. Adapting that encoder further on this corpus produced the deployed model.

| Metric | Value |
|---|---|
| Precision | 0.96 |
| Recall | 0.83 |
| F1 | 0.89 |
| Accuracy | 0.90 |

*Table 3. Layer 1 held-out performance. n = 292, 47% relevant by construction. The always-positive baseline scores F1 0.61 on this set.*

### 5.2 Layer 2: the multi-label tagger

The tagger is a single shared encoder, initialised from the same adapted base, with 16 independent sigmoid heads. A shared encoder rather than 16 binary models because the tags co-occur heavily and the representations transfer, and because it is substantially cheaper to train and serve. The loss is binary cross-entropy with two modifications: a per-element mask so abstained cells contribute no gradient, and a per-tag positive weight capped at 20, without which the rare heads collapse to all-zero predictions.

Training runs in two stages. Stage one is two epochs over 128,195 weakly labelled rows. Stage two initialises from that checkpoint and fine-tunes on the hand-labelled training split alone for 18 epochs at a lower learning rate. The separation matters: 793 gold rows mixed into 128,195 weak rows are invisible at any loss weight, and a variant anchored with weak rows to limit forgetting scores worse than one without, because the forgetting is what redraws the decision boundary.

Per-tag thresholds are set by prevalence matching on the development set rather than by maximising F1. With around 140 development rows and single-digit positives on the rare tags, F1 maximisation selects thresholds that are systematically too high, because shedding false positives looks locally optimal and costs recall the development set is too small to detect; on this data that choice costs about 0.12 macro recall. Prevalence is the one quantity a small sample estimates reliably, so each threshold is set to make the predicted positive rate match the observed rate.

Table 4 gives per-tag performance on the 424-comment gold test set, alongside the inter-rater agreement from Section 4.4.

| Tag | P | R | F1 | n+ | Corpus % | Rater kappa |
|---|---|---|---|---|---|---|
| depression | 0.89 | 0.92 | 0.91 | 74 | 17.3 | 0.90 |
| anxiety | 0.89 | 0.90 | 0.90 | 92 | 28.4 | 0.91 |
| self_harm_suicide | 0.86 | 0.95 | 0.90 | 19 | 1.6 | 0.88 |
| loneliness_isolation | 0.86 | 0.86 | 0.86 | 21 | 1.7 | 0.81 |
| help_seeking | 0.80 | 0.89 | 0.85 | 46 | 13.2 | 0.93 |
| adhd_neurodivergence | 0.73 | 0.95 | 0.83 | 20 | 2.7 | 0.96 |
| burnout_motivation | 0.72 | 0.90 | 0.80 | 29 | 9.8 | 0.65 |
| body_image_eating | 0.80 | 0.76 | 0.78 | 58 | 13.8 | 0.95 |
| sleep | 0.84 | 0.70 | 0.76 | 23 | 2.2 | 0.92 |
| stress_pressure | 0.75 | 0.73 | 0.74 | 33 | 7.6 | 0.58 |
| exercise_dependence | 0.63 | 0.86 | 0.73 | 22 | 5.5 | 0.96 |
| exercise_coping | 0.70 | 0.73 | 0.72 | 93 | 25.5 | 0.82 |
| performance_psych | 0.68 | 0.64 | 0.66 | 33 | 9.8 | 0.81 |
| substance_use | 0.87 | 0.52 | 0.65 | 25 | 1.6 | 0.96 |
| trauma_ptsd | 0.85 | 0.50 | 0.63 | 22 | 1.2 | 0.87 |
| injury_distress | 0.62 | 0.58 | 0.60 | 26 | 4.9 | 0.87 |
| Micro (pooled, 6,784 decisions) | 0.79 | 0.79 | 0.79 |  |  | 0.88 |
| Macro (mean of 16 tags) | 0.78 | 0.77 | 0.77 |  |  | 0.86 |

*Table 4. Layer 2 per-tag performance on gold test (n = 424, cue-enriched), the share of the tagged corpus each tag covers, and the second-rater kappa from Section 4.4. Per-tag accuracy is omitted: it exceeds 0.95 on the rare tags because most decisions are true negatives, which makes it uninformative here.*

The corpus tag rates in Table 4 were never fitted to and track the independent gold prevalence estimates closely, which serves as a calibration check. Every tag runs slightly below its gold estimate, which is the expected behaviour of a precision-favouring operating point. Per-tag F1 and per-tag rater agreement also move together: the tags where two humans disagree are the tags where the model scores lowest, which suggests the remaining headroom on those constructs lies in the rubric rather than in the model.

### 5.3 End-to-end performance

Layer 1 and Layer 2 are trained and evaluated separately, but a user applies them as a cascade to raw comments, where a comment the gate drops loses all of its tags. We therefore evaluate the full pipeline on a set of 580 comments combining the gold test set with a sample of control-arm comments treated as carrying no tags. Pooled across the 12 tags in that evaluation, the cascade reaches precision 0.76, recall 0.83, and F1 0.79. Gate recall on truly relevant comments is 1.00 and its false-pass rate on irrelevant comments is 3.2%, so at this operating point the cascade loses very little to the gate and end-to-end performance is close to that of the tagger alone.

## 6. What the corpus shows over six years

This section demonstrates what the resource supports and, in one respect, how it should be used. Tag series come from the full corpus tagged with the released model (134,534 comments). Analysis scripts are named in each subsection and released with the pipeline.

### 6.1 Volume and the crude rate

![figure](results_temporal/fig1_volume_and_rate.png)

*Figure 1. Monthly volume and the crude relevance rate with 95% Wilson intervals. The control arm covers 2018-2022 only. Circles mark months whose rate deviates from a 13-month centred median by more than 2.5 robust standard deviations.*

Raw monthly counts of athlete mental-health comments move with platform volume, so rates are reported as proportions of the classified comments in each month. Expressed that way, the crude relevance rate is flat across the six years: a binomial regression of the rate on time gives +0.18% per year (95% CI -0.18 to +0.54, p = .32). The control arm is also flat (+0.33% per year, p = .74), which is the expected behaviour of a control.

### 6.2 Community composition and the within-community trend

![figure](results_temporal/fig3_composition.png)

*Figure 3. Top: the mix of communities posting each month. Bottom: the crude rate against the same corpus standardised to a fixed community mix, with linear fits. Both series are computed from identical data.*

The communities in this corpus do not hold constant shares. Between the first and last twelve months, r/xxfitness loses 11.1 percentage points of volume share and r/Fitness loses 11.5, while r/tennis gains 5.1 and r/crossfit 2.9. The effective number of communities, measured as the inverse Herfindahl index, rises from 2.6 to 6.7. Since community relevance rates range from 7.7% to 48.7%, that shift moves the pooled rate on its own.

Accounting for it changes the result.

| Model | Change per year | 95% CI | p |
|---|---|---|---|
| Crude: logit(relevant) ~ time | +0.18% | -0.18 to +0.54 | .32 |
| Adjusted: + community fixed effects | +4.36% | +3.93 to +4.78 | <.001 |
| Adjusted, participant communities | +5.36% | +4.85 to +5.87 | <.001 |
| Adjusted, spectator communities | +2.08% | +1.33 to +2.83 | <.001 |

*Table 5. Binomial GLM on (month x community) cells, 583,658 comments.*

Direct standardisation gives the same answer without a model. Holding the community mix at its pooled six-year composition and recomputing each month from each community's own rate, the series moves from 21.08% in the first year to 22.76% in the last, a slope of +0.60 percentage points per year against the crude series' -0.03. Within every community in the corpus, athlete mental-health discussion rose over the six years; pooled across communities the rate appears flat because the corpus drifted toward communities that discuss it less. This is the central methodological result of the section, and any temporal claim drawn from this dataset should carry the adjustment.

### 6.3 Seasonality and the pandemic

Month-of-year effects on top of the linear trend are strong and stable (likelihood ratio 563.3 on 11 degrees of freedom, p < .001). Relative to January, relevance odds peak in August at 1.17 and are lowest in May at 0.89.

An interrupted time series at March 2020 gives a level shift of OR 1.209 (p < .001), and it holds after the composition control at OR 1.178 (p < .001), so the pandemic change reflects how often people discussed mental health rather than which communities were posting. April 2020 is the largest single deviation in the series (robust z = 3.14). The mechanism is visible in the composition data: r/nba's share of monthly volume falls from 15.0% in January 2020 to 5.8% in April, matching the NBA's suspension of its season on 11 March, while r/xxfitness rises from 23.1% to 36.7%. That the effect survives adjustment for exactly this shift is evidence the corpus tracks real-world events.

### 6.4 Events the corpus does not detect

Two events were nominated in advance as the athlete mental-health moments of this period: Naomi Osaka's withdrawal from the French Open in May 2021 and Simone Biles's withdrawal from Olympic finals in July 2021. Neither produced a detectable change (robust z of -0.02 and +0.74 respectively).

The most likely explanation is scope. This corpus is dominated by people writing about their own training, only about 4% of it is professional-sport discussion, and the spectator communities that would respond to an elite athlete's press conference are the part of the corpus that discusses mental health least (8.68% against 37.61%). A corpus built to capture people writing about themselves need not respond to another person's news. We report it because a resource that claims temporal validity should state which events it does not detect.

### 6.5 Per-tag trends

![figure](results_temporal/fig4_headline_tags.png)

*Figure 4. The four themes carrying a reported finding, with 95% Wilson intervals and linear fits. Full 16-panel version with intervals in Appendix A.*

Table 6 gives each tag's annual change crude, adjusted for community with a 95% confidence interval, and with spectator communities excluded.

| Tag | Crude %/yr | Adjusted %/yr (95% CI) | No-spectator %/yr | Reading |
|---|---|---|---|---|
| adhd_neurodivergence | +20.93 | +24.57 (+22.04 to +27.16) | +28.03 | robust, largest |
| trauma_ptsd | +9.01 | +4.14 (+0.99 to +7.38) | +1.89 | attenuates |
| performance_psych | +8.52 | -4.02 (-5.14 to -2.88) | -1.12 | sign reverses |
| help_seeking | +6.13 | +5.82 (+4.79 to +6.86) | +1.64 | robust |
| burnout_motivation | +2.85 | +2.99 (+1.84 to +4.16) | +3.08 | robust |
| stress_pressure | +0.43 | +0.57 (-0.70 to +1.85) | +1.03 | no change |
| injury_distress | -1.11 | -0.80 (-2.34 to +0.76) | -0.77 | no change |
| anxiety | -4.29 | -6.38 (-7.07 to -5.67) | -4.98 | robust |
| exercise_coping | -4.42 | -2.21 (-3.01 to -1.40) | -1.74 | attenuates |
| substance_use | -4.48 | -3.31 (-5.83 to -0.72) | -5.19 | robust |
| sleep | -6.10 | -1.28 (-3.50 to +0.99) | -0.52 | attenuates to null |
| self_harm_suicide | -6.50 | -8.75 (-11.15 to -6.28) | -9.14 | robust |
| exercise_dependence | -7.11 | -3.61 (-5.01 to -2.18) | -4.50 | attenuates |
| loneliness_isolation | -8.85 | -8.02 (-10.38 to -5.60) | -7.28 | robust |
| body_image_eating | -11.30 | -2.62 (-3.59 to -1.64) | -3.05 | attenuates |
| depression | -13.38 | -12.65 (-13.43 to -11.87) | -11.40 | robust |

*Table 6. Per-tag annual change in the share of relevant comments carrying the tag, from binomial GLMs on (month x community) cells.*

Four results are worth drawing out. ADHD and neurodivergence discussion grows fastest of any theme, at +24.6% per year adjusted (95% CI +22.0 to +27.2), roughly tripling its share across the six years and rising further to +28.0% when spectator communities are excluded. Help-seeking rises steadily at +5.8% per year (95% CI +4.8 to +6.9). Depression falls at -12.7% per year (95% CI -13.4 to -11.9), the largest decline in the taxonomy. All three are robust to every control applied.

The fourth is methodological. Performance psychology appears to be one of the fastest-growing themes in the crude series at +8.5% per year, and declines at -4.0% per year within communities. It accounts for 29.8% of relevant comments in r/tennis and 18.4% in r/nba against 3.3% in r/xxfitness, so the corpus's drift toward tennis produces an apparent rise where the community-level trend runs the other way. Body image, sleep, trauma and exercise dependence attenuate in the same direction without reversing; sleep attenuates to a null.

Community-level prevalences serve as a further validity check. Body-image talk is concentrated in physique-oriented communities (25.1% in r/xxfitness, 25.5% in r/bodybuilding) and is near zero in spectator communities (1.4% in r/nba, 1.3% in r/tennis). Exercise-as-coping peaks in r/running at 49.2% and is lowest in r/tennis at 2.7%. Performance psychology runs the other way, peaking in the competitive-sport communities. Nothing in the pipeline was fitted to produce this pattern, so it is evidence that the tags track the constructs they name.

## 7. Release and access

The trained relevance and theme classifiers are released through HuggingFace Models, and the full pipeline through GitHub, with every number in this paper reproducible from a command-line argument against the scripts named in each section. For users without a programming background, a HuggingFace Space runs the cascade over uploaded text and returns the tag distribution with no code. The intended user there is a practitioner, for example a sports psychologist who wants to know what a population is raising.

STRIDE is built with the ISAAC pipeline and its corpus is therefore governed by the ISAAC Data Use Agreement (Hemmatian & Kurdi, 2025), which restricts redistribution of the underlying comment data and requires that downstream researchers apply for access through the same institutional channel. Corpus access is provided through that channel rather than by direct redistribution. Released alongside the models are the labelling rubrics in full, the weak-supervision rules, the train, development and test split identifiers, and the per-comment tag probabilities and binary flags.

*TODO: confirm with the ISAAC maintainers exactly which derived artefacts may be distributed (identifiers plus labels, or nothing beyond the models), fix the access URL and the citation once the ISAAC preprint is up, and decide the handling of the 2,115 comments carrying the self_harm_suicide tag. Options on the table are gated access for that subset, identifiers and labels without text, or paraphrase-level anonymisation.*

## 8. Limitations

- Population. The corpus covers recreational and amateur competitive athletes across 19 communities, not elite athletes and not Reddit as a whole. Roughly 4% concerns professional sport, so prevalence figures do not transfer to elite populations.
- Evaluation sample size. Per-tag metrics rest on a 424-comment test set, and the rarest tags on fewer than 25 positive instances each; self_harm_suicide rests on 19. The point estimates are usable and the intervals around them are wide. Future work will confirm that performance is robust on substantially larger labelled subsets.
- Per-tag performance varies with construct difficulty. The tags where the model scores lowest are the tags where two independent human raters also agree least, which suggests these are genuinely fuzzy boundaries rather than modelling failures. For subjective constructs of this kind, F1 in the 0.6 range is workable for aggregate analysis, though users should consult the per-tag figures before relying on any single theme.
- Enriched-sample metrics are not corpus metrics. Per-tag precision and recall are measured on a cue-enriched test set, which is necessary to obtain enough positives for the rare tags; natural prevalence is estimated only from the proportional sample.
- The control arm covers 2018 through 2022. Comparisons against the control are therefore limited to that window, and the 2023 figures in this paper are reported without a control reference.
- Spectator and participant communities are mixed in the headline corpus. They are separable in the released fields, and Section 6 shows they behave sufficiently differently that most analyses should separate them.
- A single annotation protocol underlies both layers. Second-rater agreement is measured (Section 4.4) but on one additional rater; a larger multi-rater study would tighten the estimates on the two low-agreement constructs.
## 9. Ethical considerations

The corpus is built from public Reddit comments. Author names are removed from the released data and comments carry their platform identifiers so that upstream deletions can be honoured. No attempt is made to identify individuals, link accounts across communities, or infer clinical status for any named person, and the tags are markers of discourse rather than diagnoses. The self_harm_suicide tag in particular flags language, not risk, and must not be used to target individuals for intervention. Access to the comment data is mediated by the ISAAC Data Use Agreement, which prohibits re-identification attempts and cross-referencing against Reddit archives or APIs.

## References

- American Psychiatric Association. (2022). Diagnostic and Statistical Manual of Mental Disorders (5th ed., text rev.). American Psychiatric Association Publishing.
- Barbieri, F., Camacho-Collados, J., Espinosa-Anke, L., & Neves, L. (2020). TweetEval: Unified benchmark and comparative evaluation for tweet classification. Findings of EMNLP.
- Baumgartner, J., Zannettou, S., Keegan, B., Squire, M., & Blackburn, J. (2020). The Pushshift Reddit dataset. Proceedings of ICWSM.
- Castaldelli-Maia, J. M., et al. (2019). Mental health symptoms and disorders in elite athletes: a systematic review on cultural influencers and barriers to athletes seeking treatment. British Journal of Sports Medicine, 53(11).
- Coppersmith, G., Dredze, M., & Harman, C. (2014). Quantifying mental health signals in Twitter. Proceedings of the ACL Workshop on Computational Linguistics and Clinical Psychology.
- De Choudhury, M., Gamon, M., Counts, S., & Horvitz, E. (2013). Predicting depression via social media. Proceedings of ICWSM.
- Derogatis, L. R. (1994). SCL-90-R: Administration, scoring and procedures manual. National Computer Systems.
- Gouttebarge, V., et al. (2019). Occurrence of mental health symptoms and disorders in current and former elite athletes: a systematic review and meta-analysis. British Journal of Sports Medicine, 53(11), 700-706.
- Gururangan, S., et al. (2020). Don't stop pretraining: Adapt language models to domains and tasks. Proceedings of ACL.
- Hemmatian, B., & Kurdi, B. (2025). The Illinois Social Attitudes Aggregate Corpus [Computer software]. GitHub. https://github.com/BabakHemmatian/Illinois_Social_Attitudes
- Liu, Y., et al. (2019). RoBERTa: A robustly optimized BERT pretraining approach. arXiv preprint.
- Low, D. M., et al. (2020). Natural language processing reveals vulnerable mental health support groups and heightened health anxiety on Reddit during COVID-19. Journal of Medical Internet Research, 22(10).
- Ratner, A., Bach, S. H., Ehrenberg, H., Fries, J., Wu, S., & Re, C. (2017). Snorkel: Rapid training data creation with weak supervision. Proceedings of the VLDB Endowment, 11(3).
- Reardon, C. L., et al. (2019). Mental health in elite athletes: International Olympic Committee consensus statement (2019). British Journal of Sports Medicine, 53(11), 667-699.
- Rice, S. M., et al. (2016). The mental health of elite athletes: a narrative systematic review. Sports Medicine, 46(9), 1333-1353.
## Appendix A. All 16 themes over time

![figure](results_temporal/fig2_tag_trends_ci.png)

*Figure A1. Share of relevant comments carrying each of the 16 themes, by month, with 95% Wilson intervals. Median interval half-width ranges from 0.51 percentage points (trauma_ptsd) to 2.10 (anxiety).*
