# Paper Direction & Advisor Notes

**Captured:** 2026-08-10 · **Source:** accumulated meeting notes with Babak (advisor), spanning the
Layer-1 build through the current paper-planning phase.

This file is the **intent** record — where the project is going, what venue it targets, and what
Babak asked for. It complements the **execution** records:

- `docs/relevance_layer1_process.md` / `docs/relevance_layer1_methods.md` — Layer 1
- `docs/layer2_tags_process_2026-07-25.md` / `docs/layer2_tag_rubric.md` — Layer 2
- `experiments/trials_log.md` — one row per run
- `ISAAC_intro_working_notes.md` — the sibling BRM resource-paper template from the same lab

Notes are recorded close to how they were said. Where a directive has already been implemented,
status is marked `[DONE]`, `[PARTIAL]`, or `[OPEN]` against what is actually in the repo. Those
status marks are Naveen/Claude's assessment, not Babak's words.

---

## 1. The gap this work fills

Mental health through the lens of social media discussion — Reddit as the example platform.
Two things social text offers that clinical data does not:

- **Diagnosis through statements** — people describe symptoms in their own words.
- **A better clinical picture from self-description** — peer-to-peer talk is less filtered than
  what gets said to a health professional.

**The literature gap is asymmetric, and that asymmetry is the opening:**

| Field | Has | Lacks |
|---|---|---|
| Sports psychology / clinical | Construct depth, DSM-grounded taxonomies, theory | Technical depth; scale; NLP method |
| Social-media mental-health NLP | Technical depth, scale, modeling | Mental-health theoretical depth — treats "anxiety" as one flat term |

> "Anxiety is a term. Types of anxiety go much deeper. Mental health disorders — there's a family
> of disorders with subtypes."

**The contribution is connecting the dots**: bring DSM-level construct granularity to a
computational social-science corpus.

### The sharp version of the research question

> **Can you distinguish the subtypes of anxiety disorders, in a sports context, from people's
> conversations?**

**Applied angle:** these laypeople are the same population who would walk into a sports
psychologist's office. That is what makes the corpus actionable rather than academic.

`[OPEN]` — the current Layer-2 taxonomy has a single flat `anxiety` tag. Subtype granularity
(social anxiety vs. GAD vs. performance-specific vs. health anxiety) is not yet in the rubric.
This is the single largest unclaimed novelty in the project.

---

## 2. Venue strategy

**Aim high, then downgrade based on results and how fast progress comes.**

### Target types

- **Conferences over journals.** In CS, conferences matter more; review cycles are much faster.
  (In psychology it is the reverse — journals matter more.)
- **Workshop track at a major conference could be a big boost.** What a workshop track wants:
  - a novel perspective on the problem,
  - research that offers expanse,
  - really complete results.

### Named venues

| Venue | Notes |
|---|---|
| **ICWSM** — International Conference on Web and Social Media | Computational social science work. Example: https://ojs.aaai.org/index.php/ICWSM/article/view/31413 |
| **ACL** — Association for Computational Linguistics | Year-round submission |
| **EMNLP** — Empirical Methods in Natural Language Processing | |
| **NeurIPS** *(noted as "Neiriss")* | **Workshop track** specifically |
| **ACM** | Example: https://dl.acm.org/doi/pdf/10.1145/3269206.3271732 |
| *"Chronolog of informatics research"* `[?]` | Recorded verbatim — exact venue name needs confirming with Babak |

Preprint to **arXiv** (https://arxiv.org/).

**Direction: focus the application toward a conference, not a journal.**

> Note: this is a shift from the earlier BRM (*Behavior Research Methods*) framing carried over
> from ISAAC. The BRM template in `ISAAC_intro_working_notes.md` is still the right *structural*
> model for a resource paper; the venue target has moved.

---

## 3. What the paper is: a resource paper

The shape, in Babak's ordering:

1. **A novel dataset** — this is the core resource.
2. **Analysis that showcases why the dataset matters** — ask and answer genuinely interesting
   questions with it.
3. **Models trained on it**, shared so others can reuse.
4. **Tools so other people's lives are easier** — including non-technical users.

> "The key information is the resource that you're providing (dataset); the analysis that you do is
> showcasing why this dataset [matters]; and then following that is sharing resources so other
> people's lives will be easier — ask and answer really interesting questions with that, and also
> you can train your own models."

**There probably are no good existing datasets for this.** That is the argument for building one.

### Access and distribution `[OPEN]`

- **Open-source GitHub** with paper results reproducible via CLI arguments
- **HuggingFace Datasets** — the corpus
- **HuggingFace Models** — pretrained weights, easily available
- **HuggingFace Spaces** — coding-free web app for analysis
- Data loader + model loader ability

**Coding-free web app** is the accessibility play — the user could be a sports psychologist with
no programming background. (Same argument ISAAC makes with its website; see
`ISAAC_intro_working_notes.md` §4, feature #1.)

### The classifier as a product, not just a method

- A classifier **for users of the website**, or applied to **content on the website**
- **General purpose** — multiple classifiers, chained
- First: *is this mental health?* → then analyze what proportion of people → then further
  classification

`[DONE]` — this cascade is exactly the built Layer-1 → Layer-2 architecture.

---

## 4. Analysis angles Babak wants in the paper

### Descriptive

> "Providing an analysis of people-related social-media mental health. This percentage on
> athlete-related subreddits, people are mentioning mental health, and this is the distribution of
> different types of mentions, sentiment, emotions."

So: prevalence + tag distribution + sentiment + emotion, on athlete-related subreddits.
`[PARTIAL]` — prevalence and tag distribution exist (`final_dataset_tagged.csv`); sentiment and
emotion labelers are not yet run on this corpus (the ISAAC pipeline has them).

### Correlational — the interesting-questions layer

This is where the paper earns a conference slot. Look for **cross-domain correlations**:

- Relationship of mental health to **performance** — e.g. an inverse relationship
- **Confidence issues in basketball as indicators of general anxiety** (a sports-specific symptom
  pointing at a DSM-general construct)
- How people's challenges *within* sport correlate to specific issues *outside* sport
- e.g. do people with confidence issues show more **social isolation**, or a specific
  **family structure**?
- Confidence → diet, grades, etc.

**Ground it in DSM.** Try to make each claim more specific than "athletes have anxiety."

**Report negative results too.** Find results that are positive, and also report ~3 that are not.
`[DONE as practice]` — the trials log and both process docs already record negative results
systematically; this is an established strength to carry into the paper.

---

## 5. Corpus scope — the subreddit tier list

Babak's guidance: **look at subreddits that are basketball/sport-specific and mental-health-sport
specific**, otherwise it is irrelevant to the project.

> **Important framing note from the notes:** "The data is not about sports, it's about like gym."

### Tier 1 — Seed the gold set here (dense, on-topic, low noise)
- r/sportspsychology
- r/MentalCoaching
- r/getdisciplined (sport-adjacent subset)

### Tier 2 — Individual-sport training subs (the bulk of the data)
People post about their own slumps, nerves, confidence, burnout here.
- **Endurance:** r/running, r/AdvancedRunning, r/triathlon, r/Swimming, r/cycling, r/Rowing, r/XCRunning
- **Strength:** r/powerlifting, r/weightlifting, r/bodybuilding, r/crossfit
- **Climbing:** r/climbharder, r/bouldering, r/climbing
- **Combat:** r/bjj, r/MMA, r/boxing, r/MuayThai, r/MartialArts
- **Skill/aesthetic:** r/Gymnastics, r/figureskating, r/tennis, r/golf, r/trackandfield

### Tier 3 — Team-sport *player* subs (mixed; filter hard)
Usable but fan/spectator content must be stripped.
- r/BasketballTips (player-focused — better than r/Basketball), r/bball
- r/Volleyball, r/hockeyplayers, r/footballcoaching

### ⚠️ Avoid for athlete mental health
r/nba, r/nfl, r/soccer, r/baseball, r/hockey — almost entirely fans and match threads.
**This is the "irrelevant content" flagged earlier.**

### Tier 4 — Injury / identity-loss
Concentrated injury-distress and "who am I without the sport" language.
- r/ACL, r/runninginjuries, r/injuries, r/Sprained, r/PhysicalTherapy (patient posts)

### Tier 5 — General mental-health subs (comparison / out-of-domain baseline)
Not the target population, but exactly the baseline / "mashed" contrast samples.
Use for contrast, transfer checks, and negative examples.
- r/anxiety, r/depression, r/socialanxiety, r/mentalhealth

### Status against the built corpus `[OPEN — material decision]`

The current corpus was **not** drawn from this tier list. Actual composition: 74% of relevant
comments come from r/xxfitness, r/running, r/Fitness, r/bodybuilding; only ~4% is pro-sport
discussion. `code/classify_corpus.py` **drops** the Tier-5 mental-health subreddits entirely
(`DROP_SUBS`), on the reasoning that being in r/depression does not make a comment about sport.

Three consequences worth raising with Babak:

1. The tier list is **largely consistent** with what was built (Tier 2 dominates in both), so this
   is a re-scoping and top-up, not a rebuild.
2. Tier 5 is currently *deleted*, but Babak wants it as a **contrast/baseline set**. The existing
   `baseline` arm (956k non-keyword sports comments) is a *different* kind of control. These are
   two distinct baselines and the paper should not conflate them.
3. Tier 1 and Tier 4 are thin or absent in the current draw. Tier 4 would directly raise the
   prevalence of `injury_distress` (currently F1 0.66) and `identity_retirement` (dropped at 0.1%
   prevalence for being unevaluable). A Tier-4 top-up could revive both.

---

## 6. Label / taxonomy design

**Action item as given:** think of a set of labels; use keyword sets as examples; around **4–5**
labels, e.g. anxiety, depression — whichever is **most helpful for Mentality Sports**.

- **Definitely one label is "unrelated."**
- Look at **50 from each** of the baseline and matched samples, and come up with a classification
  scheme that works best for Mentality Sports.
- Different categories of relevance are possible — Athletes × MH, Fans × MH, etc. Keeping them
  separate gives the option to change the pipeline later. **Focus: Athletes × Mental Health.**

`[DONE]` — the two-dimension `mh` × `sport` decomposition implements exactly this, and Layer 2 now
carries 16 tags (well past 4–5). The `[OPEN]` piece is DSM **subtype** granularity (§1).

---

## 7. Modeling directives (mostly historical — Layer 1 era)

Recorded for provenance; most are implemented. See `experiments/trials_log.md` for the runs.

**Relevance and filtering**
- Comments may flag sports *and* MH but not be relevant *to each other* `[DONE — the AND gate]`
- Add sports-related word filtering `[DONE]`
- The relevance target must be about athletes **and** mental health `[DONE]`
- Ask it in **two separate prompts** — "is this related to mental health?" and "is this related to
  athletes?" — then aggregate `[DONE — this is the κ 0.38 → 0.81/0.86 fix]`
- Use Claude to generate ~1500 labels; try to do some by hand `[DONE — 1199, then expanded to 2516]`

**Class balance and weighting**
- Train on 65/35 — no need to force 90/10 `[DONE]`
- Weight to push the classifier to learn "irrelevant" better: since ⅔ is relevant and ⅓ irrelevant,
  weight the ⅓ more, so getting one from the minority wrong hurts more `[DONE — balanced class weights]`
- Weighting accounts for overfitting and for some data being more common

**Thresholds**
- **Turn the threshold off first** (argmax), report performance for both models, and only add a
  threshold if precision is too low `[DONE]`
- Higher threshold → precision up, recall down. **Precision is better to have the imbalance toward.**
- Model was too liberal: rarely misses a positive, but many false positives → make it more
  conservative; try 0.55 / 0.6 `[DONE — final operating point mh ≥ 0.50, sport ≥ 0.40]`
- **Grid search** the two models' thresholds jointly for the best combined gate; different rates for
  the two models `[DONE — code/eval_holdout_grid.py]`
- **Error analysis on the AND gate** `[DONE]`

**Training mechanics**
- Use the validation set to decide when to stop. If it is not stopping, train more epochs.
- Make it unlimited epochs until F1 drops twice — **early stopping**, per-epoch, patience 2 `[DONE]`
- Try different **batch sizes** — halve or double
- **Get a GPU.** Install torch with CUDA support in the venv; `torch.cuda.is_available()` to check.
  conda is easier. `[DONE — RTX 2060, ~70× speedup]`

**Splits and metrics**
- **80 / 10 / 10** train / validation / test `[DONE]`
- Report **precision (more important), recall, F1**
- Inter-rater agreement = the kappa you calculated
- **Final gate goal: 0.8 on all of them** `[DONE — Layer 1 at P 0.96 / R 0.83 / F1 0.89]`
- Target ~2500 total labeled `[DONE — 2516]`
- Read F1 **against the base rate**, not against zero

**Caveat to state in the paper**
- Comparing psychiatrist data with online anonymous data is different — the populations and the
  disclosure conditions are not the same.

---

## 8. Temporal analysis

- **Frequency tables**, visualized over time
- X axis = time in months across years; Y axis = number of documents
- **Look at the spikes** — particular time periods where an event produced a spike in the data
- **Use proportions rather than raw frequencies** — Reddit grows over time
- **Always compare against a baseline**: total documents you have, vs. the overall volume of
  discussion. Without that, growth in the platform looks like growth in the signal.
- **Regression models** on the resulting time series
- The sample must have good **95% confidence intervals** for temporal analysis — this is *why* the
  annotation sample was month-stratified

`[PARTIAL]` — month-stratified sampling is done; the temporal analysis itself is not yet run on the
tagged dataset. `counts_curated_monthly.csv` and `freq_aggregate.png` exist from earlier work.

---

## 9. Writing the paper

**Immediate focus: write the introduction.** One page to a page and a half. Draft in Docs, convert
to Overleaf. **Look at example papers. Cite things.**

**Name the corpus.** "We present the XXXX corpus…"

### Introduction — the argument, close to as dictated

1. **State the problem.** Mental health issues in athletes matter. While there is some data on
   rates of these issues, those rates may be distorted by the difficulty of sharing struggles in a
   clinical setting or with health professionals.

2. **The move.** One way to cross-validate the rate of different concerns is by looking at
   peer-to-peer conversations, for instance on social media. Advances in natural language
   processing allow such evaluations to be performed on natural conversations at scale.

3. **The contribution.** "We present the XXXX corpus to support the text-based identification of
   mental health concerns among athletes." Then, briefly:
   - describe the **strengths** of the dataset
   - **note limitations**
   - the **range of people who could benefit** from it
   - **highlight the ways you make the dataset conveniently accessible to them**

### Methods — what you actually did

1. **Source data**, then **how you filtered it**: first by keywords, then by multiple trained
   classifiers.
2. **Justify the classifiers you chose.** The justification is: you wanted the ability of modern
   LLMs, but also **scalability**.
3. **Descriptive statistics of the finished dataset** — number of documents, the subreddits it
   focuses on, number of documents over time.
4. **Describe the labeling** — mental-health-concern tagging.
5. **Performance table.** Overall, per layer, and per tag.

`[READY]` — every one of these five has a source already written:
source/filtering and the LLM-vs-scalability justification are in
`docs/relevance_layer1_process.md`; descriptive stats are in `docs/layer2_tags_process_2026-07-25.md`
§1 and §7; labeling is `docs/layer2_tag_rubric.md`; the per-tag performance table is in
`docs/meeting_report_2026-08-05_layer2_expansion.md` §3b.

### Metric definitions to include

- **Precision** — of everything the model predicted positive, how many actually were positive.
  `TP / (TP + FP)`. Answers *"when the model says yes, how often is it right?"*
- **Recall** — of everything that actually was positive, how many the model caught.
  `TP / (TP + FN)`. Answers *"how many of the real positives did we find?"*
- **F1** — harmonic mean of precision and recall: `2 × (P × R) / (P + R)`. High only when both are
  high, so it punishes models that are great at one and bad at the other.
- **Intuition:** a spam filter with high precision rarely flags real email as spam; one with high
  recall rarely lets spam through. F1 balances the two.

---

## 10. Literature review

- **Read and review articles.** In CS, review articles usually get published in journals.
- **Look at two different types of review article:**
  - *Sports psychology* — usually mental/construct depth, but not technical depth
  - *Mental health and social media* — usually deserves technical depth, not mental/theoretical depth
- **Search for sports-psych articles and mental-health-in-social-media studies**; send them to Babak.
- Use the review to **see where the gaps are**, then connect the dots (§1).

---

## 11. Process

- Babak will **share a weekly plan**.
- Babak will **share the training script**. `[DONE — this became code/train_relevance.py]`
- **Read the codebase, generate ideas.**
- **Share an initial outline with him — multiple ideas**, not one.
- Go over the training script and take notes.
- **Run different trials; keep records; allow tweaks to be recorded and edited. Good version
  control. Take notes on everything.** `[DONE — experiments/trials_log.md + the process docs]`
- Send him performance numbers.

---

## 12. Open items and decisions needed

| # | Item | Why it matters |
|---|---|---|
| 1 | **Confirm the venue.** Conference (ICWSM / ACL / EMNLP / NeurIPS workshop) vs. the earlier BRM journal framing. | Changes length, structure, and deadline. Conference framing is the stated direction. |
| 2 | **Anxiety/disorder subtype granularity.** | The stated novelty (§1) — currently not in the taxonomy. |
| 3 | **Re-scope the corpus to the tier list?** Especially Tier 1, Tier 4, and re-admitting Tier 5 as a contrast set rather than deleting it. | Material change to the dataset; affects `injury_distress` and `identity_retirement`. |
| 4 | **Name the corpus.** | Needed for the intro's first sentence. |
| 5 | **Sentiment + emotion labels** on this corpus. | Babak explicitly listed them in the descriptive analysis. Not yet run. |
| 6 | **Temporal analysis** (proportions, baseline-corrected, with CIs) and regression. | Explicitly requested; the sampling was designed for it. |
| 7 | **Second-rater κ for Layer 2.** | Blocks calling the Layer-2 labels "gold" in print. Layer 1 has it (κ 0.81/0.86); Layer 2 has none. |
| 8 | **Confirm the "Chronolog of informatics research" venue name.** | Recorded verbatim; may be a mis-transcription. |
| 9 | **Three truncated pastes** in the source notes (`#2 +67 lines`, `#5 +65 lines`, `#8 +32 lines`) were not captured in full. | May contain directives not reflected here. Worth re-pasting. |

---

## 13. Reference links from the notes

- ICWSM example paper — https://ojs.aaai.org/index.php/ICWSM/article/view/31413
- ACM example paper — https://dl.acm.org/doi/pdf/10.1145/3269206.3271732
- arXiv — https://arxiv.org/
