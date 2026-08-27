# STRIDE

**S**port and **T**raining **R**eddit corpus for **I**dentifying **D**istress and **E**motion — a
corpus of 129,834 athlete mental health comments, the two-stage classifier cascade that built it,
and the analysis pipeline behind the paper.

> *What Athletes Tell Each Other Versus What Screening Instruments Find*
> Naveen Gunawardana, Babak Hemmatian — in preparation, JMIR Mental Health

Prevalence estimates for athlete mental health come almost entirely from screening instruments and
surveys, all of which require the athlete to disclose. STRIDE provides an unprompted comparison:
what athletes say to each other, unasked, on public forums. Comparing the two shows that peer
discourse and clinical screening disagree systematically about which problems are prominent.

## Headline results

| | |
|---|---|
| Comments read | 1,625,377 across 19 sport/training/fitness subreddits, 2018–2023 |
| Relevant (matched arm) | **129,834** (22.24%) against 0.53% in the control arm — 42× separation |
| Themes | 16, multi-label, mean 1.51 per comment |
| Layer 1 (relevance gate) | P 0.96 / R 0.83 / **F1 0.89**; second-rater κ 0.819 |
| Layer 2 (theme tagger) | micro **F1 0.79**, macro 0.77; pooled second-rater κ 0.879 |
| Full cascade, end to end | P 0.76 / R 0.83 / F1 0.79 on 580 raw comments |

Sleep (26.4% in the reference meta-analysis) and alcohol misuse (18.8%) appear in 2.2% and 1.7% of
relevant comments; after a stratified recall audit both rise to ~3.5%, leaving gaps of 7.5× and
5.4×. Exercise used as a coping strategy appears in 25.5% of comments and has no counterpart in the
prevalence literature at all.

## The pipeline

Two layers. Layer 1 is a gate, so its errors cascade — that is why both precision and recall are
reported at every stage.

```
raw monthly Reddit archives
  └─ keyword filter (disorder-anchored taxonomy, built for recall)
       ├─ matched arm   2018–2023
       └─ control arm   2018–2022   (same communities, no MH keyword)
           │
  Layer 1  ├─ p_mh    ≥ 0.50   twitter-roberta + domain-adaptive pretraining
           └─ p_sport ≥ 0.40   ── relevant = mh AND sport
           │
  Layer 2  └─ 16 sigmoid heads, one shared encoder
              weak supervision with abstention (128,195 rule-labelled rows)
              → fine-tuned on 1,358 hand-labelled comments
              → per-theme thresholds set by prevalence matching
           │
           └─ stride_release.csv — 129,834 comments × (16 probabilities + 16 binary flags)
```

**The 16 themes.** Condition: `depression`, `anxiety`, `stress_pressure`, `burnout_motivation`,
`performance_psych`, `body_image_eating`, `injury_distress`, `self_harm_suicide`, `sleep`,
`substance_use`, `trauma_ptsd`, `adhd_neurodivergence`, `loneliness_isolation`,
`exercise_dependence`. Response: `help_seeking`, `exercise_coping`.

All 16 are released rather than filtered to a quality bar — a single global F1 cutoff would decide
on the user's behalf which constructs are usable. Per-theme precision, recall, F1 and inter-rater κ
are published so users can make that call themselves, and every row keeps its per-theme probability
so thresholds can be reset rather than inherited.

**The 19 communities.** `xxfitness` `Fitness` `bodybuilding` `weightroom` `crossfit` (strength) ·
`running` `triathlon` `Swimming` `Rowing` `trackandfield` (endurance) · `nba` `CollegeBasketball`
`Basketball` `BasketballTips` `volleyball` (team ball) · `tennis` `climbing` `Gymnastics`
`sportspsychology` (individual skill).

## Repository layout

```
code/                   the full pipeline — filtering, training, evaluation, analysis
scripts/                pipeline runners (run from the repo root) and Windows launchers
data/
  data_relevance_ratings/     Layer-1 annotation sets and the 292-comment held-out set
  data_relevance_QAratings/   post-filter QA samples behind the false-positive-rate claims
  data_layer2_ratings/        Layer-2 hand-labelled gold (6,637 rows across seven draws)
  layer2_audit/               stratified recall audit of sleep and substance use
docs/
  PLAN.md                     the original layered-classifier plan
  paper/                      manuscript lineage, v2 → v5, each .md mirroring its .docx
  methods/                    annotation rubrics and per-stage process records
  reports/                    meeting reports, advisor notes, revision plan
  isaac/                      inherited ISAAC documentation and its figures
experiments/            trials_log.md — every training run, including the negative results
keywords/               keyword lists inherited from the ISAAC pipeline
results_temporal/       month × community aggregate counts and every paper figure
run_logs/               stdout from each pipeline run
isaac-data-loader/      terms-gated loader for the upstream ISAAC corpus
```

`docs/methods/layer2_tag_rubric.md` is the annotation rubric in full — inclusion and exclusion criteria and
anchor examples for all 16 themes. `experiments/trials_log.md` records every trial including the
ones that failed, which is where the domain-mismatch diagnosis and the four abandoned levers live.

## Reproducing the paper

Requires Python 3.11 and a CUDA GPU for the training and tagging steps.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

The runners in `scripts/` hardcode the Windows interpreter path `.venv/Scripts/python.exe`;
adjust `PY=` at the top of each on POSIX. Run them from the repository root — the paths
inside them are root-relative.

| Step | Command |
|---|---|
| Keyword filter | `python code/filter_keywords.py -g mental_health` (see note below) |
| Train Layer 1 | `python code/train_relevance.py -r train_relevance -t comments -g mental_health` |
| Evaluate the gate | `python code/eval_holdout_gate.py` |
| Classify the corpus | `python code/driver_classify.py` |
| Assemble the arms | `python code/build_final_dataset.py` |
| Build the silver set | `python code/layer2_silver.py` |
| Train Layer 2 | `bash scripts/train_layer2_v3b.sh` |
| Evaluate Layer 2 | `python code/eval_layer2.py --model models/layer2_tags_v3b --gold data/layer2/gold_test.csv --dev data/layer2/gold_dev.csv` |
| End-to-end cascade | `python code/eval_joint.py` |
| Tag the release corpus | `python code/tag_corpus_final.py` |
| Temporal analysis | `python code/temporal_analysis.py` |
| Composition adjustment | `python code/rate_adjusted.py` |
| Vocabulary baseline | `python code/vocab_baseline.py` |
| Recall audit | `python code/audit_score.py` |
| Inter-rater agreement | `python code/metrics_interrater_l1.py` · `metrics_interrater_l2.py` |

`code/demo_gate.py` runs the whole cascade over typed text in a browser on `localhost:8000`,
stdlib only.

**Missing input: the keyword taxonomy.** `code/filter_keywords.py` reads its term lists from
`keywords/<group>_<value>.txt`. The lists committed here are the ISAAC social-group sets
(ability, age, race, sexuality, skin tone, weight); the STRIDE mental-health and sport lists
(`mental_health_*.txt`, `sport_*.txt`) are **not in the repository**, so the first pipeline step
cannot currently be re-run from a clean clone. They need to be committed before release — the
paper describes the taxonomy as disorder-anchored, following the IOC consensus statement and
DSM-5-TR symptom vocabulary translated into lay phrasing.

**Shipped model: `models/layer2_tags_v3b`.** A v6 reproduction run from committed code confirms it
(micro 0.78 against v3b's 0.79, within run-to-run noise) rather than replacing it; the corpus
tagging and the entire temporal analysis are built on v3b. See `experiments/trials_log.md`.

## What is and is not in this repository

Committed: all code, all hand-labelled rating sets, the aggregate month × community counts behind
every figure, and the full document trail.

Not committed (`.gitignore`): trained model weights (`models/`), the raw and classified comment
corpora (`comments/`, `data/classified/`), the Layer-2 gold splits (`data/layer2/`), and ~20 GB of
local run artifacts (`results/`). De-anonymisation key files mapping blinded IDs back to authors are
excluded permanently and must never be committed.

## Release and access

Planned distribution: the corpus through HuggingFace Datasets, the classifiers through HuggingFace
Models, and a coding-free HuggingFace Space that runs the cascade over uploaded text for
practitioners who do not write code. **URLs are not yet live.**

Two questions are open and both are blocking:

1. **ISAAC Data Use Agreement.** STRIDE is built with the ISAAC pipeline and inherits its DUA, which
   restricts redistribution of raw comment text to third parties. The release may therefore carry
   platform identifiers plus derived labels rather than text, with text access routed through the
   ISAAC institutional channel. To be confirmed with the corpus owners.
2. **Sensitive-content controls.** 2,115 comments carry `self_harm_suicide`. Gated access for that
   subset, identifiers-plus-labels in place of text, or paraphrase-level anonymisation — to be
   decided alongside (1).

## Ethics

The corpus is built from public Reddit comments. Author identifiers are removed and comments carry
their platform identifiers so upstream deletions can be honoured. No attempt is made to identify
individuals, link accounts across communities, or infer clinical status for any named person.

**The theme labels are markers of discourse, not diagnoses.** The `self_harm_suicide` theme flags
language rather than risk and must not be used to target individuals for intervention. Re-identification
and cross-referencing against Reddit archives or APIs are prohibited under the ISAAC DUA.

## Citation

```bibtex
@misc{gunawardana_stride,
  author = {Gunawardana, Naveen and Hemmatian, Babak},
  title  = {STRIDE: Sport and Training Reddit corpus for Identifying Distress and Emotion},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```

STRIDE is built with filtering and sampling components from the
[Illinois Social Attitudes Aggregate Corpus (ISAAC)](https://github.com/BabakHemmatian/Illinois_Social_Attitudes)
pipeline (Hemmatian & Kurdi, 2025).

## Contact

Naveen Gunawardana · Babak Hemmatian, Ph.D. — [babak.hemmatian@gmail.com](mailto:babak.hemmatian@gmail.com)

## License

TBA — pending the ISAAC Data Use Agreement review described above.
