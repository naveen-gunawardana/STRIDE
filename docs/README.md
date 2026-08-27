# Documentation index

## `PLAN.md`
The original layered-classifier plan: Layer 1 as a relevance gate, Layer 2 as issue type, and the
two-dimension decomposition (`mh AND sport`) that the whole pipeline still rests on.

## `paper/` — the manuscript
Each `.md` is a mirror of its `.docx`, committed so that drafts can be diffed in git. Edit the
`.docx`; regenerate the `.md`.

| File | What it is |
|---|---|
| `STRIDE_JMIRMH_v5.md` | **Current draft.** JMIR Mental Health; peer discourse against screening prevalence |
| `STRIDE_paper_v4.md` | Resource-paper framing, CS-conference register; §7 is the fullest release statement |
| `AMHC_paper_JMIRMH_v3.md` | First JMIR restructure, pre-rename |
| `AMHC_paper_draft_v2.md` | First full draft |
| `AMHC_introduction_draft_v1.docx` | Introduction only |

## `methods/` — rubrics and process records
Written in the BRM style the lab uses: record the reasoning *and* the failures, so a decision can be
re-litigated later without re-running it.

| File | What it is |
|---|---|
| `layer2_tag_rubric.md` | **The annotation rubric.** All 16 themes, inclusion/exclusion criteria, anchor examples |
| `mh_rubric_broad.md` | Layer-1 mental-health rubric after the broadening to performance psychology |
| `relevance_layer1_methods.md` | Layer-1 methods as written for the paper |
| `relevance_layer1_process.md` | How Layer 1 was actually built, including the domain-mismatch diagnosis |
| `layer2_tags_process_2026-07-25.md` | How the 16-theme tagger was built: weak supervision, abstention, thresholds |
| `temporal_analysis_2026-08-14.md` | The composition-adjustment problem and how the trend models handle it |
| `literature_comparison_draft.md` | The prevalence comparison that became the paper's spine |

## `reports/` — meetings and progress
| File | What it is |
|---|---|
| `revision_plan_2026-08-18.md` | Response to advisor review; the release blockers live in §4 |
| `paper_direction_and_advisor_notes.md` | What the paper is meant to be; §3 sets out the four release channels |
| `meeting_report_2026-08-05_layer2_expansion.md` | Layer-2 expansion to 16 themes |
| `meeting_report_2026-07-26.md` | Layer-2 first results |
| `OVERNIGHT_REPORT.md` | Layer-1 completion report |

## `isaac/` — inherited documentation
STRIDE is built with filtering and sampling components from the
[ISAAC](https://github.com/BabakHemmatian/Illinois_Social_Attitudes) pipeline. These documents
describe that upstream pipeline, not STRIDE, and are kept because the shared code paths and the
column definitions still apply.

| File | What it is |
|---|---|
| `variable_list.md` | Column definitions for ISAAC corpus rows, inherited by STRIDE's base columns |
| `ISAAC_intro_working_notes.md` | The sibling resource paper, used as a structural template |
| `label_location_internals.md` | Operational detail for the `label_location` resource |

Figures in this folder (`Generalization_Labels.jpg`, `ISAAC_Logo_1.png`,
`line_overlay_distinctions.png`, `freq_aggregate.png`) belong to those documents.
