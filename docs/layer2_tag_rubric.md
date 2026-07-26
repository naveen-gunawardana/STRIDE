# Layer-2 tag labeling rubric (v1, 2026-07-25)

**Input:** a Reddit comment that **Layer 1 already marked relevant** (athlete × mental health).
You are no longer deciding *whether* it is athlete mental health — assume it is. You are deciding
**which mental-health themes it contains**.

**Multi-label:** a comment can carry **zero, one, or several** tags. Tags are independent — do not
force a choice between them. (Zero is legitimate: Layer 1 has ~4% residual false positives, and some
true positives are generic "mental health is important" talk with no specific theme.)

## DECIDED RULES (read these first)

1. **Judge only the text shown.** No subreddit, no author, no thread context.
2. **The theme may belong to anyone** — the author, a teammate, a pro athlete, a partner, a
   generic "people". Third-party and general-discussion mentions count. *"Osaka clearly has anxiety
   or depression"* → `anxiety` + `depression`.
3. **Passing mentions count**, exactly as in Layer 1. *"...struggle with focus (thanks adhd)"* counts
   for its theme even in a clause.
4. **Lean YES when genuinely uncertain** — consistent with the broadened Layer-1 rubric. But
   "uncertain" means *the theme is plausibly present*, not *a keyword is present*.
5. **Keyword ≠ theme.** The single biggest error mode. Every tag below has a *required construct*;
   the word alone never earns the tag. See the ✗ lines.
6. **Negation and absence flip to NO.** *"I don't get anxious before races"* → **no** `anxiety`.
   *"I never had an eating disorder"* → **no** `body_image_eating`. (Layer 1 was hardened this way;
   Layer 2 inherits the rule.)
7. **Figurative / joke usage is NO.** *"ptsd from playing on double rims"*, *"this offense is
   depressing"*, *"the fanbase is clinically depressed"* → no tag.

---

## Condition tags — what the person is experiencing

### `depression`
Low mood as a **state or condition**: depression (clinical or colloquial), hopelessness,
worthlessness, anhedonia / loss of enjoyment, persistent sadness, "in a dark place", grief that is
weighing on the person, diagnosed depression, being on antidepressants **for** depression.
- ✓ *"running is my main protection against the constant threat of depression"*
- ✓ *"i started feeling down and like i'm choosing to bust my ass and feel bad instead of being happy"*
- ✗ "depressing" about a *game/team/economy* — figurative → no tag.
- ✗ Momentary disappointment that resolves (*"gutted we lost"*).

### `anxiety`
Anxiety, panic, worry, fear, nerves, dread, being on edge, social anxiety, health anxiety,
gym-timidity ("gymtimidation"), panic attacks — **not tied to competitive performance**
(that is `performance_psych`).
- ✓ *"i just took a break from the gym because my anxiety was really bad"*
- ✓ *"open water swimming can cause anxiety/panic for some"*
- ✓ *"anxious about grunting in a public gym"* (social anxiety)
- ✗ "anxious to get started" = eager, not anxious → no tag.
- ✗ Negated: *"i don't get anxious"*.

### `stress_pressure`
Stress, overwhelm, being under pressure, or expectations/demands weighing on the person — from
life, work, school, family, coaches, fans, or the sport itself. Includes stress **as a load being
managed**.
- ✓ *"stress got to me, i was very anxious"* (also `anxiety`)
- ✓ *"press conferences can be a huge burden and extremely exhausting"*
- ✗ **Physiological/training stress** — *"box squats put more stress on my back"*, "stress
  fracture", "stressing the muscle" → no tag. This is the dominant false positive in a fitness
  corpus.

### `burnout_motivation`
**Psychological** burnout, mental exhaustion, loss of motivation/drive, dreading training, going
through the motions, overtraining framed as a mental/emotional problem, needing a break to recover
mentally.
- ✓ *"i'm kinda mentally burnt out"*, *"struggling with motivation"*, *"dreading going to the gym"*
- ✗ **Physical fatigue** — *"i went to failure and burnout for the rest of the day"*, *"leaves me
  really tired after gym sessions"*, "muscle burnout", DOMS, being sore → no tag. Burnout in a
  fitness corpus is usually muscular; require a mental/motivational construct.

### `performance_psych`
Sports/performance psychology, **in scope for `mh` since the v2 Layer-1 rubric**: performance
anxiety, competition nerves, choking, the yips, mental blocks, confidence / self-doubt / self-esteem
**tied to performing**, fear of failure, overthinking technique, feeling watched or judged while
competing or training, pressure from parents/coaches/crowd affecting play.
- ✓ *"self doubt is starting to seep in"* (before a race)
- ✓ *"i don't play well when my parents are at the game because i tense up"* (Layer-1 anchor case)
- ✓ *"she says it's the yips"*, *"enough to execute and not choke"*
- ✗ "choke" about a *team losing* with no psychological framing → no tag.
- ✗ Pure technique talk (*"my elbow drops on the jumper"*).

### `body_image_eating`
Disordered eating (anorexia, bulimia, bingeing, purging, restriction, orthorexia, "ed"), body-image
distress, hating one's body/appearance, weight or food that has become a **source of psychological
distress or compulsion**.
- ✓ *"we would starve ourselves and make ourselves throw up"*
- ✓ *"my body image got too out of whack"*, *"i used to suffer from anorexia"*
- ✓ *"obsessively scanning everything i put in my mouth"* (compulsion)
- ✗ **Ordinary diet/nutrition talk** — calories, macros, cutting, bulking, "trying to lose 5 lbs"
  with no distress → no tag. This is the dominant false positive in a fitness corpus.
- ✗ "binge **drinking**" → that is `substance`-adjacent, not eating (and `substance` is not a v1 tag).

### `injury_distress`
Injury, illness, or being sidelined causing a **psychological** consequence: depression/frustration
after injury, fear of re-injury, identity loss from not training, distress about rehab timelines.
**Requires injury AND distress.**
- ✓ *"don't let a knee injury derail your life and get you depressed"*
- ✓ *"i had to take 6 months off for injuries and i was miserable"*
- ✓ *"i'm terrified of injury"* (fear of injury)
- ✗ Pure injury/rehab mechanics — *"tore my acl, 9 months of rehab, here's my protocol"* with no
  emotional content → no tag.

### `self_harm_suicide`
Suicidal ideation, attempts, or death by suicide; self-harm/cutting; "wanted to end it";
"saved my life" said about a real suicidal period. Includes third-party and news/pro-athlete cases.
- ✓ *"i was severely depressed to the point where i'd have suicidal thoughts everyday"*
- ✓ *"she didn't just die, she committed suicide"*
- ✓ *"my boyfriend told me he had been feeling suicidal"*
- ✗ Hyperbole — *"i'm gonna kill myself if we lose again"*, *"this workout is killing me"* → no tag.

---

## Response tags — what is being done about it

### `help_seeking`
Therapy, counselling, psychiatry, psychology, medication (SSRIs, antidepressants), diagnosis,
hospitalisation, hotlines, or **advice to seek** any of these; also reaching out to people for
mental-health support.
- ✓ *"get some therapy my man"*, *"i conquered this with my therapist during cbt"*
- ✓ *"i have been prescribed ssris in the past"*
- ✗ **Physical** therapy / physio / chiropractor / doctor for an injury → no tag.
  (*"went to a lot of therapy, both physical and psychological"* → ✓, because psychological is named.)

### `exercise_coping`
Sport, training, or exercise used **to manage mental health** — as an outlet, escape, release,
"what keeps me sane", "my therapy", mood regulation — **or** the flip side: exercise/rest failing to
help or making mood worse, and mental-health-motivated training decisions.
- ✓ *"running is about the only thing that keeps me sane"*
- ✓ *"exercise is my escape and release"*
- ✓ *"lifting improving mental health is not guaranteed, it can go the other way — i feel worse after
  leg day"* (negative direction still counts: it is about the exercise↔mood link)
- ✗ Generic "exercise is healthy" with no mental/emotional link → no tag.
- ✗ Training for performance goals alone.

---

## Tags considered and **excluded from v1** (recorded, not forgotten)

`substance_use`, `loneliness_isolation`, `sleep`, `trauma_ptsd`, `adhd_neurodivergence`,
`identity_retirement`. Each is real in the corpus (probe prevalence 1.9–5.6%) but was cut from v1
for one of two reasons: **(a)** the construct is dominated by non-MH senses in a fitness corpus
("drinking" = water, "alone" = training solo, "sleep" = recovery), so a defensible label needs its
own rubric pass; or **(b)** prevalence is too low to both train and *honestly evaluate* at the 0.8
bar. They are candidates for v2 — see the process doc for the reasoning.

## Labels file format

`data/data_layer2_ratings/layer2_sample_{n}_rated.csv` — one row per comment, one **0/1 column per
tag**, plus `notes` for judgement calls. `x` in a tag column = unclear (excluded from metrics, as in
Layer 1).
