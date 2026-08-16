"""Weak-supervision labeling functions for the Layer-2 tag model.

Design (Snorkel-style, three-valued -- this is the key decision):

  HP[tag]  high-PRECISION cue  -> silver label 1
  HR[tag]  high-RECALL  cue    -> if it does NOT fire, silver label 0
  HR fires but HP does not     -> ABSTAIN (masked out of the loss entirely)

Why abstain instead of a plain binary rule set: if every comment without the precise cue were
labeled 0, then paraphrases the rules cannot express ("i just don't want to get out of bed
anymore" for depression) would be trained as NEGATIVES, actively teaching the model not to
generalize beyond the lexicon. Abstaining on the uncertain band leaves the model free to learn
those from context, which is the entire point of using a transformer over the rules.

Guards applied before any cue is accepted:
  * NEGATION  -- a negator within ~5 tokens before the cue kills it ("i don't get anxious").
    Layer 1 needed the same hardening; Layer 2 inherits it.
  * FIGURATIVE/SENSE -- per-tag kill patterns for the dominant wrong-sense in a fitness corpus
    ("muscle burnout", "stress fracture", "this workout is killing me", "depressing offense").
"""
import os, re

TAGS = ["depression", "anxiety", "stress_pressure", "burnout_motivation", "performance_psych",
        "body_image_eating", "injury_distress", "self_harm_suicide", "help_seeking",
        "exercise_coping",
        # v3 (2026-07-26, per Babak): tags added to reduce forcing off-taxonomy themes.
        # identity_retirement dropped 2026-08-05 (per directive): 0.1% prevalence, ~0 silver, only 7
        # gold-test positives -> not honestly evaluable at the 0.8 bar (original taxonomy cut it too).
        "substance_use", "loneliness_isolation", "sleep", "trauma_ptsd",
        "adhd_neurodivergence", "exercise_dependence"]

# Co-occurrence window (characters) for the v2 span-crossing patterns. Swept against gold
# train+dev with code/tune_lexicon.py; see the process doc for the sweep table.
W = int(os.environ.get("L2_WINDOW", "80"))

NEG = (r"(?:not|n't|never|no|without|rarely|hardly|dont|doesnt|didnt|isnt|wasnt|arent|"
       r"free of|free from|nothing to do with|instead of|other than)")
# "don't LET a knee injury get you depressed" negates the verb, not the cue -- these are
# exhortations that assert the theme rather than deny it. Same for "not sure/not saying".
NEG_EXEMPT = re.compile(r"\b(?:let|want|mean|saying|sure|only|just|alone)\b", re.I)

def _neg_before(text, start, window=32):
    """A negator within ~3 tokens before the cue, excluding the exemptions above."""
    pre = text[max(0, start - window):start]
    m = re.search(NEG + r"(\W+(?:\w+\W+){0,2})$", pre, re.I)
    return bool(m) and not NEG_EXEMPT.search(m.group(1))

# Tags whose cue is a single concept, where a preceding negator really does deny it.
# The co-occurrence patterns (injury+distress, exercise+mood) span a clause, so a negator
# near the first element says nothing about the construct -- guard is off for those.
NEG_GUARDED = {"depression", "anxiety", "stress_pressure", "burnout_motivation",
               "performance_psych", "body_image_eating", "self_harm_suicide", "help_seeking",
               # v3 new tags are single-concept -> a preceding negator really denies them
               "substance_use", "loneliness_isolation", "sleep", "trauma_ptsd",
               "adhd_neurodivergence", "identity_retirement", "exercise_dependence"}

def _hit(text, pattern, kills=(), guard=True):
    """-> 1 fired clean | 0 fired but every occurrence was negated | None never fired."""
    negated = False
    for m in pattern.finditer(text):
        ctx = text[max(0, m.start() - 60):m.end() + 60]
        if any(k.search(ctx) for k in kills):
            continue
        if guard and _neg_before(text, m.start()):
            negated = True
            continue
        return 1
    return 0 if negated else None

def _cooc(text, pairs, kills=(), window=None):
    """Tri-state like _hit, for a within-`window`-char co-occurrence of two word-class matches.

    Replaces the regex form `A[\\s\\S]{0,W}B|B[\\s\\S]{0,W}A`, which backtracks catastrophically on
    long unpunctuated comments. Here each class is matched independently (linear), then we check
    whether any A-span and B-span lie within `window` chars -- O(#A * #B), both tiny in practice.
    Guard is intentionally off: a negator near one element says nothing about a spanning construct
    (same rationale as the v2 span tags, which are excluded from NEG_GUARDED)."""
    if window is None:
        window = W
    for ra, rb in pairs:
        a_spans = [(m.start(), m.end()) for m in ra.finditer(text)]
        if not a_spans:
            continue
        b_spans = [(m.start(), m.end()) for m in rb.finditer(text)]
        for as_, ae in a_spans:
            for bs, be in b_spans:
                if max(bs - ae, as_ - be) > window:     # gap between the two spans (0 if overlap)
                    continue
                lo, hi = min(as_, bs), max(ae, be)
                ctx = text[max(0, lo - 60):hi + 60]
                if any(k.search(ctx) for k in kills):    # wrong-sense guard, same as _hit
                    continue
                return 1
    return None

# ---------------------------------------------------------------- per-tag definitions
# kills = wrong-sense contexts that must not count (validated against corpus samples)
KILLS = {
    "depression": [r"\bdepress\w*\s+(?:offense|defense|team|game|season|economy|market|weather)\b",
                   r"\b(?:offense|defense|team|game|season|economy|market|weather|record)\b[^.]{0,20}\bdepress"],
    "stress_pressure": [r"\bstress\s*(?:fracture|response|reaction)\b",
                        r"\b(?:muscle|joint|knee|back|spine|tendon|bone|shoulder|hip|ankle)s?\b[^.]{0,25}\bstress\b",
                        r"\bstress(?:es|ing|ed)?\b[^.]{0,25}\b(?:muscle|joint|knee|back|spine|tendon|bone|shoulder|hip|ankle)s?\b",
                        r"\btime under (?:tension|stress)\b", r"\bblood pressure\b", r"\bpressure\s*(?:point|wash|cooker|plate)"],
    "burnout_motivation": [r"\b(?:muscle|arm|leg|forearm|quad|calf|lung|bicep)s?\b[^.]{0,25}\bburn",
                           r"\bburn(?:ed|t)?\s*out\b[^.]{0,20}\b(?:set|rep|lift|muscle|failure)\b",
                           r"\bburn(?:s|ed|ing)?\s+(?:calorie|fat|carb)"],
    "self_harm_suicide": [r"\b(?:kill|killing|dying|death)\b[^.]{0,25}\b(?:workout|leg day|wod|set|cardio|hill|run)\b",
                          r"\b(?:workout|leg day|wod|set|cardio|hill|run|climb)\b[^.]{0,25}\b(?:kill|killing)\b",
                          r"\bkill myself\s+(?:if|when)\b", r"\bdead\s+(?:lift|tired|legs)\b",
                          # gym-joke construction: "high lat insertions give suicidal ideation"
                          r"\b(?:insertion|genetics|physique|lat|calve|form)s?\b[^.]{0,30}\bsuicid",
                          r"\bsuicid\w*\b[^.]{0,20}\b(?:insertion|genetics|physique|lift|grip)\b"],
    "body_image_eating": [r"\bbinge\s*(?:drink|watch|read)"],
    "help_seeking": [r"\b(?:physical|physio|sports?|massage|occupational|hand|knee|shoulder)\s+therap",
                     r"\btherapy\b[^.]{0,15}\b(?:knee|shoulder|back|ankle|acl|rehab|injur)",
                     r"\b(?:knee|shoulder|back|ankle|acl|rehab|injur)\w*\b[^.]{0,20}\btherap"],
    "anxiety": [r"\banxious to\b", r"\bcan'?t wait\b"],
    # ---- v3 new tags: fitness wrong-sense guards ----
    "substance_use": [r"\bdrink(?:ing|s)?\s+(?:water|fluid|enough|more|plenty|electrolyte)",
                      r"\b(?:water|fluid|electrolyte|protein|pre.?workout|caffeine|creatine|energy drink|"
                      r"shake|smoothie|coffee)\b[^.]{0,15}\bdrink", r"\bstay(?:ing)? hydrated\b",
                      r"\baddicted to (?:the )?(?:gym|working out|exercise|running|lifting|training|feeling|endorphins?|pump|gains|grind)\b"],
    "loneliness_isolation": [r"\b(?:train|training|run|running|lift|lifting|work(?:ing)? out|workout|"
                             r"climb\w*|hik\w+|ride|riding|cycl\w+|swim\w*|exercis\w+|gym)\b[^.]{0,20}\b(?:alone|solo|by myself)\b",
                             r"\b(?:alone|solo|by myself)\b[^.]{0,20}\b(?:gym|lift|rep|set|\bpr\b|workout|train|run|climb|ride)\b",
                             r"\bleave me alone\b", r"\balone time\b", r"\bhome alone\b"],
    "sleep": [r"\b(?:recovery|muscle|gains?|growth|hypertrophy|nutrition|protein|repair)\b[^.]{0,25}\bsleep",
              r"\bsleep\b[^.]{0,25}\b(?:recovery|muscle|gains?|growth|hypertrophy|for gains|repair)\b",
              r"\brest day\b", r"\b(?:get|getting|need|needs)\s+(?:more |enough |good |quality |8 |7 |eight |seven )?sleep\b"],
    "trauma_ptsd": [r"\bptsd\b[^.]{0,30}\b(?:rim|shot|miss|missed|loss|lost|game|ref|call|meme|double|"
                    r"free ?throw|penalty|layup|3pt|three)\b",
                    r"\b(?:rim|shot|game|ref|loss|double|penalty|layup|missed)\b[^.]{0,25}\bptsd\b",
                    r"\btrigger(?:ed|s|ing)?\b[^.]{0,15}\b(?:muscle|set|rep|dom|soreness|nerve|the bar)\b"],
    "adhd_neurodivergence": [r"\badd(?:ed|ing|s)?\s+(?:weight|set|rep|volume|plate|mile|pound|kg|lb|more|muscle|size|cardio)",
                             r"\b(?:weight|set|rep|volume|plate|mile|pound|muscle|size)\b[^.]{0,10}\badd\b"],
    "exercise_dependence": [r"\baddicted to (?:the )?(?:feeling|endorphins?|pump|gains|grind|results|progress|"
                            r"dopamine|high|runner'?s high)\b"],
    "identity_retirement": [r"\bidentity theft\b", r"\bbrand identity\b"],
}

# ---------------------------------------------------------------- shared vocabulary
# v2: the six tags whose rules under-fired (rule recall 0.11-0.45 against gold train+dev) are
# rewritten as CO-OCCURRENCE patterns over these shared word classes, with a window that may
# cross sentence boundaries. v1 used tight single-clause patterns and taught the model a much
# narrower concept than the rubric defines -- see the process doc, section 5.
_EX = (r"(?:run|runs|running|ran|lift|lifts|lifted|lifting|exercis\w+|workout|workouts|"
       r"working out|work out|gym|train|trains|training|swim\w*|climb\w*|cycl\w+|bik\w+|yoga|"
       r"cardio|sports?|walk\w*|jog\w*|crossfit|pilates|row\w+|hik\w+|being active|move my body)")
_MOOD = (r"(?:mental health|mentally|mental|depress\w*|anxi\w*|mood|stress\w*|sane|sanity|"
         r"therapy|therapeutic|escape|outlet|release|head ?space|cope|coping|well.?being|"
         r"self.?care|emotional\w*|emotions?|\bsad\b|happy|happier|calm|relax\w*|endorphins?|"
         r"dopamine|clarity|my head|my mind|psyche|burn.?out|panic)")
_LINK = (r"(?:help\w*|improv\w*|benefit\w*|good for|great for|works? for|reliev\w*|relief|"
         r"reduc\w*|manag\w*|cope|coping|deal\w* with|keeps? me|kept me|saved|makes? me feel|"
         r"made me feel|feel better|felt better|better|worse|boost\w*|clear\w*|escape|outlet|"
         r"release|therapy|therapeutic|distract\w*|fight\w*|combat\w*|battl\w*|treat\w*|"
         r"impact\w*|effect\w*|because of|due to|thanks to|struggl\w*|affect\w*)")
_COMP = (r"(?:race|races|racing|game|games|match\w*|meet|meets|competition|compet\w+|tournament|"
         r"season|perform\w*|shot|shooting|shoot|lift|lifts|lifting|\bpr\b|prs|max|attempt|"
         r"routine|event|practice|play|playing|played|swing|serve|jumper|free ?throw)")
_PSYCH = (r"(?:nerves|nervous|anxi\w*|chok\w+|yips|self.?doubt|doubt|mental block|overthink\w*|"
          r"psych\w+ (?:myself|him|her|out)|in my (?:own )?head|head game|mental game|composure|"
          r"confidence|confident|fear of fail\w*|feel like a failure|freeze up|froze|tense up|"
          r"pressure to perform|mental fortitude|mentality|mental side|self.?worth|"
          r"performance anxiety|jelly legs|self.?conscious)")
_EMO = (r"(?:depress\w*|anxi\w*|miserabl\w*|frustrat\w*|devastat\w*|mental health|mentally|"
        r"emotional\w*|discourag\w*|scared|afraid|terrified|fear|upset|\bsad\b|sadness|cry|cried|"
        r"crying|tears|hopeless|worthless|resentful|\bslump\b|going crazy|losing my mind|"
        r"feel like shit|feel weak|pathetic|hard mentally|took a toll|heartbr\w*|gutted|"
        r"restless|weak and pathetic|struggl\w*)")

# v3 (2026-08-05): co-occurrence word classes for the under-firing new tags (prevalence ratio < 0.5
# -> the model learned too narrow a concept). Broadened the same way the original 6 tags were in v2:
# a term class + a context class within L2_WINDOW chars, either order. Water/rest-day/etc. still
# removed by KILLS before the match is accepted.
_SUB = (r"(?:alcohol\w*|drink\w*|drunk|beer|wine|liquor|booze|vodka|whiskey|weed|marijuana|drugs?|"
        r"cocaine|coke|heroin|meth|pills?|xanax|adderall|opioid\w*|shots?|getting high|substance)")
_SUBCTX = (r"(?:cope|coping|numb|forget|escape|deal with|too much|every ?night|problem|addict\w*|"
           r"depend\w*|self.?medicat\w*|relaps\w*|sober|sobriety|to feel|to get through|blackout|"
           r"can'?t stop|hungover|drown\w*|abuse|habit)")
_DEPCTX = (r"(?:addict\w*|compuls\w*|can'?t (?:stop|skip|miss|take)|guilt\w*|obsess\w*|"
           r"control(?:s|ling)? my life|dependenc\w*|over.?train\w*|over.?exercis\w*|"
           r"through (?:the )?injur|twice a day|ruining my|taking over my|crippling|punish\w*|"
           r"even (?:injured|hurt|sick)|anxious (?:if|when) i (?:miss|skip))")
_TRAUMA = (r"(?:trauma\w*|\bptsd\b|abuse\w*|assault\w*|molest\w*|flashback\w*|survivor|"
           r"childhood|abused|violat\w*)")
_TRCTX = (r"(?:haunt\w*|never (?:got|gotten) over|struggl\w*|scarred|nightmare\w*|re.?liv\w*|"
          r"triggered by|can'?t forget|messed me up|still (?:affects|haunts|think about)|"
          r"deal\w* with|depress\w*|anxi\w*|ruined me)")

# High-precision: the construct is essentially unambiguous when these fire un-negated.
HP = {
    "depression": r"\bdepress(?:ion|ive)\b|\b(?:i'?m|im|feeling|been|felt|was|get)\s+(?:so\s+|really\s+|very\s+|pretty\s+)?depressed\b|\bclinical(?:ly)? depress|\bmajor depressive\b|\bhopeless\b|\bworthless\b|\bdark place\b|\bno joy\b|\blost interest in (?:everything|things)\b|\bcan'?t feel (?:anything|happy)\b|\bantidepressant|\bmy depression\b",
    "anxiety": r"\banxiety\b|\bpanic attack|\bpanic disorder\b|\bsocial anxiety\b|\bgeneralized anxiety\b|\bi'?m anxious\b|\bfeel anxious\b|\bgets? anxious\b|\bso anxious\b|\bgym ?timidation\b|\bagoraphob|\bconstantly worry|\bworry (?:constantly|all the time)\b",
    # v2: allow bare "stress\w*" / "pressure" -- the physical senses (stress fracture, stress on
    # my back, time under tension) are removed by KILLS, so the narrow v1 phrase list was costing
    # recall (0.17) for nothing.
    "stress_pressure": r"\bstress(?:ed|ful|es|ing|or|ors)?\b|\bstress\b|\boverwhelm\w*|\btoo much pressure\b|\bpressure (?:from|to perform|of|on)\b|\bunder (?:a lot of |so much )?pressure\b|\bmental(?:ly)? (?:load|drain)|\bmy stress\b|\bburden\w*|\bexpectations\b|\btoo much on my plate\b|\bstretched thin\b|\bjuggl\w+|\bso much going on\b|\bbreaking point\b|\btook its? toll\b|\btaking a toll\b",
    "burnout_motivation": r"\bburn(?:ed|t)? ?out\b|\bburnout\b|\bmental(?:ly)? (?:exhaust|drain|fried|fatigue)|\blost (?:my |the )?(?:motivation|drive|eye of the tiger)\b|\bno motivation\b|\bzero motivation\b|\black of motivation\b|\b(?:struggl|trouble)\w* with (?:my )?motivation\b|\bunmotivated\b|\bde.?motivat\w+|\bmotivation (?:has |just )?(?:dropped|declined|gone|is gone|tanked)\b|\bdread(?:ing)? (?:going to |the )?(?:gym|training|practice|running|workout)|\bdon'?t want to (?:train|work ?out|run|practice|go to the gym)\b|\bgoing through the motions\b|\bovertrain|\bin a (?:workout |gym |training )?rut\b|\bfell off (?:track|the wagon)\b|\bcan'?t get (?:back )?(?:in|into) the groove\b|\bforce myself to (?:go|train|run|work ?out)\b|\btired of (?:training|running|the gym|dieting)\b|\bslacked\b|\bsick of (?:training|running|the gym)\b",
    "performance_psych": (r"\bperformance anxiety\b|\bthe yips\b|\byips\b|\bmental block\b|"
        r"\bfear of fail\w*|\bself.?doubt\b|\bchok\w+ under pressure\b|\bcompetition nerves\b|"
        r"\brace (?:day )?nerves\b|\bpre.?(?:game|race|meet|comp)\w* (?:nerves|anxiety|jitters)\b|"
        r"\bgame day (?:nerves|anxiety)\b|\blost (?:my )?confidence\b|"
        r"\bconfidence (?:issues|problems|is shot)\b|\bperform(?:ance)? under pressure\b|"
        r"\bpressure to perform\b|\bmental (?:game|side|fortitude)\b|\bsports? psycholog\w+|"
        # v2: co-occurrence of a psychological difficulty with a competition/performance context,
        # allowed to span sentences. v1 required both inside one clause -> rule recall 0.11.
        + _PSYCH + rf"[\s\S]{{0,{W}}}?" + _COMP + r"|" + _COMP + rf"[\s\S]{{0,{W}}}?" + _PSYCH),
    "body_image_eating": r"\beating disorder|\banorexi|\bbulimi|\borthorexi|\bed recovery\b|\bbinge(?:ing|d)?\b(?!\s*(?:drink|watch))|\bpurg(?:e|ed|ing)\b|\brestrict(?:ing|ion)\b[^.]{0,30}\b(?:food|eat|calorie)|\bbody image\b|\bbody dysmorph|\bhate my body\b|\bhate how i look\b|\bdisordered eating\b|\bstarv(?:e|ed|ing)\s+(?:myself|ourselves|herself|himself|themselves)\b|\bmake (?:myself|ourselves|herself|himself) (?:throw up|sick|vomit)\b|\b(?:threw|throwing) up\b[^.]{0,30}\b(?:food|eat|calorie|meal|weight)|\bthigh gap\b|\bfeel fat\b|\bobsess\w+ (?:over|about|with) (?:my )?(?:weight|calorie|food|body|scale)|\b(?:insecure|self.?conscious|unhappy|disgusted|embarrassed) (?:about|with|by|over) (?:my|how i look|the way i look|how my)[\s\S]{0,25}(?:body|weight|physique|shape|look|arms|legs|stomach|thighs|shoulders|skin)|\bbody (?:image|dysmorph\w*)|\bhate (?:how i look|my (?:body|weight|stomach|thighs|arms|legs))\b|\brelationship with (?:food|my body)\b|\bfood (?:guilt|anxiety|fear)\b|\bguilty (?:about|for) (?:eating|food)\b|\bemotion(?:al)? eating\b|\beat(?:ing)? my feelings\b|\bcompulsive (?:exercis|eating)\w*|\bstop(?:ped)? eating\b|\bcan'?t (?:eat|stop eating)\b|\bnot eating (?:enough|nearly enough)\b|\bmy ed\b|\bed (?:recovery|feelings|tendencies|habits)\b|\bdisordered eating\b|\bunder ?weight\b|\bsuicide cut\w*",
    # v2: widened the window to span sentences and expanded both sides. Requires injury/illness
    # AND distress -- pure rehab mechanics must stay out (rubric) -- but v1's 60-char same-clause
    # window only caught 23% of gold positives.
    "injury_distress": (r"\bafraid of (?:getting )?(?:injur|hurt)|\bterrified of injur|"
        r"\bscared of (?:re.?)?injur|\bfear of (?:re.?)?injur|\bpost.?injury (?:anxiety|depress)|"
        r"(?:\binjur\w+|\bsidelined\b|\bsurgery\b|\brehab\w*|\btorn\b|\bfracture\w*|\bsprain\w*|"
        r"\bacl\b|\bconcussion\b|\bchronic pain\b|\bcan'?t (?:train|run|lift|play|walk)\b|"
        r"\bout for \d+ (?:week|month)|\bbroke my \w+)" + rf"[\s\S]{{0,{W}}}?" + _EMO + r"|"
        + _EMO + rf"[\s\S]{{0,{W}}}?" + r"(?:\binjur\w+|\bsidelined\b|\bsurgery\b|\brehab\w*|"
        r"\bacl\b|\bconcussion\b|\bchronic pain\b|\bbroke my \w+)"),
    "self_harm_suicide": r"\bsuicid(?:e|al)\b|\bkill(?:ed|ing)? (?:him|her|my|them)self\b|\btook (?:his|her|their) own life\b|\bself.?harm|\bcutting myself\b|\bend (?:it all|my life)\b|\bwant(?:ed)? to die\b|\bnot want to be (?:here|alive)\b",
    # NOTE: "saved my life" was in HP and leaked hyperbole ("basketball saved my life" about a
    # rough patch, not suicidality). Demoted to HR only -- it now abstains instead of asserting.
    "help_seeking": r"\btherapist\b|\btherapy\b|\bcounsell?or\b|\bcounsell?ing\b|\bpsychiatrist\b|\bpsychologist\b|\bsports? psych\w*\b|\bcbt\b|\bssri\b|\bantidepressant|\bmental health (?:professional|treatment|care|services)\b|\bsee (?:a|someone|somebody) (?:professional|about (?:it|this))\b|\bgot (?:diagnosed|help)\b|\bon medication for\b|\bprescribed\b[^.]{0,30}\b(?:ssri|antidepressant|lexapro|zoloft|prozac|wellbutrin)|\b(?:lexapro|zoloft|prozac|wellbutrin|celexa|effexor)\b",
    # v2: the worst under-firer -- rule recall 0.16 against a 27% gold prevalence, while rule
    # PRECISION was 1.00, i.e. all headroom and no risk. Now a general
    # exercise <-> mood link (either direction) plus the explicit idioms.
    "exercise_coping": (r"\bkeeps? me sane\b|\bmy (?:therapy|escape|outlet|sanity)\b|"
        r"\b(?:exercise|running|lifting|training|climbing|swimming|yoga|the gym) is my\b|"
        r"\bmental health benefits?\b|\bfor my mental health\b|\bmental(?:ly)? (?:clarity|clear)\b|"
        r"\bruns? (?:is|as) (?:my )?(?:therapy|meditation|escape|outlet)\b|"
        r"\bclears? my (?:head|mind)\b|\bhead ?space\b|"
        + _EX + rf"[\s\S]{{0,{W}}}?" + _LINK + rf"[\s\S]{{0,{W}}}?" + _MOOD + r"|"
        + _MOOD + rf"[\s\S]{{0,{W}}}?" + _LINK + rf"[\s\S]{{0,{W}}}?" + _EX),
    # ---- v3 new tags ----
    "substance_use": r"\balcoholic\b|\bdrinking problem\b|\bdrink(?:ing)?\b[^.]{0,25}\bto (?:cope|forget|numb|deal|escape|feel|get through|handle)\b|\bdrink(?:ing)? (?:every|most|all|each) (?:night|day|evening|weekend)\b|\bdrink(?:ing)? (?:heavily|a lot|too much|myself)\b|\bsubstance (?:abuse|use disorder|problem)\b|\brelaps(?:e|ed|ing)\b|\b(?:getting|been|staying|stay|days?|weeks?|months?|years?) sober\b|\bsobriety\b|\baddicted to (?:alcohol|drinking|drugs|weed|pills|cocaine|opioid|xanax|adderall|booze)\b|\bdrug (?:problem|addiction|habit|abuse)\b|\balcohol(?:ism)?\b[^.]{0,20}\b(?:problem|cope|numb|depress|anxi|escape)|\bnumb(?:ing)? (?:myself|it|the pain|my feelings|everything) with (?:alcohol|drink|weed|drugs|booze|pills)\b|\bblackout drunk\b|\bdrown(?:ing)? (?:my|the) (?:sorrows|pain|feelings)\b|\bself.?medicat\w+|\bpill(?:s)? (?:problem|habit|addiction)\b|\b(?:coke|heroin|meth|opioid)\b[^.]{0,12}\b(?:habit|problem|addict|abus)",
    # substance/trauma/exercise_dependence co-occurrence moved to _COOC (Python proximity check):
    # the regex form A[\s\S]{0,W}B|B[\s\S]{0,W}A backtracks catastrophically on long unpunctuated
    # comments (a single comment pegged a core for minutes). The Python check is linear and exact.
    "loneliness_isolation": r"\blonel(?:y|iness)\b|\bfeel(?:ing)? (?:so |really |very |completely |utterly |kinda )?alone\b|\bisolat(?:ed|ion|ing)\b|\bno (?:real |close )?friends\b|\bnobody (?:to talk to|who|cares|understands|gets it|there)\b|\bno one (?:to talk to|cares|understands|gets me|there)\b|\bsocially (?:isolated|withdrawn|disconnected|anxious)\b|\bfeel (?:so |really )?disconnected\b|\bfeel(?:ing)? left out\b|\balienat(?:ed|ion)\b|\bcut off from (?:everyone|people|friends|the world)\b|\ball by myself\b|\bhave no one\b|\bfeel invisible\b|\bwithdrawn from (?:people|everyone|friends|society)\b|\bstarv(?:ed|ing) for (?:human )?(?:connection|contact)\b",
    "sleep": r"\binsomnia\b|\bcan'?t (?:sleep|fall asleep|stay asleep)\b|\bcant (?:sleep|fall asleep)\b|\btrouble (?:sleeping|falling asleep|staying asleep)\b|\bsleepless\b|\bnot sleeping (?:well|at all|much|enough)\b|\blying (?:there )?awake\b|\bawake (?:all|half the) night\b|\bup all night\b|\bsleep (?:problem|issue|deprivation|anxiety|deprived|schedule is|is wrecked)\b|\bracing (?:thoughts|mind)\b[^.]{0,25}\b(?:night|sleep|bed)|\bnightmares?\b|\bwaking up (?:at \d|every|multiple|throughout|drenched)|\bcan'?t (?:shut|turn) (?:my brain|my mind|it) off\b|\bhaven'?t slept\b|\bno sleep\b[^.]{0,20}\b(?:anxi|stress|worry|depress)|\bexhausted (?:but|and) can'?t sleep\b|\bwide awake at \d",
    "trauma_ptsd": r"\bptsd\b|\bpost.?traumatic\b|\btraumati[sz](?:ed|ing|e)\b|\btrauma\b|\bflashback\w*|\bchildhood trauma\b|\b(?:sexual|physical|emotional)(?:ly)? abus(?:e|ed|ive)\b|\bwas abused\b|\bsurvivor of\b|\bassault(?:ed)?\b|\bmolest\w+|\bhaunt(?:s|ed) me\b|\bnever (?:got|gotten) over\b[^.]{0,25}\b(?:trauma|it happened|the incident|abuse|attack)|\btriggered by\b[^.]{0,25}\b(?:memory|memories|past|reminds|abuse|trauma|him|her)|\bre.?liv(?:e|ing) (?:the|it|that)\b|\bnightmares about\b|\bcomplex trauma\b|\bcptsd\b",
    "adhd_neurodivergence": r"\badhd\b|\ba\.?d\.?h\.?d\.?\b|\bautis(?:m|tic)\b|\basperger'?s?\b|\bneurodiverg\w+|\bneurodivergent\b|\bexecutive (?:function|dysfunction)\b|\bon the (?:autism )?spectrum\b|\bocd\b|\bobsessive.?compulsive\b|\bdyslexi\w+|\bdyspraxi\w+|\bstim(?:ming)?\b|\bhyperfocus\w*|\btime blindness\b|\bsensory (?:overload|issues|processing)\b|\bmy (?:adhd|autism|ocd)\b|\bexecutive.?function\w*",
    "identity_retirement": r"\bretir(?:e|ed|ing|ement)\b[^.]{0,45}\b(?:depress|lost|struggl|identity|miss|hard|adjust|purpose|empty|hole|void|nothing|who i am)|\b(?:lost|losing|loss of) (?:my )?(?:sense of )?(?:identity|purpose|self)\b|\bwho (?:am|was) i without\b|\bmy (?:whole |entire )?identity (?:is|was|is tied|revolved|wrapped up)\b|\b(?:running|lifting|the gym|climbing|my sport|competing)\b[^.]{0,20}\b(?:is|was) my (?:identity|whole life|everything|life|purpose)\b|\bdefined (?:by|me) (?:my sport|as an athlete|by (?:running|lifting|the gym))\b|\blife after (?:sport|the sport|competing|running|my career|basketball)\b|\bdon'?t know who i am (?:without|anymore|now)\b|\bidentity (?:crisis|loss)\b",
    "exercise_dependence": r"\bexercise (?:addiction|dependenc\w+|bulimia)\b|\baddicted to (?:the gym|working out|exercise|running|training|lifting|the treadmill)\b|\bgym addict\w*|\bcan'?t (?:take|skip|miss) (?:a |any )?(?:rest )?days?\b|\bguilty (?:if|when|about|for) (?:i )?(?:miss|skip|don'?t|not|didn'?t)\w* (?:a )?(?:workout|the gym|gym|run|training|exercise|lifting)|\bcompulsiv\w+ (?:exercis|train|running|workout)|\bhave to (?:work ?out|train|run|exercise|hit the gym) (?:every ?day|no matter|even (?:when|if)|or i)|\bover.?exercis\w+|\bexercis\w+ (?:is )?(?:crippling|ruining|controlling|taking over|destroying)\b|\bmy (?:climbing|running|gym|training|lifting|workout|exercise) addiction\b|\bpunish(?:ing)? myself (?:with|at the gym|with exercise|by (?:running|training))\b|\bwork(?:ing)? out (?:twice a day|through (?:the )?injur|even (?:injured|sick|hurt))\b|\bcan'?t stop (?:working out|training|running|going to the gym)\b|\bworkout (?:controls|runs|ruining) my life\b|\banxious (?:if|when) i (?:miss|skip|can'?t) (?:the gym|a workout|training)",
}

# High-recall: anything that could plausibly be the theme. Not matching => safe negative.
HR = {
    "depression": r"depress|hopeless|worthless|\bsad\b|\bsadness\b|\blow mood\b|\bdown\b|\bmiserable\b|\bempty\b|\bnumb\b|\bdark\b|\bcry|\bmood\b|\bgrief\b|\bgrieving\b|\bmourn|\bjoy\b|\bmotivation\b|\bapath|\bdespair|\bbleak\b|\bmental health\b|\bunhappy\b|\bblue\b",
    "anxiety": r"anxi|panic|\bnervous\b|\bnerves\b|\bworry\b|\bworried\b|\bworrying\b|\bafraid\b|\bfear\b|\bscared\b|\bterrified\b|\bdread\b|\bon edge\b|\bstress|\bphobia\b|\bfreak(?:ing)? out\b|\bintimidat|\bself.?conscious\b|\buncomfortable\b|\bmental health\b|\btense\b|\bshaky\b",
    "stress_pressure": r"\bstress|\bpressure\b|\boverwhelm|\bburden\b|\bexpectation|\bdemand|\btoo much\b|\bcop(?:e|ing)\b|\bstrain\b|\bmental health\b|\bexhaust|\bdrain|\bjuggl|\bbalanc\w+ (?:work|school|life)|\bstretched thin\b|\bstruggl",
    "burnout_motivation": r"burn|\bexhaust|\bfatigue|\bmotivat|\bdrain|\bdread|\btired\b|\bworn out\b|\bfried\b|\bdone with\b|\bquit|\bbreak\b|\bslump\b|\bplateau\b|\bapath|\bovertrain|\bstale\b|\bboring\b|\bbored\b|\bforce myself\b|\bmental",
    "performance_psych": r"\bchok|\byips\b|\bconfidence\b|\bconfident\b|\bdoubt\b|\bnerv|\banxi|\bmental\b|\bhead\b|\boverthink|\bpressure\b|\bfail|\bperform|\bcompet|\bmeet\b|\brace\b|\bgame\b|\bjudged?\b|\bwatch(?:ed|ing)\b|\bblock\b|\bfocus\b|\bpsych",
    "body_image_eating": r"\beat|\bfood\b|\bdiet\b|\bweight\b|\bcalorie|\bfat\b|\bskinny\b|\bthin\b|\bbody\b|\bmirror\b|\blook\b|\bappearance\b|\bbinge|\bpurg|\bstarv|\brestrict|\bmacro|\bscale\b|\bbulk|\bcut(?:ting)?\b|\banorex|\bbulim|\bed\b|\bimage\b|\bugly\b|\bshape\b",
    "injury_distress": r"\binjur|\bhurt\b|\bpain\b|\btorn?\b|\bacl\b|\bsurgery\b|\brehab|\bsidelined\b|\bbroken\b|\bfracture|\bstrain\b|\bsprain|\bphysio|\bpt\b|\brecover|\bsetback\b|\bout for\b|\bcan'?t (?:train|run|lift|play)\b|\bmiss(?:ed|ing)? (?:the )?season\b",
    "self_harm_suicide": r"suicid|\bkill\b|\bkilling\b|\bdie\b|\bdying\b|\bdead\b|\bdeath\b|\bself.?harm|\bcut(?:ting)?\b|\bend it\b|\bharm\b|\bhurt myself\b|\boverdose\b|\bhotline\b|\bsaved my life\b|\bnot be here\b",
    "help_seeking": r"therap|counsel|psych|\bcbt\b|\bssri\b|\bmedicat|\bprescri|\bdiagnos|\bdoctor\b|\bdr\.|\bclinic\b|\btreatment\b|\bhelp\b|\bprofessional\b|\bhotline\b|\bmeds\b|\bantidepress|\bmental health\b",
    "exercise_coping": r"\bmental health\b|\bmental\b|\bdepress|\banxi|\bmood\b|\bsane\b|\bsanity\b|\bstress|\btherapy\b|\bescape\b|\boutlet\b|\brelease\b|\bhead ?space\b|\bclears? my head\b|\bfeel better\b|\bhelps? me\b|\bcoping\b|\bcope\b|\bwellbeing\b|\bwell.being\b|\bself.?care\b",
    # ---- v3 new tags ----
    "substance_use": r"\bdrink|\balcohol|\bdrunk\b|\bsober|\bsobriety\b|\bweed\b|\bdrugs?\b|\bhungover\b|\baddict|\bsubstance\b|\brelapse\b|\bcocaine|\bopioid|\bxanax|\bnarcotic|\bbeer\b|\bwine\b|\bliquor\b|\bbooze\b|\bpills?\b|\bnumb\b|\bmedicat|\bhigh\b|\bcoke\b|\bheroin\b|\bmeth\b",
    "loneliness_isolation": r"\blonel|\balone\b|\bisolat|\bfriends?\b|\bsocial\b|\bnobody\b|\bno one\b|\bdisconnect|\bwithdrawn\b|\bleft out\b|\bcompany\b|\bconnection\b|\bby myself\b|\binvisible\b|\balienat|\bempty\b|\boutcast\b",
    "sleep": r"\bsleep|\binsomni|\bawake\b|\btired\b|\brest\w*|\bnightmare|\bnight\b|\bexhaust|\bfatigue|\brestless\b|\bwaking\b|\bbed\b|\binsomnia\b|\bnap\b|\bdrowsy\b|\bcircadian\b",
    "trauma_ptsd": r"\btrauma|\bptsd\b|\bflashback|\btrigger|\babus|\bassault|\bmolest|\bhaunt|\bnightmare|\bpast\b|\bviolence\b|\bsurvivor\b|\bincident\b|\bre.?live|\bmemories\b",
    "adhd_neurodivergence": r"\badhd\b|\badd\b|\bautis|\basperger|\bneurodiverg|\bspectrum\b|\bocd\b|\bfocus\b|\battention\b|\bhyperfocus\b|\bexecutive\b|\bstim\b|\bsensory\b|\bdyslex|\bdistract|\bimpuls",
    "identity_retirement": r"\bretir|\bidentity\b|\bpurpose\b|\bwho i am\b|\bwho am i\b|\bmeaning\b|\bdefine|\bwithout (?:the )?sport\b|\bafter (?:sport|competing|running|basketball)\b|\bself.?worth\b|\bempty\b|\bvoid\b|\bmy sport\b|\bcareer\b",
    "exercise_dependence": r"\baddict|\bcompuls|\bcan'?t stop\b|\bover.?train|\bover.?exercis|\brest day\b|\bguilt|\bevery ?day\b|\bobsess|\bcontrol|\bhave to (?:train|run|work ?out|go)\b|\bcrippling\b|\bpunish|\bdependenc|\btwice a day\b|\bmiss(?:ing)? (?:a )?(?:workout|day|the gym)\b|\bskip",
}

_HP = {k: re.compile(v, re.I) for k, v in HP.items()}
_HR = {k: re.compile(v, re.I) for k, v in HR.items()}
_KILLS = {k: tuple(re.compile(p, re.I) for p in v) for k, v in KILLS.items()}

# v3 co-occurrence: (term-class, context-class) pairs checked within W chars via _cooc (Python,
# no regex backtracking). A tag fires if the base HP fires OR its co-occurrence pair does.
# Round-2 finding (trials L2v3b/v3c): broadening substance_use and trauma_ptsd added silver noise for
# no measurable gain (F1 flat) while the extra positives dragged the shared encoder's good tags down.
# Only exercise_dependence genuinely benefited (0.20 -> 0.73). So keep co-occurrence ONLY there;
# substance/trauma revert to their narrow high-precision rules (their base HP still stands).
_COOC = {
    "exercise_dependence": [(_EX, _DEPCTX)],
}
_COOC = {tag: [(re.compile(a, re.I), re.compile(b, re.I)) for a, b in pairs]
         for tag, pairs in _COOC.items()}

def label(text):
    """-> {tag: 1 | 0 | None}.  None = abstain (excluded from the loss).

    A cue that fires but is *negated* becomes an explicit 0 rather than an abstain: those are
    the highest-value negatives in the set, because they teach the model the difference between
    "my anxiety was bad" and "i don't get anxious" instead of letting it key on the keyword.
    """
    out = {}
    for t in TAGS:
        hp = _hit(text, _HP[t], _KILLS.get(t, ()), guard=(t in NEG_GUARDED))
        if hp != 1 and t in _COOC:      # base HP did not fire clean -> try the co-occurrence pair
            c = _cooc(text, _COOC[t], _KILLS.get(t, ()))
            if c == 1:
                hp = 1
        if hp == 1:
            out[t] = 1
        elif hp == 0:
            out[t] = 0               # cue present but negated -> hard negative
        elif _HR[t].search(text):
            out[t] = None            # plausible but unproven -> abstain
        else:
            out[t] = 0
    return out

def label_row(text):
    d = label(text)
    return [d[t] for t in TAGS]

def hr_match(text, tag):
    """Does the broad high-recall cue for `tag` appear? (candidate finder for sampling)"""
    return bool(_HR[tag].search(text))

def any_hr(text):
    """True if any tag's high-recall cue appears -- i.e. the text is on-theme for something."""
    return any(r.search(text) for r in _HR.values())
