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
        "exercise_coping"]

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
               "performance_psych", "body_image_eating", "self_harm_suicide", "help_seeking"}

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
        + _PSYCH + rf"[\s\S]{{0,{W}}}" + _COMP + r"|" + _COMP + rf"[\s\S]{{0,{W}}}" + _PSYCH),
    "body_image_eating": r"\beating disorder|\banorexi|\bbulimi|\borthorexi|\bed recovery\b|\bbinge(?:ing|d)?\b(?!\s*(?:drink|watch))|\bpurg(?:e|ed|ing)\b|\brestrict(?:ing|ion)\b[^.]{0,30}\b(?:food|eat|calorie)|\bbody image\b|\bbody dysmorph|\bhate my body\b|\bhate how i look\b|\bdisordered eating\b|\bstarv(?:e|ed|ing)\s+(?:myself|ourselves|herself|himself|themselves)\b|\bmake (?:myself|ourselves|herself|himself) (?:throw up|sick|vomit)\b|\b(?:threw|throwing) up\b[^.]{0,30}\b(?:food|eat|calorie|meal|weight)|\bthigh gap\b|\bfeel fat\b|\bobsess\w+ (?:over|about|with) (?:my )?(?:weight|calorie|food|body|scale)|\b(?:insecure|self.?conscious|unhappy|disgusted|embarrassed) (?:about|with|by|over) (?:my|how i look|the way i look|how my)[\s\S]{0,25}(?:body|weight|physique|shape|look|arms|legs|stomach|thighs|shoulders|skin)|\bbody (?:image|dysmorph\w*)|\bhate (?:how i look|my (?:body|weight|stomach|thighs|arms|legs))\b|\brelationship with (?:food|my body)\b|\bfood (?:guilt|anxiety|fear)\b|\bguilty (?:about|for) (?:eating|food)\b|\bemotion(?:al)? eating\b|\beat(?:ing)? my feelings\b|\bcompulsive (?:exercis|eating)\w*|\bstop(?:ped)? eating\b|\bcan'?t (?:eat|stop eating)\b|\bnot eating (?:enough|nearly enough)\b|\bmy ed\b|\bed (?:recovery|feelings|tendencies|habits)\b|\bdisordered eating\b|\bunder ?weight\b|\bsuicide cut\w*",
    # v2: widened the window to span sentences and expanded both sides. Requires injury/illness
    # AND distress -- pure rehab mechanics must stay out (rubric) -- but v1's 60-char same-clause
    # window only caught 23% of gold positives.
    "injury_distress": (r"\bafraid of (?:getting )?(?:injur|hurt)|\bterrified of injur|"
        r"\bscared of (?:re.?)?injur|\bfear of (?:re.?)?injur|\bpost.?injury (?:anxiety|depress)|"
        r"(?:\binjur\w+|\bsidelined\b|\bsurgery\b|\brehab\w*|\btorn\b|\bfracture\w*|\bsprain\w*|"
        r"\bacl\b|\bconcussion\b|\bchronic pain\b|\bcan'?t (?:train|run|lift|play|walk)\b|"
        r"\bout for \d+ (?:week|month)|\bbroke my \w+)" + rf"[\s\S]{{0,{W}}}" + _EMO + r"|"
        + _EMO + rf"[\s\S]{{0,{W}}}" + r"(?:\binjur\w+|\bsidelined\b|\bsurgery\b|\brehab\w*|"
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
        + _EX + rf"[\s\S]{{0,{W}}}" + _LINK + rf"[\s\S]{{0,{W}}}" + _MOOD + r"|"
        + _MOOD + rf"[\s\S]{{0,{W}}}" + _LINK + rf"[\s\S]{{0,{W}}}" + _EX),
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
}

_HP = {k: re.compile(v, re.I) for k, v in HP.items()}
_HR = {k: re.compile(v, re.I) for k, v in HR.items()}
_KILLS = {k: tuple(re.compile(p, re.I) for p in v) for k, v in KILLS.items()}

def label(text):
    """-> {tag: 1 | 0 | None}.  None = abstain (excluded from the loss).

    A cue that fires but is *negated* becomes an explicit 0 rather than an abstain: those are
    the highest-value negatives in the set, because they teach the model the difference between
    "my anxiety was bad" and "i don't get anxious" instead of letting it key on the keyword.
    """
    out = {}
    for t in TAGS:
        hp = _hit(text, _HP[t], _KILLS.get(t, ()), guard=(t in NEG_GUARDED))
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
