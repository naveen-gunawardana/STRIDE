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
import re

TAGS = ["depression", "anxiety", "stress_pressure", "burnout_motivation", "performance_psych",
        "body_image_eating", "injury_distress", "self_harm_suicide", "help_seeking",
        "exercise_coping"]

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

# High-precision: the construct is essentially unambiguous when these fire un-negated.
HP = {
    "depression": r"\bdepress(?:ion|ive)\b|\b(?:i'?m|im|feeling|been|felt|was|get)\s+(?:so\s+|really\s+|very\s+|pretty\s+)?depressed\b|\bclinical(?:ly)? depress|\bmajor depressive\b|\bhopeless\b|\bworthless\b|\bdark place\b|\bno joy\b|\blost interest in (?:everything|things)\b|\bcan'?t feel (?:anything|happy)\b|\bantidepressant|\bmy depression\b",
    "anxiety": r"\banxiety\b|\bpanic attack|\bpanic disorder\b|\bsocial anxiety\b|\bgeneralized anxiety\b|\bi'?m anxious\b|\bfeel anxious\b|\bgets? anxious\b|\bso anxious\b|\bgym ?timidation\b|\bagoraphob|\bconstantly worry|\bworry (?:constantly|all the time)\b",
    "stress_pressure": r"\bstressed out\b|\bso stressed\b|\breally stressed\b|\bstress (?:is|was) (?:getting|killing|too)\b|\boverwhelmed\b|\bfeeling overwhelmed\b|\btoo much pressure\b|\bpressure (?:from|to perform|of)\b|\bunder (?:a lot of |so much )?pressure\b|\bmental(?:ly)? (?:load|drain)|\bstress (?:got to me|management|relief|eating)\b|\bmy stress\b|\bcope with (?:the )?stress\b",
    "burnout_motivation": r"\bburn(?:ed|t)? ?out\b|\bburnout\b|\bmental(?:ly)? (?:exhaust|drain|fried|fatigue)|\blost (?:my )?motivation\b|\bno motivation\b|\bzero motivation\b|\black of motivation\b|\bstruggl\w+ with motivation\b|\bunmotivated\b|\bdread(?:ing)? (?:going to |the )?(?:gym|training|practice|running|workout)|\bdon'?t want to (?:train|work ?out|run|practice)\b|\bgoing through the motions\b|\bovertrain",
    "performance_psych": r"\bperformance anxiety\b|\bthe yips\b|\byips\b|\bmental block\b|\bfear of failure\b|\bself.?doubt\b|\bchoke[sd]? under pressure\b|\bchoking under pressure\b|\bcompetition nerves\b|\brace (?:day )?nerves\b|\bpre.?(?:game|race|meet|comp) (?:nerves|anxiety)\b|\bgame day (?:nerves|anxiety)\b|\bin my (?:own )?head\b|\boverthink\w*\b[^.]{0,40}\b(?:lift|form|shot|swing|race|game|technique)|\blost (?:my )?confidence\b|\bconfidence (?:issues|problems|is shot)\b|\bperform(?:ance)? under pressure\b",
    "body_image_eating": r"\beating disorder|\banorexi|\bbulimi|\borthorexi|\bed recovery\b|\bbinge(?:ing|d)?\b(?!\s*(?:drink|watch))|\bpurg(?:e|ed|ing)\b|\brestrict(?:ing|ion)\b[^.]{0,30}\b(?:food|eat|calorie)|\bbody image\b|\bbody dysmorph|\bhate my body\b|\bhate how i look\b|\bdisordered eating\b|\bstarv(?:e|ed|ing)\s+(?:myself|ourselves|herself|himself|themselves)\b|\bmake (?:myself|ourselves|herself|himself) (?:throw up|sick|vomit)\b|\b(?:threw|throwing) up\b[^.]{0,30}\b(?:food|eat|calorie|meal|weight)|\bthigh gap\b|\bfeel fat\b|\bobsess\w+ (?:over|about|with) (?:my )?(?:weight|calorie|food|body)",
    "injury_distress": r"\binjur\w+[^.]{0,60}\b(?:depress|anxi|miserabl|frustrat|devastat|mental health|emotional|hard mentally|mentally|crushed|heartbr)\w*|\b(?:depress|anxi|miserabl|devastat|mental health|mentally)\w*[^.]{0,60}\binjur|\bafraid of (?:getting )?(?:injur|hurt)|\bterrified of injur|\bscared of (?:re.?)?injur|\bfear of (?:re.?)?injur|\bcan'?t (?:train|run|lift|play)\b[^.]{0,40}\b(?:depress|miserable|going crazy|losing my mind)|\bsidelined\b[^.]{0,50}\b(?:depress|miserable|frustrat|mental)",
    "self_harm_suicide": r"\bsuicid(?:e|al)\b|\bkill(?:ed|ing)? (?:him|her|my|them)self\b|\btook (?:his|her|their) own life\b|\bself.?harm|\bcutting myself\b|\bend (?:it all|my life)\b|\bwant(?:ed)? to die\b|\bnot want to be (?:here|alive)\b",
    # NOTE: "saved my life" was in HP and leaked hyperbole ("basketball saved my life" about a
    # rough patch, not suicidality). Demoted to HR only -- it now abstains instead of asserting.
    "help_seeking": r"\btherapist\b|\btherapy\b|\bcounsell?or\b|\bcounsell?ing\b|\bpsychiatrist\b|\bpsychologist\b|\bsports? psych\w*\b|\bcbt\b|\bssri\b|\bantidepressant|\bmental health (?:professional|treatment|care|services)\b|\bsee (?:a|someone|somebody) (?:professional|about (?:it|this))\b|\bgot (?:diagnosed|help)\b|\bon medication for\b|\bprescribed\b[^.]{0,30}\b(?:ssri|antidepressant|lexapro|zoloft|prozac|wellbutrin)|\b(?:lexapro|zoloft|prozac|wellbutrin|celexa|effexor)\b",
    "exercise_coping": r"\b(?:running|lifting|exercise|working out|the gym|training|swimming|climbing|cycling|yoga|sport)\b[^.]{0,60}\b(?:helps? (?:my |with )?(?:mental|depress|anxi|mood)|keeps? me sane|saved my life|my therapy|my escape|my outlet|for my mental health|mental health)\b|\b(?:mental health|depress\w*|anxi\w*|mood|sanity)\b[^.]{0,60}\b(?:is why i (?:run|lift|train)|running|lifting|exercise|working out|the gym)\b[^.]{0,40}\b(?:helps?|better|improve)|\bexercise is my\b|\btraining is my (?:therapy|outlet|escape)\b|\brun(?:ning)? (?:is|as) (?:my )?(?:therapy|meditation|escape|outlet)\b|\blift(?:ing)? (?:is|as) (?:my )?(?:therapy|outlet|escape)\b|\bworkout(?:s)? (?:help|improve)\w*\b[^.]{0,30}\b(?:mood|mental|depress|anxi)",
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
