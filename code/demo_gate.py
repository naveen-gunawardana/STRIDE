"""Two-layer demo — type a comment, see the full pipeline run live in your browser.

  Layer 1 (gate):  is this athlete mental health?   relevant = P(mh)>=0.50 AND P(sport)>=0.40
  Layer 2 (tags):  which of 10 MH themes?            (runs only if Layer 1 says relevant)

All three are transformer models: mh + sport (twitter-roberta) and a 10-head multi-label
tag model (DAPT base). No Flask -- stdlib http.server only.

Run:   .venv\\Scripts\\python.exe code\\demo_gate.py     (or double-click demo_gate.bat)
Then open http://127.0.0.1:8000
"""
import html, json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layer2_lexicon import TAGS                      # canonical tag order

MH_THR, SP_THR = 0.50, 0.40                          # Layer-1 deliverable operating point
L2_DIR = "models/layer2_tags"                        # == layer2_tags_r5_e18 deliverable
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PORT = 8000

# the 4 tags validated >=0.8 on P/R/F1 (see meeting report); the rest are provisional
VALIDATED = {"anxiety", "depression", "burnout_motivation", "self_harm_suicide"}
NICE = {"depression": "Depression", "anxiety": "Anxiety", "stress_pressure": "Stress / pressure",
        "burnout_motivation": "Burnout / motivation", "performance_psych": "Performance psych",
        "body_image_eating": "Body image / eating", "injury_distress": "Injury distress",
        "self_harm_suicide": "Self-harm / suicide", "help_seeking": "Help-seeking",
        "exercise_coping": "Exercise as coping"}

print(f"[demo] loading models on {DEVICE} ...")
_M = {}
for g in ("mh", "sport"):
    p = f"models/filter_relevance_{g}"
    _M[g] = (AutoTokenizer.from_pretrained(p),
             AutoModelForSequenceClassification.from_pretrained(p).to(DEVICE).eval())
_L2_TOK = AutoTokenizer.from_pretrained(L2_DIR)
_L2 = AutoModelForSequenceClassification.from_pretrained(L2_DIR).to(DEVICE).eval()
_L2_THR = json.load(open(os.path.join(L2_DIR, "thresholds.json"), encoding="utf-8"))
print("[demo] ready.")


def prob(group, text):
    tok, model = _M[group]
    with torch.no_grad():
        enc = tok([text], truncation=True, padding=True, max_length=512, return_tensors="pt").to(DEVICE)
        return float(model(**enc).logits.softmax(1)[0, 1])


def layer2_tags(text):
    with torch.no_grad():
        enc = _L2_TOK([text], truncation=True, padding=True, max_length=256, return_tensors="pt").to(DEVICE)
        probs = torch.sigmoid(_L2(**enc).logits)[0].float().cpu().tolist()
    out = [{"tag": t, "p": probs[i], "thr": _L2_THR[t], "fires": probs[i] >= _L2_THR[t]}
           for i, t in enumerate(TAGS)]
    out.sort(key=lambda d: d["p"], reverse=True)
    return out


def classify(text):
    p_mh, p_sp = prob("mh", text), prob("sport", text)
    relevant = (p_mh >= MH_THR) and (p_sp >= SP_THR)
    return {"p_mh": p_mh, "p_sport": p_sp, "mh_yes": p_mh >= MH_THR,
            "sport_yes": p_sp >= SP_THR, "relevant": relevant,
            "tags": layer2_tags(text) if relevant else None}


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Athlete-MH classifier demo</title>
<style>
 body{{font-family:system-ui,sans-serif;max-width:720px;margin:36px auto;padding:0 16px;color:#1a1a2e}}
 h1{{font-size:1.4rem;margin-bottom:2px}} .sub{{color:#666;margin-top:0}}
 textarea{{width:100%;height:110px;font-size:1rem;padding:10px;border:1px solid #ccc;border-radius:8px;box-sizing:border-box}}
 button{{margin-top:10px;padding:10px 22px;font-size:1rem;border:0;border-radius:8px;background:#3a5;color:#fff;cursor:pointer}}
 .card{{border:1px solid #e6e6ee;border-radius:10px;padding:14px 16px;margin:16px 0;background:#fafafd}}
 .lbl{{font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:#8a8aa0;font-weight:700}}
 .verdict{{font-size:1.25rem;font-weight:700;margin:4px 0 8px}} .yes{{color:#1a7f37}} .no{{color:#b42318}}
 .bar{{height:16px;border-radius:5px;background:#ececf2;overflow:hidden;margin:3px 0 8px}}
 .fill{{height:100%;background:#3a5}} .fill.lo{{background:#c9ccd6}}
 .row{{margin:9px 0}} .tag{{display:flex;justify-content:space-between;font-size:.95rem;margin-top:10px}}
 .fires{{font-weight:700}} .prov{{font-size:.7rem;color:#b8860b;border:1px solid #e0c060;border-radius:4px;padding:0 4px;margin-left:6px}}
 .val{{font-size:.7rem;color:#1a7f37;border:1px solid #8ec99b;border-radius:4px;padding:0 4px;margin-left:6px}}
 .muted{{color:#999}} .thr{{color:#999;font-size:.8rem;margin-top:18px}}
</style></head><body>
<h1>Athlete mental-health classifier</h1>
<p class="sub">Layer 1 decides <b>relevance</b>; Layer 2 tags the <b>themes</b>. Both are transformer models.</p>
<form method="post">
<textarea name="comment" placeholder="Paste a Reddit comment...">{comment}</textarea>
<button type="submit">Classify</button>
</form>
{result}
<p class="thr">Layer 1: relevant if P(mh) &ge; {mh} AND P(sport) &ge; {sp} (held-out F1 0.89).
Layer 2: 10-head multi-label, per-tag thresholds; <span class="val">validated</span> = &ge;0.8 on P/R/F1,
<span class="prov">provisional</span> = below.</p>
</body></html>"""


def barhtml(p, thr):
    pct = int(round(p * 100))
    cls = "fill" if p >= thr else "fill lo"
    return f'<div class="bar"><div class="{cls}" style="width:{pct}%"></div></div>'


def result_html(text):
    if not text.strip():
        return ""
    r = classify(text)
    # ---- Layer 1 card ----
    v = ('<div class="verdict yes">&#10003; RELEVANT &mdash; athlete mental health</div>'
         if r["relevant"] else '<div class="verdict no">&#10007; not relevant</div>')
    def line(label, p, yes, thr):
        mark = "&#10003;" if yes else "&#10007;"
        return (f'<div class="row"><b>{label}</b> {p:.2f} '
                f'<span class="{"yes" if yes else "no"}">{mark} {"pass" if yes else "below"} '
                f'(thr {thr})</span>{barhtml(p, thr)}</div>')
    l1 = ('<div class="card"><div class="lbl">Layer 1 &mdash; relevance gate</div>' + v
          + line("P(mental health)", r["p_mh"], r["mh_yes"], MH_THR)
          + line("P(sport)", r["p_sport"], r["sport_yes"], SP_THR) + '</div>')
    # ---- Layer 2 card ----
    if not r["relevant"]:
        l2 = ('<div class="card muted"><div class="lbl">Layer 2 &mdash; theme tags</div>'
              'Skipped &mdash; Layer 1 said not relevant, so no tags are applied.</div>')
    else:
        fired = [t for t in r["tags"] if t["fires"]]
        head = ('<b>Tags: ' + ", ".join(NICE[t["tag"]] for t in fired) + '</b>'
                if fired else '<b class="muted">No specific theme tag fired</b>')
        rows = ""
        for t in r["tags"]:
            badge = ('<span class="val">validated</span>' if t["tag"] in VALIDATED
                     else '<span class="prov">provisional</span>')
            name = (f'<span class="fires">{NICE[t["tag"]]}</span>' if t["fires"]
                    else f'<span class="muted">{NICE[t["tag"]]}</span>')
            mark = "&#10003; " if t["fires"] else ""
            rows += (f'<div class="tag"><span>{mark}{name}{badge}</span>'
                     f'<span>{t["p"]:.2f} <span class="muted">/ thr {t["thr"]:.2f}</span></span></div>'
                     + barhtml(t["p"], t["thr"]))
        l2 = ('<div class="card"><div class="lbl">Layer 2 &mdash; theme tags</div>'
              + head + '<div style="margin-top:8px">' + rows + '</div></div>')
    return l1 + l2


class H(BaseHTTPRequestHandler):
    def _send(self, body):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def do_GET(self):
        if self.path.startswith("/favicon"):
            self.send_response(204); self.end_headers(); return
        self._send(PAGE.format(comment="", result="", mh=MH_THR, sp=SP_THR))

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        data = parse_qs(self.rfile.read(n).decode("utf-8"))
        comment = data.get("comment", [""])[0]
        self._send(PAGE.format(comment=html.escape(comment),
                               result=result_html(comment), mh=MH_THR, sp=SP_THR))

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    print(f"[demo] open  http://127.0.0.1:{PORT}   (Ctrl+C to stop)")
    HTTPServer(("127.0.0.1", PORT), H).serve_forever()
