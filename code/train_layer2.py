"""Train the Layer-2 multi-label tag classifier from the DAPT-adapted base.

Architecture: one RoBERTa encoder + 10 sigmoid heads (multi-label), initialised from
`models/twitter-roberta-dapt` -- the same domain-adapted base that lifted the Layer-1 mh model
(gate F1 0.85 -> 0.89). A single shared encoder rather than 10 binary models because the tags
co-occur heavily (anxiety+depression, body_image+anxiety) and shared representations transfer;
it is also ~10x cheaper to train and serve.

Loss: BCEWithLogits with TWO modifications
  * per-element MASK -- silver label -1 (rule abstained) contributes no gradient at all
  * per-tag pos_weight -- tags run 0.5%-15% positive; without this the rare heads collapse to 0

Gold rows (hand-labeled) can be mixed in with a higher sample weight via --gold.

Usage:
  .venv/Scripts/python.exe code/train_layer2.py [--silver F] [--gold F] [--epochs N]
                                                [--gold-weight W] [--out DIR]
"""
import csv, os, random, sys
import numpy as np, torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, Trainer,
                          TrainingArguments, set_seed)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layer2_lexicon import TAGS
csv.field_size_limit(2**31 - 1)

BASE = "models/twitter-roberta-dapt" if os.path.isdir("models/twitter-roberta-dapt") \
       else "cardiffnlp/twitter-roberta-base"
MAX_LEN = 256
set_seed(1)

def arg(flag, default=None, cast=None):
    if flag not in sys.argv:
        return default
    return (cast or str)(sys.argv[sys.argv.index(flag) + 1])

def read_labeled(path, tags=TAGS):
    """Read a csv with a `text` column + one column per tag holding 1/0/-1 (or blank = -1)."""
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            t = (r.get("text") or "").strip()
            if not t:
                continue
            y = []
            for k in tags:
                v = (r.get(k) or "").strip()
                y.append(float(v) if v in ("0", "1") else -1.0)
            rows.append((t, y))
    return rows

class MultiLabelDS(Dataset):
    def __init__(self, rows, tok, weights=None):
        self.rows, self.tok = rows, tok
        self.w = weights or [1.0] * len(rows)
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        t, y = self.rows[i]
        e = self.tok(t, truncation=True, max_length=MAX_LEN)
        e["labels"] = y
        e["sample_weight"] = self.w[i]
        return e

class Collator:
    def __init__(self, tok): self.tok = tok
    def __call__(self, feats):
        labels = torch.tensor([f.pop("labels") for f in feats], dtype=torch.float)
        sw = torch.tensor([f.pop("sample_weight") for f in feats], dtype=torch.float)
        batch = self.tok.pad(feats, return_tensors="pt")
        batch["labels"] = labels
        batch["sample_weight"] = sw
        return batch

class MaskedBCETrainer(Trainer):
    """BCE that ignores abstained (-1) elements and reweights rare positives."""
    def __init__(self, *a, pos_weight=None, **kw):
        super().__init__(*a, **kw)
        self.pos_weight = pos_weight
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        sw = inputs.pop("sample_weight", None)
        logits = model(**inputs).logits
        mask = (labels >= 0).float()                 # -1 == abstain -> no gradient
        tgt = labels.clamp(min=0)
        pw = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        per = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, tgt, reduction="none", pos_weight=pw)
        per = per * mask
        if sw is not None:
            per = per * sw.to(per.device).unsqueeze(1)
        loss = per.sum() / mask.sum().clamp(min=1.0)
        return (loss, {"logits": logits}) if return_outputs else loss

def main():
    silver = arg("--silver", "data/layer2/silver.csv")
    gold = arg("--gold")
    epochs = arg("--epochs", 2, int)
    gw = arg("--gold-weight", 8.0, float)
    out = arg("--out", "models/layer2_tags")
    lr = arg("--lr", 2e-5, float)
    bs = arg("--batch", 16, int)
    # --init lets stage 2 start from the silver-trained checkpoint instead of the DAPT base, so
    # gold fine-tuning can correct the decision boundary the rules taught. 204 gold rows mixed
    # into 121k silver rows barely moves it, however high the loss weight.
    init = arg("--init")
    no_silver = "--no-silver" in sys.argv

    tok = AutoTokenizer.from_pretrained(init or BASE)
    rows = [] if no_silver else read_labeled(silver)
    w = [1.0] * len(rows)
    print(f"[l2] init={init or BASE}")
    print(f"[l2] silver rows: {len(rows):,}" + ("  (SKIPPED --no-silver)" if no_silver else ""))
    if gold:
        for p in gold.split(","):
            if not os.path.exists(p):
                print(f"[l2] WARNING: gold file {p} not found -- skipping", flush=True)
                continue
            g = read_labeled(p)
            rows += g; w += [gw] * len(g)
            print(f"[l2] + gold {p}: {len(g):,} rows at weight {gw}")

    idx = list(range(len(rows)))
    random.Random(1).shuffle(idx)
    rows = [rows[i] for i in idx]; w = [w[i] for i in idx]
    cut = max(1, int(len(rows) * 0.05))
    val_rows, val_w = rows[:cut], w[:cut]
    tr_rows, tr_w = rows[cut:], w[cut:]

    # pos_weight per tag = (#neg / #pos) over the *observed* (non-abstained) training labels,
    # capped so the rarest tags don't dominate the gradient entirely.
    Y = np.array([r[1] for r in tr_rows])
    pw = []
    for j, t in enumerate(TAGS):
        col = Y[:, j]
        p, ng = float((col == 1).sum()), float((col == 0).sum())
        pw.append(min(ng / max(p, 1.0), 20.0))
    print(f"[l2] pos_weight: " + ", ".join(f"{t}={v:.1f}" for t, v in zip(TAGS, pw)))
    print(f"[l2] train {len(tr_rows):,} / val {len(val_rows):,}", flush=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        init or BASE, num_labels=len(TAGS), problem_type="multi_label_classification")
    args = TrainingArguments(
        output_dir="./results_layer2", num_train_epochs=epochs,
        per_device_train_batch_size=bs, per_device_eval_batch_size=64,
        learning_rate=lr, weight_decay=0.01, warmup_ratio=0.06,
        fp16=torch.cuda.is_available(), logging_steps=100,
        eval_strategy="epoch", save_strategy="no", report_to=[],
        remove_unused_columns=False,
    )
    trainer = MaskedBCETrainer(
        model=model, args=args,
        train_dataset=MultiLabelDS(tr_rows, tok, tr_w),
        eval_dataset=MultiLabelDS(val_rows, tok, val_w),
        data_collator=Collator(tok),
        pos_weight=torch.tensor(pw, dtype=torch.float),
    )
    trainer.train()
    os.makedirs(out, exist_ok=True)
    trainer.save_model(out); tok.save_pretrained(out)
    with open(os.path.join(out, "tags.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(TAGS))
    print(f"[l2] DONE -> {out}", flush=True)

if __name__ == "__main__":
    main()
