"""Domain-adaptive pretraining (DAPT): continue MLM pretraining of twitter-roberta-base on the
raw matched-corpus text (no labels), so the base learns this corpus's vocabulary/register before
we fine-tune the mh classifier from it.

Output: models/twitter-roberta-dapt  (a domain-adapted base for downstream fine-tuning).
Run:    .venv/Scripts/python.exe code/dapt_pretrain.py
"""
import csv, glob, os, random, torch
from transformers import (AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments,
                          DataCollatorForLanguageModeling)
csv.field_size_limit(2**31 - 1)

BASE = "cardiffnlp/twitter-roberta-base"
OUT = "models/twitter-roberta-dapt"
MAX_LEN = 256              # MLM doesn't need full 512 to learn domain vocab; 256 is ~2x faster
MLM_PROB = 0.15
EPOCHS = 1
BATCH = 16
SUBSET = 200_000          # ~2.3h on the 6GB card; full 574k would be ~6.8h. subset is plenty for DAPT.

MATCHED = "comments/2018-2022/MS_comments_2018_2022/MS_comments_2018_2022/matched"

print("[dapt] loading corpus text ...", flush=True)
texts = []
for f in sorted(glob.glob(MATCHED + "/*.csv")):
    with open(f, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            t = (r.get("text") or "").strip()
            if len(t) > 15:                     # skip near-empty comments
                texts.append(t)
random.seed(1)
if SUBSET and len(texts) > SUBSET:
    texts = random.sample(texts, SUBSET)
print(f"[dapt] {len(texts):,} comments for MLM (subset of full matched corpus)", flush=True)

tok = AutoTokenizer.from_pretrained(BASE)

class TextDS(torch.utils.data.Dataset):
    def __init__(self, texts): self.texts = texts
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        e = tok(self.texts[i], truncation=True, max_length=MAX_LEN)
        return {"input_ids": e["input_ids"], "attention_mask": e["attention_mask"]}

ds = TextDS(texts)
collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=True, mlm_probability=MLM_PROB)
model = AutoModelForMaskedLM.from_pretrained(BASE)

args = TrainingArguments(
    output_dir="./results_dapt",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH,
    fp16=torch.cuda.is_available(),
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_ratio=0.05,
    logging_steps=200,
    save_strategy="no",          # save once at the end via trainer.save_model
)
trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)
print("[dapt] starting MLM pretraining ...", flush=True)
trainer.train()
os.makedirs(OUT, exist_ok=True)
trainer.save_model(OUT)
tok.save_pretrained(OUT)
print(f"[dapt] DONE -> {OUT}", flush=True)
