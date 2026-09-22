"""Fine-tune t5-small on Spider's training split, in the same prompt format
as the off-the-shelf checkpoint, to fix the failures found in serve.py probes
(bare COUNT hallucinating a WHERE, ORDER BY, JOIN, and "mention/list" phrasing
picking the wrong column).

Stage-based: `--stage simple` trains on the join/nested/HAVING-free subset of
Spider train (3420 of 7000 examples) -- the "train small and simple first"
step. `--stage all` trains on everything. Each stage saves to its own
artifacts/model_<stage>/ directory and is evaluated with the same ladder and
Spider-dev harness used on the original checkpoint, so before/after is a fair
comparison.

Train databases are disjoint from the 20 dev databases (checked in
spider_data.verify-adjacent logic), so dev accuracy after fine-tuning still
measures generalization, not memorization.
"""
import argparse
import random
import re
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer

import spider_data
from spider_data import ROOT, load_train, schema_text

TOKENIZER_SOURCE = "t5-small"
BASE_CHECKPOINT = "t5-small"  # start from plain t5-small, not the buggy checkpoint --
                                # we want to see what OUR fine-tuning teaches it, not
                                # inherit cssupport's biases


def prompt_for(x):
    return "tables:\n" + schema_text(x["db_id"]) + "\nquery for:" + x["question"]


def is_simple(query):
    s = query.lower()
    return not (" join " in s or s.count("select") > 1 or "having" in s)


class SqlDataset(Dataset):
    def __init__(self, examples, tok, max_len=512, max_target_len=128):
        self.examples, self.tok, self.max_len, self.max_target_len = examples, tok, max_len, max_target_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        x = self.examples[i]
        return prompt_for(x), x["query"]

    def collate(self, batch):
        prompts, targets = zip(*batch)
        enc = self.tok(list(prompts), padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        lab = self.tok(text_target=list(targets), padding=True, truncation=True, max_length=self.max_target_len, return_tensors="pt")
        labels = lab.input_ids
        labels[labels == self.tok.pad_token_id] = -100  # ignored by the loss
        return enc.input_ids, enc.attention_mask, labels


def train(stage, epochs, batch_size, lr, seed, device):
    random.seed(seed)
    torch.manual_seed(seed)

    # download in-process (a separate `python spider_data.py` subprocess on
    # Kaggle exited 0 but left train_spider.json missing -- root cause
    # unconfirmed, possibly stdout buffering hiding a real failure; calling
    # the function directly here removes the cross-process boundary entirely)
    spider_data.download()
    assert (spider_data.SPIDER_DIR / "train_spider.json").exists(), \
        f"train_spider.json missing after download() in {spider_data.SPIDER_DIR}"

    data = load_train()
    if stage == "simple":
        data = [x for x in data if is_simple(x["query"])]
    print(f"stage={stage}: {len(data)} training examples")

    tok = T5Tokenizer.from_pretrained(TOKENIZER_SOURCE)
    model = T5ForConditionalGeneration.from_pretrained(BASE_CHECKPOINT).to(device)

    ds = SqlDataset(data, tok)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=ds.collate)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        t0, total_loss, n = time.time(), 0.0, 0
        for input_ids, attn, labels in dl:
            input_ids, attn, labels = input_ids.to(device), attn.to(device), labels.to(device)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            out.loss.backward()
            opt.step()
            opt.zero_grad()
            total_loss += out.loss.item() * input_ids.size(0)
            n += input_ids.size(0)
        print(f"  epoch {epoch + 1}/{epochs}  loss={total_loss / n:.4f}  ({time.time() - t0:.0f}s)")

    out_dir = ROOT / "artifacts" / f"model_{stage}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"saved to {out_dir}")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["simple", "all"], default="simple")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    train(args.stage, args.epochs, args.batch_size, args.lr, args.seed, device)
