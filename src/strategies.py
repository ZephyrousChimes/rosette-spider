"""The three NL->SQL strategies, with identical schema text for each.

  zero_shot   : google/flan-t5-small, schema + question only
  rag_fewshot : same model + top-3 (question, SQL) pairs retrieved by TF-IDF
                from Spider train (7000 pairs; its databases are disjoint from
                dev, so examples carry SQL structure, never the target schema)
  fine_tuned  : cssupport/t5-small-awesome-text-to-sql, in its training format

Generation is greedy, batched, and cached per strategy in artifacts/ so a
rerun (or a Kaggle session restart) doesn't repeat it.
"""
import json
import time

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

from spider_data import ROOT, schema_text

BASE_MODEL = "google/flan-t5-small"
FINETUNED_MODEL = "cssupport/t5-small-awesome-text-to-sql"
# the fine-tuned checkpoint ships without tokenizer files; it was trained from
# t5-small, so its vocabulary is t5-small's
TOKENIZER_FOR = {BASE_MODEL: BASE_MODEL, FINETUNED_MODEL: "t5-small"}
MODEL_FOR = {"zero_shot": BASE_MODEL, "rag_fewshot": BASE_MODEL, "fine_tuned": FINETUNED_MODEL}


class Retriever:
    def __init__(self, bank):
        self.bank = bank
        self.vec = TfidfVectorizer().fit([b["question"] for b in bank])
        self.mat = self.vec.transform([b["question"] for b in bank])

    def top_k(self, question, k=3):
        sims = cosine_similarity(self.vec.transform([question]), self.mat)[0]
        return [self.bank[i] for i in sims.argsort()[::-1][:k]]


def zero_shot_prompt(x):
    return (
        "Translate the question to a SQL query given this schema.\n"
        f"Schema:\n{schema_text(x['db_id'])}\n"
        f"Question: {x['question']}\nSQL:"
    )


def rag_prompt(x, retriever):
    examples = "\n".join(f"Q: {e['question']}\nSQL: {e['query']}" for e in retriever.top_k(x["question"]))
    return (
        "Translate the question to a SQL query given this schema. "
        "Here are similar examples.\n"
        f"Schema:\n{schema_text(x['db_id'])}\n"
        f"{examples}\n"
        f"Q: {x['question']}\nSQL:"
    )


def fine_tuned_prompt(x):
    return "tables:\n" + schema_text(x["db_id"]) + "\nquery for:" + x["question"]


def build_prompts(dev, retriever):
    return {
        "zero_shot": [zero_shot_prompt(x) for x in dev],
        "rag_fewshot": [rag_prompt(x, retriever) for x in dev],
        "fine_tuned": [fine_tuned_prompt(x) for x in dev],
    }


def generate(model_name, prompts, device, batch_size=32, max_new_tokens=256):
    tok = T5Tokenizer.from_pretrained(TOKENIZER_FOR[model_name])
    # fp32 on purpose: T5 activations overflow in fp16 and produce NaNs
    model = T5ForConditionalGeneration.from_pretrained(model_name, use_safetensors=False).to(device).eval()
    # no truncation: ~8% of prompts exceed 512 tokens, and truncating from the
    # right would cut off the question. T5's relative position bias accepts
    # longer inputs.
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out = [None] * len(prompts)
    for s in tqdm(range(0, len(order), batch_size), desc=model_name.split("/")[-1]):
        idx = order[s:s + batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, num_beams=1, do_sample=False)
        for i, g in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
            out[i] = g.strip()
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def run_or_load(name, prompts, device, batch_size=32):
    """Generate for one strategy, or load cached predictions if they exist for
    exactly these prompts."""
    path = ROOT / "artifacts" / f"preds_{name}.jsonl"
    if path.exists():
        cached = [json.loads(l) for l in path.read_text().splitlines()]
        if [c["prompt"] for c in cached] == prompts:
            print(f"{name}: loaded {len(cached)} cached predictions from {path.name}")
            return [c["pred_sql"] for c in cached]
    t0 = time.time()
    preds = generate(MODEL_FOR[name], prompts, device, batch_size)
    print(f"{name}: generated {len(preds)} in {time.time() - t0:.0f}s")
    path.parent.mkdir(exist_ok=True)
    path.write_text("".join(json.dumps({"prompt": p, "pred_sql": s}) + "\n" for p, s in zip(prompts, preds)))
    return preds
