"""Evaluate a fine-tuned checkpoint (from finetune.py) on Spider dev, using
the same execution-accuracy harness and reporting as spider_eval.py, so
before/after numbers are directly comparable to the original cssupport
checkpoint's results in artifacts/accuracy.csv.

Also re-runs the hand-written clause ladder from the serve.py probing session
(bare COUNT, WHERE, ORDER BY, JOIN, "mention" vs "names" phrasing) as a
targeted regression test -- the natural Spider distribution mixes clause
types together, so the ladder is what actually isolates whether each
specific failure got fixed.
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from finetune import prompt_for
from metrics import mcnemar_exact, results_match, run_sql, wilson
import spider_data
from spider_data import ROOT, load_dev

LADDER = [
    # (db_id, question, gold_sql, label)
    ("concert_singer", "How many singers are there?", "SELECT count(*) FROM singer", "bare COUNT"),
    ("concert_singer", "How many singers are older than 30?", "SELECT count(*) FROM singer WHERE Age > 30", "COUNT + WHERE"),
    ("concert_singer", "What are the names of singers older than 30?", "SELECT Name FROM singer WHERE Age > 30", "SELECT col + WHERE"),
    ("concert_singer", "Mention all singers that are older than 45.", "SELECT Name FROM singer WHERE Age > 45", "SELECT col + WHERE, vague phrasing"),
    ("concert_singer", "What are the names of singers, ordered by age?", "SELECT Name FROM singer ORDER BY Age", "ORDER BY"),
    ("concert_singer", "What is the name of the oldest singer?", "SELECT Name FROM singer ORDER BY Age DESC LIMIT 1", "ORDER BY + LIMIT"),
    ("concert_singer", "How many singers are there of each country?", "SELECT Country, count(*) FROM singer GROUP BY Country", "GROUP BY"),
    ("concert_singer", "What are the names of stadiums that have had a concert?",
     "SELECT DISTINCT T1.Name FROM stadium AS T1 JOIN concert AS T2 ON T1.Stadium_ID = T2.Stadium_ID", "JOIN"),
    ("pets_1", "How many students are there?", "SELECT count(*) FROM Student", "bare COUNT (2nd db)"),
    ("pets_1", "Mention all students older than 20.", "SELECT Fname FROM Student WHERE Age > 20", "SELECT col + WHERE, vague phrasing (2nd db)"),
]


def load_model(model_dir, device):
    tok = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir).to(device).eval()
    return tok, model


def generate_batch(tok, model, prompts, device, batch_size=32, max_new_tokens=128):
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out = [None] * len(prompts)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, num_beams=1, do_sample=False)
        for i, g in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
            out[i] = g.strip()
    return out


def run_ladder(tok, model, device):
    print("\n=== clause ladder ===")
    ok_count = 0
    for db, q, gold, label in LADDER:
        prompt = prompt_for({"db_id": db, "question": q})
        pred = generate_batch(tok, model, [prompt], device)[0]
        gold_rows, gerr = run_sql(db, gold)
        assert gerr is None, gerr
        pred_rows, err = run_sql(db, pred)
        ok = results_match(gold, gold_rows, pred_rows)
        ok_count += ok
        print(f"{'OK ' if ok else 'ERR' if err else 'BAD'} [{label}] {q}")
        print(f"      pred: {pred}" + (f"  -> {err}" if err else ""))
    print(f"\nladder: {ok_count}/{len(LADDER)}")
    return ok_count, len(LADDER)


def run_dev(tok, model, device, seen, limit=None):
    spider_data.download()  # in-process; see the note in finetune.py
    dev = load_dev()
    if limit:
        dev = dev[:limit]
    prompts = [prompt_for(x) for x in dev]
    preds = generate_batch(tok, model, prompts, device)
    records = []
    for x, pred in zip(dev, preds):
        gold_rows, gerr = run_sql(x["db_id"], x["query"])
        assert gerr is None, gerr
        pred_rows, err = run_sql(x["db_id"], pred)
        records.append({
            "db_id": x["db_id"], "question": x["question"], "gold_sql": x["query"], "pred_sql": pred,
            "seen_in_ft_training": x.get("question", "") in seen,
            "correct": results_match(x["query"], gold_rows, pred_rows), "exec_error": err,
        })
    return records


def report(records, label):
    print(f"\n=== {label}: Spider dev, n={len(records)} ===")
    k = sum(r["correct"] for r in records)
    n = len(records)
    lo, hi = wilson(k, n)
    err = sum(r["exec_error"] is not None for r in records) / n
    print(f"  exec_acc={k / n:.1%}  95% CI [{lo:.1%}, {hi:.1%}]  invalid SQL={err:.1%}")
    return {"n": n, "exec_acc": k / n, "ci95": [lo, hi], "invalid_sql": err}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--dev_limit", type=int, default=None)
    args = ap.parse_args()
    label = args.label or Path(args.model_dir).name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| model:", args.model_dir)

    tok, model = load_model(args.model_dir, device)
    ladder_ok, ladder_n = run_ladder(tok, model, device)
    records = run_dev(tok, model, device, seen=set(), limit=args.dev_limit)
    summary = report(records, label)
    summary["ladder"] = f"{ladder_ok}/{ladder_n}"

    out = ROOT / "artifacts" / f"eval_{label}.json"
    out.write_text(json.dumps(summary, indent=1))
    with open(ROOT / "artifacts" / f"records_{label}.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\nsaved {out}")
