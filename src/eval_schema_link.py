"""Does pruning the schema with schema_link.py before generation help the
off-the-shelf fine-tuned checkpoint? Reuses the already-computed baseline
(artifacts/spider_records.jsonl from spider_eval.py, full unpruned schema)
and generates only the pruned-schema predictions fresh, so this runs
independently of -- and without competing for GPU time with -- the
finetune.py experiment.

Paired comparison (same questions, same model, only the schema differs), so
McNemar's exact test on the discordant pairs is the right significance test,
same as spider_eval.py uses between strategies.
"""
import json
import time

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from metrics import mcnemar_exact, results_match, run_sql, wilson
from schema_link import pruned_schema_text
from spider_data import ROOT
from strategies import FINETUNED_MODEL, TOKENIZER_FOR

BASELINE = ROOT / "artifacts" / "spider_records.jsonl"


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


def main():
    baseline = [json.loads(l) for l in BASELINE.read_text().splitlines()]
    print(f"loaded {len(baseline)} baseline records from {BASELINE.name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    tok = T5Tokenizer.from_pretrained(TOKENIZER_FOR[FINETUNED_MODEL])
    model = T5ForConditionalGeneration.from_pretrained(FINETUNED_MODEL).to(device).eval()

    prompts = ["tables:\n" + pruned_schema_text(r["db_id"], r["question"]) + "\nquery for:" + r["question"]
               for r in baseline]
    t0 = time.time()
    preds = generate_batch(tok, model, prompts, device)
    print(f"generated {len(preds)} in {time.time() - t0:.0f}s")

    records = []
    for r, pred in zip(baseline, preds):
        gold_rows, gerr = run_sql(r["db_id"], r["gold_sql"])
        assert gerr is None, gerr
        pred_rows, err = run_sql(r["db_id"], pred)
        records.append({
            "db_id": r["db_id"], "question": r["question"], "gold_sql": r["gold_sql"], "pred_sql": pred,
            "seen_in_ft_training": r["seen_in_ft_training"],
            "correct": results_match(r["gold_sql"], gold_rows, pred_rows), "exec_error": err,
            "baseline_correct": r["fine_tuned"]["correct"],
        })

    for subset_name, rs in [("all", records), ("unseen", [r for r in records if not r["seen_in_ft_training"]])]:
        n = len(rs)
        k_pruned = sum(r["correct"] for r in rs)
        k_base = sum(r["baseline_correct"] for r in rs)
        lo, hi = wilson(k_pruned, n)
        blo, bhi = wilson(k_base, n)
        n10, n01, p = mcnemar_exact([r["correct"] for r in rs], [r["baseline_correct"] for r in rs])
        print(f"\n=== {subset_name} (n={n}) ===")
        print(f"  baseline (full schema):   {k_base}/{n} = {k_base / n:.1%}  95% CI [{blo:.1%}, {bhi:.1%}]")
        print(f"  pruned schema:            {k_pruned}/{n} = {k_pruned / n:.1%}  95% CI [{lo:.1%}, {hi:.1%}]")
        print(f"  McNemar: pruned-only-right={n10}  baseline-only-right={n01}  p={p:.3g}")

    out = ROOT / "artifacts" / "records_schema_link.jsonl"
    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
