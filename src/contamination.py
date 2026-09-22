"""Which Spider dev questions did the fine-tuned checkpoint see in training?

cssupport/t5-small-awesome-text-to-sql was trained on b-mc2/sql-create-context
and Clinton/Text-to-sql-v1 (per its model card). sql-create-context is built
from WikiSQL *and Spider*, so a dev question can sit verbatim in its training
data -- in which case scoring it measures recall, not generalization.

This streams both training sets (the second is ~630MB, never written to disk),
matches dev questions after lowercasing and whitespace/punctuation
normalization, and writes the matched dev indices to artifacts/. Exact match
only: paraphrased leakage is not caught, so "unseen" is an upper bound on
truly unseen.
"""
import json
import re
import urllib.request

from spider_data import ROOT, load_dev

SCC_URL = "https://huggingface.co/datasets/b-mc2/sql-create-context/resolve/main/sql_create_context_v4.json"
T2S_URL = "https://huggingface.co/datasets/Clinton/Text-to-sql-v1/resolve/main/texttosqlv2.jsonl"
SEEN_PATH = ROOT / "artifacts" / "spider_dev_seen.json"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip(" ?.")


def compute(out=SEEN_PATH) -> dict:
    dev = load_dev()
    by_q: dict[str, list[int]] = {}
    for i, x in enumerate(dev):
        by_q.setdefault(norm(x["question"]), []).append(i)

    seen: dict[int, set[str]] = {}
    with urllib.request.urlopen(SCC_URL) as r:
        for row in json.load(r):
            for i in by_q.get(norm(row["question"]), []):
                seen.setdefault(i, set()).add("sql-create-context")
    with urllib.request.urlopen(T2S_URL) as r:
        for line in r:
            row = json.loads(line)
            for i in by_q.get(norm(row["instruction"]), []):
                seen.setdefault(i, set()).add(f"Text-to-sql-v1/{row.get('source')}")

    result = {str(i): sorted(s) for i, s in sorted(seen.items())}
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"{len(seen)}/{len(dev)} dev questions appear verbatim in the fine-tuned model's training data")
    return result


def load() -> set[int]:
    return {int(i) for i in json.loads(SEEN_PATH.read_text())}
