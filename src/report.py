"""Tables for the notebook: accuracy with CIs per subset, paired tests,
breakdown by query structure, and a failure taxonomy."""
import re
from collections import Counter

import pandas as pd

from metrics import mcnemar_exact, wilson

STRATEGIES = ["zero_shot", "rag_fewshot", "fine_tuned"]


def subsets(records):
    return {
        "all": records,
        "unseen by fine_tuned": [r for r in records if not r["seen_in_ft_training"]],
        "seen by fine_tuned": [r for r in records if r["seen_in_ft_training"]],
    }


def accuracy_table(records):
    rows = []
    for sname, rs in subsets(records).items():
        n = len(rs)
        for name in STRATEGIES:
            k = sum(r[name]["correct"] for r in rs)
            lo, hi = wilson(k, n)
            rows.append({
                "subset": sname, "n": n, "strategy": name,
                "exec_acc": k / n if n else 0.0, "ci95_lo": lo, "ci95_hi": hi,
                "invalid_sql": sum(r[name]["exec_error"] is not None for r in rs) / n if n else 0.0,
                "old_set_metric": sum(r[name]["correct_set_metric"] for r in rs) / n if n else 0.0,
            })
    return pd.DataFrame(rows)


def paired_tests(records):
    rows = []
    for sname, rs in subsets(records).items():
        for a, b in [("fine_tuned", "rag_fewshot"), ("fine_tuned", "zero_shot"), ("rag_fewshot", "zero_shot")]:
            n10, n01, p = mcnemar_exact([r[a]["correct"] for r in rs], [r[b]["correct"] for r in rs])
            rows.append({"subset": sname, "a": a, "b": b, "only_a_right": n10, "only_b_right": n01, "p_value": p})
    return pd.DataFrame(rows)


def structure_table(records):
    rows = []
    for sname, rs in list(subsets(records).items())[1:]:
        for st in ["single-table", "join", "nested/set-op"]:
            g = [r for r in rs if r["structure"] == st]
            row = {"subset": sname, "structure": st, "n": len(g)}
            for name in STRATEGIES:
                row[name] = sum(r[name]["correct"] for r in g) / len(g) if g else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def failure_kind(r, name):
    s = r[name]
    if s["correct"]:
        return "correct"
    err = s["exec_error"]
    if err is None:
        return "runs, wrong result"
    if "interrupted" in err:
        return "timeout"
    if re.search(r"no such (column|table)", err):
        return "unknown table/column"
    if "syntax error" in err or "incomplete input" in err:
        return "syntax error"
    return "other exec error"


def failure_table(records):
    return pd.DataFrame(
        {name: Counter(failure_kind(r, name) for r in records) for name in STRATEGIES}
    ).fillna(0).astype(int)
