"""Execution accuracy and the statistics around it.

Result comparison: ordered list comparison when the gold query has ORDER BY,
multiset (Counter) comparison otherwise. The older `set(rows)` comparison --
which ignores row order and collapses duplicates -- is computed alongside to
show how much it over-credits.

Not the official Spider evaluator: column order is not permuted and values are
compared exactly, so this is slightly stricter than test-suite EX.
"""
import math
import re
import sqlite3
import time
from collections import Counter

from spider_data import db_path

QUERY_TIMEOUT_S = 5.0


def run_sql(db_id, sql, timeout_s=QUERY_TIMEOUT_S):
    """Read-only execution with a wall-clock timeout (a generated cartesian
    join can otherwise run for minutes)."""
    conn = sqlite3.connect(f"file:{db_path(db_id)}?mode=ro", uri=True)
    conn.text_factory = lambda b: b.decode(errors="ignore")
    deadline = time.monotonic() + timeout_s
    conn.set_progress_handler(lambda: time.monotonic() > deadline, 10_000)
    try:
        return conn.execute(sql).fetchall(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    finally:
        conn.close()


def has_order_by(sql):
    return re.search(r"\border\s+by\b", sql, re.I) is not None


def results_match(gold_sql, gold_rows, pred_rows):
    if pred_rows is None:
        return False
    if has_order_by(gold_sql):
        return pred_rows == gold_rows
    return Counter(pred_rows) == Counter(gold_rows)


def structure(sql):
    s = sql.lower()
    if s.count("select") > 1:
        return "nested/set-op"
    if " join " in s:
        return "join"
    return "single-table"


def score(records, name, preds, gold):
    for r, pred, (gold_rows, _) in zip(records, preds, gold):
        rows, err = run_sql(r["db_id"], pred)
        r[name] = {
            "pred_sql": pred,
            "exec_error": err,
            "correct": results_match(r["gold_sql"], gold_rows, rows),
            "correct_set_metric": rows is not None and set(rows) == set(gold_rows),
        }


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(a, b):
    """Two-sided exact McNemar on paired booleans: a binomial test on the
    discordant pairs. The strategies answer the same questions, so this is the
    right test -- not two independent proportions."""
    n10 = sum(x and not y for x, y in zip(a, b))
    n01 = sum(y and not x for x, y in zip(a, b))
    n = n10 + n01
    if n == 0:
        return n10, n01, 1.0
    p = 2 * sum(math.comb(n, i) for i in range(min(n10, n01) + 1)) / 2 ** n
    return n10, n01, min(1.0, p)
