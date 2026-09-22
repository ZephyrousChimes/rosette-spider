"""Schema linking: pick the tables relevant to a question and prune the
schema to just those, before handing it to the generation model.

Named prior art: RESDSQL (Li et al., AAAI 2023) does this as a trained
ranker. This is the simple version -- no training, pure lexical overlap
between the question's words and each table's own name plus its column
names -- because our earlier oracle-schema probe (giving the model only the
gold query's tables) already showed a real, if modest, gain: 4/24 -> 5/24 on
hand-written questions. A trained linker is the natural upgrade if this
heuristic version helps.

A table is kept if it's among the top-k lexical-overlap scorers. An earlier
version also pulled in any table connected by a foreign key to a kept one,
to catch join/bridge tables the question never names (e.g. "students who
have pets" needs Has_Pet, named by neither word) -- but on a schema where
every table is FK-connected to every other (car_1: 6 tables, one FK chain),
that pull-in cascaded through the *entire* connected component in a single
pass and kept all 6 tables regardless of the question, defeating pruning
entirely. Dropped rather than fixed properly (would need shortest-path-
between-kept-tables logic, not "anything touching a kept table"), so this
version can genuinely miss an unnamed bridge table -- a documented
limitation, and exactly the gap a trained linker like RESDSQL closes.
"""
import re
from collections import defaultdict

from spider_data import _tables, schema_text

_STOP = {"the", "a", "an", "of", "in", "is", "are", "was", "were", "for", "to", "and", "or",
         "what", "which", "who", "how", "many", "list", "show", "find", "all", "each",
         "with", "that", "have", "has", "do", "does"}


def _stem(w: str) -> str:
    # naive plural stripping ("singers" -> "singer"), not real stemming --
    # enough to fix the exact-match miss without pulling in a dependency
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("es") and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def _words(s: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"[a-z]+", s.lower()) if w not in _STOP and len(w) > 2}


def _camel_split(s: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", " ", s).replace("_", " ")


def relevant_tables(db_id: str, question: str, extra: int = 1) -> set[str]:
    """Returns the set of table names (original casing) judged relevant:
    every table with nonzero lexical overlap with the question, plus `extra`
    more of the next-highest scorers as a safety margin (score-0 tables
    might still be needed, e.g. via an unnamed join table)."""
    t = _tables()[db_id]
    q_words = _words(question)
    col_words_by_table: dict[int, set[str]] = defaultdict(set)
    for ti, col in t["column_names_original"]:
        if ti >= 0:
            col_words_by_table[ti] |= _words(_camel_split(col))

    scores = []
    for i, name in enumerate(t["table_names_original"]):
        table_words = _words(_camel_split(name))
        score = len(q_words & table_words) * 2 + len(q_words & col_words_by_table[i])
        scores.append((score, i, name))
    scores.sort(reverse=True)

    n_positive = sum(1 for s, _, _ in scores if s > 0)
    top_n = max(1, min(len(scores), n_positive + extra))  # capped: always a real prune when n_positive < len(scores)
    kept = {i for _, i, _ in scores[:top_n]}
    return {t["table_names_original"][i] for i in kept}


def pruned_schema_text(db_id: str, question: str) -> str:
    keep = relevant_tables(db_id, question)
    parts = schema_text(db_id).split("; ")
    pruned = [p for p in parts if p.split()[2] in keep]
    return "; ".join(pruned) if pruned else schema_text(db_id)  # never prune to nothing
