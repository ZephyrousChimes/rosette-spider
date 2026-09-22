"""Rule-based front end for one narrow, unambiguous intent: "show me [all /
first N] <table>" -- no filter, no ranking, no aggregate. Handled entirely
without the model.

Why this needs its own path rather than better training data: Spider's
training set contains 1,107 LIMIT examples and *every single one* also has
ORDER BY (almost always `GROUP BY ... ORDER BY count(*) DESC`, the
"most-common-X-per-group" template -- see finetune.py's is_simple() and the
training-data analysis in mentor/round1b_rosette_spider_deep_dive.md). A bare
"give me any N rows" has zero examples anywhere in Spider, train or dev --
it's not a sampling gap, the pattern genuinely doesn't occur in the
benchmark. So the model has learned "LIMIT implies GROUP BY + ORDER BY
count(*) DESC" as an exceptionless rule and applies it even to "show all
stadiums" (no LIMIT at all), hallucinating a ranking that was never asked
for. Rather than manufacture synthetic training data for a pattern the real
benchmark never tests, this intent is simple enough to extract directly:
find the table the question names, optionally find a row count, done.

Deliberately conservative: any filter/aggregate/ranking word in the question
(older, average, most, each, ...) aborts extraction and falls back to the
model, because at that point this is no longer the simple "give me rows"
intent.
"""
import re

from schema_link import _stem, _words
from spider_data import _tables

_TRIGGER = re.compile(
    r"^(show|list|mention|give|find|display)\b.*?\b(all|every|the|some|me)?\b", re.I
)
_COUNT = re.compile(r"\b(\d+)\b")
_ORDINAL_ALL = re.compile(r"\b(first|top|last)\b", re.I)
# any of these means the question wants a filter, aggregate, or ranking --
# not a bare "give me rows" request, so don't extract
_DISQUALIFY = re.compile(
    r"\b(how\s+many|count|average|avg|sum|total|maximum|maxim|minimum|min|"
    r"most|least|each|per|group|older|younger|greater|less|more\s+than|"
    r"at\s+least|at\s+most|equal|named|whose|that\s+(are|have|is)|"
    r"who\s+(have|are|is)|located|older|between|before|after|order(ed)?\s+by|"
    r"sort(ed)?|distinct|unique)\b", re.I
)


def try_extract(db_id: str, question: str) -> dict | None:
    """Returns {"sql": ...} if this question is confidently the bare
    "show/list N rows from table T" intent, else None (caller should fall
    back to the model)."""
    if _DISQUALIFY.search(question):
        return None
    if not _TRIGGER.match(question.strip()):
        return None

    q_words = _words(question)
    tables = _tables()[db_id]["table_names_original"]
    matches = [t for t in tables if _stem(t.lower()) in q_words or _stem(t.lower()) + "s" in q_words]
    # also allow a table whose stemmed name appears as a substring word
    # (handles multi-word/underscored names like singer_in_concert only
    # loosely -- fine, since a false negative here just falls back to the model)
    if len(matches) != 1:
        return None
    table = matches[0]

    n = _COUNT.search(question)
    limit = f" LIMIT {n.group(1)}" if n else (" LIMIT 5" if _ORDINAL_ALL.search(question) and not n else "")
    return {"sql": f"SELECT * FROM {table}{limit}", "extracted_table": table}
