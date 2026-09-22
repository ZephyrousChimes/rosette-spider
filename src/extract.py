"""Rule-based front end for two narrow, unambiguous intents that don't need
the model at all:

  1. "show me [all / first N] <table>"               -> SELECT * FROM t [LIMIT N]
  2. "show me [all] <table> [from/in/named/...] <V>"  -> SELECT * FROM t WHERE c = V

Why (1) needs its own path: see finetune.py's is_simple() and
mentor/round1b_rosette_spider_deep_dive.md Part 7b -- every LIMIT example in
Spider's training data also has an ORDER BY, so the model has no examples of
a bare "give me any N rows" and hallucinates a ranking it was never asked
for.

Why (2) is *grounded in the actual database* rather than keyword-matched:
a fixed list of trigger words ("from", "named", "whose", ...) is brittle --
it either misses phrasings ("show all singer from netherlands" silently
dropped its filter in an earlier version of this file) or overreaches into
questions that need real reasoning. Instead: after removing the trigger
verb, the matched table name, and ordinary stopwords, whatever words are
left over are checked against the *actual distinct values* in that table's
text columns. If a leftover word or short phrase is a real cell value
("Netherlands" really is a value of singer.Country), that's strong,
phrasing-independent evidence of what filter was meant -- "from Netherlands",
"whose country is Netherlands", and "in netherlands" all leave the same
leftover token and ground to the same column. This is a simplified form of
*value linking* / cell grounding, the same idea BRIDGE (Lin, Socher & Xiong,
2020) uses text-to-SQL, just without a trained retrieval model behind it.

Safety: any leftover words that *don't* ground to a real value, beyond a
small noise tolerance (filler words like "please"/"kindly"), abort
extraction and fall back to the model rather than silently ignoring a filter
we can't confidently resolve. Aggregate/ranking/comparison words
(how many, average, older than, ...) are disqualified up front, since those
need real reasoning this extractor doesn't attempt.
"""
import re
import sqlite3

from schema_link import _stem, _words
from spider_data import _tables, db_path

_TRIGGER_VERBS = {"show", "list", "mention", "give", "find", "display"}
_LEAD_WINDOW = 4  # trigger verb must appear within this many leading words --
                  # tolerates "please show me..." without also matching a
                  # trigger word that shows up incidentally, later, in an
                  # otherwise-unrelated question
_COUNT = re.compile(r"\b(\d+)\b")
_ORDINAL = re.compile(r"\b(first|top|last)\b", re.I)
# structural words that mean "this needs real reasoning" -- aggregation,
# ranking, or a comparison rather than an equality filter. Left in as an
# up-front bail-out; equality filters are instead resolved by grounding
# against real data below.
_DISQUALIFY = re.compile(
    r"\b(how\s+many|count|average|avg|sum|total|maximum|maxim|minimum|min|"
    r"most|least|each|per|group|older|younger|greater|less|more\s+than|"
    r"at\s+least|at\s+most|between|before|after|order(ed)?\s+by|"
    r"sort(ed)?|distinct|unique)\b", re.I
)
# extra stopwords on top of schema_link._words' own list -- trigger verbs,
# fillers, and relative-clause scaffolding ("whose", "named", "is") that
# carry no filter content once we've grounded the actual value
_EXTRA_STOP = {"show", "list", "mention", "give", "find", "display", "me", "us", "please", "kindly",
               "whose", "named", "called", "first", "top", "last", "from",
               # conversational fillers -- without these, a short common word
               # ("you", "can") can coincidentally equal a real short cell
               # value and ground to the wrong column (found on "can you
               # kindly show..." colliding with a song literally named "You")
               "can", "could", "would", "will", "you", "your", "i", "we", "they",
               "do", "does", "did", "here", "there", "now"}
_MIN_GROUND_LEN = 4  # a candidate phrase shorter than this is too likely to
                      # coincidentally collide with an unrelated short cell value

_TEXT_TYPES = {"text", "others"}


def _table_index(db_id: str, table: str) -> int:
    return _tables()[db_id]["table_names_original"].index(table)


def _text_columns(db_id: str, table: str) -> list[str]:
    t = _tables()[db_id]
    ti = _table_index(db_id, table)
    return [col for (tj, col), ty in zip(t["column_names_original"], t["column_types"])
            if tj == ti and ty in _TEXT_TYPES]


def _value_map(db_id: str, table: str, column: str) -> dict[str, str]:
    """lowercase cell value -> its real casing, for one column."""
    conn = sqlite3.connect(f"file:{db_path(db_id)}?mode=ro", uri=True)
    conn.text_factory = lambda b: b.decode(errors="ignore")
    try:
        rows = conn.execute(f'SELECT DISTINCT "{column}" FROM "{table}"').fetchall()
        return {str(r[0]).lower(): str(r[0]) for r in rows if r[0] is not None}
    except Exception:
        return {}
    finally:
        conn.close()


def _ground_value(db_id: str, table: str, leftover_tokens: list[str]) -> tuple[str, str] | None:
    """Try every contiguous n-gram (longest first) of the leftover raw
    tokens against every text column's real values. Returns (column, actual
    cell value) on a confident, unambiguous match, else None."""
    columns = _text_columns(db_id, table)
    value_maps = {c: _value_map(db_id, table, c) for c in columns}

    best = None  # (n_gram_length, column, value)
    for n in (3, 2, 1):
        for i in range(len(leftover_tokens) - n + 1):
            phrase = " ".join(leftover_tokens[i:i + n]).lower()
            if len(phrase) < _MIN_GROUND_LEN:
                continue
            hits = [(c, value_maps[c][phrase]) for c in columns if phrase in value_maps[c]]
            if len(hits) == 1:
                if best is None or n > best[0]:
                    best = (n, *hits[0])
            elif len(hits) > 1:
                return None  # ambiguous (matches two different columns) -- don't guess
        if best is not None:
            return best[1], best[2]
    return None


def try_extract(db_id: str, question: str) -> dict | None:
    """Returns {"sql": ...} for the two intents described in the module
    docstring, else None (caller falls back to the model)."""
    if _DISQUALIFY.search(question):
        return None
    lead_words = re.findall(r"[A-Za-z]+", question)[:_LEAD_WINDOW]
    if not any(w.lower() in _TRIGGER_VERBS for w in lead_words):
        return None

    tables = _tables()[db_id]["table_names_original"]
    q_words = _words(question)  # stemmed, schema_link's own stopwords already removed
    matches = [t for t in tables if _stem(t.lower()) in q_words or _stem(t.lower()) + "s" in q_words]
    if len(matches) != 1:
        return None
    table = matches[0]

    # raw (unstemmed) alphabetic tokens, for grounding against real cell
    # values -- stemming "Netherlands" would corrupt the literal string we
    # need to match, and \d+ is excluded here so a row-count number never
    # counts as leftover "content" below
    raw_tokens = re.findall(r"[A-Za-z]+", question)
    table_stem = _stem(table.lower())
    leftover = [w for w in raw_tokens
                if _stem(w.lower()) not in ({table_stem} | _EXTRA_STOP)
                and w.lower() not in _EXTRA_STOP
                and _words(w)]  # drops schema_link's own stopwords (a/the/of/is/...)

    grounded = _ground_value(db_id, table, leftover) if leftover else None
    if grounded is not None:
        column, value = grounded
        escaped = value.replace("'", "''")
        return {"sql": f"SELECT * FROM {table} WHERE \"{column}\" = '{escaped}'",
                "extracted_table": table, "extracted_filter": (column, value)}

    # no grounded value -- proceed as bare/unfiltered ONLY if there is truly
    # nothing left over. Any leftover word here has already survived the
    # stopword/filler removal above, so by construction it's real content --
    # tolerating a *count* of leftover words (an earlier version of this
    # file did that) let genuine unresolvable filters through silently, e.g.
    # "show all students from Bangalore" on a table that stores city codes,
    # not names: "Bangalore" never grounds, and a tolerant count treated it
    # as harmless noise and silently dropped the filter. Zero tolerance for
    # ungrounded content means: abstain, let the model try, instead of
    # guessing wrong silently.
    if leftover:
        return None

    n = _COUNT.search(question)
    limit = f" LIMIT {n.group(1)}" if n else (" LIMIT 5" if _ORDINAL.search(question) else "")
    return {"sql": f"SELECT * FROM {table}{limit}", "extracted_table": table}
