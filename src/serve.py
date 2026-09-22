"""FastAPI server: send a question, get SQL back (and optionally its result).

    uvicorn serve:app --app-dir src --port 8000

    POST /sql   {"question": "...", "db_id": "concert_singer", "execute": true}
    POST /sql   {"question": "...", "schema": "CREATE TABLE t (a INTEGER, b VARCHAR)"}
    GET  /databases        Spider dev databases available for db_id / execute
    GET  /schema/{db_id}   the schema text the model is given for that database
    GET  /examples/{db_id} Spider dev questions for that database, with gold SQL
    GET  /metrics          the offline Spider numbers, so callers see how far to trust it
    GET  /                 a small web page over the same endpoints

If the question is a Spider dev question for that database, the response also
carries the gold SQL and whether the prediction's result matches it.

Returns the generated SQL itself, not just an answer, so a caller can audit
what would run. On Spider dev this model is right ~10% of the time; see
/metrics before trusting its output.
"""
import csv
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from metrics import results_match, run_sql
from nl2sql import NL2SQL
from spider_data import ROOT, SPIDER_DIR, _tables, load_dev, schema_text

_model = None


@asynccontextmanager
async def lifespan(app):
    model()  # load weights once at startup, so the first request isn't slow
    yield


app = FastAPI(title="rosette-spider: NL to SQL", lifespan=lifespan)


def model() -> NL2SQL:
    global _model
    if _model is None:
        _model = NL2SQL()
    return _model


_GOLD = None


def gold_for(db_id, question):
    global _GOLD
    if _GOLD is None:
        _GOLD = {}
        if (SPIDER_DIR / "dev.json").exists():
            for x in load_dev():
                _GOLD.setdefault(x["db_id"], {})[x["question"].strip().lower()] = x["query"]
    return _GOLD.get(db_id, {}).get(question.strip().lower())


def _require_db(db_id):
    if db_id not in _tables() or not (SPIDER_DIR / "database" / db_id).exists():
        raise HTTPException(status_code=404, detail=f"unknown database '{db_id}'; see /databases")


class Query(BaseModel):
    question: str
    db_id: str | None = None
    # "schema" in the JSON body; the attribute name differs because
    # `schema` shadows a BaseModel attribute
    schema_sql: str | None = Field(default=None, alias="schema")
    execute: bool = False


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/databases")
def databases():
    if not (SPIDER_DIR / "tables.json").exists():
        return {"databases": {}, "note": "Spider data not downloaded; run `python -c 'import spider_data; spider_data.download()'`"}
    return {"databases": {db: t["table_names_original"] for db, t in sorted(_tables().items())
                          if (SPIDER_DIR / "database" / db).exists()}}


@app.get("/schema/{db_id}")
def schema(db_id: str):
    _require_db(db_id)
    return {"db_id": db_id, "schema": schema_text(db_id)}


@app.get("/examples/{db_id}")
def examples(db_id: str, n: int = 10):
    _require_db(db_id)
    return {"db_id": db_id, "examples": [{"question": x["question"], "gold_sql": x["query"]}
                                         for x in load_dev() if x["db_id"] == db_id][:n]}


@app.post("/sql")
def sql(q: Query):
    if q.db_id is not None:
        _require_db(q.db_id)
    try:
        result = model().ask(q.question, db_id=q.db_id, schema=q.schema_sql, execute=q.execute)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    gold = gold_for(q.db_id, q.question) if q.db_id else None
    if gold:
        gold_rows, _ = run_sql(q.db_id, gold)
        pred_rows, _ = run_sql(q.db_id, result["sql"])
        result["gold_sql"] = gold
        result["gold_rows"] = gold_rows[:50]
        result["matches_gold"] = results_match(gold, gold_rows, pred_rows)
    return result


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/metrics")
def metrics():
    path = ROOT / "artifacts" / "accuracy.csv"
    if not path.exists():
        return {"note": "no evaluation results yet; run the notebook"}
    with open(path) as f:
        return {"spider_dev_execution_accuracy": list(csv.DictReader(f))}
