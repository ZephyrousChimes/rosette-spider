"""FastAPI server: send a question, get SQL back (and optionally its result).

    uvicorn serve:app --app-dir src --port 8000

    POST /sql   {"question": "...", "db_id": "concert_singer", "execute": true}
    POST /sql   {"question": "...", "schema": "CREATE TABLE t (a INTEGER, b VARCHAR)"}
    GET  /databases        Spider dev databases available for db_id / execute
    GET  /metrics          the offline Spider numbers, so callers see how far to trust it

Returns the generated SQL itself, not just an answer, so a caller can audit
what would run. On Spider dev this model is right ~10% of the time; see
/metrics before trusting its output.
"""
import csv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from nl2sql import NL2SQL
from spider_data import ROOT, SPIDER_DIR, _tables

app = FastAPI(title="rosette-spider: NL to SQL")
_model = None


def model() -> NL2SQL:
    global _model
    if _model is None:
        _model = NL2SQL()
    return _model


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


@app.post("/sql")
def sql(q: Query):
    try:
        return model().ask(q.question, db_id=q.db_id, schema=q.schema_sql, execute=q.execute)
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=f"{type(e).__name__}: {e}")


@app.get("/metrics")
def metrics():
    path = ROOT / "artifacts" / "accuracy.csv"
    if not path.exists():
        return {"note": "no evaluation results yet; run the notebook"}
    with open(path) as f:
        return {"spider_dev_execution_accuracy": list(csv.DictReader(f))}
