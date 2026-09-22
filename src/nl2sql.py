"""Question in, SQL out: the fine-tuned strategy (best on Spider dev) wrapped
for single queries. Used by both the notebook's "try it" section and serve.py.

The schema comes from either a Spider dev database (`db_id`) or any
`CREATE TABLE ...` text you pass in. Execution is only possible against a
Spider database, read-only and with a timeout.
"""
import time

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from strategies import FINETUNED_MODEL, TOKENIZER_FOR


class NL2SQL:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = T5Tokenizer.from_pretrained(TOKENIZER_FOR[FINETUNED_MODEL])
        self.model = T5ForConditionalGeneration.from_pretrained(FINETUNED_MODEL).to(self.device).eval()

    def generate(self, question: str, schema: str) -> str:
        prompt = "tables:\n" + schema + "\nquery for:" + question
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=256, num_beams=1, do_sample=False)
        return self.tok.decode(out[0], skip_special_tokens=True).strip()

    def ask(self, question: str, db_id: str | None = None, schema: str | None = None, execute: bool = False) -> dict:
        if (db_id is None) == (schema is None):
            raise ValueError("pass exactly one of db_id or schema")
        if db_id is not None:
            from spider_data import schema_text
            schema = schema_text(db_id)
        t0 = time.perf_counter()
        sql = self.generate(question, schema)
        result = {"question": question, "db_id": db_id, "sql": sql,
                  "generation_ms": round((time.perf_counter() - t0) * 1000, 1)}
        if execute:
            if db_id is None:
                raise ValueError("execute=True needs a db_id (a Spider database to run against)")
            from metrics import run_sql
            rows, err = run_sql(db_id, sql)
            result["rows"] = rows[:50] if rows is not None else None
            result["error"] = err
        return result
