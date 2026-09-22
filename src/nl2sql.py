"""Question in, SQL out: the fine-tuned strategy (best on Spider dev) wrapped
for single queries. Used by both the notebook's "try it" section and serve.py.

The schema comes from either a Spider dev database (`db_id`) or any
`CREATE TABLE ...` text you pass in. Execution is only possible against a
Spider database, read-only and with a timeout.

Set MODEL_DIR to point at a local checkpoint (e.g. artifacts/model_simple,
our own fine-tune) instead of the off-the-shelf cssupport one -- both use
t5-small's tokenizer, so this is just a path swap.
"""
import os
import time

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from spider_data import ROOT
from strategies import FINETUNED_MODEL, TOKENIZER_FOR

DEFAULT_MODEL_DIR = os.environ.get("MODEL_DIR", FINETUNED_MODEL)


class NL2SQL:
    def __init__(self, device=None, model_dir=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_dir = model_dir or DEFAULT_MODEL_DIR
        # a bare name like "model_simple" means artifacts/model_simple;
        # anything else (a HF hub id, or an absolute/relative path) is used as-is
        local = ROOT / "artifacts" / model_dir
        source = str(local) if local.exists() else model_dir
        self.model_name = source
        tok_source = TOKENIZER_FOR.get(source, "t5-small")
        self.tok = T5Tokenizer.from_pretrained(tok_source)
        self.model = T5ForConditionalGeneration.from_pretrained(source).to(self.device).eval()

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
        extracted = None
        if db_id is not None:
            from extract import try_extract
            extracted = try_extract(db_id, question)
        if extracted is not None:
            sql, source = extracted["sql"], "extracted"
        else:
            sql, source = self.generate(question, schema), "model"
        result = {"question": question, "db_id": db_id, "sql": sql, "source": source,
                  "generation_ms": round((time.perf_counter() - t0) * 1000, 1)}
        if execute:
            if db_id is None:
                raise ValueError("execute=True needs a db_id (a Spider database to run against)")
            from metrics import run_sql_with_columns
            rows, columns, err = run_sql_with_columns(db_id, sql)
            result["rows"] = rows[:50] if rows is not None else None
            result["columns"] = columns
            result["error"] = err
        return result
