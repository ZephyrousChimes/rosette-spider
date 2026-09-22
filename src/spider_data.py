"""Fetch the Spider dev set (Yu et al., 2018) and load it for evaluation.

Spider is the standard cross-domain NL->SQL benchmark: 1034 dev questions over
20 SQLite databases, none of which appear in Spider's training split. Only the
dev databases are extracted -- the full archive unpacks to ~1.8GB, the dev
databases to ~100MB.

The archive comes from a HuggingFace mirror of the official spider_data.zip,
so `verify()` checks every dev row against the official xlangai/spider release
before anything is scored on it.
"""
import json
import os
import urllib.request
import zipfile
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPIDER_DIR = Path(os.environ.get("SPIDER_DIR", ROOT / "data" / "spider"))
ZIP_URL = "https://huggingface.co/datasets/HAL-9001/spider-databases/resolve/main/spider_data.zip"
OFFICIAL_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows?dataset=xlangai/spider"
    "&config=spider&split=validation&offset={off}&length=100"
)
FILES = ["dev.json", "tables.json", "train_spider.json"]


def download():
    if (SPIDER_DIR / "dev.json").exists():
        print(f"already present: {SPIDER_DIR}")
        return
    SPIDER_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = SPIDER_DIR.parent / "spider_data.zip"
    if not zip_path.exists():
        print("downloading spider_data.zip (~206MB)...")
        urllib.request.urlretrieve(ZIP_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        for f in FILES:
            (SPIDER_DIR / f).write_bytes(z.read(f"spider_data/{f}"))
        dev_dbs = {x["db_id"] for x in load_dev()}
        for name in z.namelist():
            parts = name.split("/")
            if name.startswith("spider_data/database/") and len(parts) > 3 and parts[2] in dev_dbs and parts[-1]:
                out = SPIDER_DIR / "database" / parts[2] / parts[-1]
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(z.read(name))
    zip_path.unlink()
    print(f"extracted dev set and {len(dev_dbs)} databases to {SPIDER_DIR}")


def verify():
    official = []
    for off in range(0, 1034, 100):
        with urllib.request.urlopen(OFFICIAL_ROWS_URL.format(off=off)) as r:
            official += [row["row"] for row in json.load(r)["rows"]]
    dev = load_dev()
    same = sum(
        a["db_id"] == b["db_id"] and a["question"] == b["question"] and a["query"] == b["query"]
        for a, b in zip(dev, official)
    )
    assert len(dev) == len(official) == same, f"mirror differs from official: {same}/{len(official)}"
    print(f"verified: {same}/{len(official)} dev rows identical to xlangai/spider")


def load_dev() -> list[dict]:
    return json.loads((SPIDER_DIR / "dev.json").read_text())


def load_train() -> list[dict]:
    return json.loads((SPIDER_DIR / "train_spider.json").read_text())


def db_path(db_id: str) -> Path:
    return SPIDER_DIR / "database" / db_id / f"{db_id}.sqlite"


_TYPES = {"number": "INTEGER", "text": "VARCHAR", "time": "VARCHAR", "boolean": "VARCHAR", "others": "VARCHAR"}


@lru_cache(maxsize=None)
def _tables() -> dict:
    return {t["db_id"]: t for t in json.loads((SPIDER_DIR / "tables.json").read_text())}


@lru_cache(maxsize=None)
def schema_text(db_id: str) -> str:
    """Compact `CREATE TABLE t (col TYPE, ...); ...` -- the format the
    fine-tuned checkpoint was trained on (per its model card). All three
    strategies get the identical schema string, so the comparison stays fair."""
    t = _tables()[db_id]
    cols: dict[int, list[str]] = {}
    for (ti, col), ty in zip(t["column_names_original"], t["column_types"]):
        if ti >= 0:
            cols.setdefault(ti, []).append(f"{col} {_TYPES[ty]}")
    return "; ".join(
        f"CREATE TABLE {name} ({', '.join(cols[i])})" for i, name in enumerate(t["table_names_original"])
    )
