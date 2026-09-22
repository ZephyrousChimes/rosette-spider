# rosette-spider

Three NL→SQL strategies compared on the **Spider dev set** (1034 questions, 20 databases
unseen in Spider's training split). This scales up the evaluation from
[rosette](https://github.com/ZephyrousChimes/rosette), which used a 10-question toy database.

| strategy | model | extra input |
|---|---|---|
| `zero_shot` | `google/flan-t5-small` | schema + question |
| `rag_fewshot` | `google/flan-t5-small` | + top-3 similar (question, SQL) pairs from Spider train, via TF-IDF |
| `fine_tuned` | `cssupport/t5-small-awesome-text-to-sql` | schema + question, in its training format |

## Results (Spider dev, Kaggle T4 run)
Execution accuracy with 95% Wilson intervals. *Unseen* means the dev question does not appear
verbatim in the fine-tuned model's training data; only that row measures generalization.

| subset | n | zero_shot | rag_fewshot | fine_tuned |
|---|---|---|---|---|
| all | 1034 | 0.0% [0.0, 0.4] | 2.6% [1.8, 3.8] | **10.1%** [8.4, 12.0] |
| unseen by fine_tuned | 468 | 0.0% [0.0, 0.8] | 1.7% [0.9, 3.3] | **9.6%** [7.3, 12.6] |
| seen by fine_tuned | 566 | 0.0% [0.0, 0.7] | 3.4% [2.2, 5.2] | 10.4% [8.2, 13.2] |
| invalid SQL (all) | 1034 | 99.5% | 93.5% | 73.2% |

- **The ranking is significant, even on unseen questions.** In exact McNemar tests,
  fine_tuned beats rag_fewshot 45 to 8 on discordant pairs (p = 2e-7), and rag_fewshot beats
  zero_shot 8 to 0 (p = 0.008).
- **Contamination barely helps.** Accuracy is 10.4% on seen questions and 9.6% on unseen ones.
  A likely reason is that the training contexts list only the tables each query uses, while
  here the model gets the full schema, so memorized answers don't transfer.
- **The dominant failure is schema linking, not grammar.** 648 of fine_tuned's 930 failures
  reference a table or column that doesn't exist, mostly a real column under the wrong table
  alias. Only 76 are syntax errors. Accuracy on unseen questions is 13.1% for single-table
  queries and 2.6% for joins.
- **The toy 10-question result overstated it.** On that set fine_tuned scored 40% with 0%
  invalid SQL; on Spider it scores 10% with 73% invalid.

Full tables are in `artifacts/*.csv`. `rosette_spider.ipynb` is committed **with the outputs of the
Kaggle T4 run**, so GitHub renders every table and example, and `results/rosette_spider.html` is a
standalone copy. Three independent Kaggle runs produced identical tables, since decoding is greedy.

## Ask it a question (API)
```bash
pip install -r requirements.txt
python -c "import sys; sys.path.insert(0, 'src'); import spider_data; spider_data.download()"   # only needed for db_id / execute
uvicorn serve:app --app-dir src --port 8000
```
```bash
# against a Spider database, and run the SQL
curl -X POST localhost:8000/sql -H 'content-type: application/json' \
     -d '{"question": "What is the average age of all singers?", "db_id": "concert_singer", "execute": true}'

# against any schema you give it (generation only)
curl -X POST localhost:8000/sql -H 'content-type: application/json' \
     -d '{"question": "How many customers are there?", "schema": "CREATE TABLE customers (customer_id INTEGER, name VARCHAR, city VARCHAR)"}'
```
Open **http://localhost:8000** for a small web page: pick a Spider database or paste your own schema,
ask a question, and see the SQL and its result. For a Spider dev question, the page also shows the gold
SQL and whether the results match. Interactive API docs are at `/docs`.

Endpoints: `POST /sql`, `GET /databases`, `GET /schema/{db_id}`, `GET /examples/{db_id}`,
`GET /metrics` (the Spider numbers above), `GET /health`.
The response contains the generated SQL itself, so a caller can check what would run. Given the
~10% accuracy, treat it as a demo of the harness, not a production model.

## Run it
**Kaggle:** import `rosette_spider.ipynb`, set *Accelerator → GPU T4* and *Internet → On*, then
Run All. The first cell clones this repo if the scripts aren't next to the notebook.

**Kaggle, headless, with outputs saved:** `kaggle kernels push -p kaggle` runs
`kaggle/run_notebook.py`, which executes the notebook with nbconvert and writes
`rosette_spider_executed.ipynb` and `.html` to the run's outputs. Download them with
`kaggle kernels output <owner>/rosette-spider-run`.

**Locally:** `pip install -r requirements.txt`, then run the notebook. Set `LIMIT=40` in the
environment for a quick subset run.

## Evaluation choices
- **Execution accuracy.** Predicted and gold SQL are both run, read-only with a 5 s timeout.
  Results are compared as an ordered list if the gold query has `ORDER BY` and as a multiset
  otherwise. The older `set(rows)` comparison, which ignores order and collapses duplicates, is
  reported alongside it to show how much it over-credits.
- **Contamination split.** 566 of the 1034 dev questions (54.7%) appear verbatim in the
  fine-tuned checkpoint's training data (`b-mc2/sql-create-context` is built from WikiSQL and
  Spider). Every number is reported on *seen*, *unseen* and *all*. The seen-list is in
  `artifacts/spider_dev_seen.json`, and `src/contamination.py` rebuilds it. Matching is exact
  after normalization, so paraphrased leakage isn't caught.
- **Same schema text for every strategy**, in the compact `CREATE TABLE` format the fine-tuned
  model was trained on. There's no truncation: about 8% of prompts exceed 512 tokens, and
  truncating would cut off the question.
- **Paired statistics.** All strategies answer the same questions, so comparisons use an exact
  McNemar test on the discordant pairs, and each accuracy gets a 95% Wilson interval.
- **Data integrity.** The Spider archive comes from a HuggingFace mirror, and `verify()` checks
  every dev row against the official `xlangai/spider` release before scoring.

This isn't the official Spider evaluator. Column order isn't permuted, so it's slightly stricter
than test-suite execution accuracy.

## Layout
```
rosette_spider.ipynb     # the whole pipeline, top to bottom (committed with outputs)
results/rosette_spider.html  # the same executed notebook as HTML
src/spider_data.py       # download, verify, schema text
src/contamination.py     # which dev questions the fine-tuned model trained on
src/strategies.py        # prompts, retrieval, batched generation (cached per strategy)
src/metrics.py           # execution, result matching, Wilson CI, McNemar
src/report.py            # result tables
src/nl2sql.py            # question -> SQL with the fine-tuned model (used by notebook + API)
src/serve.py             # FastAPI server
src/static/index.html    # the web page served at /
kaggle/                  # headless Kaggle runner that saves the executed notebook
artifacts/               # seen-list and result tables (committed); predictions (gitignored)
```
