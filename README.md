# rosette-spider

Three NL→SQL strategies compared on the **Spider dev set** (1034 questions, 20 databases
unseen in Spider's training split). This scales up the evaluation from
[rosette](https://github.com/ZephyrousChimes/rosette), which used a 10-question toy database.

| strategy | model | extra input |
|---|---|---|
| `zero_shot` | `google/flan-t5-small` | schema + question |
| `rag_fewshot` | `google/flan-t5-small` | + top-3 similar (question, SQL) pairs from Spider train, via TF-IDF |
| `fine_tuned` | `cssupport/t5-small-awesome-text-to-sql` | schema + question, in its training format |

## Run it
**Kaggle:** import `rosette_spider.ipynb`, set *Accelerator → GPU T4* and *Internet → On*, then
Run All. The first cell clones this repo if the scripts aren't next to the notebook.

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
rosette_spider.ipynb     # the whole pipeline, top to bottom
src/spider_data.py       # download, verify, schema text
src/contamination.py     # which dev questions the fine-tuned model trained on
src/strategies.py        # prompts, retrieval, batched generation (cached per strategy)
src/metrics.py           # execution, result matching, Wilson CI, McNemar
src/report.py            # result tables
artifacts/               # seen-list (committed) and run outputs
```
