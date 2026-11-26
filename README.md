# Bitemporal-RAG

**Bitemporal Retrieval-Augmented Generation over Evolving Regulations via Incremental Fact Indexing**

Bitemporal-RAG answers questions about evolving rules using two dates. The
**asking date** determines which revisions were visible, and the **business
date** determines which rule was in force. Field-Level Incremental Maintenance
(Flim) updates affected fact versions and preserves earlier beliefs. Retrieval
filters by both dates and by applicability before ranking evidence and generating
an answer with source citations.

## Install and test

Use Python 3.10 or newer. From the repository root, run the following commands
in an activated virtual environment. The tests run locally without an API key,
model downloads, or a GPU.

```sh
python -m pip install -e ".[llm,dev]"
python -m pytest -q
python scripts/bench_condition_split.py
```

## Query two dates

A fictional licensing fee rises from 100 to 120. The revision was published on
May 8 and takes effect on June 1. For the same June business date, the answer
changes once the revision becomes visible. This example runs without a model.

```python
from datetime import date

from asof.schema import Fact, Notice, RevisionType
from asof.store import BitemporalStore, LocateImpact, flim_update
from asof.temporal import Interval

store = BitemporalStore()
original = Interval.of("2025-01-01", "INF")
store.append(Fact("license", "fee", "100", original, original, doc_id="original"))

notice = Notice("revision", date(2025, 5, 8), "The fee is now 120.",
                effective=date(2025, 6, 1), cited_ids=["original"])
pending = Interval.of("INF", "INF")
new_fact = Fact("license", "fee", "120", pending, pending)
flim_update(store, notice, [new_fact],
            classify=lambda fact, candidates: (RevisionType.REPLACE, None),
            locate=LocateImpact())

for asking_date in (date(2025, 5, 7), date(2025, 5, 8)):
    print([fact.value for fact in store.query(asking_date, date(2025, 6, 15))])
# ['100']
# ['120']
```

## Run the native pipeline

The experiment runner reads `notices.jsonl` and `questions.jsonl` from
`data/canonical/<domain>/`. To generate a synthetic domain, run
`python scripts/build_cond_demo.py`. To prepare your own raw sources, put them
under `datasets/` or set `paths.datasets` in `configs/default.yaml`. The loaders
in `src/asof/data/loaders/` define the input layouts. Run
`asof-prepare --domains <domain>` for a configured loader.

For an API-backed, BM25-only run of the synthetic domain, set
`SILICONFLOW_API_KEY` in your environment and run these commands.

```sh
python scripts/build_cond_demo.py
asof-run --domain cond_demo --method asof --override configs/providers/qwen_api.yaml
```

Provider overrides for DeepSeek, OpenAI, and local vLLM live in
`configs/providers/`. Install `.[gpu,llm]` to use local dense embeddings and a
reranker. Use `asof-run --help` to see the native baselines, `asof-ablation` to
run the bundled variants, and `asof-tables` to summarize the resulting JSON.

## Code map

- [`src/asof/store/`](src/asof/store/) contains the bitemporal store, Flim, and impact lookup.
- [`src/asof/extract/`](src/asof/extract/) contains fact extraction and revision classification.
- [`src/asof/methods/asof.py`](src/asof/methods/asof.py) connects updates, retrieval, and answer generation.
- [`src/asof/retrieval/`](src/asof/retrieval/) contains hybrid retrieval and rank fusion.
- [`src/asof/eval/`](src/asof/eval/) contains the native evaluation and cost accounting.
- [`tests/`](tests/) covers visibility, partial revisions, evidence attribution, and rebuild agreement.

This release covers the native bitemporal indexing and QA pipeline under the
[MIT License](LICENSE). The paper's large-scale evaluations require separate
benchmark inputs and model services. Benchmark datasets retain their upstream
licenses.
