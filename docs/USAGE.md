# Benchmark usage

Run commands from the repository root after `./setup.sh` and `source .local/env.sh`. See the [quickstart](../README.md) for the synthetic workflow.

## Engine integration

Scheduling policies depend on the `Predictor` interface and occurrence state. Engine access is isolated behind `QueryExecutor.execute(query_text)`: return only after consuming the full result, and raise an exception on failure. Executors may be called concurrently up to the configured MPL and must support that use.

The CLI's `--endpoint` selects the included HTTP SPARQL executor. To use another client or engine, inject an executor through the Python API:

```python
from sparql_wm_bench.policies import Policy
from sparql_wm_bench.runtime import replay

class EngineExecutor:
    def __init__(self, submit):
        self.submit = submit

    def execute(self, query_text):
        # submit returns an iterable over the complete result stream.
        for row in self.submit(query_text):
            pass

# trace and catalog are loaded benchmark inputs; submit_query is your client.
run = replay(trace, catalog, Policy("fifo", mpl=10), predictor=None,
             executor=EngineExecutor(submit_query))
```

For learned policies, supply a predictor implementing `predict` and, where required, `predict_joint`. Predictions are keyed by occurrence ID. Train and evaluate on measured execution durations from the target engine; keep test query identities separate from training. The supplied graph-based models and workload presets target SPARQL queries.

Keep traces, query texts, engine configuration, and resources fixed across policies. Compare engine configurations in separate experiment directories, using a matching FIFO baseline for each.

## Catalogs and immutable workloads

CSV/JSON catalogs contain `query_id`, `query_text` (or `query_file`), `dataset`, `isolated_runtime_s`, `peak_cpu_pct`, optional `status`. Import only full-result isolated profiles; CPU peaks must use 10-ms samples with 100% representing one core. Missing CPU is eligible only for S. Non-success profiles are excluded. No query text normalization changes quoted SPARQL literals; identities use stripped text hashes.

```bash
.venv/bin/sparql-wm-bench catalog validate --input /path/to/queries.csv
.venv/bin/sparql-wm-bench workload generate --catalog /path/to/queries.csv \
  --config configs/paper/dbpedia_heterogeneous_rcb.json --exclude runs/inputs/D1/bag.json --out runs/inputs/dhet
.venv/bin/sparql-wm-bench run --trace runs/inputs/dhet/trace.csv \
  --catalog runs/inputs/dhet/bag.json --policy fifo --mpl 10 \
  --endpoint http://localhost:8080/sparql --run-id fifo_01 --out runs/fifo_01
```

Replace `/path/to/queries.csv` with your profiled catalog. Generate W1/D1 first, then pass their saved bags with `--exclude` when generating W2/D2 and profile mixtures. Every competing policy must read the same saved trace. Never regenerate a trace per policy. Selection uses no replacement and fails if distinct candidates are insufficient. Repeat each scheduling setting twice with distinct run IDs.

Presets use portable JSON (no YAML dependency). W/D selection is 320 below the dataset's 85th percentile plus 30 at/above it. `configs/paper/protocol.json` records repeat counts, prediction MPLs, rates and timing matrix. Mixture counts are defined once in `workloads.COMPOSITIONS`.

P uses an exponential initial gap. Each group's entry gap follows the preceding group's distribution, including PAR transitions. Bursts have fixed 0/.5 endpoints and sorted uniform interior draws. Completion events precede equal-time arrivals; arrivals are ordered by trace index. RR starts with fast, alternates when both queues are occupied, and retains its turn when only one queue is occupied; class assignments are retained while waiting. RTQ scans in arrival order and spills only toward slow slots. Reeval policies never move physical slots. IconQ uses MPL10, L2, alpha .2, lambda .5, no short bypass, wait floor 3 s and overrun factor .15. Regression scheduling applies and logs a 1e-6-second floor.

## CQPP, graph extraction and models

Run `./setup.sh --ml --java` and `source .local/env.sh` first. ML requires a compatible PyTorch/DGL environment; Java 17 and Maven build logical graph extraction. No module import starts Java or initializes a model.

```bash
MAVEN_OPTS="-Djava.io.tmpdir=$TMPDIR" mvn -f java/qpe/pom.xml \
  -Dmaven.repo.local="$PWD/.local/m2" package
.venv/bin/sparql-wm-bench trace build --runs runs/fifo_training \
  --catalog /path/to/queries.json --workload-id W1 --mpl 10 --out derived/cqpp
.venv/bin/sparql-wm-bench model train --config configs/paper/ml.json \
  --graph-config examples/graph-config.json --train derived/cqpp/train.jsonl \
  --validation derived/cqpp/validation.jsonl --out models/bi_class
.venv/bin/sparql-wm-bench model evaluate --model-dir models/bi_class \
  --graph-config examples/graph-config.json --test derived/cqpp/test.jsonl --out reports/prediction
```

Edit graph-config paths to actual external **predicate/entity/literal Meta-KG statistics**. `in_dim` must match the resulting featurizer dimension; the example ML value is a configurable placeholder, not a fixed model dimension. Trace building requires at least two FIFO run groups for an 80/20 grouped split; W2/D2 traces are supplied separately with `--test-runs`. Training sources must be W1/D1 at MPL1/5/10. Each `trace build` invocation takes one workload/MPL; combine JSONL outputs explicitly when training across those sources.

Model registry: `planrgcn_class`, `scqpp_class`, `bi_scqpp_class`, `bi_scqpp_reg`, `p_iconq_class`. Change `model_type` in the ML config. All use logical graph RGCN embeddings. SCQPP uses candidate-only CE; Bi-SCQPP uses joint CE or joint RMSE. P+IconQ uses target-relative query features. Checkpoints store architecture, thresholds, data hashes, training time, and fitted feature scalers; load only your own trusted feature pickle files. Checkpoint architecture, feature schema, and runtime thresholds must match the configured model. The supported feature-schema tag is `logical-query-graph-13-v1`.

For real predictive runs add `--model-dir` and `--graph-config` to `run`. `model serve` offers optional length-prefixed JSON inference; `service.Remote` implements the client. `model benchmark --test <60-query-items.json> ...` measures the required running/waiting matrix after graph cache warmup. Model evaluation reports role-specific metrics and latency; GPU timings synchronize CUDA.

## Analysis and limitations

`analyze` requires complete, matched repeats. It averages latency per occurrence before distributions, computes S50 as median(policy/FIFO10) ratios, exports CDF differences and summaries, and refuses mismatched traces. Use one workload/scenario/resource setting per input directory. Separate FIFO MPL settings into separate analysis inputs; `--reference fifo_mpl10` identifies the FIFO MPL10 baseline. Boxplots use 1.5-IQR whiskers; CDF denominators include all completed occurrences. Failures are preserved and prevent paired comparisons.

**Validation boundaries:** Synthetic runs exercise scheduling and accounting. Validate full-result correctness, performance, GPU inference, dataset profiles, and learned checkpoints against each target engine. Preprofiled catalogs are required; automatic CPU profiling is not included. See [benchmark design](DESIGN.md), [components](COMPONENTS.md), and [validation](VALIDATION.md).

A four-occurrence fixture with a repeated query identity is also included:

```bash
.venv/bin/sparql-wm-bench run --catalog examples/catalog.json --trace examples/trace.csv \
  --policy fifo --mpl 1 --synthetic-durations examples/durations.json --out runs/tiny
```
