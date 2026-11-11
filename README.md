# Workload-management benchmark

A standalone benchmark for evaluating learned, engine-agnostic workload management. Compare scheduling policies and runtime predictors using fixed arrival traces, recorded executions, and paired metrics. Python 3.10+ is required.

The scheduling core uses pluggable executor and predictor interfaces. SPARQL workloads, logical query-graph models, and an HTTP SPARQL executor are included. See [engine integration](docs/USAGE.md#engine-integration) to connect a backend.

## Quickstart

Run from the repository root:

```bash
./setup.sh
source .local/env.sh
python examples/smoke.py
sparql-wm-bench analyze --runs runs/smoke --out reports/smoke
```

The smoke workflow needs no database, Java, GPU, or external dataset. It runs four policies twice on the same 350-query trace. Results go to `runs/smoke/`; analysis goes to `reports/smoke/`.

Setup creates or reuses `.venv/`, installs the package and test dependencies, and writes environment settings to `.local/env.sh`. Source that file in each new shell. All generated files stay inside the repository.

- `./setup.sh --plot` adds plotting; append `--plot` to the analysis command.
- `./setup.sh --ml --java` adds ML dependencies and builds query-graph extraction; requires Java 17 and Maven. PyTorch/DGL compatibility depends on your platform.
- `PYTHON=python3.11 ./setup.sh` selects Python when creating the environment.

Run the core tests without ML dependencies:

```bash
pytest tests/test_core.py tests/test_integration_offline.py tests/test_packaging.py
```

## Run against a SPARQL endpoint

Provide a profiled CSV/JSON catalog containing `query_id`, `query_text` (or `query_file`), `dataset`, `isolated_runtime_s`, and `peak_cpu_pct`. CPU profiles must use 10-ms sampling, with 100% representing one core. Replace the input path and endpoint below:

```bash
sparql-wm-bench catalog validate --input /path/to/queries.csv
sparql-wm-bench workload generate --catalog /path/to/queries.csv \
  --config configs/paper/D1_0.6.json --out runs/inputs/D1
for repeat in 1 2; do
  sparql-wm-bench run --trace runs/inputs/D1/trace.csv \
    --catalog runs/inputs/D1/bag.json --policy fifo --mpl 10 \
    --endpoint http://localhost:8080/sparql \
    --run-id "fifo_$repeat" --out "runs/D1/fifo_$repeat"
done
sparql-wm-bench analyze --runs runs/D1 --out reports/D1
```

Reuse the saved trace for every competing policy. W2/D2 and profile-mixture generation require `--exclude` with the training-workload bag. Predictive policies additionally require a trained model and Meta-KG statistics.

See [detailed usage](docs/USAGE.md) for training, evaluation, graph configuration, and policy behavior; [experiment presets](configs/paper/experiments.json) define repeatable evaluation matrices. Use `sparql-wm-bench --help` for commands.

## Limitations

Live engine performance and GPU inference require validation in your environment. Import preprofiled catalogs; automatic CPU profiling is not included. See [benchmark design](docs/DESIGN.md), [components](docs/COMPONENTS.md), and [validation](docs/VALIDATION.md).
