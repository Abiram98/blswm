# Non-intrusive Learned SPARQL Workload Management Benchmark

A standalone benchmark for evaluating learned, engine-agnostic workload management. 
The goal with this benchmark is to compare scheduling policies and runtime predictors using different workload compositions. 
Python 3.10+ is required.

The implementation has three self-contained modules:

- [`src/wrk_gen`](src/wrk_gen): query grouping, runtime/CPU profiling, sampling and arrivals.
- [`src/policies`](src/policies): online policy execution.
- [`src/qpp`](src/qpp): model training, evaluation and TCP serving.

## Quickstart

Run from the repository root:

```bash
./setup.sh
source .local/env.sh
python -m wrk_gen --help
python -m policies --help
python -m qpp --help
```

Setup creates or reuses `.venv/`, installs the package and test dependencies, and writes environment settings to `.local/env.sh`. Source that file in each new shell. Choose new output directories for each run. Workload profiling can also write to explicit paths such as `/data/temp/wiki_qc`.

- `./setup.sh --ml --java` adds ML dependencies and builds query-graph extraction; requires Java 17 and Maven. PyTorch/DGL compatibility depends on your platform.
- `PYTHON=python3.11 ./setup.sh` selects Python when creating the environment.

## Data

DBepdia and Wikidata RDF data can be downloaded via ...
The RDF files can be loaded into any triplestore of choice.
We use Virtuoso.

The workload data and CQPP model data is available on this shared drive: [Google Drive](https://drive.google.com/drive/folders/10BAlgoJeQMEC4zmbD4Lsut6TVSTzmJZo?usp=sharing)

## Model serving

Install the ML dependencies and build the graph extractor once with `./setup.sh --ml --java`. In each server terminal, run `source .local/env.sh`.

Run each server in its own terminal and leave it running during policy execution. Stop it with Ctrl+C. Both commands use CPU; use `--device cuda:0` for an available GPU.

### CQPP interval classifier

Serves `run_queryset_joint_dwns_base_120` for RTQ, RTQ-Reeval, SQA, SQA-Reeval and RR. Classes are `[0, 120)` and `[120, infinity)` seconds; policy `--threshold-seconds` defaults to 120.

```bash
python -m qpp serve \
  --model-dir /data/models/cqpp_wrk/queryset_cqpp_dataset_w1/run_queryset_joint_dwns_base_120 \
  --host 127.0.0.1 --port 9010 --device cpu
```

### CQPP runtime regressor

Serves `run_queryset_joint_dwns_base_regression` for IconQShed. Predictions are total execution runtimes in seconds for each query in the supplied query set.

```bash
python -m qpp serve \
  --model-dir /data/models/cqpp_wrk/queryset_cqpp_dataset_w1/run_queryset_joint_dwns_base_regression \
  --host 127.0.0.1 --port 9011 --device cpu
```

Both servers use the local `java/qpe/target/qpe-1.0-SNAPSHOT.jar`; override with `--jar` if needed.

## Policy execution

Each policy has its own Python CLI and options: `python -m policies <policy> --help`. Run from the repository root after `source .local/env.sh`. Start the required model server from the section above before running a predictive policy.

The examples use `data/test_wkl_short`: 14 queries selected from the original workload with recorded execution times below 30 seconds (`ql.csv` and `arrival_time.csv`). Each command below includes all required arguments.

Run policies sequentially. Each `--out` must be a new directory; results include execution records, predictions, decisions and running-query updates.

### FIFO

No predictor required. `--mpl` limits concurrent queries.

```bash
python -m policies fifo \
  --workload-dir data/test_wkl_short \
  --endpoint http://wikidata_virtuoso:8890/sparql \
  --query-timeout 30 --mpl 10 --out runs/fifo
```

### RTQ

Uses the classifier on port 9010. `--class-capacities` sets fast/slow pool sizes; add `--no-spillover` to prevent fast queries from using slow slots.

```bash
python -m policies rtq \
  --workload-dir data/test_wkl_short \
  --endpoint http://wikidata_virtuoso:8890/sparql \
  --query-timeout 30 --predictor-port 9010 \
  --class-capacities 5 5 --mpl 10 --out runs/rtq
```

### RTQ-Reeval

Updates running-query classes while preserving physical slot allocations.

```bash
python -m policies rtq-reeval \
  --workload-dir data/test_wkl_short \
  --endpoint http://wikidata_virtuoso:8890/sparql \
  --query-timeout 30 --predictor-port 9010 \
  --class-capacities 5 5 --mpl 10 --out runs/rtq-reeval
```

### SQA

Uses the classifier on port 9010. `--class-costs` sets fast/slow costs; `--budget` limits their sum.

```bash
python -m policies sqa \
  --workload-dir data/test_wkl_short \
  --endpoint http://wikidata_virtuoso:8890/sparql \
  --query-timeout 30 --predictor-port 9010 \
  --budget 10 --class-costs 1 3 --mpl 10 --out runs/sqa
```

### SQA-Reeval

Updates running-query classes and costs before checking the budget.

```bash
python -m policies sqa-reeval \
  --workload-dir data/test_wkl_short \
  --endpoint http://wikidata_virtuoso:8890/sparql \
  --query-timeout 30 --predictor-port 9010 \
  --budget 10 --class-costs 1 3 --mpl 10 --out runs/sqa-reeval
```

### RR

Uses the classifier on port 9010 and alternates between fast/slow FIFO queues.

```bash
python -m policies rr \
  --workload-dir data/test_wkl_short \
  --endpoint http://wikidata_virtuoso:8890/sparql \
  --query-timeout 30 --predictor-port 9010 --mpl 10 --out runs/rr
```

### IconQShed

Uses the runtime regressor on port 9011. `--alpha` weights candidate slowdown, `--starve-penalty` weights aging, and `--steps-into-future` controls lookahead.

```bash
python -m policies iconqshed \
  --workload-dir data/test_wkl_short \
  --endpoint http://wikidata_virtuoso:8890/sparql \
  --query-timeout 30 --predictor-port 9011 \
  --mpl 10 --alpha .2 --starve-penalty .5 --steps-into-future 2 \
  --reconsider-interval 5 --out runs/iconqshed
```

See each command's `--help` for additional options. The earlier `python -m policies --policy <policy> ...` syntax remains supported.

## Model training and evaluation

Initialize from a trusted model directory and provide separate JSONL training/validation files:

```bash
python -m qpp train --model-dir /path/to/trained/model \
  --data /path/to/train.jsonl --validation /path/to/validation.jsonl \
  --epochs 10 --out models/cqpp
python -m qpp evaluate --model-dir models/cqpp \
  --data /path/to/test.jsonl --out reports/cqpp.json
```

Rows contain `S` with inline `query_text`, plus aligned class labels in `y`. Regression targets use each query's `observed_runtime` or an aligned `y_runtime` array, in seconds. Classification reports cross-entropy, accuracy and macro F1; regression reports RMSE and MAE. Only load trusted checkpoint/pickle artifacts.

## Query-log characterization and workload generation

`python -m wrk_gen` characterizes query logs against a real endpoint, samples a workload composition, then assigns order and arrivals. Run from the repository root after `source .local/env.sh`. Each command has `--help`; choose new output directories. Characterization executes queries; grouping, sampling and arrival generation run offline.

### 1. Prepare query groups (offline)

Read `id`/`query_id` and `text`/`query_text`/`queryString` columns. This example selects 20 queries with prior runtimes between 0.5 and 2 seconds and creates four groups of five. Omit `--sample-size` and the runtime bounds to group the complete log. Change `--group-size` to choose X.

```bash
python -m wrk_gen groups \
  --input /data/wkls/wikidata/qc/wikidata_ql.csv \
  --sample-size 20 --runtime-min 0.5 --runtime-max 2 \
  --group-size 5 --seed 42 --out /data/temp/wiki_qc/groups
```

Prior runtimes are optional. Filters recognize `runtime_seconds`, `isolated_runtime_s`, `ex time (secs)` or `mean_latency`; use `--runtime-column NAME` for another column. For a complete log without prior measurements, omit sampling and runtime filters:

```bash
python -m wrk_gen groups \
  --input /data/wkls/wikidata/qc/wikidata_ql.csv \
  --group-size 100 --seed 42 --out /data/temp/wiki_qc/all_groups
```

These are collection batches, separate from the runtime/CPU profile groups used for workload composition. Input query IDs must be unique. `groups.json` records the selection and batch hashes.

### 2. Characterize queries against the endpoint

Install the OS package `collectl` locally and on the database host; remote collection also needs Python 3 and SSH access. Use SSH keys or `--ssh-ask-password`. The monitored host must contain the endpoint's database process. Use `--local --pid PID` only when that process is visible locally.

```bash
python -m wrk_gen profile \
  --groups /data/temp/wiki_qc/groups \
  --endpoint http://wikidata_virtuoso:8890/sparql \
  --ssh-host wikidata_virtuoso --ssh-user root --ssh-ask-password \
  --query-timeout 30 --repetitions 1 --out /data/temp/wiki_qc/profiles
```

`--groups` also accepts individual group directories. Queries run sequentially. Each group retains collectl raw logs, playback, database-process CPU/RSS samples and execution records. `catalog.csv` combines measured runtimes and peak CPU. Defaults use 1-ms CPU-counter sampling and 10-ms CPU windows (`100%` means one core); `--cpu-interval`, `--cpu-window` and `--collectl-interval` are configurable. CPU covers the database process, including background activity. Queries shorter than a complete CPU window have an unknown peak rather than an invented value. Prior runtime filters are selection criteria, not guarantees of new runtimes.

Use `--groups /data/temp/wiki_qc/all_groups` in the profiling command to characterize the complete log. Database PID discovery requires a unique database process; select one with `--pid` when necessary. `--repetitions` controls executions per query and `--query-gap` controls spacing. Across repetitions, catalog runtime is averaged and peak CPU is the maximum available peak. The catalog also contains mean CPU, peak RSS, status and measurement settings. Failed executions stop profiling and are recorded in the manifest. Keep `catalog.csv` beside its collection manifest for timeout screening, and use consistent CPU windows when comparing or combining catalogs.

### 3. Sample a frozen workload bag

Choose the source CSV with `--input`, define query groups by runtime/CPU bounds in `--profiles`, and set how many queries to draw from each group with repeated `--group NAME=COUNT` arguments. Names and counts are parameters; no workload family or size is fixed in code.

```bash
python -m wrk_gen sample \
  --input /data/temp/wiki_qc/profiles/catalog.csv \
  --profiles configs/workloads/profiles_short.json \
  --group subsecond=10 --group moderate=10 \
  --name short_20 --dataset wikidata --seed 42 \
  --reuse distinct_only --screen-timeouts --out /data/temp/wiki_qc/bag
```

`profiles_short.json` defines `subsecond` as runtime below 1 second and `moderate` as 1–30 seconds. Edit the bounds or supply another profiles file. Sampling deduplicates exact query texts, assigns distinct texts across overlapping groups, and preserves source IDs and measurements. `--reuse prefer_distinct` fills shortages with balanced repeated occurrences; `--reuse distinct_only` rejects shortages. Each occurrence gets its own ID. `--exclude` accepts training-query CSVs. `--screen-timeouts` excludes timeout/partial observations and runtimes at recorded collection deadlines; it requires collection manifests for usable observations.

Alternatively, `--composition /path/to/composition.json` supplies the workload ID, dataset, group counts, sampling seed and reuse rule. Existing dataset `profiles.json` and `composition.json` files are accepted directly. `--group`, `--seed`, `--name`, `--dataset` and `--reuse` override individual settings. For example, a composition's counts can be `{"S":20,"M":210,"L10":60,"V100":10,"H_M":30,"H_L":20}`; the supplied profiles define those names. The combined `--recipe configs/workloads/sampling_20.json` form also remains available.

#### Composition design: saved DBpedia workloads

The families under `/data/wkls/dbpedia/wkls2` each contain two independently sampled instances (`<family>_1` and `<family>_2`) of 350 occurrences. `profiles.json` defines group eligibility, `composition.json` records counts and sampling settings, and `bag.json` freezes the selected queries. These saved profiles differ from the short example above:

| Profile | Isolated runtime (seconds) | Peak CPU (%) |
|---|---|---|
| `S` | 0.1 ≤ t < 1 | 100 ≤ CPU ≤ 200 |
| `M` | 1 ≤ t < 10 | 100 ≤ CPU ≤ 200 |
| `L10` | 10 ≤ t ≤ 299 | 100 ≤ CPU ≤ 200 |
| `V100` | 100 ≤ t ≤ 110 | 100 ≤ CPU ≤ 200 |
| `H_M` | 1 ≤ t < 10 | CPU > 200 |
| `H_L` | 10 ≤ t ≤ 299 | CPU > 200 |

`V100` overlaps `L10`, reserving a narrow long-runtime cohort for a controlled burst. Distinct assignment across overlapping profiles avoids unnecessary reuse. Counts describe assigned occurrences: an occurrence assigned to `V100` is counted once. A high CPU peak identifies a high-peak group; it does not establish sustained CPU demand.

| Family | S | M | L10 | V100 | H_M | H_L | Design interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| `hom_mod` | 0 | 350 | 0 | 0 | 0 | 0 | Homogeneous moderate-runtime baseline. |
| `rt_skew` | 0 | 280 | 60 | 10 | 0 | 0 | Moderate majority with a long-runtime tail. |
| `hi_cpu` | 0 | 250 | 35 | 10 | 30 | 25 | Explicit high-peak CPU cohorts alongside moderate queries. |
| `lon_ex` | 0 | 140 | 180 | 30 | 0 | 0 | More long-running queries, exposing prolonged slot occupancy. |
| `het` | 20 | 210 | 60 | 10 | 30 | 20 | Mixed short, moderate, long and high-peak queries. |

These interpretations describe the comparisons supported by the recorded mixes; policy effects require execution measurements. Both instances use identical counts and `prefer_distinct` reuse. Their sampling seeds differ: `hom_mod` 101/1101, `rt_skew` 102/1102, `hi_cpu` 103/1103, `lon_ex` 104/1104 and `het` 105/1105. Reuse can change the number of distinct texts; inspect `validation.json`.

To apply a saved design to your own characterized CSV, pass its `profiles.json` and `composition.json` to `sample`. Identical selection requires identical input measurements, exclusions and settings. Use the saved bag when the exact existing selection is required. CSVs, profile bounds, counts, reuse rules and seeds remain parameters.

### 4. Assign query order and arrival times (P1–P4)

All patterns reuse the same frozen bag. Sampling, ordering (`--order-seed`) and timing (`--arrival-seed`) have separate seeds. Rates are queries per second.

| Pattern | Query order and arrivals |
|---|---|
| P1 | Random order, Poisson arrivals at `--rate`. |
| P2 | Bursts from `--heavy-profile`, interleaved with Poisson background groups. |
| P3 | `--long-profile` occurrences first in a burst, then Poisson arrivals. |
| P4 | Occurrences spread across phases with separate counts and arrival rates. |

These commands use the two groups in the 20-query example above. P2 and P3 explicitly select `moderate` for a short live run; dataset scenarios can instead select their recorded `H_M` and `V100` groups.

```bash
python -m wrk_gen arrivals --bag /data/temp/wiki_qc/bag \
  --pattern P1 --rate 10 --order-seed 202 --arrival-seed 303 \
  --out /data/temp/wiki_qc/P1
python -m wrk_gen arrivals --bag /data/temp/wiki_qc/bag \
  --pattern P2 --heavy-profile moderate --background-profile subsecond \
  --rate 10 --bursts 2 --burst-count 5 --background-counts 5 5 \
  --burst-window 0.05 --order-seed 202 --arrival-seed 303 \
  --out /data/temp/wiki_qc/P2
python -m wrk_gen arrivals --bag /data/temp/wiki_qc/bag \
  --pattern P3 --long-profile moderate --long-count 10 \
  --fast-profile subsecond --fast-count 10 --rate 10 --burst-window 0.05 \
  --order-seed 202 --arrival-seed 303 --out /data/temp/wiki_qc/P3
python -m wrk_gen arrivals --bag /data/temp/wiki_qc/bag \
  --pattern P4 --phase-counts 6 8 6 --phase-rates 10 20 10 \
  --order-seed 202 --arrival-seed 303 --out /data/temp/wiki_qc/P4
```

To reproduce a dataset scenario, pass its JSON file with `--scenario /path/to/P2/config.json` instead of supplying individual options. 
Both a standalone scenario object and the saved `{"scenario": ...}` wrapper are accepted. 
Burst endpoints span the requested window. 
The gap into each phase follows the preceding phase's arrival assignment, where the first arrival uses an initial gap unless `--first-arrival zero` is given.

The bag exports `bag.json`, `occurrences.csv`, `profiles.json` and `composition.json`. Each scenario exports `ql.csv` (`id,text,ex time (secs)`), `arrival_time.csv` (`query_no,arrival_time`), `groups.csv`, `sampled_measurements.csv`, configuration, validation and hashes. These formats and seeded sampling/order/arrival algorithms match the existing dataset workloads. Use the generated scenario directory as a policy's `--workload-dir`.
