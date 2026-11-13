# Non-intrusive Learned SPARQL Workload Management Benchmark

A standalone benchmark for evaluating learned, engine-agnostic workload management. 
The goal with this benchmark is to compare scheduling policies and runtime predictors using different workload compositions. 
Python 3.10+ is required.

The implementation has four self-contained modules:

- [`src/wrk_gen`](src/wrk_gen): query grouping, runtime/CPU profiling, sampling and arrivals.
- [`src/policies`](src/policies): online policy execution.
- [`src/qpp`](src/qpp): model training, evaluation and TCP serving.
- [`src/analysis`](src/analysis): workload and prediction analysis.

## Setup

The experiments run the same Docker image in two containers: one for Virtuoso and one for the workload manager. 
Docker is required. 
Run these commands from the repository root:

```bash
docker build -f docker/virtuoso/Dockerfile -t blswm:latest docker/virtuoso
docker network create virtuoso_net
mkdir -p runs/virtuoso-db runs/virtuoso-import
cp -n docker/virtuoso/virtuoso.ini runs/virtuoso.ini
```

### Database container

`runs/virtuoso-db` is an example persistent database directory. Replace its host path with the directory containing your Virtuoso database if you already have one. The bundled [configuration](docker/virtuoso/virtuoso.ini) reproduces the live endpoint settings; adjust its copy at `runs/virtuoso.ini` for your available memory before starting the container. The `128g` memory and `26` CPU limits below are example values.

```bash
docker run -dit --name wikidata_virtuoso --network virtuoso_net \
  --memory 128g --cpus 26 \
  -v "$(pwd)/runs/virtuoso-db:/database" \
  -v "$(pwd)/runs/virtuoso.ini:/database/virtuoso.ini:ro" \
  -v "$(pwd)/runs/virtuoso-import:/import:ro" \
  blswm:latest /bin/bash -c "cd /database && /opt/virtuoso-opensource/bin/virtuoso-t -c virtuoso.ini +foreground"
```

The database container retains its files in the mounted host directory. The manager reaches its endpoint at `http://wikidata_virtuoso:8890/sparql`. To expose it to the host as well, add `-p 127.0.0.1:8890:8890` to `docker run`.

### Workload-manager container

Start a second container on the same network, mounting this repository and the model data:

```bash
docker run -dit --name wrk_man --network virtuoso_net \
  --memory 12g --cpus 12 \
  -v "$(pwd):/blswm" -v /data:/data:ro -w /blswm \
  blswm:latest /bin/bash
docker exec -it wrk_man /bin/bash
```

Replace `/data` with the host directory containing the model data if needed. The bind mount exposes it read-only. In the workload-manager container, set up the repository environment:

```bash
./setup.sh --ml --analysis --java
source .local/env.sh
python -m wrk_gen --help
python -m policies --help
python -m qpp --help
python -m analysis --help
```

`setup.sh` creates or reuses `.venv/` and writes `.local/env.sh`; source that file in each new shell. It requires Python 3.10+, Java 17 and Maven for `--java`. Use `--tune` to install Ray Tune, or `PYTHON=python3.11` to select Python when creating the environment. Choose new output directories under the repository; keep `/data` read-only.

## Data

In principle, any triplestore with a SPARQL endpoint can be used. 
Our experiments use Virtuoso. 
The [DBpedia 2016.10 file list](data/dbpedia2016/files.txt) contains the RDF downloads used for DBpedia. 
For Wikidata, download the WDBench [truthy direct-properties RDF file](https://figshare.com/s/50b7544ad6b1f51de060), `truthy_direct_properties.nt.bz2`. 
Keep the datasets in separate databases when reproducing experiments.

For DBpedia, download the listed Turtle files into the import directory created during setup:

```bash
while IFS= read -r url; do
  curl -fL "$url" -o "runs/virtuoso-import/$(basename "$url")" || break
done < data/dbpedia2016/files.txt
```

Check that all listed files downloaded successfully.

Once the database is ready, run `docker exec -it wikidata_virtuoso /opt/virtuoso-opensource/bin/isql 1111 dba dba` on the Docker host. In the SQL prompt, run the following for DBpedia; for Wikidata, replace `*.ttl.bz2` with `*.nt.bz2`:

```sql
ld_dir('/import', '*.ttl.bz2', 'http://localhost:8890/dataspace');
rdf_loader_run();
checkpoint;
SELECT ll_file, ll_state, ll_error FROM DB.DBA.load_list;
```

Note, multiple ```rdf_loader_run();``` enables concurrent loading in Virtuoso.
Check the load list for failures and use the appropriate endpoint host in the policy commands. 

**The [shared drive](https://drive.google.com/drive/folders/10BAlgoJeQMEC4zmbD4Lsut6TVSTzmJZo?usp=sharing) contains trained CQPP models, workload executions, and query logs with runtime/CPU characterization.**

## Model serving

Install the ML dependencies and build the engine-agnostic query plan constructor once with `./setup.sh --ml --java`. 
In each server terminal, run `source .local/env.sh`.

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

Both servers use the local `src/qpp/java/qpe/target/qpe-1.0-SNAPSHOT.jar`; override with `--jar` if needed.

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

## Build CQPP datasets

The dataset builder converts completed **FIFO** workload executions into training data. Run from the repository root after `source .local/env.sh`. First execute the workload; use the same workload directory when building the dataset:

```bash
python -m policies fifo \
  --workload-dir data/test_wkl_short \
  --endpoint http://wikidata_virtuoso:8890/sparql \
  --query-timeout 30 --mpl 5 --out runs/dataset_fifo
```

Pass one or more completed run directories through `--runs`. Each must belong to the supplied workload. Repeat the FIFO command with different `--out` directories to collect more runs or MPL settings. The builder reads the workload's `ql.csv` and each run's `run.json`. It uses recorded admission and completion times to identify the queries running at each decision.

### Incoming-query datasets: DWNS and PlanRGCN

Use `--format candidate` when the model predicts the incoming query's performance. DWNS receives the running queries as context; PlanRGCN receives the incoming query alone.

```bash
python -m qpp dataset --workload data/test_wkl_short \
  --runs runs/dataset_fifo --format candidate --split train_validation \
  --validation-fraction 0.2 --threshold 120 --seed 42 \
  --out runs/datasets/candidate_trainval
python -m qpp dataset --workload /path/to/test_workload \
  --runs runs/test_fifo_1 runs/test_fifo_2 --format candidate --split test \
  --threshold 120 --out runs/datasets/candidate_test
```

The training command writes `train.csv` and `val.csv` for DWNS, and `planrgcn_train.tsv` and `planrgcn_validation.tsv` for PlanRGCN. The test command combines all supplied runs into `test.csv` and `planrgcn_test.tsv`. CSV columns follow the incoming-query dataset format, including `Execution_Time`, `Running_Queries_At_Start` and `Queries_Started_During`. Set `RUNTIME_COL_NAME=Execution_Time` for the DWNS reader; the PlanRGCN TSV uses `mean_latency`. Start and end times are seconds relative to the run start. Resource measurements, query file names and measurement-coverage flags are blank when unavailable.

### Joint query-set datasets: QSPS and IconQShed models

Use `--format queryset` when the model predicts performance for the candidate and running queries. Each JSONL state contains query texts, roles, elapsed times, aligned class labels in `y`, and observed runtimes in seconds for regression.

```bash
python -m qpp dataset --workload data/test_wkl_short \
  --runs runs/dataset_fifo --format queryset --split train_validation \
  --validation-fraction 0.2 --sample-mode all_query_start_events \
  --threshold 120 --seed 42 --out runs/datasets/queryset_trainval
python -m qpp dataset --workload /path/to/test_workload \
  --runs runs/test_fifo_1 runs/test_fifo_2 --format queryset --split test \
  --sample-mode all_query_start_events --threshold 120 \
  --out runs/datasets/queryset_test
```

The training command writes `train.jsonl` and `validation.jsonl`; the test command combines the supplied runs into `test.jsonl`. `--sample-mode all_query_start_events` samples each query start. `all_decision_events` also samples completions, and `aligned_hard_prefix` selects comparable cumulative hard-query cutpoints across runs. A single short trace may have only one hard-prefix value and therefore too few groups for a train/validation split; use multiple traces or start-event sampling.

### Split and labeling settings

`--threshold` sets the binary class boundary in seconds; measured runtimes remain available for regression. `--validation-fraction` and `--seed` configure the training/validation split. Candidate datasets group repeated candidate query texts together; query-set datasets group by decision cutpoint. The builder chooses the least query-text overlap among 512 seeded candidate splits and records remaining context overlap, input hashes and sample counts in `manifest.json`. Check that overlap before training. This heuristic does not reproduce the saved datasets' complete stratification or split optimization.

For held-out evaluation, supply an independent test workload and its FIFO runs. The test command does not automatically check overlap against training data. Every `--out` must be a new directory.

## Model training and evaluation


```bash
python -m qpp fit --model-type qsps \
  --data runs/datasets/queryset_trainval/train.jsonl \
  --validation runs/datasets/queryset_trainval/validation.jsonl \
  --featurizer /data/models/cqpp_wrk/queryset_cqpp_dataset_w1/run_queryset_joint_dwns_base_120/featurizer.pickle \
  --init-checkpoint /data/models/cqpp_wrk/queryset_cqpp_dataset_w1/run_queryset_joint_dwns_base_120/best_model.pt \
  --hidden-dim1 512 --hidden-dim2 128 --lstm-hidden-dim 64 --lstm-num-layers 2 \
  --thresholds 120 --epochs 10 --lr 0.001 --device cpu \
  --out models/qsps
```

To fit features from local knowledge-graph statistics instead, create a featurizer and pass its `featurizer.pickle` to `qpp fit` or `qpp tune`:

```bash
python -m qpp featurizer --predicate-stats /path/to/predicate_stats \
  --entity-stats /path/to/entity_stats --literal-stats /path/to/literal_stats \
  --bins 10 --scaling binner --out models/graph_features
```

Each statistics directory needs a `freq` subdirectory containing SPARQL-result JSON files. Binned features also require predicate `ents`/`lits` and entity `subj`/`obj` directories with their respective counts. Omit `--literal-stats` if unavailable. The output manifest records input hashes.

Omit `--init-checkpoint` to initialize a new model from scratch.

For Ray Tune, install the optional dependency with `./setup.sh --ml --tune --java`. `qpp tune` uses the same model and training options plus a JSON search space, seeded random selection, ASHA scheduling and explicit trial resources; it uses the chosen validation selection metric. The best checkpoint and featurizer are saved in the output directory alongside trial results. `best_selection_score` is validation loss, or `1 - macro_f1` when selecting macro-F1; lower is better. Edit `configs/qpp/search.json` to choose candidate values, including optional `feature_bins` and `feature_scaling` (`binner` or `none`); do not vary dimensions when initializing from a fixed-shape checkpoint. Use `--gpus-per-trial` and `--device cuda` when allocating GPUs.

```bash
python -m qpp tune --model-type qsps \
  --data runs/datasets/queryset_trainval/train.jsonl \
  --validation runs/datasets/queryset_trainval/validation.jsonl \
  --featurizer /data/models/cqpp_wrk/queryset_cqpp_dataset_w1/run_queryset_joint_dwns_base_120/featurizer.pickle \
  --search configs/qpp/search.json --samples 4 --cpus 2 \
  --cpus-per-trial 1 --max-concurrent-trials 2 \
  --epochs 10 --device cpu --out models/qsps_tuned
```

The existing `qpp train` command fine-tunes a compatible joint CQPP checkpoint directly:

Initialize from a trusted model directory and provide separate JSONL training/validation files:

```bash
python -m qpp train --model-dir /path/to/trained/model \
  --data /path/to/train.jsonl --validation /path/to/validation.jsonl \
  --epochs 10 --out models/cqpp
python -m qpp evaluate --model-dir models/cqpp \
  --data /path/to/test.jsonl --out reports/cqpp.json
```


## Workload and prediction analysis

Install plotting dependencies with `./setup.sh --analysis` and source `.local/env.sh`. After running the FIFO and RTQ examples, supply paths and parameters directly:

```bash
python -m analysis prepare --runs runs/fifo runs/rtq --out runs/analysis_prepared
python -m analysis workloads --prepared runs/analysis_prepared \
  --run FIFO runs/fifo --run RTQ runs/rtq \
  --reference FIFO --out runs/analysis_workloads
python -m analysis predictions --data runs/datasets/queryset_test/test.jsonl \
  --port 9010 --out runs/analysis_predictions
```

Repeat `--run NAME PATH` for each policy; give multiple paths after a name to average repeated executions per occurrence. `--metric` selects latency, waiting or execution time. To analyze an existing evaluation report, replace `--port 9010` with `--report reports/cqpp.json`; for a candidate classification CSV also provide `--thresholds 120`. `python -m analysis compare-predictions --results runs/predictions_qsps runs/predictions_dwns --out runs/model_comparison` compares model analyses made from the same labeled examples.

The same operations can be configured in Python:

```python
from analysis import (PreparationConfig, WorkloadRunConfig, WorkloadAnalysisConfig,
                      PredictionAnalysisConfig, PredictionComparisonConfig)

PreparationConfig(["runs/fifo", "runs/rtq"], "runs/py_prepared").run()
WorkloadAnalysisConfig("runs/py_prepared", [
    WorkloadRunConfig("runs/fifo", "FIFO"),
    WorkloadRunConfig("runs/rtq", "RTQ"),
], "runs/py_workloads", reference="FIFO").run()
PredictionAnalysisConfig("runs/datasets/queryset_test/test.jsonl",
                         "runs/py_predictions", port=9010).run()
PredictionComparisonConfig(["runs/predictions_qsps", "runs/predictions_dwns"],
                           "runs/py_model_comparison").run()
```

Both interfaces generate PDF figures, LaTeX tables, CSV data and JSON metadata. 
Results include S50 relative to the selected reference, backlog-weighted concurrency, classification recall/F1 and regression MAE/RMSE. 

## Query-log characterization and workload generation

`python -m wrk_gen` characterizes query logs against a real endpoint, samples a workload composition, then assigns order and arrivals. Run from the repository root after `source .local/env.sh`. Each command has `--help`; choose new output directories. Characterization executes queries; grouping, sampling and arrival generation run offline.

### 1. Prepare query groups (offline)

Read `id`/`query_id` and `text`/`query_text`/`queryString` columns. This example selects 20 queries with prior runtimes between 0.5 and 2 seconds and creates four groups of five. Omit `--sample-size` and the runtime bounds to group the complete log. Change `--group-size` to choose X.

```bash
python -m wrk_gen groups \
  --input /data/wkls/wikidata/qc/wikidata_ql.csv \
  --sample-size 20 --runtime-min 0.5 --runtime-max 2 \
  --group-size 5 --seed 42 --out runs/wiki_qc/groups
```

Prior runtimes are optional. Filters recognize `runtime_seconds`, `isolated_runtime_s`, `ex time (secs)` or `mean_latency`; use `--runtime-column NAME` for another column. For a complete log without prior measurements, omit sampling and runtime filters:

```bash
python -m wrk_gen groups \
  --input /data/wkls/wikidata/qc/wikidata_ql.csv \
  --group-size 100 --seed 42 --out runs/wiki_qc/all_groups
```

These are collection batches, separate from the runtime/CPU profile groups used for workload composition. Input query IDs must be unique. `groups.json` records the selection and batch hashes.

### 2. Characterize queries against the endpoint

Install the OS package `collectl` locally and on the database host; remote collection also needs Python 3 and SSH access. Use SSH keys or `--ssh-ask-password`. The monitored host must contain the endpoint's database process. Use `--local --pid PID` only when that process is visible locally.

```bash
python -m wrk_gen profile \
  --groups runs/wiki_qc/groups \
  --endpoint http://wikidata_virtuoso:8890/sparql \
  --ssh-host wikidata_virtuoso --ssh-user root --ssh-ask-password \
  --query-timeout 30 --repetitions 1 --out runs/wiki_qc/profiles
```

`--groups` also accepts individual group directories. Queries run sequentially. Each group retains collectl raw logs, playback, database-process CPU/RSS samples and execution records. `catalog.csv` combines measured runtimes and peak CPU. Defaults use 1-ms CPU-counter sampling and 10-ms CPU windows (`100%` means one core); `--cpu-interval`, `--cpu-window` and `--collectl-interval` are configurable. CPU covers the database process, including background activity. Queries shorter than a complete CPU window have an unknown peak rather than an invented value. Prior runtime filters are selection criteria, not guarantees of new runtimes.

Use `--groups runs/wiki_qc/all_groups` in the profiling command to characterize the complete log. Database PID discovery requires a unique database process; select one with `--pid` when necessary. `--repetitions` controls executions per query and `--query-gap` controls spacing. Across repetitions, catalog runtime is averaged and peak CPU is the maximum available peak. The catalog also contains mean CPU, peak RSS, status and measurement settings. Failed executions stop profiling and are recorded in the manifest. Keep `catalog.csv` beside its collection manifest for timeout screening, and use consistent CPU windows when comparing or combining catalogs.

### 3. Sample a frozen workload bag

Choose the source CSV with `--input`, define query groups by runtime/CPU bounds in `--profiles`, and set how many queries to draw from each group with repeated `--group NAME=COUNT` arguments. Names and counts are parameters; no workload family or size is fixed in code.

```bash
python -m wrk_gen sample \
  --input runs/wiki_qc/profiles/catalog.csv \
  --profiles configs/workloads/profiles_short.json \
  --group subsecond=10 --group moderate=10 \
  --name short_20 --dataset wikidata --seed 42 \
  --reuse distinct_only --screen-timeouts --out runs/wiki_qc/bag
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
python -m wrk_gen arrivals --bag runs/wiki_qc/bag \
  --pattern P1 --rate 10 --order-seed 202 --arrival-seed 303 \
  --out runs/wiki_qc/P1
python -m wrk_gen arrivals --bag runs/wiki_qc/bag \
  --pattern P2 --heavy-profile moderate --background-profile subsecond \
  --rate 10 --bursts 2 --burst-count 5 --background-counts 5 5 \
  --burst-window 0.05 --order-seed 202 --arrival-seed 303 \
  --out runs/wiki_qc/P2
python -m wrk_gen arrivals --bag runs/wiki_qc/bag \
  --pattern P3 --long-profile moderate --long-count 10 \
  --fast-profile subsecond --fast-count 10 --rate 10 --burst-window 0.05 \
  --order-seed 202 --arrival-seed 303 --out runs/wiki_qc/P3
python -m wrk_gen arrivals --bag runs/wiki_qc/bag \
  --pattern P4 --phase-counts 6 8 6 --phase-rates 10 20 10 \
  --order-seed 202 --arrival-seed 303 --out runs/wiki_qc/P4
```

To reproduce a dataset scenario, pass its JSON file with `--scenario /path/to/P2/config.json` instead of supplying individual options. 
Both a standalone scenario object and the saved `{"scenario": ...}` wrapper are accepted. 
Burst endpoints span the requested window. 
The gap into each phase follows the preceding phase's arrival assignment, where the first arrival uses an initial gap unless `--first-arrival zero` is given.

The bag exports `bag.json`, `occurrences.csv`, `profiles.json` and `composition.json`. Each scenario exports `ql.csv` (`id,text,ex time (secs)`), `arrival_time.csv` (`query_no,arrival_time`), `groups.csv`, `sampled_measurements.csv`, configuration, validation and hashes. These formats and seeded sampling/order/arrival algorithms match the existing dataset workloads. Use the generated scenario directory as a policy's `--workload-dir`.
