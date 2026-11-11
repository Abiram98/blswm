# Porting report

## Initial audit
Target: `/blswm`. Existing README inspected; task.md preserved. Source mounts are read-only.
No LICENSE/COPYING files were found in either source tree: copied implementations retain provenance but no redistribution license is asserted.
DWNet contains pre-existing changes in analysis scripts; these are preserved.
/data contains DBpedia profile/execution logs and CQPP model artifacts; availability is not proof of paper-compatible data, thresholds or checkpoints.

Selected origins: DWNet prof/arrival_time.py (burst and transition semantics), wrk_man/policy/scheduling (capacity rules), iconqshed/policy.py (lookahead scoring); sqpp models and q_graph (logical graph features).
Specification overrides: MPL 10 rather than IconQ source default 20; short bypass disabled; physical capacities enforced during reevaluation.
JointDWNS and JSONL trainer inspected: joint classification exists; candidate-only regression in dwns.py does not meet all-position regression contract. Implement joint RMSE explicitly.

## Implemented extraction and deliberate changes

`docs/SOURCE_MAP.json` records copied file hashes. Only the logical graph dependency closure and its Maven sources were copied from sqpp; no source history, dataset, checkpoint or built binary is included. Python imports were namespaced under `sparql_wm_bench.vendor`. The Java adapter uses explicit configuration and lazy JVM startup.

New graph RGCN layers use the source two-layer ReLU + mean-pooling semantics with 13 relations. The all-position bidirectional LSTM follows JointDWNS base, without elapsed/role embeddings. The source trainer's cross-entropy path is replaced by a small explicit task registry supporting joint CE and joint RMSE. Candidate-only regression source classes were not mislabeled as joint regression. P+IconQ was not located by searches of both source repositories' Python files; target conditioning follows task.md. Unknown historical hyperparameters are configurable implementation defaults (128 hidden units, one LSTM layer, dropout .1, Adam .001, 10 epochs), not asserted paper values.

IconQ policy is directly adapted from DWNet/iconqshed/policy.py. The wrapper uses occurrence IDs as policy keys and validates joint vectors. Paper overrides MPL10 and disables short bypass; source constants wait_floor=3 and overrun_extension_factor=.15 are retained. A 1e-6 seconds floor is applied only by the scheduling predictor wrapper and both raw and floored values are logged.

The source Featurizer2 `scaling="None"` path attempted to concatenate zero-dimensional statistic scalars. The port wraps these with `np.atleast_1d`; Java→logical graph→DGL integration exposed and verifies this correction. The binned feature representation is preserved. Source profile predicates support additional CPU bounds and families; the port implements only the specification's disjoint profiles, including S ignoring CPU and L100 exclusion.

Schedules preserve source initial-gap, previous-group transition-gap, and fixed burst endpoint semantics. Independent selection/arrival seeds and bag/schedule hashes are persisted. For text disjointness the artifact uses exact stripped query text: whitespace inside SPARQL literals is never rewritten. Semantically equivalent but differently formatted query text is not canonicalized by a full SPARQL equivalence engine; imported identity quality remains a prerequisite.

## Verification and explicit limits

Editable core installation succeeded in a repository-local virtual environment. Unit/offline integration tests cover profiles, selection, all patterns, all policy replay paths, slot/budget invariants, inference charging, FIFO ordering, errors, path escapes, grouped CQPP states, paired metrics, HTTP complete-body consumption, protocol framing, and five CPU train/reload/predict paths. Actual-source differential tests cover Poisson timing/burst endpoints, RTQ spillover capacities, SQA costs and IconQ choices. These are behavioral checks, not exhaustive source equivalence: original source JSONL and server-decoding paths have not been differentially verified. RR queue choices follow the inspected source choose() rule.

The Maven build succeeded from copied Java17 sources with repository-local Maven cache and temporary directory. A separately marked integration test uses generated empty Meta-KG maps with unscaled features and verifies Java query extraction and DGL conversion. It does not certify real KG statistics. Core imports have no Torch/DGL/JPype dependency, JVM startup or live endpoint initialization.

Unverified external work: real Virtuoso runs/result-set correctness, original KG and query-population validation, original checkpoint conversion, GPU timing matrix and all paper numerical results. The available /data/models artifacts include thresholds other than 120 seconds in names; they were not silently loaded or declared compatible. No database, GPU or source checkpoint is necessary for synthetic acceptance tests.

Unimplemented optional capability: automatic isolated Virtuoso+Collectl profiling orchestration. The source profiling stack includes remote/service-specific infrastructure (prof/isolated_cpu.py, prof/runner.py); no portable engine/collector configuration is provided. Preprofiled catalog import is the supported path; callers must supply 10-ms, per-core peak CPU values. Runtime HTTP endpoint execution is implemented separately. Source license terms are absent from the discovered trees; provenance is retained and no license to redistribute third-party sources is inferred.

Analysis emits unavailable diagnostics explicitly and rejects incomplete populations. Full-result HTTP consumption does not prove semantic result correctness without an external oracle. Graph caches are static, in-memory and scoped to one provider; context-dependent predictions are never persisted across decisions. Checkpoints are portable within this artifact's versioned schema, not wire/state-dict compatible with arbitrary legacy checkpoints.
