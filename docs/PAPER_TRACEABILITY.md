# Specification traceability

| task.md | Target | Origin | Verification |
|---|---|---|---|
| §0, §7 | io.py, packaging | new | output escapes, import isolation |
| §2 | catalog.py, workloads.py | DWNet/prof | boundaries, counts, deterministic schedules |
| §4 | policies.py, runtime.py | DWNet scheduling + iconqshed | synthetic admission and completion invariants |
| §3.1 | dataset.py | DWNet trace builder semantics | state ordering, labels, grouping |
| §3.2–3.3 | ml.py | sqpp architecture + specification | task metadata, reload, ordering |
| §5 | analysis.py | specified paired aggregation | S50, latency, completeness |
| §6–9 | configs, examples, tests | task.md | smoke, packaging, differential tests |
