# Executed validation

All commands ran from /blswm. Installation and test commands used local TMPDIR/cache/pycache settings documented in README.

- `python -m venv --system-site-packages .venv`
- `.venv/bin/pip install -e '.[test]'` — succeeded.
- `.venv/bin/python -m pytest --basetemp=/blswm/.local/pytest-release2 -q` — 52 passed, including the separately marked Java integration test.
- `MAVEN_OPTS=-Djava.io.tmpdir=/blswm/.local/tmp mvn -q -f java/qpe/pom.xml -Dmaven.repo.local=/blswm/.local/m2 -DskipTests package` — succeeded.
- `.venv/bin/python examples/smoke.py` — four policies × two repeats × 350 offered occurrences, all successful synthetic executions.
- `.venv/bin/sparql-wm-bench analyze --runs runs/smoke --out reports/smoke --plot` — succeeded after replacing the newer matplotlib `tick_labels` argument with supported `labels`.
- `.venv/bin/python -m sparql_wm_bench --help` — succeeded.
- `.venv/bin/python -m compileall -q src` — succeeded.

Initial test failures found simulator final-completion termination and missing graph dependency imports; these were corrected. The Java graph integration additionally found the source unscaled scalar concatenation bug, corrected in the port. Earlier failing runs are not counted as passing evidence.

Source Git status was inspected before and after: sqpp stayed clean, and DWNet's pre-existing modified/untracked analysis files remained. No writes were intentionally made to source/data mounts. This is not a byte-for-byte filesystem checksum attestation: a complete pre-run checksum of all mounts was not captured.

No original paper numerical experiment or external Virtuoso/GPU experiment was executed. Generated runs, figures, binaries, caches, environments and checkpoints are ignored by Git.

Final smoke and CLI checks were rerun after setting IDs and predictor-error accounting were added. `run --catalog examples/catalog.json --trace examples/trace.csv --policy fifo --mpl 1 --synthetic-durations examples/durations.json --out runs/tiny` also succeeded.
