# Validation

Run from the project root after setup:

```bash
source .local/env.sh
pytest tests/test_core.py tests/test_integration_offline.py tests/test_packaging.py
python examples/smoke.py
sparql-wm-bench analyze --runs runs/smoke --out reports/smoke
```

Core tests cover profiles, sampling, open-loop arrivals, admission rules, latency accounting, failure handling, protocol framing, HTTP full-result consumption, and output-path isolation. They require no database, Java, GPU, or trained checkpoint. The smoke workflow records 2,800 synthetic occurrences across four policies and two repeats.

For model and graph tests:

```bash
./setup.sh --ml --java
source .local/env.sh
pytest -q
```

The full suite has 52 tests, including five CPU train/save/reload/predict paths and a separately marked Java graph-extraction integration test. The Java test uses synthetic statistics and requires the built JAR; it skips when the JAR is unavailable. An environment with the optional dependencies and JAR passed all 52 tests.

Synthetic tests do not measure live engine performance. Each engine evaluation must validate result completeness, profiles, resource settings, checkpoint compatibility, and identical trace replay across policies. Preserve failed executions and reject incomplete or unmatched populations for paired analysis. GPU timing requires a compatible GPU environment and synchronized measurements.
