"""GreedyScheduler's policy, independent of SPARQL execution and model serving.

Reference: /IconqSched/scheduler/greedy_scheduler.py, especially ingest_query.
All runtime estimates here originate from the supplied predictor, never labels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class PolicyConfig:
    max_concurrency_level: int = 20
    alpha: float = 0.2
    starve_penalty: float = 0.5
    steps_into_future: int = 2
    ignore_short_running: bool = False
    short_running_threshold: float = 5.0
    future_wait_floor: float = 3.0
    overrun_extension_factor: float = 0.15

    def __post_init__(self):
        if self.max_concurrency_level < 1 or self.steps_into_future < 1:
            raise ValueError("max_concurrency_level and steps_into_future must be positive")
        for name in ("alpha", "starve_penalty", "short_running_threshold", "future_wait_floor"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.overrun_extension_factor) or self.overrun_extension_factor <= 0:
            raise ValueError("overrun_extension_factor must be finite and positive")


@dataclass
class WaitingQuery:
    query_id: str
    text: str
    arrived_at: float
    baseline: float | None = None


@dataclass
class RunningQuery:
    query_id: str
    text: str
    started_at: float
    predicted_runtime: float
    finish_at: float


@dataclass
class Decision:
    query_id: str | None
    reason: str
    predicted_runtime: float | None = None
    running_predictions: dict[str, float] = field(default_factory=dict)
    candidates: list[dict] = field(default_factory=list)


class IconQSchedPolicy:
    def __init__(self, predictor, config: PolicyConfig = PolicyConfig()):
        self.predictor = predictor
        self.config = config

    def select(self, waiting: list[WaitingQuery], running: list[RunningQuery], now: float) -> Decision:
        return self._select(waiting, running, now, self._predict_scenario)

    def _predict_scenario(self, candidate, running, at):
        """One ordered set, one vector of total runtimes, including outgoing effects."""
        if hasattr(self.predictor, "predict_set"):
            return self.predictor.predict_set(candidate, running, at)
        # Compatibility for scalar test fixtures and the historical DWNS adapter.
        # Production uses QSPSRuntimePredictor, never this retargeting path.
        values = {candidate.query_id: self.predictor.predict(candidate.text, [r.text for r in running])}
        values.update({r.query_id: self.predictor.predict(
            r.text, [other.text for other in running if other.query_id != r.query_id] + [candidate.text])
            for r in running})
        return values

    def select_current(self, waiting, running, clock):
        """Collect predictions, then score at a fresh clock without more inference.

        Keep all joint vectors for this decision even if the bounded LRU evicts
        them. A fresh clock can change next-completion ordering; collect any new
        scenario, then refresh again before returning. State-aware models sample
        elapsed features once per scenario in this decision (not indefinitely
        chasing the clock during slow inference). The tested base model ignores
        elapsed features; later reconsiderations refresh state-aware predictions.
        Speculative passes must not alter the running queries' finish estimates.
        """
        predictions = {}

        def predict(candidate, running, at):
            key = (candidate.query_id, candidate.text,
                   tuple((r.query_id, r.text) for r in running))
            if key not in predictions:
                predictions[key] = self._predict_scenario(candidate, running, at)
            return predictions[key]

        snapshot = [replace(r) for r in running]
        while True:
            now = clock()
            previous_count = len(predictions)
            working = [replace(r) for r in snapshot]
            decision = self._select(waiting, working, now, predict)
            if len(predictions) == previous_count:
                for original, refreshed in zip(running, working):
                    original.finish_at = refreshed.finish_at
                return now, decision

    def _select(self, waiting, running, now, predict):
        cfg = self.config
        if not waiting:
            return Decision(None, "empty_queue")
        if len(running) >= cfg.max_concurrency_level:
            return Decision(None, "concurrency_limit")
        for q in waiting:
            if q.baseline is None:
                # Empty context replaces Stage's average-runtime estimate.
                q.baseline = predict(q, [], now)[q.query_id]
        if cfg.ignore_short_running:
            for q in waiting:
                if q.baseline < cfg.short_running_threshold:
                    return Decision(q.query_id, "short_query", q.baseline)
        if not running:
            q = min(waiting, key=lambda q: q.baseline)
            return Decision(q.query_id, "idle_shortest_query", q.baseline)

        # The original real-execution branch extends imminent/overdue estimates
        # with random jitter. Use a fixed factor close to its mean for reproducibility.
        for r in running:
            if r.finish_at < now + cfg.future_wait_floor:
                r.finish_at = now + cfg.overrun_extension_factor * r.predicted_runtime
        future = sorted(running, key=lambda r: r.finish_at)[:cfg.steps_into_future]
        old_predictions = {r.query_id: max(r.predicted_runtime, now - r.started_at)
                           for r in running}
        rows = []
        eligible = []
        for q in waiting:
            joint = predict(q, running, now)
            current = joint[q.query_id]
            new_predictions = {r.query_id: joint[r.query_id] for r in running}
            delta_own = current - q.baseline
            delta_running = sum(new_predictions[r.query_id] - old_predictions[r.query_id]
                                for r in running)
            removed = set()
            comparisons = []
            for next_finish in future:
                removed.add(next_finish.query_id)
                survivors = [r for r in running if r.query_id not in removed]
                # Preserve running-query order and roles. The candidate remains
                # last even when reading another query's outgoing-effect output.
                later_joint = predict(q, survivors, next_finish.finish_at)
                later = later_joint[q.query_id]
                own_delta = (current - later
                             - max(next_finish.finish_at - now, cfg.future_wait_floor)
                             * max(len(waiting) - 1, 1))
                running_delta = sum(
                    new_predictions[r.query_id] - later_joint[r.query_id]
                    for r in survivors
                )
                future_starvation = max(next_finish.finish_at - q.arrived_at, 0.1) * cfg.starve_penalty
                comparisons.append(dict(finish_query_id=next_finish.query_id,
                                        finish_at=next_finish.finish_at,
                                        own_delta=own_delta, running_delta=running_delta,
                                        starvation_penalty=future_starvation,
                                        delta=own_delta + running_delta - future_starvation))
            weighted_benefit = cfg.alpha * delta_own + delta_running
            should_submit = (weighted_benefit < -current
                             or max(row["delta"] for row in comparisons) < 0
                             or (cfg.ignore_short_running and current < cfg.short_running_threshold))
            score = weighted_benefit - max(0.0, now - q.arrived_at) * cfg.starve_penalty
            row = dict(query_id=q.query_id, baseline=q.baseline, predicted_runtime=current,
                       delta_own=delta_own, delta_running=delta_running, score=score,
                       should_submit=should_submit, future_comparisons=comparisons,
                       running_predictions=new_predictions)
            rows.append(row)
            if should_submit:
                eligible.append(row)
        if not eligible:
            return Decision(None, "wait_for_event", candidates=rows)
        best = min(eligible, key=lambda row: row["score"])
        return Decision(best["query_id"], "minimum_score", best["predicted_runtime"],
                        best["running_predictions"], rows)
