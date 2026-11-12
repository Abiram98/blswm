"""Open-loop replay with a single controller and measured HTTP execution."""
import copy
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from urllib.request import Request, urlopen
from urllib.parse import urlencode
import hashlib
import json

def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()
from .schema import Occurrence, Running, validate_predictions

class HTTPExecutor:

    def __init__(self, endpoint, timeout=300, headers=None):
        self.endpoint, self.timeout, self.headers = (endpoint, timeout, headers or {})

    def execute(self, text):
        req = Request(self.endpoint, data=urlencode({'query': text}).encode(), headers={'Accept': 'application/sparql-results+json', **self.headers})
        with urlopen(req, timeout=self.timeout) as response:
            while response.read(65536):
                pass

class MeasuredPredictor:

    def __init__(self, predictor, clock, logs, task='classification'):
        self.predictor, self.clock, self.logs, self.task = (predictor, clock, logs, task)

    def _call(self, method, candidate, running, now):
        start = self.clock()
        keys = [r.occurrence.occurrence_id for r in running] + [candidate.occurrence_id]
        result = getattr(self.predictor, method)(candidate, running, now)
        values = result if method == 'predict_joint' else {candidate.occurrence_id: result}
        validate_predictions(values, keys if method == 'predict_joint' else [candidate.occurrence_id], self.task)
        raw_values = dict(values)
        if self.task == 'regression':
            values = {k: max(1e-06, float(v)) for k, v in values.items()}
            result = values if method == 'predict_joint' else values[candidate.occurrence_id]
        self.logs.append(dict(method=method, occurrence_id=candidate.occurrence_id, state_keys=keys, start_s=start, duration_s=self.clock() - start, predictions=values, raw_predictions=raw_values, regression_floor_s=1e-06 if self.task == 'regression' else None))
        return result

    def predict(self, *args):
        return self._call('predict', *args)

    def predict_joint(self, *args):
        return self._call('predict_joint', *args)

def replay(trace, catalog, policy, predictor, *, executor, run_id='run1'):
    """Replay open arrivals with measured predictions and real executor completions."""
    occurrences = [Occurrence(str(r['occurrence_id']), str(r['query_id']), float(r['arrival_s']), int(r['trace_index'])) for r in trace]
    if len({o.trace_index for o in occurrences}) != len(occurrences):
        raise ValueError('Duplicate trace index')
    if len({o.occurrence_id for o in occurrences}) != len(occurrences):
        raise ValueError('Duplicate occurrence')
    if any((not math.isfinite(o.arrival_s) or o.arrival_s < 0 for o in occurrences)):
        raise ValueError('Invalid arrival')
    occurrences.sort(key=lambda o: (o.arrival_s, o.trace_index))
    queries = {r['query_id']: r['query_text'] for r in catalog}
    metadata = {k: catalog[0].get(k) if catalog else None for k in ('dataset', 'workload', 'scenario')}
    if any((o.query_id not in queries for o in occurrences)):
        raise ValueError('Missing query text')
    if executor is None:
        raise ValueError('Real replay requires executor')
    origin = time.monotonic()
    now = 0.0
    cursor = 0
    waiting = []
    running = []
    results = []
    predictions = []
    decisions = []
    state = {}
    pending = []
    running_updates = []
    admission_values = {}
    clock = lambda: time.monotonic() - origin
    measured = MeasuredPredictor(predictor, clock, predictions, policy.prediction_task)
    pool = ThreadPoolExecutor(max_workers=policy.mpl)

    def work(text):
        try:
            executor.execute(text)
            status = 'success'
            error = None
        except Exception as e:
            status = 'failed'
            error = f'{type(e).__name__}: {e}'
        return (clock(), status, error)

    def complete(r, end, status='success', error=None):
        # Pending executions retain admission snapshots; reevaluation updates the running record.
        r = next((active for active in running if active.occurrence.occurrence_id == r.occurrence.occurrence_id))
        o = r.occurrence
        results.append(dict(run_id=run_id, policy=policy.name, occurrence_id=o.occurrence_id, query_id=o.query_id, scheduled_arrival=o.arrival_s, arrival_s=o.arrival_s, admission_s=r.admission_s, completion_s=end, wait_s=r.admission_s - o.arrival_s, execution_s=end - r.admission_s, latency_s=end - o.arrival_s, status=status, error=error, dataset=metadata.get('dataset'), workload=metadata.get('workload'), scenario=metadata.get('scenario'), slot=r.slot, cost=r.cost, prediction=r.prediction, admission_cost=admission_values[o.occurrence_id]['cost'], admission_prediction=admission_values[o.occurrence_id]['prediction']))
        running.remove(r)

    def commit_running_updates(updates, admitted):
        for update in updates:
            values = update['values']
            refreshed = []
            for active in running:
                key = active.occurrence.occurrence_id
                latest = replace(active, **values[key])
                running_updates.append(dict(time_s=now, decision_index=len(decisions) - 1, candidate_occurrence_id=update['candidate_occurrence_id'], candidate_admitted=update['candidate_occurrence_id'] == admitted, occurrence_id=key, slot=active.slot, previous_prediction=active.prediction, prediction=latest.prediction, previous_cost=active.cost, cost=latest.cost))
                refreshed.append(latest)
            running[:] = refreshed
    controller_error = None
    previous_snapshot = None
    previous_decision_s = -math.inf
    try:
        while cursor < len(occurrences) or waiting or running:
            now = clock()
            for future, r in list(pending):
                if future.done():
                    end, status, error = future.result()
                    complete(r, end, status, error)
                    pending.remove((future, r))
            while cursor < len(occurrences) and occurrences[cursor].arrival_s <= now:
                waiting.append(occurrences[cursor])
                cursor += 1
            if cursor == len(occurrences) and (not waiting) and (not running):
                break
            snapshot = (tuple((o.occurrence_id for o in waiting)), tuple((r.occurrence.occurrence_id for r in running)))
            if snapshot == previous_snapshot and (policy.prediction_task != 'regression' or now - previous_decision_s < policy.reconsider_interval):
                time.sleep(0.005)
                continue
            previous_snapshot = snapshot
            previous_decision_s = now
            started = clock()
            proposed_state = copy.deepcopy(state)
            d = policy.decide(tuple(waiting), tuple(running), now, measured, proposed_state)
            now = clock()
            # Inference may span completions; reconcile them before committing a decision.
            stale = any((entry[0].done() for entry in pending))
            admitted = d.occurrence.occurrence_id if d and (not stale) else None
            decisions.append(dict(time_s=now, duration_s=now - started, waiting=len(waiting), running=len(running), admitted=admitted, committed=not stale, details=proposed_state.pop('decision_details', {})))
            if stale:
                previous_snapshot = None
                continue
            commit_running_updates(proposed_state.pop('running_updates', []), admitted)
            state = proposed_state
            if d:
                waiting.remove(d.occurrence)
                r = Running(d.occurrence, now, d.slot, d.cost, d.prediction)
                running.append(r)
                admission_values[d.occurrence.occurrence_id] = dict(cost=d.cost, prediction=d.prediction)
                pending.append((pool.submit(work, queries[d.occurrence.query_id]), r))
                continue
            if waiting and (not running) and (cursor == len(occurrences)):
                raise RuntimeError('Policy deadlock')
            delay = min(0.01, max(0, occurrences[cursor].arrival_s - now)) if cursor < len(occurrences) else 0.001
            time.sleep(delay)
    except Exception as exc:
        controller_error = f'{type(exc).__name__}: {exc}'
        for future, r in pending:
            end, status, error = future.result()
            complete(r, end, status, error)
        recorded = {r['occurrence_id'] for r in results}
        for o in occurrences:
            if o.occurrence_id not in recorded:
                results.append(dict(run_id=run_id, policy=policy.name, occurrence_id=o.occurrence_id, query_id=o.query_id, scheduled_arrival=o.arrival_s, arrival_s=o.arrival_s, admission_s=None, completion_s=None, wait_s=None, execution_s=None, latency_s=None, status='controller_failure', error=controller_error, dataset=metadata.get('dataset'), workload=metadata.get('workload'), scenario=metadata.get('scenario'), slot=None, cost=None, prediction=None, admission_cost=None, admission_prediction=None))
    finally:
        if pool:
            pool.shutdown(wait=True)
    return dict(records=results, predictions=predictions, decisions=decisions, running_updates=running_updates, trace=trace, run_id=run_id, controller_error=controller_error, setting_id=f'{policy.name}_mpl{policy.mpl}', mpl=policy.mpl, catalog_hash=digest(queries), mode='real', complete=controller_error is None and len(results) == len(occurrences) and all((r['status'] == 'success' for r in results)))
