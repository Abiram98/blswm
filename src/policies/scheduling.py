"""Pure admission rules; the runtime owns query execution and policy state."""
from collections import Counter
from dataclasses import replace
import math
from .schema import Decision, validate_predictions

NAMES = ('fifo', 'rr', 'rtq', 'rtq-reeval', 'sqa', 'sqa-reeval', 'iconq', 'iconqshed')


class Policy:
    def __init__(self, name='fifo', mpl=10, budget=10, **options):
        if name not in NAMES or isinstance(mpl, bool) or not isinstance(mpl, int) or mpl < 1:
            raise ValueError('Invalid policy configuration')
        self.capacities = tuple(options.pop('class_capacities', (5, 5)))
        self.costs = tuple(options.pop('class_costs', (1., 3.)))
        self.spillover = options.pop('spillover', True)
        if len(self.capacities) != 2 or any(isinstance(c, bool) or not isinstance(c, int) or c < 1 for c in self.capacities):
            raise ValueError('Two positive integer class capacities are required')
        if len(self.costs) != 2 or any(not math.isfinite(c) or c <= 0 for c in self.costs):
            raise ValueError('Two positive finite class costs are required')
        if not math.isfinite(budget) or budget <= 0 or (name.startswith('sqa') and max(self.costs) > budget):
            raise ValueError('Budget must fit each class cost')
        self.name, self.mpl, self.budget = name, mpl, budget
        self.prediction_task = 'regression' if name in ('iconq', 'iconqshed') else 'classification'
        self.reconsider_interval = float(options.pop('reconsider_interval', 5.))
        if not math.isfinite(self.reconsider_interval) or self.reconsider_interval <= 0:
            raise ValueError('reconsider_interval must be positive and finite')
        self.options = options
        if self.prediction_task == 'regression':
            from .iconq import PolicyConfig
            PolicyConfig(max_concurrency_level=self.mpl, **self.options)

    def decide(self, waiting, running, now, predictor, state):
        # Per-decision updates are committed by the runtime only if the active
        # snapshot survives inference. Keep them separate from persistent state.
        state.pop('running_updates', None)
        state.pop('decision_details', None)
        if not waiting or len(running) >= self.mpl:
            return None
        if self.name == 'fifo':
            return Decision(waiting[0], reason='fifo')
        if self.prediction_task == 'regression':
            return self._iconq(waiting, running, now, predictor, state)
        if self.name == 'rr':
            return self._round_robin(waiting, running, now, predictor, state)

        physical = Counter(r.slot for r in running)
        for candidate in waiting:
            joint = None
            if self.name.endswith('reeval'):
                joint, running = self._reevaluate(candidate, running, now, predictor, state)
                predicted_class = joint[candidate.occurrence_id]
            else:
                predicted_class = predictor.predict(candidate, running, now)
            validate_predictions({candidate.occurrence_id: predicted_class}, [candidate.occurrence_id], 'classification')
            predicted_class = int(predicted_class)

            if self.name.startswith('rtq'):
                virtual = Counter(joint[r.occurrence.occurrence_id] for r in running) if joint else physical
                slots = range(predicted_class, 2) if self.spillover else (predicted_class,)
                for slot in slots:
                    if physical[slot] < self.capacities[slot] and virtual[slot] < self.capacities[slot]:
                        return Decision(candidate, slot=slot, prediction=predicted_class, reason='class_capacity')
            else:
                used = (sum(self.costs[int(joint[r.occurrence.occurrence_id])] for r in running)
                        if joint else sum(r.cost for r in running))
                cost = self.costs[predicted_class]
                if used + cost <= self.budget:
                    return Decision(candidate, cost=cost, prediction=predicted_class, reason='budget')
        return None

    def _reevaluate(self, candidate, running, now, predictor, state):
        joint = predictor.predict_joint(candidate, running, now)
        keys = [r.occurrence.occurrence_id for r in running] + [candidate.occurrence_id]
        validate_predictions(joint, keys, 'classification')
        values = {}
        for active in running:
            predicted_class = int(joint[active.occurrence.occurrence_id])
            values[active.occurrence.occurrence_id] = dict(
                prediction=predicted_class,
                cost=self.costs[predicted_class] if self.name == 'sqa-reeval' else active.cost,
            )
        if values:
            state.setdefault('running_updates', []).append(dict(
                candidate_occurrence_id=candidate.occurrence_id, values=values,
            ))
        # A later candidate in this decision sees the latest validated classes
        # and costs, while the caller's immutable snapshot stays unchanged.
        return joint, tuple(replace(r, **values[r.occurrence.occurrence_id]) for r in running)

    def _round_robin(self, waiting, running, now, predictor, state):
        classes = state.setdefault('classes', {})
        for occurrence in waiting:
            if occurrence.occurrence_id not in classes:
                value = predictor.predict(occurrence, running, now)
                validate_predictions({occurrence.occurrence_id: value}, [occurrence.occurrence_id], 'classification')
                classes[occurrence.occurrence_id] = value
        queues = [[o for o in waiting if classes[o.occurrence_id] == c] for c in (0, 1)]
        turn = state.get('turn', 0)
        selected = turn if queues[turn] else 1 - turn
        if all(queues):
            state['turn'] = 1 - turn
        occurrence = queues[selected][0]
        classes.pop(occurrence.occurrence_id)
        return Decision(occurrence, prediction=selected, reason='class_round_robin')

    def _iconq(self, waiting, running, now, predictor, state):
        from .iconq import IconQSchedPolicy, PolicyConfig, WaitingQuery, RunningQuery

        class Adapter:
            def predict_set(self, candidate, active, at):
                occurrence = next(o for o in waiting if o.occurrence_id == candidate.query_id)
                active_keys = {a.query_id for a in active}
                context = tuple(r for r in running if r.occurrence.occurrence_id in active_keys)
                return predictor.predict_joint(occurrence, context, at)

        baselines = state.setdefault('iconq_baselines', {})
        queued = [WaitingQuery(o.occurrence_id, o.query_id, o.arrival_s, baselines.get(o.occurrence_id)) for o in waiting]
        active = [RunningQuery(r.occurrence.occurrence_id, r.occurrence.query_id, r.admission_s,
                               r.prediction, r.admission_s + r.prediction) for r in running]
        config = PolicyConfig(max_concurrency_level=self.mpl, **self.options)
        scheduler = IconQSchedPolicy(Adapter(), config)
        clock = getattr(predictor, 'clock', None)
        if clock is None:
            decision = scheduler.select(queued, active, now)
        else:
            _, decision = scheduler.select_current(queued, active, clock)
        state['iconq_baselines'] = {q.query_id: q.baseline for q in queued if q.query_id != decision.query_id}
        state['decision_details'] = dict(reason=decision.reason, candidates=decision.candidates,
                                        predicted_runtime=decision.predicted_runtime)
        if decision.query_id is None:
            return None
        if decision.running_predictions:
            state['running_updates'] = [dict(candidate_occurrence_id=decision.query_id, values={
                r.occurrence.occurrence_id: dict(prediction=decision.running_predictions[r.occurrence.occurrence_id], cost=r.cost)
                for r in running
            })]
        occurrence = next(o for o in waiting if o.occurrence_id == decision.query_id)
        return Decision(occurrence, prediction=decision.predicted_runtime, reason=decision.reason)
