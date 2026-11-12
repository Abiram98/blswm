"""Policy-facing client; occurrence IDs are kept separate from query identities."""
import math
from .protocol import request, VERSION


class RemotePredictor:
    def __init__(self, catalog, host='127.0.0.1', port=9010, timeout=60, thresholds=(120.,), task='classification'):
        self.texts = {str(q['query_id']): q['query_text'] for q in catalog}
        self.host, self.port, self.timeout = host, port, timeout
        self.thresholds = list(thresholds)
        if task not in ('classification', 'regression'):
            raise ValueError('Unknown prediction task')
        self.task = task
        self.metadata = self._request(dict(type='info'))
        self._check(self.metadata)

    def _request(self, payload):
        return request(self.host, self.port, payload, self.timeout)

    def _check(self, response):
        common = (response.get('protocol_version') == VERSION
                  and response.get('prediction_task') == self.task
                  and response.get('returns_per_query_predictions') is True)
        if self.task == 'regression':
            valid = (response.get('prediction_unit') == 'seconds'
                     and response.get('runtime_semantics') == 'total_runtime'
                     and response.get('output_dim') == 1)
        else:
            valid = (response.get('prediction_unit') == 'class_index'
                     and response.get('runtime_semantics') == 'runtime_interval'
                     and response.get('runtime_thresholds') == self.thresholds
                     and response.get('num_classes') == len(self.thresholds) + 1)
        if not common or not valid:
            raise ValueError(f'Predictor {self.task} contract mismatch')

    def predict_joint(self, candidate, running, now):
        active = sorted(running, key=lambda r: (r.admission_s, r.occurrence.trace_index))
        queries = [dict(query_id=r.occurrence.occurrence_id, query_text=self.texts[r.occurrence.query_id],
                        role='running', pos=i, elapsed_at_decision=max(0., now-r.admission_s))
                   for i, r in enumerate(active)]
        queries.append(dict(query_id=candidate.occurrence_id, query_text=self.texts[candidate.query_id],
                            role='candidate', pos=len(queries), elapsed_at_decision=0.))
        response = self._request(dict(type='predict', queries=queries))
        self._check(response)
        predictions = response.get('predictions')
        if not isinstance(predictions, list) or len(predictions) != len(queries):
            raise ValueError('Predictor returned an incomplete query set')
        values = {}
        for item in predictions:
            index = item.get('input_index')
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(queries):
                raise ValueError('Invalid prediction input index')
            key = queries[index]['query_id']
            if key in values or item.get('query_id') != key:
                raise ValueError('Prediction identities do not match request')
            if self.task == 'regression':
                value = item.get('predicted_runtime_seconds')
                if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                    raise ValueError('Invalid runtime prediction')
                # Preserve raw predictions; the scheduler adapter logs and floors
                # nonpositive estimates before using them in timing arithmetic.
                values[key] = float(value)
                continue
            cls = item.get('predicted_class')
            if key in values or item.get('query_id') != key or isinstance(cls, bool) or not isinstance(cls, int) or not 0 <= cls <= len(self.thresholds):
                raise ValueError('Prediction identities or classes do not match request')
            probs = item.get('probabilities', {})
            if (set(probs) != {str(i) for i in range(len(self.thresholds)+1)}
                    or any(not math.isfinite(float(v)) or not 0 <= float(v) <= 1 for v in probs.values())
                    or not math.isclose(sum(probs.values()), 1., abs_tol=1e-5)):
                raise ValueError('Invalid class probabilities')
            values[key] = cls
        return values

    def predict(self, candidate, running=(), now=0.):
        return self.predict_joint(candidate, running, now)[candidate.occurrence_id]
