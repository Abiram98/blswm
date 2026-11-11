from dataclasses import dataclass
from typing import Protocol
import math

@dataclass(frozen=True)
class Query:
    query_id: str
    query_text: str

@dataclass(frozen=True)
class Occurrence:
    occurrence_id: str
    query_id: str
    arrival_s: float
    trace_index: int

@dataclass(frozen=True)
class Running:
    occurrence: Occurrence
    admission_s: float
    slot: int = 0
    cost: float = 1
    prediction: float = 0

@dataclass(frozen=True)
class Decision:
    occurrence: Occurrence
    slot: int = 0
    cost: float = 1
    prediction: float = 0
    reason: str = 'capacity'

class Predictor(Protocol):
    def predict(self, candidate: Occurrence, running: tuple[Running, ...], now: float) -> float: ...
    def predict_joint(self, candidate: Occurrence, running: tuple[Running, ...], now: float) -> dict[str, float]: ...

class QueryExecutor(Protocol):
    def execute(self, query_text: str) -> None: ...

def validate_predictions(values, keys, task):
    if set(values) != set(keys):
        raise ValueError('Prediction occurrence keys mismatch')
    if any(not math.isfinite(float(v)) for v in values.values()):
        raise ValueError('Nonfinite prediction')
    if task == 'classification' and any(v not in (0, 1) for v in values.values()):
        raise ValueError('Invalid class')
    return values

class FakePredictor:
    """Explicit synthetic fixture: values are predictions, never online labels."""
    def __init__(self, values=None, default=0):
        self.values, self.default = values or {}, default
    def predict(self, candidate, running=(), now=0):
        return self.values.get(candidate.query_id, self.default)
    def predict_joint(self, candidate, running=(), now=0):
        return {o.occurrence_id: self.predict(o, (), now) for o in [*(r.occurrence for r in running), candidate]}
