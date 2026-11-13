"""Python configuration objects for the analysis commands."""
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

from .workloads import WorkloadAnalyzer, WorkloadRunConfig


@dataclass
class PreparationConfig:
    runs: Sequence[str | Path]
    output: str | Path
    time_base: str = 'epoch'
    resource_interval: float = .1

    def run(self):
        from .prepare import prepare
        return prepare(self.runs,self.output,time_base=self.time_base,resource_interval=self.resource_interval)


@dataclass
class WorkloadAnalysisConfig:
    prepared: str | Path
    runs: Sequence[WorkloadRunConfig]
    output: str | Path
    metric: str = 'latency_s'
    reference: str | None = None
    include_incomplete: bool = False

    def run(self):
        return WorkloadAnalyzer(self.runs,self.prepared).analyze(self.output,self.metric,self.reference,self.include_incomplete)


@dataclass
class PredictionAnalysisConfig:
    data: str | Path
    output: str | Path
    report: str | Path | None = None
    host: str = '127.0.0.1'
    port: int | None = None
    timeout: float = 60.
    thresholds: Sequence[float] | None = None

    def run(self):
        from .predictions import analyze
        if (self.report is None)==(self.port is None):raise ValueError('Specify exactly one of report or port')
        if self.port is not None and (isinstance(self.port,bool) or not isinstance(self.port,int) or not 1<=self.port<=65535):raise ValueError('Invalid TCP port')
        if not math.isfinite(self.timeout) or self.timeout<=0:raise ValueError('Timeout must be positive')
        thresholds=list(self.thresholds) if self.thresholds is not None else None
        return analyze(self.data,self.output,report=self.report,host=self.host,port=self.port,timeout=self.timeout,thresholds=thresholds)


@dataclass
class PredictionComparisonConfig:
    results: Sequence[str | Path]
    output: str | Path

    def run(self):
        from .model_comparison import compare
        return compare(self.results,self.output)
