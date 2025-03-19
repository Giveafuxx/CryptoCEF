from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StrategyPerformance:
    window: Optional[any] = None
    threshold: Optional[any] = None
    sharpe: Optional[any] = None
    sortino: Optional[any] = None
    pos_avg_cor: Optional[any] = None
    pnl_avg_cor: Optional[any] = None
    para_weight: Optional[any] = None
    final_weight: Optional[any] = None
    calmar: Optional[any] = None
    mdd: Optional[any] = None
    trade: Optional[any] = None
    long_trades: Optional[any] = None
    short_trades: Optional[any] = None
    annual_return: Optional[any] = None


@dataclass
class ExecutionDetail:
    front_run: bool = False
    resolution: str = "24h"
    delay_mode: bool = False
    delay_time: int = 10
    prd_mode: bool = False


@dataclass
class StrategyDetail:
    strategy_code: Optional[any] = None
    model: Optional[any] = None
    direction: Optional[any] = None
    resolution: Optional[any] = None
    long_short: Optional[any] = None
    currency: str = "NATIVE"
    window: Optional[any] = None
    threshold: Optional[any] = None


@dataclass
class MetricDetail:
    metric: Optional[any] = None
    asset_input: str = "BTC"
    asset_output: Optional[str] = None
    metric_type: str = "close"
    data_source: str = "glassnode"
    front_run_mode: str = "last"
    transformType: str = "raw"
    transformPeriod: Optional[any] = None
    operator: Optional[any] = None

    def __post_init__(self):
        if self.asset_output is None:
            self.asset_output = self.asset_input
