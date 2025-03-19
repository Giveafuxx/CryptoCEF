import logging
import pandas as pd
from typing import Tuple, Dict

from parameter import StrategyPerformance
from analyzer import PerformanceAnalyzer

logger = logging.getLogger(__name__)


def cal_pnl(temp_df: pd.DataFrame):
    temp_df["pos_t-1"] = temp_df["pos"].shift(1)
    temp_df["trade"] = abs(temp_df["pos"] - temp_df["pos_t-1"])
    temp_df["cost"] = temp_df["trade"] * 0.06 / 100

    # Calculate number of long and short trades
    temp_df["long_trade"] = ((temp_df["pos"] > temp_df["pos_t-1"]) & (temp_df["pos"] != 0)).astype(int)
    temp_df["short_trade"] = ((temp_df["pos"] < temp_df["pos_t-1"]) & (temp_df["pos"] != 0)).astype(int)

    temp_df["pnl"] = temp_df["pos_t-1"] * temp_df["chg"] - temp_df["cost"]

    return temp_df


def backtesting(
    window: int,
    threshold: float,
    df: pd.DataFrame,
    resolution: str,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Perform backtesting for a given model and parameters.

    Args:
        window (int): The window size for the model.
        threshold (float): The threshold value for the model.
        df (pd.DataFrame): The input data frame containing market data.
        model (str): The model to be used for backtesting.
        long_or_short (str): Indicates whether the strategy is long or short.
        direction (str): The direction of the trade (e.g., 'reversion, 'momentum').
        resolution (str): The time resolution for the backtest (e.g., '24h', '1h', '10m').
        daily_rebalancing (bool): Whether to perform daily rebalancing.

    Returns:
        Tuple[pd.Series, pd.DataFrame]: A series containing the backtest results and the modified data frame.
    """
    try:
        df["cumu"] = df["pnl"].cumsum()
        df["dd"] = df["cumu"].cummax() - df["cumu"]

        df["bnh_pnl"] = df["chg"]

        df.iloc[0 : (window - 1), df.columns.get_loc("bnh_pnl")] = 0
        df["bnh_cumu"] = df["bnh_pnl"].cumsum()

        performance_info = StrategyPerformance(
            window=window,
            threshold=threshold,
            sharpe=PerformanceAnalyzer.Calculate.sharpe_ratio(df, resolution, window),
            sortino=PerformanceAnalyzer.Calculate.sortino_ratio(df, resolution, window),
            calmar=PerformanceAnalyzer.Calculate.calmar_ratio(df, window, resolution),
            mdd=PerformanceAnalyzer.Calculate.mdd(df),
            trade=df["long_trade"].sum() + df["short_trade"].sum(),
            long_trades=df["long_trade"].sum(),
            short_trades=df["short_trade"].sum(),
            annual_return=PerformanceAnalyzer.Calculate.annual_return(df, window, resolution)
        )

        return performance_info, df
    except Exception as e:
        logger.error(
            f"Error in backtesting with parameters: window={window}, threshold={threshold}, resolution={resolution}"
        )
        logger.error(f"Error: {e}")
        raise
