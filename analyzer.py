import numpy as np
import pandas as pd
from statsmodels.regression.rolling import RollingOLS
from dataclasses import dataclass
from typing import Tuple, List
import logging


logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    class Calculate:
        @staticmethod
        def annual_return(
            df: pd.DataFrame, window: int, resolution: str = "24h"
        ) -> float:
            """
            Calculate the annualized return based on the specified time resolution.

            Parameters:
            - df (pd.DataFrame): DataFrame containing a 'pnl' column with returns.
            - resolution (str): Time resolution for returns ("24h", "1h", "10m").

            Returns:
            - float: Annualized return rounded to three decimal places.
            """
            # count the number of row in the dataframe
            avg_return = PerformanceAnalyzer.Calculate.avg_return(df, window)

            if resolution == "24h":
                return round(avg_return * 365, 3)
            elif resolution == "1h":
                return round(avg_return * 365 * 24, 3)
            elif resolution == "10m":
                return round(avg_return * 365 * 24 * 6, 3)
            else:
                return 0

        @staticmethod
        def avg_return(df: pd.DataFrame, window: int) -> float:
            """
            Calculate the average return over a specified window.

            Parameters:
            - df (pd.DataFrame): DataFrame containing a 'pnl' column with returns.
            - window (int): The number of observations to consider from the DataFrame.

            Returns:
            - float: The average return over the specified window.
            """
            return df.loc[window : len(df), "pnl"].mean()

        @staticmethod
        def calmar_ratio(df: pd.DataFrame, window, resolution: str = "24h") -> float:
            """
            Calculate the Calmar Ratio (annual return divided by maximum drawdown).

            Parameters:
            - df (pd.DataFrame): DataFrame containing 'pnl' and 'dd' columns.
            - resolution (str): Time resolution for returns ("24h", "1h", "10m").

            Returns:
            - float: Calmar Ratio rounded to three decimal places.
            """
            annual_return = PerformanceAnalyzer.Calculate.annual_return(
                df, window, resolution
            )
            mdd = PerformanceAnalyzer.Calculate.mdd(df)
            return round(annual_return / mdd, 3) if mdd != 0 else 0

        @staticmethod
        def downside_deviation(returns: pd.DataFrame) -> float:
            """
            Calculate the downside deviation of returns (only considering negative deviations).

            Parameters:
            - returns (pd.DataFrame): Series or DataFrame column of returns.

            Returns:
            - float: Downside deviation.
            """
            return np.sqrt(np.mean(np.minimum(0, returns) ** 2))

        @staticmethod
        def mdd(df: pd.DataFrame) -> float:
            """
            Calculate the maximum drawdown from the 'dd' column.

            Parameters:
            - df (pd.DataFrame): DataFrame containing a 'dd' column.

            Returns:
            - float: Maximum drawdown, rounded to three decimal places.
            """
            return round(df["dd"].max(), 3)

        @staticmethod
        def returns(df: pd.DataFrame) -> float:
            """
            Calculate the total return from the 'cumu' column.

            Parameters:
            - df (pd.DataFrame): DataFrame containing a 'cumu' column.

            Returns:
            - float: Final cumulative return.
            """
            return df["cumu"].iloc[-1]

        @staticmethod
        def sd(df: pd.DataFrame, window: int) -> float:
            """
            Calculate the standard deviation of returns over a specified window.

            Parameters:
            - df (pd.DataFrame): DataFrame containing a 'pnl' column with returns.
            - window (int): The number of observations to consider from the DataFrame.

            Returns:
            - float: Standard deviation of returns over the specified window.
            """
            return df.loc[window : len(df), "pnl"].std()

        @staticmethod
        def sharpe_ratio(df: pd.DataFrame, resolution: str, window: int) -> float:
            """
            Calculate the Sharpe ratio of returns.

            Parameters:
            - df (pd.DataFrame): DataFrame containing a 'pnl' column with returns.
            - resolution (str): Time resolution for returns ("24h", "1h", "10m").
            - window (int): The number of observations to consider from the DataFrame.

            Returns:
            - float: Sharpe Ratio rounded to three decimal places.
            """
            avg_return = PerformanceAnalyzer.Calculate.avg_return(df, window)
            sd = PerformanceAnalyzer.Calculate.sd(df, window)

            if sd == 0:
                return 0
            else:
                if resolution == "24h":
                    return round(avg_return / sd * np.sqrt(365), 3)
                elif resolution == "1h":
                    return round(avg_return / sd * np.sqrt(365 * 24), 3)
                elif resolution == "10m":
                    return round(avg_return / sd * np.sqrt(365 * 24 * 6), 3)
                else:
                    return 0

        @staticmethod
        def rolling_sharpe_ratio(
            df: pd.DataFrame, resolution: str, window: int
        ) -> pd.DataFrame:
            if resolution == "24h":
                num_of_timeframe = window
            elif resolution == "1h":
                num_of_timeframe = window * 24
            elif resolution == "10m":
                num_of_timeframe = window * 24 * 6
            df["rolling_sharpe"] = (
                df["pnl"].rolling(window=window).mean()
                * num_of_timeframe
                / (df["pnl"].rolling(window=window).std() * np.sqrt(num_of_timeframe))
            )
            return df["rolling_sharpe"]

        @staticmethod
        def sortino_ratio(
            df: pd.DataFrame,
            resolution: str,
            window: int,
            risk_free_rate=0,
            target_return=0,
        ):
            """
            Calculate the Sortino Ratio of returns.

            Parameters:
            - df (pd.DataFrame): DataFrame containing a 'pnl' column with returns.
            - resolution (str): Time resolution for returns ("24h", "1h").
            - window (int): The number of observations to consider from the DataFrame.
            - risk_free_rate (float): Risk-free rate as a decimal.
            - target_return (float): Minimum acceptable return.

            Returns:
            - float: The Sortino Ratio.
            """
            if window == 0:
                returns = df["pnl"]
            else:
                returns = df.loc[window:, "pnl"]

            # Calculate excess returns over risk-free rate
            excess_returns = returns - risk_free_rate

            # Calculate downside deviation (only negative excess returns)
            downside_deviation = PerformanceAnalyzer.Calculate.downside_deviation(
                excess_returns - target_return
            )

            # Calculate the mean of excess returns
            mean_excess_return = np.mean(excess_returns)

            # Calculate Sortino Ratio
            sortino_ratio = (
                mean_excess_return / downside_deviation
                if downside_deviation != 0
                else np.nan
            )

            if resolution == "24h":
                return sortino_ratio * np.sqrt(365)
            elif resolution == "1h":
                return sortino_ratio * np.sqrt(365 * 24)
            elif resolution == "10m":
                return sortino_ratio * np.sqrt(365 * 24 * 10)
            else:
                return 0

        @staticmethod
        def rolling_sortino_ratio(
            df: pd.DataFrame, resolution: str, window: int
        ) -> pd.DataFrame:
            if resolution == "24h":
                num_of_timeframe = window
            elif resolution == "1h":
                num_of_timeframe = window * 24
            elif resolution == "10m":
                num_of_timeframe = window * 24 * 6
            df["rolling_mean"] = df["pnl"].rolling(window).mean() * num_of_timeframe
            df["rolling_downside_dev"] = (
                df["pnl"]
                .rolling(window)
                .apply(lambda x: np.std(x[x < 0]) * np.sqrt(num_of_timeframe), raw=True)
            )
            df["rolling_sortino"] = df["rolling_mean"] / df["rolling_downside_dev"]
            return df["rolling_sortino"]

        @staticmethod
        def rolling_beta(df: pd.DataFrame, window: int) -> pd.DataFrame:
            df["rolling_cov"] = df["pnl"].rolling(window).cov(df["bnh_pnl"])
            df["rolling_var_market"] = df["bnh_pnl"].rolling(window).var()
            df["annualized_rolling_beta"] = df["rolling_cov"] / df["rolling_var_market"]
            return df["annualized_rolling_beta"]

        @staticmethod
        def rolling_alpha(df: pd.DataFrame, window: int) -> pd.DataFrame:
            # Add a constant term for the intercept (alpha) in the regression
            df["constant"] = 1

            # Perform rolling regression with strategy returns as dependent and benchmark + constant as independent
            model = RollingOLS(df["pnl"], df[["constant", "bnh_pnl"]], window=window)
            rolling_results = model.fit()

            # Extract the rolling alpha (intercept term)
            df["rolling_alpha"] = rolling_results.params["constant"]

            return df["rolling_alpha"]

        @staticmethod
        def _hourly_to_daily(df: pd.DataFrame) -> pd.DataFrame:
            """
            Convert hourly returns to daily returns by compounding hourly returns.

            Parameters:
            - df (pd.DataFrame): DataFrame with a 'pnl' column and datetime index.

            Returns:
            - pd.DataFrame: DataFrame with daily returns.
            """
            temp_df = df[["t", "pnl"]]
            temp_df["t"] = pd.to_datetime(temp_df["t"])

            # Set "t" as index
            temp_df.set_index("t", inplace=True)

            # Group by date and calculate daily return by compounding hourly returns
            daily_returns = temp_df.resample("D").apply(
                lambda x: np.prod(1 + x["pnl"]) - 1
            )


class DrawdownAnalyzer:
    """Analyzer for computing and analyzing strategy drawdowns"""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize DrawdownAnalyzer with PnL data

        Parameters:
        df: DataFrame with 'pnl' column and datetime index
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        if 'pnl' not in df.columns:
            raise ValueError("DataFrame must contain 'pnl' column")

        self.df = df
        self.cumulative = self.df['pnl'].cumsum()
        self._calculate_drawdown_series()

    def _calculate_drawdown_series(self) -> None:
        """Calculate running drawdown series"""
        rolling_max = self.cumulative.expanding().max()
        self.drawdown_series = (self.cumulative - rolling_max)

    def get_max_drawdown(self) -> float:
        """Return maximum drawdown value"""
        return self.drawdown_series.min()

    def get_current_drawdown(self) -> float:
        """Return current drawdown value"""
        return self.drawdown_series.iloc[-1]

    def get_drawdown_periods(self, threshold: float = None) -> List[DrawdownPeriod]:
        """
        Get list of drawdown periods

        Parameters:
        threshold: Optional minimum drawdown to consider

        Returns:
        List of DrawdownPeriod objects
        """
        periods = []
        in_drawdown = False
        peak_idx = None

        for i in range(len(self.cumulative)):
            if not in_drawdown:
                if self.drawdown_series.iloc[i] < 0:
                    in_drawdown = True
                    peak_idx = i - 1
                    valley_idx = i
            else:
                if self.drawdown_series.iloc[i] < self.drawdown_series.iloc[valley_idx]:
                    valley_idx = i
                elif self.cumulative.iloc[i] >= self.cumulative.iloc[peak_idx]:
                    # Drawdown period complete
                    depth = self.drawdown_series.iloc[valley_idx]

                    # Skip if doesn't meet threshold
                    if threshold is None or depth <= threshold:
                        period = DrawdownPeriod(
                            start_date=self.cumulative.index[peak_idx],
                            end_date=self.cumulative.index[valley_idx],
                            recovery_date=self.cumulative.index[i],
                            depth=depth,
                            duration_total=(i - peak_idx),
                            duration_drawdown=(valley_idx - peak_idx),
                            duration_recovery=(i - valley_idx)
                        )
                        periods.append(period)

                    in_drawdown = False

        return periods

    def get_drawdown_stats(self) -> dict:
        """
        Calculate comprehensive drawdown statistics

        Returns:
        Dictionary with drawdown statistics
        """
        periods = self.get_drawdown_periods()

        if not periods:
            return {
                'drawdown': {"avg": 0,
                             "sd": 0,
                             "max": 0},
                'duration': {"avg": 0,
                             "sd": 0,
                             "max": 0},
                'recovery': {"avg": 0,
                             "sd": 0,
                             "max": 0},
                'num_drawdowns': 0
            }

        drawdown = [p.depth for p in periods]
        duration = [p.duration_total for p in periods]
        recovery = [p.duration_recovery for p in periods]

        duration30d = [p.duration_total for p in periods if p.duration_total > 30]
        drawdown30d = [p.depth for p in periods if p.duration_total > 30]
        return {
            'drawdown': {"avg": np.mean(drawdown),
                         "sd": np.std(drawdown),
                         "max": self.get_max_drawdown()},
            'duration': {"avg": np.mean(duration),
                         "sd": np.std(duration),
                         "max": np.max(duration)},
            'recovery': {"avg": np.mean(recovery),
                         "sd": np.std(recovery),
                         "max": np.max(recovery)},
            'num_drawdowns': len(periods),
            'num_drawdown30': len(duration30d),
            'avg_drawdown30': np.mean(drawdown30d),
            'avg_duration30': np.mean(duration30d)
        }

    def plot_drawdowns(self, min_depth: float = -0.05, figsize=(15, 10)):
        """
        Create a comprehensive drawdown analysis visualization dashboard

        Parameters:
        min_depth: minimum drawdown depth to highlight (default: -0.05)
        figsize: figure size for the entire dashboard (default: (15, 10))
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            periods = self.get_drawdown_periods()
            drawdown_depths = [p.depth for p in periods]
            drawdown_durations = [p.duration_total for p in periods]
            recovery_durations = [p.duration_recovery for p in periods]

            # Create subplot grid
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(3, 2)

            # 1. Main Drawdown Timeline (larger plot)
            ax1 = fig.add_subplot(gs[0, :])
            ax1.plot(self.drawdown_series, color='gray', alpha=0.5, label='Drawdown')

            # Highlight major drawdowns
            for period in periods:
                if period.depth <= min_depth:
                    ax1.axvspan(period.start_date, period.recovery_date,
                                color='red', alpha=0.2)

            ax1.grid(True)
            ax1.set_title('Drawdown Timeline')
            ax1.set_ylabel('Drawdown (%)')
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

            # 2. Drawdown Depth Distribution
            ax2 = fig.add_subplot(gs[1, 0])
            sns.histplot(drawdown_depths, ax=ax2, kde=True)
            ax2.set_title('Drawdown Depth Distribution')
            ax2.set_xlabel('Drawdown Depth (%)')
            ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))

            # 3. Duration Distribution
            ax3 = fig.add_subplot(gs[1, 1])
            sns.histplot(drawdown_durations, ax=ax3, kde=True)
            ax3.set_title('Drawdown Duration Distribution')
            ax3.set_xlabel('Duration (days)')

            # 4. Recovery vs Depth Scatter
            ax4 = fig.add_subplot(gs[2, 0])
            sns.scatterplot(x=drawdown_depths, y=recovery_durations, ax=ax4)
            ax4.set_title('Recovery Time vs Drawdown Depth')
            ax4.set_xlabel('Drawdown Depth (%)')
            ax4.set_ylabel('Recovery Duration (days)')
            ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))

            # 5. Duration vs Depth Scatter
            ax5 = fig.add_subplot(gs[2, 1])
            sns.scatterplot(x=drawdown_depths, y=drawdown_durations, ax=ax5)
            ax5.set_title('Drawdown Duration vs Depth')
            ax5.set_xlabel('Drawdown Depth (%)')
            ax5.set_ylabel('Drawdown Duration (days)')
            ax5.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))

            # Add stats annotations
            stats = self.get_drawdown_stats()
            stats_text = (
                f"Total\n"
                f"---------------------------------------\n"
                f"# Drawdown: {stats['num_drawdowns']}\n"
                f"Max Drawdown: {stats['drawdown']['max'] * 100:.2f}%\n"
                f"Avg Drawdown: {stats['drawdown']['avg'] * 100:.2f}%\n"
                f"Avg Duration: {stats['duration']['avg']:.1f} days\n"
                f"Avg Recovery: {stats['recovery']['avg']:.1f} days\n\n"
                f"> 30 days\n"
                f"---------------------------------------\n"
                f"# Drawdown: {stats['num_drawdown30']}\n"
                f"Avg Drawdown: {stats['avg_drawdown30'] * 100:.2f}%\n"
                f"Avg Duration: {stats['avg_duration30']:.1f} days\n"
            )

            # Position the stats text in the upper right corner of the main drawdown plot
            ax3.text(
                0.98, 0.1,  # x, y coordinates in axes coordinates
                stats_text,
                transform=ax3.transAxes,  # Use axes coordinates
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(
                    facecolor='white',
                    alpha=0.8,
                    edgecolor='none',
                    pad=5
                ),
                fontsize=9
            )

            plt.tight_layout()
            plt.show()

        except ImportError:
            logger.warning("Matplotlib and seaborn are required for plotting")

            return daily_returns.reset_index()

