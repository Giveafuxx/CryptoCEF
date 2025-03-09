import numpy as np
import pandas as pd
from statsmodels.regression.rolling import RollingOLS


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

            return daily_returns.reset_index()

