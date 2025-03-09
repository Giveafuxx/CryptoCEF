"""
Cryptocurrency Factor Trading Strategy Implementation Module

This module implements a factor-based trading strategy framework for cryptocurrency markets.
It includes functionality for data fetching, strategy backtesting, and performance optimization.

Main Components:
- Strategy: Base class for all trading strategies
- FactorStrategy: Implementation of factor-based trading strategy
- Data processing and metric calculations
- Strategy optimization and backtesting

Dependencies:
    - pandas: Data manipulation and analysis
    - numpy: Numerical computations
    - logging: Error and activity logging
"""

import logging

import copy
import pandas as pd
from typing import Dict, Optional, Tuple, Any

from backtest import backtesting, cal_pnl
from model import Model
from metric import MetricFetcher, apply_transformation, apply_operator
from parameter import StrategyPerformance, ExecutionDetail, StrategyDetail, MetricDetail
from optimizer import FactorStrategyOptimizer

from plot import FactorStrategyPlotter
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


class Strategy:
    """
    Base Strategy class providing core functionality for trading strategies.

    Attributes:
        model (Model): Trading model instance (e.g., "bband", "log_bband", "robust", "tanh", "ma_diff", "roc", ""rsi",
                                                     "min_max", "percentile", "raw")
        strategy_info (StrategyDetail): Strategy configuration details
        metric_info (Dict[str, MetricDetail]): Dictionary containing metric configurations
        resolution (str): Time resolution for the strategy (e.g., "24h", "1h", "10m")
        execution_info (ExecutionDetail): Execution-related parameters
        performance_info (StrategyPerformance): Strategy performance metrics
        final_df (pd.DataFrame): Final processed data and results
        data (Dict): Raw input data
        optimizer (Optional[Any]): Strategy optimizer instance
        heatmap_para (Optional[Dict]): Parameters for heatmap visualization
    """
    def __init__(self, **kwargs):
        """
        Initialize Strategy instance.

        Args:
            **kwargs: Keyword arguments including:
                - strategy_code (str): Unique identifier for the strategy
                - model (str): Model type to use
                - direction (str): Trading direction ('momentum' or 'reversion')
                - resolution (str): Time resolution
                - long_short (str): Trading direction type
                - currency (str): Base currency
                - window (int): Look-back window
                - threshold (float): Trading threshold
        """
        self.model = Model()

        self.strategy_info = StrategyDetail(strategy_code=kwargs.get("strategy_code", "test"),
                                            model=kwargs.get("model"),
                                            direction=kwargs.get("direction", "momentum"),
                                            resolution=kwargs.get("resolution", "24h"),
                                            long_short=kwargs.get("long_short", "long short"),
                                            currency=kwargs.get("currency", "NATIVE"),
                                            window=kwargs.get("window"),
                                            threshold=kwargs.get("threshold"))

        metric1 = MetricDetail(metric=kwargs.get("metric1"),
                               asset_input=kwargs.get("asset_input1"),
                               asset_output=kwargs.get("asset_output1"),
                               operator=kwargs.get("operator"))

        metric2 = MetricDetail(metric=kwargs.get("metric2"),
                               asset_input=kwargs.get("asset_input2", metric1.asset_input),
                               asset_output=kwargs.get("asset_output2", metric1.asset_output),
                               operator=kwargs.get("operator"))

        self.metric_info = {"metric1": metric1,
                            "metric2": metric2}

        self.resolution = kwargs.get("resolution", "24h")

        self.execution_info = ExecutionDetail(resolution=kwargs.get("resolution", "24h"))
        self.performance_info = StrategyPerformance()
        self.final_df = pd.DataFrame()

        self.data = kwargs.get("data", {})

        self.optimizer = None
        self.heatmap_para = kwargs.get("heatmap_para", None)

    @staticmethod
    def _clean_params(**params):
        data = params.get("data", {})
        params.pop("data", None)
        cleaned_params = {k: None if pd.isna(v) else v for k, v in params.items()}
        cleaned_params["data"] = data
        return cleaned_params

    def _preprocess_data(self, data: dict):
        pass

    def fetch_strategy_data(self, **kwargs: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Fetch and process strategy-specific market data metrics.

        This function retrieves market data metrics and optionally combines them with price data.
        It's commonly used in our data pipeline for strategy calculations.

        Args:
            **kwargs: Dictionary containing fetch parameters:
                - metric (str): Name of the metric to fetch (e.g., "volume", "active_addresses")
                - with_price (bool, optional): Whether to include price data. Defaults to True
                - start_time (int, optional): Unix timestamp for start of data
                - end_time (int, optional): Unix timestamp for end of data
                - resolution (str, optional): Time resolution (e.g., "1h", "24h")

        Returns:
            Optional[pd.DataFrame]: Processed DataFrame with columns:
                - If with_price=True: ['t', 'value', 'price', 'metric_name']
                - If with_price=False: ['t', 'value']
                Returns None if an error occurs.

        Raises:
            ValueError: If fetched metric data or price data is empty
            Exception: For other unexpected errors during data fetching
        """
        try:
            # Initialize metric fetcher with provided parameters
            metric_fetcher: MetricFetcher = MetricFetcher(**kwargs)

            # Fetch the main metric data
            df_value: pd.DataFrame = metric_fetcher.fetch_metric(
                metric=kwargs.get("metric")
            )

            # Validate metric data is not empty
            if df_value.empty:
                raise ValueError(
                    f"Empty data for metric {self.metric_info['metric1'].metric}"
                )

            # Check if price data is required
            if kwargs.get("with_price", True):
                # Fetch price data
                df_price: pd.DataFrame = metric_fetcher.fetch_price()

                # Validate price data is not empty
                if df_price.empty:
                    raise ValueError(
                        "Empty data for metric market/price_usd_close"
                    )

                # Combine metric and price data
                df: pd.DataFrame = apply_transformation(df_value, df_price)
                return df
            else:
                # Return metric data only with renamed column
                df_value = df_value.rename(columns={"v": "value"})
                return df_value

        except Exception as e:
            # Log any errors that occur during data fetching
            logger.error(
                "Error in fetching data for strategy \n"
                f"Strategy details: {self.__dict__}"
            )
            logger.error(f"Error details: {str(e)}")
            return None

    def backtest(self, **kwargs):
        pass

    def plot_equity_curve(self, mode: str = "advanced") -> None:
        """
        Generate and display equity curve plot for the trading strategy.

        Creates a visualization of strategy performance using FactorStrategyPlotter.
        The plot includes equity curve and related metrics based on the selected mode.

        Args:
            mode (str, optional): Plot mode selection. Defaults to "advanced"
                - "basic": Shows simplified equity curve
                - "advanced": Shows detailed metrics including drawdown and performance indicators

        Notes:
            - Requires self.final_df to be populated with strategy results
            - Uses self.strategy_info for strategy configuration details
            - Uses self.metric_info for metric information
            - Displays plot directly using plotly's show() method
        """
        # Create plotter instance with strategy results and configuration
        fig = FactorStrategyPlotter(
            df_plot=self.final_df,  # DataFrame containing strategy results
            mode=mode,  # Plotting mode ("simple" / "advanced")
            result=self.performance_info,  # Strategy configuration details
            metric=self.metric_info["metric1"].metric  # Primary metric used
        )

        # Display the plot
        fig.show()

    def _fetch_single_coin_raw_data(
            self,
            metric_obj: MetricDetail,
            since: int,
            until: int,
            with_price: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Fetch raw data for a single cryptocurrency metric.

        Helper method that combines metric details and execution parameters to fetch
        raw market data for a specific time period.

        Args:
            metric_obj (MetricDetail): Object containing metric configuration details
                including asset, metric name, and other parameters
            since (int): Start timestamp for data fetching (Unix timestamp)
            until (int): End timestamp for data fetching (Unix timestamp)
            with_price (bool, optional): Whether to include price data. Defaults to True

        Returns:
            Optional[pd.DataFrame]: DataFrame containing raw metric data with columns:
                - If with_price=True: ['t', 'value', 'price']
                - If with_price=False: ['t', 'value']
                Returns None if fetch fails

        Notes:
            - Uses self.fetch_strategy_data internally
            - Combines parameters from metric_obj and execution_info
        """
        # Combine metric details and execution parameters for data fetching
        return self.fetch_strategy_data(
            **metric_obj.__dict__,  # Metric-specific parameters
            **self.execution_info.__dict__,  # Execution-related parameters
            since=since,  # Start timestamp
            until=until,  # End timestamp
            with_price=with_price  # Price data inclusion flag
        )

    def fetch_raw_data(
            self,
            since: int,
            until: int,
            with_price: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch raw data for all metrics defined in the strategy.

        Iterates through all metrics in self.metric_info and fetches corresponding
        market data for each metric within the specified time range.

        Args:
            since (int): Start timestamp for data fetching (Unix timestamp)
            until (int): End timestamp for data fetching (Unix timestamp)
            with_price (bool, optional): Whether to include price data. Defaults to True

        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing raw data for each metric
                - Key: metric name (str)
                - Value: DataFrame containing metric data
                    If metric exists: Contains fetched data
                    If no metric: Empty DataFrame
        """
        # Initialize dictionary to store results
        temp_dict: Dict[str, pd.DataFrame] = {}

        # Iterate through each metric in strategy configuration
        for metric_name, metric_obj in self.metric_info.items():
            if metric_obj.metric:
                # Fetch data if metric is defined
                temp_dict[metric_name] = self._fetch_single_coin_raw_data(
                    metric_obj=metric_obj,  # Metric configuration object
                    since=since,  # Start timestamp
                    until=until,  # End timestamp
                    with_price=with_price  # Price data inclusion flag
                )
            else:
                # Return empty DataFrame if no metric defined
                temp_dict[metric_name] = pd.DataFrame()

        return temp_dict

    def _standardize_metric_value(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize metric values using the specified model parameters.

        Private helper method that applies standardization to raw metric values
        using the strategy's model configuration and window settings.

        Args:
            df (pd.DataFrame): Input DataFrame containing raw metric values
                Expected columns: ['t', 'value', 'price'] or ['t', 'value']

        Returns:
            pd.DataFrame: Standardized DataFrame with additional columns:
                - Original columns plus standardized metric values
                - Standardization depends on model type (e.g., percentile, bband, etc)

        Notes:
            - Uses self.model.standardize_metric internally
            - Window size from self.strategy_info.window
            - Model type from self.strategy_info.model
        """
        # Apply standardization using model configuration
        return self.model.standardize_metric(
            df=df,  # Input DataFrame
            window=self.strategy_info.window,  # Look-back window
            model=self.strategy_info.model  # Model type (e.g., 'bband', 'percentile')
        )


class FactorStrategy(Strategy):
    """
    Factor-based trading strategy implementation.

    Extends the base Strategy class with specific factor-based trading logic.
    Implements optimization, backtesting, and position calculation based on factor signals.

    Additional Attributes:
        optimizer (FactorStrategyOptimizer): Optimizer for factor strategy parameters
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def optimize(self, since: int, until: int) -> None:
        """
        Optimize strategy parameters with default model for a given time period.

        This method performs strategy optimization by either:
        1. Fetching new data if self.data is empty
        2. Using existing data with a deep copy to prevent modifications
        Then initializes optimizer if needed and runs optimization process.

        Args:
            since (int): Start timestamp for optimization period (Unix timestamp)
            until (int): End timestamp for optimization period (Unix timestamp)

        Notes:
            - Creates new FactorStrategyOptimizer if not initialized
            - Uses deep copy of existing data to prevent modifications
            - Preprocesses data before optimization
            - Updates optimizer with strategy configuration
        """
        # Fetch or copy data
        if not self.data:
            data: Dict[str, pd.DataFrame] = self.fetch_raw_data(
                since=since,  # Start time
                until=until  # End time
            )
        else:
            # Deep copy to prevent modifications to original data
            data: Dict[str, pd.DataFrame] = copy.deepcopy(self.data)

        # Preprocess data for optimization
        data = self._preprocess_data(data)

        # Initialize optimizer if needed
        if self.optimizer is None:
            self.optimizer = FactorStrategyOptimizer(
                since=since,  # Start timestamp
                until=until,  # End timestamp
                data_preprocessed=data,  # Preprocessed data
                **self.__dict__  # Strategy configuration
            )

        # Run optimization with current model
        self.optimizer.optimize(self.strategy_info.model)

    def batch_optimize(self, since: int, until: int) -> None:
        """
        Perform batch optimization across multiple model configurations.

        This method executes optimization across a predefined set of models and parameters
        to find the best performing configuration. Unlike single optimize(), this method
        tests multiple model combinations.

        Args:
            since (int): Start timestamp for batch optimization period (Unix timestamp)
            until (int): End timestamp for batch optimization period (Unix timestamp)

        Notes:
            - Initializes data pipeline if self.data is empty
            - Uses deep copy of existing data to prevent modifications
            - Creates new FactorStrategyOptimizer if not initialized
            - Runs batch optimization across all configured models
            - More computationally intensive than single optimize()

        Technical Details:
            - Data Structure: Dict[str, pd.DataFrame] containing time-series data
            - Process Flow:
                1. Data preparation (fetch or copy)
                2. Data preprocessing
                3. Optimizer initialization
                4. Batch optimization execution
        """
        # Fetch new data or use existing data
        if not self.data:
            data: Dict[str, pd.DataFrame] = self.fetch_raw_data(
                since=since,  # Start timestamp
                until=until  # End timestamp
            )
        else:
            # Deep copy to prevent modifications to original data
            data: Dict[str, pd.DataFrame] = copy.deepcopy(self.data)

        # Preprocess data for optimization
        data = self._preprocess_data(data)

        # Initialize optimizer if needed
        if self.optimizer is None:
            self.optimizer = FactorStrategyOptimizer(
                since=since,  # Start timestamp
                until=until,  # End timestamp
                data_preprocessed=data,  # Preprocessed data
                **self.__dict__  # Strategy configuration
            )

        # Execute batch optimization
        self.optimizer.batch_optimize()

    def __cal_pos(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate position signals based on strategy parameters and model configuration.

        Private method that generates trading signals using the specified model and
        parameters from strategy configuration.

        Args:
            df (pd.DataFrame): Input DataFrame with standardized metric values
                Required columns depend on model type but typically include:
                - 't': timestamp
                - 'value': metric value
                - 'standardized_value': processed metric value

        Returns:
            pd.DataFrame: DataFrame with additional position signals
                Added columns include:
                - 'position': Trading position signals (-1, 0, 1)
                - Model-specific indicator columns
        """
        # Calculate positions using Model class
        return Model().cal_pos(
            df=df,                                          # Input DataFrame
            model=self.strategy_info.model,                 # Model type
            threshold=self.strategy_info.threshold,         # Signal threshold
            long_or_short=self.strategy_info.long_short,    # Trading constraints
            direction=self.strategy_info.direction          # Signal direction
        )

    def _preprocess_data(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Preprocess raw data for strategy implementation.

        This method handles data preparation by:
        1. Creating deep copies of input data to prevent modifications
        2. Applying specified operator between metric1 and metric2
        3. Preparing data structure for strategy calculations

        Args:
            data (Dict[str, pd.DataFrame]): Raw data dictionary
                Expected structure:
                {
                    "metric1": pd.DataFrame,  # Primary metric data
                    "metric2": pd.DataFrame   # Secondary metric data (if applicable)
                }
                Each DataFrame should contain:
                - 't': timestamp
                - 'value': metric values
                - 'price': price data (optional)

        Returns:
            pd.DataFrame: Preprocessed data ready for strategy implementation
                Contains combined metric data after operator application

        Notes:
            - Uses deep copy to prevent modifications to original data
            - Applies operator specified in metric1's configuration
            - Common operators include: 'A + B', 'A - B', 'A * B', 'A / B', 'ratio'
            - Handles data alignment and timestamp matching automatically
        """
        # Initialize dictionary for processed data
        temp_dict: Dict[str, pd.DataFrame] = {}

        # Create deep copies of input data
        for metric_name, metric_obj in self.metric_info.items():
            temp_dict[metric_name] = copy.deepcopy(data[metric_name])

        # Apply operator between metric1 and metric2
        return apply_operator(
            data1=temp_dict["metric1"],  # Primary metric
            data2=temp_dict["metric2"],  # Secondary metric
            operator=self.metric_info["metric1"].operator  # Operation to apply
        )

    def backtest(self, since: int, until: int) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Perform strategy backtesting over specified period.

        Executes a complete backtesting pipeline:
        1. Data preparation (fetch or copy)
        2. Data preprocessing and standardization
        3. Position calculation
        4. PnL calculation
        5. Performance analysis

        Args:
            since (int): Start timestamp for backtest (Unix timestamp)
            until (int): End timestamp for backtest (Unix timestamp)

        Returns:
            Tuple[pd.Series, pd.DataFrame]: Empty series and DataFrame if error occurs

        Updates Instance Attributes:
            - self.performance_info (pd.Series): Strategy performance metrics including:
                * Sharpe ratio
                * Max drawdown
                * Annual returns
                * Other risk metrics
            - self.final_df (pd.DataFrame): Detailed backtest results containing:
                * Timestamp index
                * Position signals
                * PnL calculations
                * Cumulative returns
                * Other strategy-specific metrics

        Technical Details:
            - Data Flow:
                1. Raw data fetching/copying
                2. Preprocessing via _preprocess_data
                3. Standardization via _standardize_metric_value
                4. Position calculation via __cal_pos
                5. PnL calculation via cal_pnl
                6. Performance analysis via backtesting
        """
        try:
            # Data preparation
            if not self.data:
                data: pd.DataFrame = self.fetch_raw_data(since=since, until=until)
            else:
                data: pd.DataFrame = copy.deepcopy(self.data)

            # Data processing pipeline
            data = self._preprocess_data(data)  # Preprocess raw data
            data = self._standardize_metric_value(data)  # Standardize metrics

            # Calculate positions and PnL
            data["pos"] = self.__cal_pos(data)  # Generate position signals
            data = cal_pnl(data)  # Calculate PnL

            # Perform backtesting analysis
            self.performance_info, self.final_df = backtesting(
                window=self.strategy_info.window,           # Look-back window
                threshold=self.strategy_info.threshold,  # Signal threshold
                df=data,  # Processed data
                resolution=self.strategy_info.resolution  # Time resolution
            )

            # Set timestamp as index for final results
            self.final_df.set_index("t", inplace=True)

            logger.info(f"FactorStrategy {self.strategy_info.strategy_code} analyzed successfully")

        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return pd.Series(), pd.DataFrame()


if __name__ == "__main__":
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    heatmap_para = {"window": (5, 200, 10),
                    "threshold": (0, 2.1, 0.1)}

    # Initialize the FactorStrategy class
    strategy = FactorStrategy(
        model="bband",
        metric1="addresses/active_count",
        # metric2="addresses/active_count",
        # operator="A / B",
        direction="momentum",
        resolution="24h",
        asset_input1="ETH",
        asset_output1="ETH",
        long_short="long short",
        window=50,
        threshold=1,
        # heatmap_para=heatmap_para
    )
    start_date = "2020-01-01"
    end_date = "2024-01-01"
    start_timestamp = int(pd.Timestamp(start_date).timestamp())
    end_timestamp = int(pd.Timestamp(end_date).timestamp())

    strategy.batch_optimize(start_timestamp, end_timestamp)
    strategy.plot_equity_curve()



