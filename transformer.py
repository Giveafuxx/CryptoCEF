import pandas as pd
import numpy as np
import logging
import copy

logger = logging.getLogger(__name__)


class Transformer:
    """
    Data Pre-Processor for time-series DataFrame.

    Notes
    -----
    df : pandas.DataFrame
        Ensure df contains columns "t" and "value".  (df[["t", "value"]])
    """

    def transform(self, df: pd.DataFrame, method: str, period: int):
        self.__data_validate(df, method)
        if df.empty:
            return df
        if method == "raw":
            return self.raw(df)
        elif method == "log":
            return self.log(df)
        elif method == "roc":
            return self.rate_of_change(df, period)
        elif method == "sqrt":
            return self.sqrt(df)
        elif method == "cbrt":
            return self.cbrt(df)
        elif method == "diff":
            return self.difference(df, period)
        elif method == "diff_pct":
            return self.difference_pct(df, period)
        elif method == "mean":
            return self.rolling_mean(df, period)
        elif method == "inverse":
            return self.inverse(df)
        elif method == "absolute":
            return self.absolute(df)
        elif method == "fft":
            return self.fft(df, period)
        elif method == "cumsum":
            return self.cumsum(df)
        elif method == "ratio2pct":
            return self.ratio2pct(df)
        else:
            return self.raw(df)

    @staticmethod
    def __data_validate(df: pd.DataFrame, method: str):
        required_cols = ['t', 'value']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if not df.empty:
            if missing_cols:
                raise logger.warning(f"Missing df columns during transformation: {missing_cols}")

            if method is not None:
                if not isinstance(method, str):
                    logger.warning(f"Transformation error in datatype: {type(method)}, required datatype 'str'.")

    @staticmethod
    def __period_validation(period):
        if not isinstance(period, int):
            logger.warning(f"Transformation period error in datatype: {type(period)}, required datatype 'int'.")
        elif int <= 0:
            raise logger.warning(f"Transformation period must be a positive integer")

    @staticmethod
    def raw(df: pd.DataFrame) -> pd.DataFrame:
        """
        Return original DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame containing a 'value' column with right-skewed data.

        Returns
        -------
        pandas.DataFrame
            Return original DataFrame.
        """
        return df

    @staticmethod
    def log(df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform right-skewed/log-normal distributed data using natural logarithm.
        Commonly used for financial metrics that exhibit multiplicative behavior.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame containing a 'value' column with right-skewed data.

        Returns
        -------
        pandas.DataFrame
            DataFrame with updated 'value' column containing log-transformed values.

        Notes
        -----
        - Adds 1 to values before taking log to handle zeros
        - Uses natural logarithm (base e)
        - Original values are overwritten

        Use Cases
        ---------
        - Normalizing right-skewed distributions for statistical analysis
        - Preparing multiplicative data for machine learning models
        - Visualizing patterns across different scales
        - Detecting anomalies in log-normal distributed data

        Examples
        --------
        import pandas as pd
        df = pd.DataFrame({'value': [100, 1000, 10000, 0]})
        transformed_df = Transformer.log(df)
        print(transformed_df['value'])
        0    4.615121
        1    6.907755
        2    9.210340
        3    0.000000

        Warnings
        --------
        - Ensure input data is positive before transformation
        - Large values might need float64 dtype to prevent overflow
        - Original values will be overwritten

        See Also
        --------
        - np.log : NumPy natural logarithm function
        """
        try:
            df['value'] = np.log(df['value'])
            return df
        except Exception as e:
            logger.error("Error in transforming data using 'log' function.")
            logger.error(e)
            raise

    @staticmethod
    def rate_of_change(df: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
        """
        Calculate the rate of change (ROC) as a percentage change over a specified
        time period. The formula is: ((current - previous) / previous) * 100.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame containing the data to calculate ROC.
        periods : int, default=1
            Number of periods to shift for calculating the change.

        Returns
        -------
        pandas.DataFrame
            DataFrame with updated 'value' column containing ROC values.

        Notes
        -----
        - Results are expressed as percentages
        - First 'periods' rows will contain NaN values
        - Infinite values may occur when previous value is zero

        Use Cases
        ---------
        - Measuring price/volume momentum
        - Identifying trend acceleration/deceleration
        - Momentum oscillator calculations
        - Volatility analysis

        Examples
        --------
        import pandas as pd
        df = pd.DataFrame({'price': [100, 110, 105, 115]})
        roc_df = Transformer.rate_of_change(df, 'price', periods=1)
        print(roc_df['price_roc'])
        0     NaN
        1    10.0
        2    -4.5
        3     9.5

        Warnings
        --------
        - Check for NaN/inf values in output
        - Consider handling zero values in denominator
        - Be aware of look-ahead bias when using in trading strategies

        See Also
        --------
        - pct_change : Similar pandas built-in method
        - momentum_transform : Alternative momentum calculation
        """
        try:
            df["value"] = ((df["value"] - df["value"].shift(periods)) / df["value"].shift(periods)) * 100
            return df
        except Exception as e:
            logger.error("Error in transforming data using 'rate_of_change' function.")
            logger.error(e)
            raise

    @staticmethod
    def sqrt(df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply square root transformation while preserving sign of the original data.
        Useful for moderating right-skewed distributions.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame containing a 'value' column.

        Returns
        -------
        pandas.DataFrame
            DataFrame with 'value' column updated with sign-preserving square root values.

        Notes
        -----
        - Preserves sign using: sign(x) * sqrt(abs(x))
        - Less aggressive than log transform
        - Original values are overwritten

        Examples
        --------
        df = pd.DataFrame({'v': [100, -49, 25, 0]})
        sqrt_df = Transformer.sqrt(df)
        print(sqrt_df['value'])
        0    10.000000
        1    -7.000000
        2     5.000000
        3     0.000000
        """
        try:
            df["value"] = np.sign(df["value"]) * np.sqrt(np.abs(df["value"]))
            return df
        except Exception as e:
            logger.error("Error in transforming data using 'sqrt' function.")
            logger.error(e)
            raise

    @staticmethod
    def cbrt(df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply cube root transformation to handle both positive and negative values.
        More moderate transformation compared to square root.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame containing a 'value' column.

        Returns
        -------
        pandas.DataFrame
            DataFrame with 'value' column updated with cube root values.

        Notes
        -----
        - Naturally preserves sign unlike square root
        - More linear near zero than square root
        - Original values are overwritten

        Examples
        --------
        df = pd.DataFrame({'value': [27, -8, 1, 0]})
        cbrt_df = Transformer.cbrt(df)
        print(cbrt_df['value'])
        0     3.000000
        1    -2.000000
        2     1.000000
        3     0.000000
        """
        try:
            df["value"] = np.cbrt(df["value"])
            return df
        except Exception as e:
            logger.error("Error in transforming data using 'cbrt' function.")
            logger.error(e)
            raise

    @staticmethod
    def difference(df: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
        """
        Calculate period-over-period differences to make series more stationary.
        Common preprocessing step for time series analysis.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame containing a 'value' column.
        periods : int, default=1
            Number of periods to shift for calculating difference.

        Returns
        -------
        pandas.DataFrame
            DataFrame with 'value' column updated with differenced values.

        Notes
        -----
        - First 'periods' rows will contain NaN
        - Removes trends and seasonality
        - Original values are overwritten

        Examples
        --------
        df = pd.DataFrame({'value': [10, 15, 13, 17]})
        diff_df = Transformer.difference(df)
        print(diff_df['value'])
        0     NaN
        1     5.0
        2    -2.0
        3     4.0
        """
        try:
            df["value"] = df["value"].diff(periods=periods)
            return df
        except Exception as e:
            logger.error("Error in transforming data using 'difference' function.")
            logger.error(e)
            raise

    @staticmethod
    def difference_pct(df: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
        """
        Calculate period-over-period differences in percentage to make series more stationary.
        Common preprocessing step for time series analysis.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame containing a 'value' column.
        periods : int, default=1
            Number of periods to shift for calculating difference.

        Returns
        -------
        pandas.DataFrame
            DataFrame with 'value' column updated with difference in percentage values.

        Notes
        -----
        - First 'periods' rows will contain NaN
        - Removes trends and seasonality
        - Original values are overwritten

        Examples
        --------
        df = pd.DataFrame({'value': [10, 20, 10, 7]})
        diff_df = Transformer.difference(df)
        print(diff_df['value'])
        0     NaN
        1     1.0
        2    -0.5
        3    -0.3
        """
        try:
            df["_"] = df["value"].shift(periods=periods)
            df["value"] = df["value"].diff(periods=periods) / df["_"] - 1
            return df.drop("_", axis=1)
        except Exception as e:
            logger.error("Error in transforming data using 'difference' function.")
            logger.error(e)
            raise

    @staticmethod
    def rolling_mean(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Calculate rolling mean using specified window size.
        Smooths data and reduces noise in time series.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame containing a 'value' column.
        window : int, default=20
            Size of the rolling window for mean calculation.

        Returns
        -------
        pandas.DataFrame
            DataFrame with 'value' column updated with rolling mean values.

        Notes
        -----
        - First (window-1) rows will contain NaN
        - Centered window can be used by setting center=True
        - Original values are overwritten

        Examples
        --------
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        roll_df = Transformer.rolling_mean(df, window=3)
        print(roll_df['value'])
        0    NaN
        1    NaN
        2    2.0
        3    3.0
        4    4.0

        Warnings
        --------
        - Consider window size impact on data loss
        - Be aware of look-ahead bias when center=True
        """
        try:
            df["value"] = df["value"].rolling(window=window).mean()
            return df
        except Exception as e:
            logger.error("Error in transforming data using 'rolling_mean' function.")
            logger.error(e)
            raise

    @staticmethod
    def inverse(df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by taking its multiplicative inverse (1/x).
        Useful for converting rates to durations or inverting ratios.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame containing a 'value' column.

        Returns
        -------
        pandas.DataFrame
            DataFrame with 'value' column updated with inverse values.

        Notes
        -----
        - Handles zero values by setting them to np.inf
        - Original values are overwritten
        - Equivalent to raising to power of -1

        Examples
        --------
        df = pd.DataFrame({'value': [2, 4, 0.5, 0.25]})
        inv_df = Transformer.inverse(df)
        print(inv_df['value'])
        0    0.500000
        1    0.250000
        2    2.000000
        3    4.000000

        Warnings
        --------
        - Check for zero values in input
        - May produce infinite values
        - Consider handling NaN values

        Use Cases
        ---------
        - Converting rates to durations
        - Inverting price ratios
        - Calculating reciprocal returns

        See Also
        --------
        - np.reciprocal : NumPy inverse function
        - power_transform : General power transformation
        """
        try:
            df['value'] = 1 / df['value']
            return df
        except Exception as e:
            logger.error("Error in transforming data using 'inverse' function.")
            logger.error(e)
            raise

    @staticmethod
    def fft(df: pd.DataFrame, period: int = 45) -> pd.DataFrame:
        try:
            fft = FastFourierTransformer(period)
            return fft.normalize_metric(df)
        except Exception as e:
            logger.error("Error in transforming data using 'fft' function.")
            logger.error(e)
            raise

    @staticmethod
    def cumsum(df: pd.DataFrame) -> pd.DataFrame:
        try:
            df["value"] = df["value"].cumsum()
            return df
        except Exception as e:
            logger.error("Error in transforming data using 'cumsum' function.")
            logger.error(e)
            raise
    
    @staticmethod
    def ratio2pct(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes asymmetrical ratio data to percentage format using value/(value+1).
        
        In financial data, ratios often have asymmetrical structures where:
        - Ratio > 1 indicates favoring one side (e.g., 3.0 means 3:1)
        - Ratio < 1 indicates favoring the opposite side (e.g., 0.5 means 1:2)
        This transformation bounds the output to [0,1] while preserving the relationship:
        - Ratio of 3.0 -> 75% (favoring upside)
        - Ratio of 0.5 -> 33% (favoring downside)
        - Ratio of 1.0 -> 50% (neutral)
    
        Args:
            df (pd.DataFrame): Input DataFrame containing a 'value' column with ratio data.
                The 'value' column must contain positive numeric data representing ratios.
    
        Returns:
            pd.DataFrame: DataFrame with 'value' column transformed to percentage (0-1 range).
                Original DataFrame structure is preserved with only 'value' column modified.
    
        Raises:
            Exception: If 'value' column is missing, contains non-numeric data,
                negative values, or any arithmetic operation fails.
        """
        try:
            df["value"] = df["value"] / (df["value"] + 1)
            return df
        except Exception as e:
            logger.error("Error in transforming data using 'ratio2pct' function.")
            logger.error(e)
            raise
        
    @staticmethod
    def absolute(df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by taking its absolute value.
        Useful for magnitude-only analysis or removing negative values.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame containing a 'value' column.

        Returns
        -------
        pandas.DataFrame
            DataFrame with 'value' column updated with absolute values.

        Notes
        -----
        - Preserves dtype of original data
        - Removes sign information
        - Zero values remain unchanged
        - NaN values remain NaN

        Examples
        --------
        df = pd.DataFrame({'value': [-2, 4, -0.5, 0.25]})
        abs_df = Transformer.absolute(df)
        print(abs_df['value'])
        0    2.00
        1    4.00
        2    0.50
        3    0.25

        Warnings
        --------
        - Information about sign is lost
        - Consider impact on statistical properties
        - Check for NaN handling requirements

        Use Cases
        ---------
        - Volatility calculations
        - Price movement magnitude analysis
        - Distance measurements
        - Deviation analysis

        See Also
        --------
        - np.abs : NumPy absolute function
        - inverse : Multiplicative inverse transformation
        """
        try:
            df['value'] = df['value'].abs()
            return df
        except Exception as e:
            logger.error("Error in transforming data using 'absolute' function.")
            logger.error(e)
            raise


class FastFourierTransformer:
    def __init__(self, window: int = 252):
        self.window_size = window
        self.stats = {}

    def normalize_metric(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize metric using rolling window FFT

        Args:
            raw_df: DataFrame with ['t', 'value'] columns

        Returns:
            DataFrame with ['t', 'value'] columns
        """
        try:
            df = copy.deepcopy(raw_df)

            # Convert to DataFrame if Series
            if isinstance(df, pd.Series):
                df = df.to_frame(name='value')
                df.index.name = 't'
                df = df.reset_index()

            # Validate input
            self._validate_input(df)

            # Convert Date to datetime
            df['t'] = pd.to_datetime(df['t'])

            # Create rolling window
            prices_series = pd.Series(df['value'].values)

            # Calculate rolling FFT normalization
            normalized_prices = self._apply_rolling_fft(prices_series)

            # Create result DataFrame
            raw_df["value"] = normalized_prices

            return raw_df
        except Exception as e:
            print(e)

    def _apply_rolling_fft(self, series: pd.Series) -> np.ndarray:
        """
        Apply rolling FFT normalization using fixed window size
        Returns NaN for initial period (window_size-1 days)
        Output is scaled to [-1,1] range
        """
        # Initialize output array with NaN
        normalized = np.full(len(series), np.nan)

        # Start processing only after initial window
        for i in range(self.window_size - 1, len(series)):
            # Get window
            window = series.iloc[i - self.window_size + 1:i + 1]

            # Apply FFT to window
            fft_result = np.fft.fft(window)
            magnitudes = np.abs(fft_result)

            # Calculate normalization factor
            norm_factor = np.sqrt(np.mean(magnitudes ** 2))

            # Initial normalization
            current_price = series.iloc[i]
            if norm_factor > 0:
                initial_norm = current_price / norm_factor

                normalized[i] = initial_norm
            else:
                normalized[i] = 0

        return normalized
    
    def _validate_input(self, df: pd.DataFrame):
        """Validate input data"""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        required_cols = ['t', 'value']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"DataFrame must contain '{col}' column")

        if df.empty:
            raise ValueError("DataFrame is empty")

        if df['value'].isnull().any():
            raise ValueError("Close prices contain NaN values")
