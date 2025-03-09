import os
import logging
import numpy as np
import pandas as pd
import json
import time
import traceback
from datetime import datetime, timedelta
from typing import List, Any, Union

from datafeed import GlassnodeDataFeed

logger = logging.getLogger(__name__)


class MetricFetcher:
    # add typing
    def __init__(self, **kwargs):
        """
        Initialize the MetricFetcher class.
        """
        self.data_source: str = kwargs.get("data_source", "glassnode")
        if self.data_source == "glassnode":
            self.data_feed = GlassnodeDataFeed()
        else:
            self.data_feed = GlassnodeDataFeed()
        self.asset_input: str = kwargs.get("asset_input")
        self.asset_output: str = kwargs.get("asset_output", self.asset_input)
        self.since: int = kwargs.get("since")
        self.until: int = kwargs.get("until")
        self.resolution: str = kwargs.get("resolution")
        self.currency: str = kwargs.get("currency", "NATIVE")

    def fetch_metric(
        self, metric: str
    ) -> Any:
        """
        Fetch metric data for a given asset_input and resolution.

        Args:
            metric (str): The metric to fetch.
        Returns:
            Any: The fetched data.
        """
        try:
            fetch_resolution = self.resolution

            df = self.data_feed.fetch_data_by_metric(
                metric=metric,
                asset=self.asset_input,
                since=self.since,
                until=self.until,
                resolution=fetch_resolution,
                currency=self.currency,
            )

            if df is None or df.empty:
                logger.error(f"No data returned for metric {metric}.")
                return pd.DataFrame()

            if "v" not in df.columns:
                return pd.DataFrame()

            return df
        except Exception as e:
            logger.error(
                f"Error fetching metric {metric} for asset_input {self.asset_input}: {e}"
            )
            logger.error(traceback.format_exc())  # Added to log the full traceback
            return None

    def fetch_two_metrics(
        self,
        metric1: str,
        metric2: str,
        operator: str,
    ) -> Any:
        """
        Fetch two metrics and apply an operator to them.

        Args:
            metric1 (str): The first metric.
            metric2 (str): The second metric.
            operator (str): The operator to apply.

        Returns:
            Any: The result of applying the operator to the two metrics.
        """
        try:
            data1 = self.fetch_metric(metric1)
            data2 = self.fetch_metric(metric2)
            return apply_operator(data1, data2, operator)
        except Exception as e:
            logger.error(
                f"Error fetching or applying operator to metrics {metric1} and {metric2} for asset_input {self.asset_input}: {e}"
            )
            return pd.DataFrame()

    def fetch_price(self, metric_type: str = "close") -> Any:
        """
        Fetch the price data for a given asset_input and resolution.

        Returns:
            Any: The fetched price data.
        """
        try:
            fetch_resolution = self.resolution

            price_metric = (
                "market/price_usd_close"
                if metric_type == "close"
                else "market/price_usd_ohlc"
            )

            logger.debug(
                f"Fetching price for asset_output {self.asset_output} with resolution {fetch_resolution}"
            )

            df = GlassnodeDataFeed().fetch_data_by_metric(
                metric=price_metric,
                asset=self.asset_output,
                since=self.since,
                until=self.until,
                resolution=fetch_resolution,
            )

            return df

        except Exception as e:
            logger.error(
                f"Error fetching price for asset_input {self.asset_input}: {e}"
            )
            return pd.DataFrame()


def apply_operator(
    data1: pd.DataFrame, data2: pd.DataFrame = pd.DataFrame(), operator: str = ""
) -> pd.DataFrame:
    """
    Apply an operator to two dataframes.

    Args:
        data1 (pd.DataFrame): The first dataframe.
        data2 (pd.DataFrame): The second dataframe.
        operator (str): The operator to apply.

    Returns:
        pd.DataFrame: The resulting dataframe.
    """
    try:
        if data2.empty:
            return data1
        df_value = pd.merge(data1, data2, how="inner", on="t")
        print(df_value)
        if operator in ["A + B", "Addition"]:
            df_value["value"] = df_value["value_x"] + df_value["value_y"]
        elif operator in ["A - B", "Subtraction"]:
            df_value["value"] = df_value["value_x"] - df_value["value_y"]
        elif operator in ["A * B", "Multiplication"]:
            df_value["value"] = df_value["value_x"] * df_value["value_y"]
        elif operator in ["A / B", "Division"]:
            df_value["value"] = df_value["value_x"] / df_value["value_y"]
        elif operator in ["A / (A + B)", "Ratio"]:
            df_value["value"] = df_value["value_x"] / (df_value["value_x"] + df_value["value_y"])
        elif operator == "B - A":
            df_value["value"] = df_value["value_y"] - df_value["value_x"]
        elif operator == "B / A":
            df_value["value"] = df_value["value_y"] / df_value["value_x"]
        elif operator == "B / (A + B)":
            df_value["value"] = df_value["value_y"] / (df_value["value_x"] + df_value["value_y"])
        df_value = df_value.drop(columns=["value_x", "value_y", "price_y", "chg_y"])
        df_value = df_value.rename(columns={"price_x": "price",
                                            "chg_x": "chg"})
        return df_value
    except Exception as e:
        logger.error(f"Error applying operator {operator}: {e}")
        return pd.DataFrame()


def apply_transformation(
    df_value: pd.DataFrame, df_price: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply a transformation to the dataframes.

    Args:
        df_value (pd.DataFrame): The value dataframe.
        df_price (pd.DataFrame): The price dataframe.

    Returns:
        pd.DataFrame: The transformed dataframe.
    """
    try:
        df = pd.merge(df_value, df_price, how="inner", on="t")
        df = df.rename(columns={"v_x": "value", "v_y": "price"})
        df["t"] = pd.to_datetime(df["t"])
        df["chg"] = df["price"].pct_change()
        return df
    except Exception as e:
        logger.error(f"Error applying transformation: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    test = MetricFetcher(
        asset_input="BTC",
        asset_output="BTC",
        since=int((datetime.now() - timedelta(days=500)).timestamp()),
        until=int(datetime.now().timestamp()),
        resolution="24h",
        front_run=False,
        delay_mode=True,
        delay_time=10,
        data_source="laevitas",
    )
    df = test.fetch_metric("Historical_Total_Volume_Open_Value_Bybit")
