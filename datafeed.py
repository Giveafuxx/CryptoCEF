import logging
import requests
import pandas as pd
from io import StringIO  # Ensure StringIO is imported
from config import *

logger = logging.getLogger(__name__)


class DataFeed:
    """
    Base class for data feeds.
    """

    def __init__(self, datasource: str) -> None:
        """
        Initialize the DataFeed with the given data source.

        Args:
            datasource (str): The data source section in the configuration.
        """
        pass


class GlassnodeDataFeed(DataFeed):
    """
    Data feed class for Glassnode API.
    """

    def __init__(self) -> None:
        """
        Initialize the GlassnodeDataFeed.
        """
        super().__init__("glassnode")

    def get_endpoints_list(self) -> pd.DataFrame:
        """
        Get the list of available endpoints from the Glassnode API.

        Returns:
            pd.DataFrame: DataFrame containing the list of endpoints.

        Raises:
            requests.RequestException: If the request to the API fails.
        """
        try:
            response = requests.get(
                f"https://api.glassnode.com/v2/metrics/endpoints",
                params={"api_key": GLASSNODE_APIKEY},
            )
            response.raise_for_status()
            return pd.read_json(StringIO(response.text), convert_dates=["t"])
        except requests.RequestException as e:
            logger.exception(f"Error fetching endpoints from Glassnode API: {e}")
            raise
        except ValueError as e:
            logger.exception(f"Error parsing JSON response: {e}")
            raise

    def get_metric_metadata(self, metric: str) -> pd.DataFrame:
        """
        Get the metric metadata from the Glassnode API.

        Parameters:
            metric (str): the metric to fetch.

        Returns:
            pd.DataFrame: DataFrame containing the list of endpoints.

        Raises:
            requests.RequestException: If the request to the API fails.
        """
        try:
            response = requests.get(
                f"https://api.glassnode.com/v1/metadata/metric",
                params={"api_key": GLASSNODE_APIKEY,
                        "path": metric}
            )
            response.raise_for_status()
            return pd.read_json(StringIO(response.text))
        except requests.RequestException as e:
            logger.exception(f"Error fetching metric metadata from Glassnode API: {e}")
            raise
        except ValueError as e:
            logger.exception(f"Error parsing JSON response: {e}")
            raise

    def fetch_data_by_metric(
            self,
            metric: str,
            asset: str,
            since: int,
            until: int,
            resolution: str,
            currency: str = "NATIVE",
    ) -> pd.DataFrame:
        """
        Fetch data for a specific metric from the Glassnode API.

        Args:
            metric (str): The metric to fetch.
            asset (str): The asset to fetch data for.
            since (int): Start timestamp.
            until (int): End timestamp.
            resolution (str): Data resolution.
            currency (str, optional): Currency type. Defaults to "NATIVE".

        Returns:
            pd.DataFrame: DataFrame containing the fetched data.

        Raises:
            requests.RequestException: If the request to the API fails.
        """
        try:
            response = requests.get(
                f"https://api.glassnode.com/v1/metrics/{metric}",
                params={
                    "a": asset,
                    "s": since,
                    "u": until,
                    "api_key": GLASSNODE_APIKEY,
                    "i": resolution,
                    "c": currency,
                },
                timeout=10,
            )
            response.raise_for_status()
            logger.debug(f"Fetched data from URL: {response.url}")
            return pd.read_json(StringIO(response.text), convert_dates=["t"])
        except requests.Timeout as e:
            logger.error(f"Glassnode API timeout at URL: https://api.glassnode.com/v1/metrics/{metric} with params: a={asset}, s={since}, u={until}, i={resolution}, c={currency}. Exception: {e}")
            raise
        except requests.RequestException as e:
            logger.exception(f"Glassnode API request error at URL: https://api.glassnode.com/v1/metrics/{metric} with params: a={asset}, s={since}, u={until}, i={resolution}, c={currency}. Response: {getattr(e.response, 'text', 'N/A')}")
            raise
        except ValueError as e:
            logger.exception(f"Error parsing JSON response: {e}")
            raise


if __name__ == "__main__":
    """
    input metric and asset
    """
