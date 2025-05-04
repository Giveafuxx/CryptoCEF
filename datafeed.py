import logging
import requests
import pandas as pd
from io import StringIO  # Ensure StringIO is imported
from datetime import datetime
from config import *
import pytz
from typing import Any, Type, Optional, Dict, List

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

class SantimentDataFeed(DataFeed):
    """
    Data feed class for Santiment API.
    """

    TOP_20_COINS = {
        "bitcoin": "bitcoin",
        "ethereum": "ethereum",
        "binance-coin": "binancecoin",
        "ripple": "xrp",
        "cardano": "cardano",
        "solana": "solana",
        "polkadot": "polkadot",
        "dogecoin": "dogecoin",
        "shiba-inu": "shiba-inu",
        "litecoin": "litecoin",
        "uniswap": "uniswap",
        "chainlink": "chainlink",
        "polygon": "matic-network",
        "avalanche": "avalanche",
        "tron": "tron",
        "stellar": "stellar",
        "aave": "aave",
        "vechain": "vechain",
        "algorand": "algorand",
        "cosmos": "cosmos"
    }

    def __init__(self) -> None:
        super().__init__('santiment')
        self.metrics_list = self.fetch_metric_list()

        self.TOP_20_COINS_SYMBOLS = {
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "BNB": "binancecoin",
            "XRP": "xrp",
            "ADA": "cardano",
            "SOL": "solana",
            "DOT": "polkadot",
            "DOGE": "dogecoin",
            "SHIB": "shiba-inu",
            "LTC": "litecoin",
            "UNI": "uniswap",
            "LINK": "chainlink",
            "MATIC": "matic-network",
            "AVAX": "avalanche",
            "TRX": "tron",
            "XLM": "stellar",
            "AAVE": "aave",
            "VET": "vechain",
            "ALGO": "algorand",
            "ATOM": "cosmos"
        }
        
    @staticmethod
    def _get_metric_query_by_asset(metric, slug, since_str, until_str, resolution):
        return f'''
        {{
          getMetric(metric: "{metric}") {{
            timeseriesData(
              slug: "{slug}"
              from: "{since_str}"
              to: "{until_str}"
              interval: "{resolution}"
              includeIncompleteData: false
            ) {{
              datetime
              value
            }}
          }}
        }}
        '''

    @staticmethod
    def _get_metric_query_by_arbitrary_search_term(metric_raw, slug, since_str, until_str, resolution):
        # refer to https://academy.santiment.net/metrics/social-volume/#definition for the text requirement
        metric_raw = metric_raw[4:]  # remove "AST_" prefix
        metric = metric_raw.split("#", 1)[0]
        text = metric_raw.split("#", 1)[1]

        return f'''
        {{
          getMetric(metric: "{metric}") {{
            timeseriesData(
              selector: {{ text: "{text}" }}
              from: "{since_str}"
              to: "{until_str}"
              interval: "{resolution}"
              includeIncompleteData: false
            ) {{
              datetime
              value
            }}
          }}
        }}
        '''
        
    def fetch_data_by_metric(self, metric: str, asset: str, since: int, until: int, resolution: str, currency:str = "NATIVE") -> pd.DataFrame:
        # convert timestamp (int) into "%Y-%m-%dT%H:%M:%SZ" (str). Example: 1672531200 -> "2023-01-01T00:00:00Z"
        since_str = datetime.fromtimestamp(since, tz=pytz.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        until_str = datetime.fromtimestamp(until, tz=pytz.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

        slug = self.TOP_20_COINS_SYMBOLS[asset]

        BASE_URL = 'https://api.santiment.net/graphql'
        API_KEY = self.api_key
        HEADERS = {
            'Authorization': f'Apikey {API_KEY}',
            'Content-Type': 'application/json'
        }

        if metric[:4] == "AST_":
            query = self._get_metric_query_by_arbitrary_search_term(metric, slug, since_str, until_str, resolution)
        else:
            query = self._get_metric_query_by_asset(metric, slug, since_str, until_str, resolution)

        response = requests.post(BASE_URL, headers=HEADERS, json={'query': query})
        result = response.json()
        timeseries_data = result.get('data', {}).get('getMetric', {}).get('timeseriesData', [])

        if not timeseries_data:
            return pd.DataFrame(columns=['t', 'v'])

        df = pd.DataFrame(timeseries_data)
        df.columns = ['t', 'v']
        df["t"] = pd.to_datetime(df["t"])

        # Format datetime based on resolution
        if resolution == "1h":  # Daily data - format as date only (YYYY-MM-DD)
            df['t'] = pd.to_datetime(df['t'].dt.strftime('%Y-%m-%d %H:%M:%S'))  # Remove timezone and time
        else:  # Hourly or other resolutions - keep full datetime
            pass  # Already has full datetime
        if resolution == "24h":
            df['t'] = pd.to_datetime(df['t'].dt.date)

        return df

    def fetch_available_slugs(self, metric: str) -> List[str]:
        BASE_URL = 'https://api.santiment.net/graphql'
        HEADERS = {
            'Authorization': f'Apikey {self.api_key}',
            'Content-Type': 'application/json'
        }

        query = f'''
        {{
          getMetric(metric: "{metric}") {{
            metadata {{
              availableProjects {{ slug }}
            }}
          }}
        }}
        '''

        response = requests.post(BASE_URL, headers=HEADERS, json={'query': query})
        result = response.json()

        try:
            all_slugs = [item['slug'] for item in
                         result.get('data', {}).get('getMetric', {}).get('metadata', {}).get('availableProjects', [])]
            return [slug for slug in all_slugs if slug in self.TOP_20_COINS.values()]
        except KeyError:
            return []

    def fetch_metric_list(self) -> List[str]:
        BASE_URL = 'https://api.santiment.net/graphql'
        HEADERS = {
            'Authorization': f'Apikey {self.api_key}',
            'Content-Type': 'application/json'
        }

        query = '{ getAvailableMetrics }'
        response = requests.post(BASE_URL, headers=HEADERS, json={'query': query})
        result = response.json()

        return result.get("data", {}).get("getAvailableMetrics", [])

if __name__ == "__main__":
    """
    input metric and asset
    """
