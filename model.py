import logging
import numpy as np
import pandas_ta as ta
import pandas as pd
from scipy.stats import percentileofscore

logger = logging.getLogger(__name__)


class Model:
    def __init__(self):
        self.standardization_model = {
            "bband": self._standardize_bBand,
            "log_bband": self._standardize_logBand,
            "robust": self._standardize_robust,
            "tanh": self._standardize_tanh,
            "ma_diff": self._standardize_maDiff,
            "roc": self._standardize_roc,
            "rsi": self._standardize_rsi,
            "min_max": self._standardize_minMax,
            "percentile": self._standardize_percentile,
            "raw": self._standardize_raw
        }
        self.calculation_model = {
            "bband": self._cal_pos_zScore,
            "log_bband": self._cal_pos_zScore,
            "robust": self._cal_pos_zScore,
            "tanh": self._cal_pos_normalized,
            "ma_diff": self._cal_pos_change,
            "roc": self._cal_pos_pctChange,
            "rsi": self._cal_pos_zeroHundred,
            "min_max": self._cal_pos_zeroOne,
            "percentile": self._cal_pos_percentile,
            "raw": self._cal_pos_zScore,
        }

    def get_model_list(self):
        return list(self.standardization_model.keys())

    def standardize_metric(self, df: pd.DataFrame, window: int, model: str, threshold: float) -> pd.DataFrame:
        model = model.lower()

        if model not in self.standardization_model:
            raise ValueError(f"Model not found: {model}, exiting...")

        try:
            return self.standardization_model[model](df, window, threshold)
        except Exception as e:
            logger.error(f"Error in cal_pos: {e}")
            raise

    def cal_pos(self, df: pd.DataFrame, model: str, threshold: float, long_or_short: str, direction: str):
        long_or_short = long_or_short.lower()
        direction = direction.lower()

        if model not in self.calculation_model:
            raise ValueError(f"Model not found: {model}, exiting...")

        try:
            return self.calculation_model[model](df, threshold, long_or_short, direction)
        except Exception as e:
            logger.error(f"Error in cal_pos: {e}")
            raise

    @staticmethod
    def _standardize_bBand(df: pd.DataFrame, window: int, threshold: float):
        df['ma'] = df['value'].rolling(window).mean()
        df['sd'] = df['value'].rolling(window).std()
        # zscore is n standard d away from mean
        df['value_de'] = (df['value'] - df['ma']) / df['sd']

        return df

    @staticmethod
    def _standardize_raw(df: pd.DataFrame, window: int, threshold: float):
        df['value_de'] = df['value']

        return df

    @staticmethod
    def _standardize_logBand(df: pd.DataFrame, window: int, threshold: float):
        df['log'] = np.log(df['value'])
        df['ma'] = df['log'].rolling(window).mean()
        df['sd'] = df['log'].rolling(window).std()
        df['value_de'] = (df['log'] - df['ma']) / df['sd']

        return df

    @staticmethod
    def _standardize_robust(df: pd.DataFrame, window: int, threshold: float):
        df["median"] = df["value"].rolling(window).median()
        df["IQR"] = df["value"].rolling(window).quantile(0.75) - df["value"].rolling(window).quantile(0.25)
        df["value_de"] = (df["value"] - df["median"]) / df["IQR"]

        return df

    @staticmethod
    def _standardize_tanh(df: pd.DataFrame, window: int, threshold: float):
        df['ma'] = df['value'].rolling(window).mean()
        df['sd'] = df['value'].rolling(window).std()
        df["value_de"] = 0.5 * (np.tanh(((df["value"] - df['ma']) / df["sd"])) + 1)

        return df

    @staticmethod
    def _standardize_maDiff(df: pd.DataFrame, window: int, threshold: float):
        df['ma'] = df['value'].rolling(window).mean()
        df['value_de'] = (df['value'] / df['ma']) - 1.0

        return df

    @staticmethod
    def _standardize_roc(df: pd.DataFrame, window: int, threshold: float):
        df['value_de'] = df['value'].pct_change(periods=window)

        return df

    @staticmethod
    def _standardize_rsi(df: pd.DataFrame, window: int, threshold: float):
        df['value_de'] = ta.rsi(df['value'], length=window)

        return df

    @staticmethod
    def _standardize_minMax(df: pd.DataFrame, window: int, threshold: float):
        df['min'] = df['value'].rolling(window).min()
        df['max'] = df['value'].rolling(window).max()
        df['value_de'] = (df['value'] - df['min']) / (df['max'] - df['min'])

        return df

    @staticmethod
    def _standardize_percentile(df: pd.DataFrame, window: int, threshold: float):
        df['percentile_high'] = df['value'].rolling(window).quantile(1 - threshold)
        df['percentile_low'] = df['value'].rolling(window).quantile(threshold)
        df["value_de"] = df['value'].rolling(window=window).apply(lambda x: percentileofscore(x, x.iloc[-1]))

        return df

    @staticmethod
    def _cal_pos_zScore(df: pd.DataFrame, threshold: float, long_or_short: str, direction: str):
        if direction == 'momentum':
            if long_or_short == 'long short':
                df['pos'] = np.where(
                    df['value_de'] > threshold, 1, np.where(df['value_de'] < -threshold, -1, 0)
                )
            elif long_or_short == 'long':
                df['pos'] = np.where(
                    df['value_de'] > threshold, 1, np.where(df['value_de'] < -threshold, 0, 0)
                )
            elif long_or_short == 'short':
                df['pos'] = np.where(
                    df['value_de'] > threshold, 0, np.where(df['value_de'] < -threshold, -1, 0)
                )
        else:
            if long_or_short == 'long short':
                df['pos'] = np.where(
                    df['value_de'] > threshold, -1, np.where(df['value_de'] < -threshold, 1, 0)
                )
            elif long_or_short == 'long':
                df['pos'] = np.where(
                    df['value_de'] > threshold, 0, np.where(df['value_de'] < -threshold, 1, 0)
                )
            elif long_or_short == 'short':
                df['pos'] = np.where(
                    df['value_de'] > threshold, -1, np.where(df['value_de'] < -threshold, 0, 0)
                )
        return df['pos']

    @staticmethod
    def _cal_pos_normalized(df: pd.DataFrame, threshold: float, long_or_short: str, direction: str):
        if direction == 'momentum':
            if long_or_short == 'long short':
                df['pos'] = np.where(
                    df['value_de'] > (1 - threshold), 1, np.where(df['value_de'] < threshold, -1, 0)
                )
            elif long_or_short == 'long':
                df['pos'] = np.where(
                    df['value_de'] > (1 - threshold), 1, np.where(df['value_de'] < threshold, 0, 0)
                )
            elif long_or_short == 'short':
                df['pos'] = np.where(
                    df['value_de'] > (1 - threshold), 0, np.where(df['value_de'] < threshold, -1, 0)
                )
        else:
            if long_or_short == 'long short':
                df['pos'] = np.where(
                    df['value_de'] > (1 - threshold), -1, np.where(df['value_de'] < threshold, 1, 0)
                )
            elif long_or_short == 'long':
                df['pos'] = np.where(
                    df['value_de'] > (1 - threshold), 0, np.where(df['value_de'] < threshold, 1, 0)
                )
            elif long_or_short == 'short':
                df['pos'] = np.where(
                    df['value_de'] > (1 - threshold), -1, np.where(df['value_de'] < threshold, 0, 0)
                )
        return df['pos']

    def _cal_pos_change(self, df: pd.DataFrame, threshold: float, long_or_short: str, direction: str):
        return self._cal_pos_zScore(df, threshold, long_or_short, direction)

    def _cal_pos_pctChange(self, df: pd.DataFrame, threshold: float, long_or_short: str, direction: str):
        return self._cal_pos_zScore(df, threshold, long_or_short, direction)

    @staticmethod
    def _cal_pos_zeroHundred(df: pd.DataFrame, threshold: float, long_or_short: str, direction: str):
        if direction == 'momentum':
            if long_or_short == 'long short':
                df['pos'] = np.where(
                    (df['value_de'] > (100 - threshold)), 1, np.where(df['value_de'] < threshold, -1, 0)
                )
            elif long_or_short == 'long':
                df['pos'] = np.where(
                    (df['value_de'] > (100 - threshold)), 1, np.where(df['value_de'] < threshold, 0, 0)
                )
            elif long_or_short == 'short':
                df['pos'] = np.where(
                    (df['value_de'] > (100 - threshold)), 0, np.where(df['value_de'] < threshold, -1, 0)
                )
        else:
            if long_or_short == 'long short':
                df['pos'] = np.where(
                    (df['value_de'] > (100 - threshold)), -1, np.where(df['value_de'] < threshold, 1, 0)
                )
            elif long_or_short == 'long':
                df['pos'] = np.where(
                    (df['value_de'] > (100 - threshold)), 0, np.where(df['value_de'] < threshold, 1, 0)
                )
            elif long_or_short == 'short':
                df['pos'] = np.where(
                    (df['value_de'] > (100 - threshold)), -1, np.where(df['value_de'] < threshold, 0, 0)
                )
        return df['pos']

    @staticmethod
    def _cal_pos_zeroOne(df: pd.DataFrame, threshold: float, long_or_short: str, direction: str):
        if direction == 'momentum':
            if long_or_short == 'long short':
                df['pos'] = np.where(
                    df['value_de'] >= (1 - threshold), 1, np.where(df['value_de'] <= threshold, -1, 0)
                )
            elif long_or_short == 'long':
                df['pos'] = np.where(
                    df['value_de'] > (1 - threshold), 1, np.where(df['value_de'] <= threshold, 0, 0)
                )
            elif long_or_short == 'short':
                df['pos'] = np.where(
                    df['value_de'] >= (1 - threshold), 0, np.where(df['value_de'] <= threshold, -1, 0)
                )
        else:
            if long_or_short == 'long short':
                df['pos'] = np.where(
                    df['value_de'] >= (1 - threshold), -1, np.where(df['value_de'] <= threshold, 1, 0)
                )
            elif long_or_short == 'long':
                df['pos'] = np.where(
                    df['value_de'] >= (1 - threshold), 0, np.where(df['value_de'] <= threshold, 1, 0)
                )
            elif long_or_short == 'short':
                df['pos'] = np.where(
                    df['value_de'] >= (1 - threshold), -1, np.where(df['value_de'] <= threshold, 0, 0)
                )

        return df['pos']

    @staticmethod
    def _cal_pos_percentile(df: pd.DataFrame, threshold: float, long_or_short: str, direction: str):
        if direction == 'momentum':
            if long_or_short == 'long short':
                df['pos'] = np.where(
                    df['value'] >= df['percentile_high'], 1,
                    np.where(df['value'] <= df['percentile_low'], -1, 0)
                )
            elif long_or_short == 'long':
                df['pos'] = np.where(
                    df['value'] >= df['percentile_high'], 1,
                    np.where(df['value'] <= df['percentile_low'], 0, 0)
                )
            elif long_or_short == 'short':
                df['pos'] = np.where(
                    df['value'] >= df['percentile_high'], 0,
                    np.where(df['value'] <= df['percentile_low'], -1, 0)
                )
        else:
            if long_or_short == 'long short':
                df['pos'] = np.where(
                    df['value'] >= df['percentile_high'], -1,
                    np.where(df['value'] <= df['percentile_low'], 1, 0)
                )
            elif long_or_short == 'long':
                df['pos'] = np.where(
                    df['value'] >= df['percentile_high'], 0,
                    np.where(df['value'] <= df['percentile_low'], 1, 0)
                )
            elif long_or_short == 'short':
                df['pos'] = np.where(
                    df['value'] >= df['percentile_high'], -1,
                    np.where(df['value'] <= df['percentile_low'], 0, 0)
                )

        return df['pos']
