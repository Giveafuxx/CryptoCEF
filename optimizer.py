import logging

import numpy as np
import pandas as pd

import os
from typing import Tuple, Dict

from parameter import StrategyDetail, MetricDetail

from backtest import cal_pnl, backtesting

import copy

from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from model import Model

import sys
import plot

logger = logging.getLogger(__name__)


class Optimizer:
    def __init__(self, **kwargs):
        """
        Initialize the Backtester class.

        Args:
            long_or_short (str): Indicates whether the strategy is long or short (e.g., 'long', 'short', 'long short').
            resolution (str): The time resolution for the backtest (e.g., '24h', '1h', '10m').
        """
        self.resolution = kwargs.get("resolution", "24h")
        self.heatmap_para = kwargs.get("heatmap_para", None)

    def _model_window(self, model: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate window and threshold parameters for a given model.

        Args:
            model (str): The model for which to generate parameters.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Arrays of window and threshold parameters.
        """

        def generate_params(window_range, threshold_range):
            return np.arange(*window_range), np.arange(*threshold_range)

        model_params = {
            "bband": {
                "24h": ((5, 375, 10), (0, 2.6, 0.1)),
                "1h": ((10, 300, 10), (0, 2.25, 0.25)),
                "10m": ((20, 300, 10), (0, 2.25, 0.25)),
            },
            "log_bband": {
                "24h": ((5, 375, 10), (0, 2.6, 0.1)),
                "1h": ((10, 300, 10), (0, 2.25, 0.25)),
                "10m": ((20, 300, 10), (0, 2.25, 0.25)),
            },
            "robust": {
                "24h": ((5, 375, 10), (0, 2.6, 0.1)),
                "1h": ((10, 300, 10), (0, 2.25, 0.25)),
                "10m": ((20, 300, 10), (0, 2.25, 0.25)),
            },
            "ma_diff": {
                "24h": ((5, 375, 10), (0, 0.11, 0.01)),
                "1h": ((10, 300, 10), (0, 0.1, 0.01)),
                "10m": ((300, 12000, 300), (0, 0.1, 0.01)),
            },
            "roc": {
                "24h": ((5, 375, 10), (0, 0.21, 0.01)),
                "1h": ((1, 200, 5), (0, 0.1, 0.005)),
                "10m": ((1, 14, 1), (0, 0.01, 0.001)),
            },
            "rsi": {
                "24h": ((5, 375, 10), (0, 52.5, 2.5)),
                "1h": ((10, 200, 5), (0, 50, 2.5)),
                "10m": ((0, 20, 1), (0, 30, 1)),
            },
            "percentile": {
                "24h": ((5, 375, 10), (0, 0.525, 0.025)),
                "1h": ((10, 300, 10), (0.03, 0.5, 0.03)),
                "10m": ((0, 20, 1), (0, 0.30, 0.02)),
            },
            "tanh": {
                "24h": ((5, 375, 10), (0, 0.525, 0.025)),
                "1h": ((10, 300, 10), (0, 0.55, 0.03)),
                "10m": ((0, 20, 1), (0, 0.30, 0.02)),
            },
            "min_max": {
                "24h": ((5, 375, 10), (0.00, 0.525, 0.025)),
                "1h": ((10, 200, 10), (0.05, 0.5, 0.05)),
                "10m": ((10, 110, 10), (0.05, 0.5, 0.05)),
            },
            "raw": {
                "24h": ((5, 375, 10), (0.00, 0.525, 0.025)),
                "1h": ((10, 200, 10), (0.05, 0.5, 0.05)),
                "10m": ((10, 110, 10), (0.05, 0.5, 0.05)),
            },
        }

        if model not in model_params or self.resolution not in model_params[model]:
            sys.exit()

        if self.heatmap_para is None:
            window_range, threshold_range = model_params[model][self.resolution]
        else:
            window_range = self.heatmap_para["window"]
            threshold_range = self.heatmap_para["threshold"]

        return generate_params(window_range, threshold_range)


class FactorStrategyOptimizer(Optimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.result_df = {}
        self.table_sharpe = {}
        self.strategy_info: StrategyDetail = kwargs.get("strategy_info")
        self.model = kwargs.get("model")
        self.direction = self.strategy_info.direction
        self.data = kwargs.get("data_preprocessed", pd.DataFrame())
        self.metric_info: Dict[str, MetricDetail] = kwargs.get("metric_info")

    def __data_existence_validation(self, model):
        if model in self.result_df:
            pass
        else:
            self.__optimize(model)

        return self.result_df[model]

    def __optimize(
        self, model: str
    ) -> pd.DataFrame:
        """
        Perform batch backtesting for a given model and direction.

        Args:
            model (str): The model to be used for backtesting.

        Returns:
            pd.DataFrame: A data frame containing the backtest results.
        """
        window_list, threshold_list = self._model_window(model)
        result_list = []
        temp_df = copy.deepcopy(self.data)

        for window in window_list:
            temp_df = Model().standardize_metric(temp_df, window, model)
            try:
                for threshold in threshold_list:
                    temp_df["pos"] = Model().cal_pos(df=temp_df,
                                                     model=model,
                                                     threshold=threshold,
                                                     long_or_short=self.strategy_info.long_short,
                                                     direction=self.strategy_info.direction)

                    temp_df = cal_pnl(temp_df)

                    result, df_placeholder = backtesting(
                        window,
                        threshold,
                        temp_df,
                        self.resolution,
                    )
                    result = pd.Series(result.__dict__)
                    result_list.append(result)
            except Exception as e:
                logger.error(f"Failed to backtest with window={window}, threshold={threshold}, model={model}, "
                             f"direction={self.strategy_info.direction}, resolution={self.resolution}")
                continue
        result_df = pd.DataFrame(result_list)
        result_df = result_df.sort_values(by="sharpe", ascending=False)
        result_df = result_df.reset_index()
        result_df["window"] = result_df["window"].round(3)
        result_df["threshold"] = result_df["threshold"].round(3)

        self.result_df[model] = result_df.sort_values(by='sharpe', ascending=False)

    def __show_top_strategy(self, model):
        result_df = self.result_df[model]
        temp_df = copy.deepcopy(self.data)
        while True:
            print(
                f"\n******************** Top 50 Sharpe Equity Curve with model:    {model}  | {self.strategy_info.direction}  ***************************")

            for i, row in result_df.iterrows():
                if i > 49: break
                print(
                    f'{i + 1}:  Sharpe: {row["sharpe"]} | Sortino: {round(row["sortino"], 3)} | Calmar: {row["calmar"]} | MDD: {row["mdd"]} | Window: {row["window"]} | Threshold: {row["threshold"]} | Trades: {row["trade"]} ')

            print(
                f"******************** Top 50 Sharpe Equity Curve with model:    {model} | {self.strategy_info.direction}  ***************************")

            scan = input("\nEnter 1 to 50, 'c' to continue, 'q' or 'x' to exit: ").strip()

            try:
                scan = int(scan)
            except:
                pass

            if isinstance(scan, int) and 1 <= scan <= 50:
                key_pressed = int(scan)
                window = int(result_df.at[(key_pressed - 1), "window"])
                threshold = result_df.at[key_pressed - 1, "threshold"]
                sharpe = result_df.at[key_pressed - 1, "sharpe"]

                print(f'[PLOTTING], sharpe: {sharpe}, window: {window}, threshold: {threshold}, model: {model}')

                temp_df = Model().standardize_metric(temp_df, window, model)

                temp_df["pos"] = Model().cal_pos(df=temp_df,
                                                 model=model,
                                                 threshold=threshold,
                                                 long_or_short=self.strategy_info.long_short,
                                                 direction=self.strategy_info.direction)

                temp_df = cal_pnl(temp_df)

                result, df_placeholder = backtesting(
                    window,
                    threshold,
                    temp_df,
                    self.resolution,
                )
                print(result)
                fig = plot.FactorStrategyPlotter(
                    df_plot=df_placeholder, mode="advanced", result=result, metric=self.metric_info["metric1"].metric
                )
                fig.show()

            if scan == 'q' or scan == 'x':
                sys.exit()

            if scan == 'c':
                break

    def update_model(self, model):
        self.model = model

    def optimize(self, model):
        self.__optimize(model)
        self.plot(model)
        self.__show_top_strategy(model)

    def batch_optimize(self):
        model_list = Model().get_model_list()
        for model in model_list:
            self.optimize(model)

    def plot(self, model: str):
        try:
            target_df = self.__data_existence_validation(model)

            self.table_sharpe[model] = target_df.pivot(index='window', columns='threshold', values='sharpe')

            heatmap_plot = sns.heatmap(self.table_sharpe[model], annot=True, cmap='RdBu', fmt='', center=0, vmin=-1.5,
                                       vmax=1.5)
            fig = plt.gcf()
            fig.set_size_inches(20, 12)

            title = f'{self.metric_info["metric1"].metric} | {self.metric_info["metric2"].metric} | ' \
                    f'{self.metric_info["metric1"].operator} | {model} | ' \
                    f'{self.metric_info["metric1"].asset_input}->{self.metric_info["metric1"].asset_output} | ' \
                    f'{self.strategy_info.currency} | {self.resolution} | {self.direction} | ' \
                    f'{self.strategy_info.long_short}\n  \
                                              Num trades of best sharpe: {round(target_df.iloc[0]["trade"])}'
            plt.title(title)

            image_file = 'image.jpg'
            plt.savefig(image_file, format='jpg', dpi=100)
            plt.close()
            # Open the image file using the default associated application (Windows-only)
            if os.name == 'nt':  # Check if the OS is Windows
                os.startfile(image_file)
            else:
                # Use Pillow to open and display the image (platform-independent)
                img = Image.open(image_file)
                img.show()
        except:
            logger.error(f"Error in plotting heatmap with model={model}, direction={self.strategy_info.direction}, "
                         f"resolution={self.resolution}")
