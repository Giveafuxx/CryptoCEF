from strategy import FactorStrategy
import pandas as pd

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    '''------------------------Config------------------------------------'''
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    heatmap_para = {"window": (5, 375, 10),
                    "threshold": (0, 0.5, 0.025)}

    start_date = "2020-01-01"
    end_date = "2023-12-31"
    start_timestamp = int(pd.Timestamp(start_date).timestamp())
    end_timestamp = int(pd.Timestamp(end_date).timestamp())

    '''-----------------------Strategy Detail------------------------------------'''
    # Initialize the FactorStrategy class
    strategy = FactorStrategy(
        model="min_max",        # "bband", "rsi", "percentile" etc, refer to model.py for more details

        metric1="addresses/accumulation_balance",
        # transformType1="fft",  # "cumsum", "mean", "sqrt", "log" etc, refer to transformer.py for more details
        # transformPeriod1=50,

        # metric2="addresses/accumulation_balance",
        # transformType2="raw",
        # transformPeriod2=50,

        # operator="A / B",     # "A + B", "A - B", "A * B", "A / B", "A / (A + B)", refer to metric.apply_operator()

        asset_input1="BTC",
        asset_input2="BTC",
        asset_output1="BTC",
        currency="NATIVE",

        long_short="long short",
        window=250,
        threshold=0.5,
        direction="momentum",   # "momentum", "reversion"
        resolution="24h",        # "1h", "24h"

        # delay_time=20          # lag price in minutes (multiple of 10)
        # heatmap_para=heatmap_para
    )

    '''-------------------------Strategy Functions------------------------------------'''
    # strategy.plot_metric(start_timestamp, end_timestamp)

    # strategy.backtest(start_timestamp, end_timestamp)
    # strategy.plot_equity_curve()

    # strategy.optimize(start_timestamp, end_timestamp)

    strategy.batch_optimize(start_timestamp, end_timestamp)

