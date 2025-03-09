from strategy import FactorStrategy
import pandas as pd

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
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
    # strategy.plot_equity_curve()

