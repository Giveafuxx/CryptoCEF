Cryptocurrency Factor Trading Strategy User Guide
==============================================

Table of Contents
----------------
0. Define a FactorStrategy Class
1. Main Functions
2. Available Models
3. Heatmap Configuration
4. Single vs Double Metric Strategy


0.Define a FactorStrategy Class
------------------------------

The FactorStrategy class is initialized with the following parameters:

```python
strategy = FactorStrategy(
    # Required Parameters
    model="bband",                      # Trading model type
    metric1="addresses/active_count",   # Primary metric
    
    # Trading Parameters
    direction="reversion",  # "momentum" or "reversion"
    resolution="24h",       # Time resolution ("24h", "1h", "10m")
    long_short="long short",# Trading direction constraints ("long", "short", "long short")
    
    # Asset Parameters
    asset_input1="BTC",    # Input asset for metric1
    asset_output1="BTC",   # Output asset for metric1
    
    # Strategy Parameters
    window=50,            # Look-back window
    threshold=1,          # Trading threshold
    
    # Optional Parameters
    metric2=None,         # Secondary metric (for double metric strategy)
    operator=None,        # Operator for combining metrics
    asset_input2=None,    # Input asset for metric2
    asset_output2=None,   # Output asset for metric2
    currency="NATIVE",    # Base currency
    heatmap_para=None     # Heatmap configuration
)
```

1.Main Functions in FactorStrategy
---------------------------------

The FactorStrategy class provides three key functions for strategy analysis:

a) backtest(since: int, until: int)
   - Purpose: Executes strategy backtesting over a specified period with default window and threshold
   - Parameters:
     * since: Start timestamp (Unix format)
     * until: End timestamp (Unix format)
   - Example:
     ```python
     start_date = "2020-01-01"
     end_date = "2024-01-01"
     start_timestamp = int(pd.Timestamp(start_date).timestamp())
     end_timestamp = int(pd.Timestamp(end_date).timestamp())
     
     strategy.backtest(start_timestamp, end_timestamp)
     strategy.plot_equity_curve()
     ```

b) optimize(since: int, until: int)
   - Purpose: Optimizes strategy parameters with default model
   - Parameters: Same as backtest()
   - Use when: Single model optimization is needed
   - Example:
     ```python
     start_date = "2020-01-01"
     end_date = "2024-01-01"
     start_timestamp = int(pd.Timestamp(start_date).timestamp())
     end_timestamp = int(pd.Timestamp(end_date).timestamp())
     
     strategy.optimize(start_timestamp, end_timestamp)
     ```

c) batch_optimize(since: int, until: int)
   - Purpose: Performs optimization across multiple models
   - Parameters: Same as backtest()
   - Use when: Testing multiple model configurations
   - Example:
     ```python
     start_date = "2020-01-01"
     end_date = "2024-01-01"
     start_timestamp = int(pd.Timestamp(start_date).timestamp())
     end_timestamp = int(pd.Timestamp(end_date).timestamp())
     
     strategy.batch_optimize(start_timestamp, end_timestamp)
     ```

2.Available Models
------------------

The system supports the following models:

1. "bband" (Bollinger Bands)
   - Traditional technical indicator using mean and standard deviation
   - Suitable for range-bound markets

2. "log_bband" (Logarithmic Bollinger Bands)
   - Modified Bollinger Bands using logarithmic returns
   - Better for volatile markets

3. "robust" (Robust Scaling)
   - Uses statistics that are robust to outliers
   - Suitable for noisy data

4. "tanh" (Hyperbolic Tangent)
   - Non-linear transformation
   - Good for normalized bounded signals

5. "ma_diff" (Moving Average Difference)
   - Compares different moving averages
   - Trend-following strategies

6. "roc" (Rate of Change)
   - Momentum indicator
   - Measures percentage change

7. "rsi" (Relative Strength Index)
   - Momentum oscillator
   - Range: 0 to 100

8. "min_max" (Min-Max Scaling)
   - Linear scaling to fixed range
   - Simple and interpretable

9. "percentile" (Percentile-based)
   - Rank-based transformation
   - Robust to outliers

10. "raw" (Raw Values)
    - No transformation
    - Direct signal usage


3.Heatmap Configuration
-----------------------

Users can customize heatmap visualization by providing heatmap_para:

```python
heatmap_para = {
    "window": (min, max, step),      # Window range
    "threshold": (min, max, step)    # Threshold range
}

# Example
heatmap_para = {
    "window": (5, 200, 10),         # From 5 to 200, step 10
    "threshold": (0, 2.1, 0.1)      # From 0 to 2.1, step 0.1
}

strategy = FactorStrategy(
    # other parameters...
    heatmap_para=heatmap_para
)
```


4.Single vs Double Metric Strategy
-----------------------

The FactorStrategy supports both single and double metric analysis:

For Single Metric Setup:
```python
strategy = FactorStrategy(
    model="bband",
    metric1="addresses/active_count",
    asset_input1="ETH",
    asset_output1="ETH",
    
    direction="momentum",
    resolution="24h",
    long_short="long short",
    
    window=75,
    threshold=1.1,
)
```
For Double Metric Setup: (Important: Include operator)
```python
strategy = FactorStrategy(
    model="bband",
    metric1="transactions/transfers_volume_sum",
    metric2="addresses/active_count",
    operator="A / B",    # Required for double metric
    asset_input1="BTC",
    asset_output1="BTC",
    asset_input2="BTC",  # Optional, defaults to asset_input1
    asset_output2="BTC",  # Optional, defaults to asset_output1
    
    direction="momentum",
    resolution="24h",
    long_short="long short",
    
    window=75,
    threshold=1.1,
)
```

Available operator for double metric:

1. Addition
    - "A + B"
    - "Addition"

2. Subtraction
    - "A - B"
    - "Subtraction"
3. Multiplication
    - "A * B"
    - "Multiplication"
4. Division
    - "A / B"
    - "Division"
5. Ratio
    - "A / (A + B)"
    - "Ratio"
