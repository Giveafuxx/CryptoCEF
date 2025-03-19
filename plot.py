import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from analyzer import PerformanceAnalyzer
from parameter import StrategyPerformance
import copy


class FactorStrategyPlotter:
    def __init__(self,
                 df_plot: pd.DataFrame = pd.DataFrame(),
                 mode: str = "simple",
                 resolution: str = "24h",
                 result: StrategyPerformance = StrategyPerformance(),
                 metric: str = ""):
        self.resolution = resolution
        self.df_plot = copy.deepcopy(df_plot)
        if "t" in self.df_plot.columns:
            self.df_plot.set_index("t", inplace=True)

        if mode == "simple":
            self.fig = self.draw_simple_mode()
        elif mode == "advanced":
            self.fig = self.draw_advanced_mode()
        else:
            self.fig = self.draw_simple_mode()

        title = f"Metric: {metric} | Window: {result.window} | Threshold: {result.threshold} | " \
                f"Sharpe: {result.sharpe} | Calmar: {result.calmar} | Annualized Return: {result.annual_return} | " \
                f"MDD: {result.mdd} | Trade: {result.trade}"

        self.fig.update_layout(
            title_text=title,  # Main title
            title_x=0.5,  # Center the title
            title_font_size=20,  # Font size
            title_font_family="Arial",  # Font family
            title_font_color="#000000"  # Font color
        )

    def draw_simple_mode(self):
        fig = make_subplots(
            rows=1, cols=1, shared_xaxes=True
        )
        fig = self.draw_main(fig=fig, row=1)

        return fig

    def draw_advanced_mode(self):
        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02,
            specs=[[{"secondary_y": True}]]*4,
            row_heights=[0.6, 0.1, 0.1, 0.1]
        )

        fig = self.draw_main(fig=fig, row=1)

        fig = self.draw_metric(fig=fig, row=2)

        fig = self.draw_rolling_sharpe_sortino(fig=fig, row=3)

        fig = self.draw_rolling_alpha_beta(fig=fig, row=4)

        return fig

    def draw_main(self, fig, row=1):
        color_map = {
            1: 'rgb(40, 200, 100)',  # Green for 'long'
            -1: 'rgb(220, 80, 90)',  # Red for 'short'
            0: 'rgb(100, 100, 100)'  # Grey for neutral/no position
        }
        colors = [color_map[val] for val in self.df_plot['pos_t-1'].fillna(0)]

        fig.add_trace(
            go.Scatter(x=self.df_plot.index, y=self.df_plot['bnh_cumu'], mode='lines', name='bnh',
                       line=dict(color='rgba(51, 153, 255, 100)'), opacity=0.3), row=row, col=1)
        fig.add_trace(
            go.Scatter(x=self.df_plot.index, y=self.df_plot['dd'], mode='lines', name='dd',
                       line=dict(color='rgb(240, 101, 111)')), row=row, col=1)
        fig.add_trace(
            go.Scatter(x=self.df_plot.index, y=self.df_plot['cumu'], mode='lines', name='cumu',
                       line=dict(color='rgb(40, 200, 100)')), row=row, col=1)
        fig.add_trace(
            go.Scatter(x=self.df_plot.index, y=self.df_plot['cumu'], mode='markers', name='cumu (with position)',
                       marker=dict(color=colors, size=4)), row=row, col=1)
        return fig

    def draw_metric(self, fig, row=2):
        # Create traces for the second subplot
        fig.add_trace(
            go.Scatter(x=self.df_plot.index, y=self.df_plot['value'], mode='lines', name='raw data',
                       line=dict(color='rgb(96, 96, 96)')), row=row, col=1)
        fig.add_trace(
            go.Scatter(x=self.df_plot.index, y=self.df_plot['value_de'], mode='lines', name='processed data',
                       line=dict(color='rgb(0, 102, 204)')), row=row, col=1, secondary_y=True)
        return fig

    def draw_rolling_sharpe_sortino(self, fig, row=3):
        # Create traces for the third subplot
        PerformanceAnalyzer.Calculate.rolling_sharpe_ratio(df=self.df_plot, resolution=self.resolution, window=60)
        fig.add_trace(
            go.Scatter(x=self.df_plot.index, y=self.df_plot['rolling_sharpe'], mode='lines',
                       name='rolling sharpe ratio', line=dict(color='rgb(0, 188, 56)')), row=row, col=1)

        # Create traces for the third subplot
        PerformanceAnalyzer.Calculate.rolling_sortino_ratio(df=self.df_plot, resolution=self.resolution, window=60)
        fig.add_trace(
            go.Scatter(x=self.df_plot.index, y=self.df_plot['rolling_sortino'], mode='lines',
                       name='rolling sortino ratio', line=dict(color='rgb(255, 128, 0)')), row=row, col=1, secondary_y=True)
        return fig

    def draw_rolling_alpha_beta(self, fig, row=3):
        # Create traces for the third subplot
        PerformanceAnalyzer.Calculate.rolling_alpha(df=self.df_plot, window=60)
        fig.add_trace(
            go.Scatter(x=self.df_plot.index, y=self.df_plot['rolling_alpha'], mode='lines',
                       name='rolling alpha', line=dict(color='rgb(187, 80, 66)')), row=row, col=1)

        # Create traces for the third subplot
        PerformanceAnalyzer.Calculate.rolling_beta(df=self.df_plot, window=60)
        fig.add_trace(
            go.Scatter(x=self.df_plot.index, y=self.df_plot['annualized_rolling_beta'], mode='lines',
                       name='rolling beta', line=dict(color='rgb(115, 74, 165)')), row=row, col=1, secondary_y=True)

        return fig

    def show(self):
        self.fig.show()

    def save_figure(self, filename: str, format: str = "png"):
        """
        Save the figure to a file
        Args:
            filename: name of the file without extension
            format: file format ('png', 'jpg', 'pdf', 'html', etc.)
        """
        if format == "html":
            self.fig.write_html(f"{filename}.html")
        else:
            self.fig.write_image(f"{filename}.{format}", scale=3)


class MetricPlotter:
    def __init__(self, **kwargs):
        self.df = kwargs.get("df")

    def plot(self):
        # Create visualization
        plt.figure(figsize=(12, 6))

        # Create twin axes for different scales
        ax1 = plt.gca()
        ax2 = ax1.twinx()

        # Plot original prices on left y-axis
        line1 = ax1.plot(self.df['t'], self.df['price'],
                         label='Original', color='blue', alpha=0.7)
        ax1.set_xlabel('t')
        ax1.set_ylabel('Price', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Plot normalized prices on right y-axis
        line2 = ax2.plot(self.df['t'], self.df['value'],
                         label='Preprocessed metric', color='gray', alpha=0.7)
        ax2.set_ylabel('Preprocessed metric', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')

        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')

        plt.title('Preprocessed metric vs Price')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.02,
        specs=[[{"secondary_y": True}],  # Row 1: Secondary y-axis
               [{"secondary_y": True}],  # Row 2: Secondary y-axis
               [{"secondary_y": True}],  # Row 3: Secondary y-axis
               [{"secondary_y": False}],  # Row 4: No secondary y-axis
               [{"secondary_y": False}]]  # Row 5: No secondary y-axis
    )

    # Adding traces to the first subplot with secondary y-axis
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6], name="Primary Y - Row 1"), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3], name="Secondary Y - Row 1"), row=1, col=1, secondary_y=True)

    # Adding traces to the second subplot with secondary y-axis
    fig.add_trace(go.Bar(x=[1, 2, 3], y=[6, 5, 4], name="Primary Y - Row 2"), row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[2, 3, 1], mode="lines+markers", name="Secondary Y - Row 2"), row=2, col=1,
                  secondary_y=True)

    # Adding traces to the third subplot with secondary y-axis
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[7, 8, 9], name="Primary Y - Row 3"), row=3, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[2, 1, 2], name="Secondary Y - Row 3"), row=3, col=1, secondary_y=True)

    # Adding trace to the fourth subplot without secondary y-axis
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[3, 2, 1], mode="lines", name="Row 4 - No Secondary Y"), row=4, col=1)

    # Adding trace to the fifth subplot without secondary y-axis
    fig.add_trace(go.Bar(x=[1, 2, 3], y=[9, 8, 7], name="Row 5 - No Secondary Y"), row=5, col=1)

    # Update layout
    fig.update_layout(title="5-Row Subplot Example with Shared X-Axis and Select Secondary Y-Axes")

    # Show the figure
    fig.show()
