import numpy as np
import pandas as pd
from load_24h_cgmacros import generate_24H_CGMacros_dataset
import matplotlib.pyplot as plt
import pdb


def plot_super_long_signals_from_df(
    df,
    figsize=(20, 3),
    title="Super Long Signals",
    xlabel="Time",
    ylabel="Value",
    save_path="super_long_signals.png",
):
    """
    Plots each column of a DataFrame as a separate subplot stacked vertically and saves the figure.

    Args:
        df (pd.DataFrame): DataFrame where each column is a signal to plot.
        figsize (tuple): Figure size in inches (width, height per subplot).
        title (str): Overall plot title.
        xlabel (str): X-axis label for all subplots.
        ylabel (str): Y-axis label for each subplot.
        save_path (str): Path to save the figure.
    """
    # Set high DPI for super high resolution
    plt.rcParams["figure.dpi"] = 300

    n_signals = df.shape[1]
    fig, axes = plt.subplots(
        n_signals, 1, figsize=(figsize[0], figsize[1] * n_signals), sharex=True
    )
    if n_signals == 1:
        axes = [axes]
    fig.suptitle(title)
    for idx, col in enumerate(df.columns):
        axes[idx].plot(df.index, df[col], linewidth=1)
        axes[idx].set_ylabel(f"{ylabel}\n{col}")
        if idx == n_signals - 1:
            axes[idx].set_xlabel(xlabel)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path)
    plt.close(fig)


def plot_super_long_signals_heatmap_from_df(
    df,
    figsize=(20, 6),
    title="Super Long Signals Heatmap",
    xlabel="Time",
    ylabel="Signals",
    save_path="super_long_signals_heatmap.png",
    cmap="viridis",
):
    """
    Plots a heatmap of the DataFrame (signals as rows, time as columns) and saves the figure.

    Args:
        df (pd.DataFrame): DataFrame where each column is a signal to plot.
        figsize (tuple): Figure size in inches (width, height).
        title (str): Overall plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        save_path (str): Path to save the figure.
        cmap (str): Colormap for the heatmap.
    """
    plt.rcParams["figure.dpi"] = 300

    fig, ax = plt.subplots(figsize=figsize)
    # Data: signals as rows, time as columns
    data = df.T.values
    im = ax.imshow(data, aspect="auto", cmap=cmap)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yticks(np.arange(len(df.columns)))
    ax.set_yticklabels(df.columns)
    ax.set_xticks([0, len(df.index) - 1])
    ax.set_xticklabels([str(df.index[0]), str(df.index[-1])])

    plt.colorbar(im, ax=ax, orientation="vertical", label="Value")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


# Example usage:
if __name__ == "__main__":
    desired_datetime = pd.to_datetime("2021-09-19 8:00")
    ts_df, meal_df = generate_24H_CGMacros_dataset()
    ts_df.index = pd.to_datetime(ts_df.index)
    # Select data from desired_datetime to desired_datetime + 6 hours
    start_time = desired_datetime
    end_time = desired_datetime + pd.Timedelta(hours=12)
    selected_df = ts_df[(ts_df.index >= start_time) & (ts_df.index < end_time)]

    plot_super_long_signals_from_df(selected_df, figsize=(30, 3))
    plot_super_long_signals_heatmap_from_df(selected_df, figsize=(30, 3))
    asd = selected_df
    # Divide 'Libre GL' into 2-hour segments
    libre_gl = asd["Libre GL"]
    segments = []
    segment_length = pd.Timedelta(minutes=30)
    start_time = libre_gl.index[0]
    end_time = libre_gl.index[-1]

    while start_time < end_time:
        segment = libre_gl[
            (libre_gl.index >= start_time)
            & (libre_gl.index < start_time + segment_length)
        ]
        if not segment.empty:
            segments.append(segment)
        start_time += segment_length
    averages = [segment.mean() for segment in segments]
    # Normalize averages to [0, 1] range
    averages = np.array(averages)
    if len(averages) > 0:
        norm_averages = (averages - averages.min()) / (averages.max() - averages.min())
    else:
        norm_averages = averages
    # segments is now a list of pd.Series, each representing a 2-hour segment
    pdb.set_trace()  # This line is for debugging purposes, can be removed in production
