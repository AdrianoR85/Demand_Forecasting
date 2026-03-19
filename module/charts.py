"""
modules/charts.py
All Matplotlib chart rendering for the Streamlit app.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np

# ── Design tokens ────────────────────────────────────────────────────────────
BG         = "#0d1117"
BG_CARD    = "#161b22"
GRID       = "#21262d"
TEXT       = "#e6edf3"
MUTED      = "#7d8590"
ACCENT_1   = "#58a6ff"   # blue  – historical
ACCENT_2   = "#f78166"   # coral – forecast
ACCENT_3   = "#3fb950"   # green – trend positive
ACCENT_RED = "#f85149"   # red   – trend negative


def _base_fig(figsize=(13, 5)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
    ax.set_facecolor(BG_CARD)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    return fig, ax


def forecast_chart(
    monthly: pd.DataFrame,
    forecast: pd.DataFrame,
    cutoff: pd.Timestamp,
    product: str,
) -> plt.Figure:
    """Main timeline chart: historical line + forecast ribbon."""
    fig, ax = _base_fig(figsize=(13, 5))

    hist_fc   = forecast[forecast["ds"] <= cutoff]
    future_fc = forecast[forecast["ds"] >  cutoff]

    # Historical band + line
    ax.fill_between(hist_fc["ds"],
                    hist_fc["yhat_lower"].clip(0), hist_fc["yhat_upper"],
                    alpha=0.12, color=ACCENT_1, zorder=1)
    ax.plot(monthly["ds"], monthly["y"],
            color=ACCENT_1, linewidth=2.2, label="Actual sales", zorder=4)
    ax.plot(hist_fc["ds"], hist_fc["yhat"],
            color=ACCENT_1, linewidth=1, linestyle="--", alpha=0.45, zorder=3)

    # Forecast band + line
    ax.fill_between(future_fc["ds"],
                    future_fc["yhat_lower"].clip(0), future_fc["yhat_upper"],
                    alpha=0.22, color=ACCENT_2, zorder=1)
    ax.plot(future_fc["ds"], future_fc["yhat"].clip(0),
            color=ACCENT_2, linewidth=2.5, label="Forecast", zorder=5)

    # Dots on forecast points
    ax.scatter(future_fc["ds"], future_fc["yhat"].clip(0),
            color=ACCENT_2, s=55, zorder=6, edgecolors=BG, linewidths=1.2)

    # Cutoff line
    ax.axvline(cutoff, color=MUTED, linestyle=":", linewidth=1.2, alpha=0.6)
    ymax = ax.get_ylim()[1]
    ax.text(cutoff, ymax * 0.97, "  Forecast start",
            color=MUTED, fontsize=8, va="top")

    ax.set_title(f"Sales Forecast — {product}",
                color=TEXT, fontsize=13, pad=14, fontweight="bold")
    ax.set_ylabel("Quantity (kg)", color=MUTED, fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30, ha="right")
    ax.legend(facecolor=BG_CARD, labelcolor=TEXT, fontsize=9,
            edgecolor=GRID, framealpha=1)

    fig.tight_layout(pad=2)
    return fig


def bar_chart(result_df: pd.DataFrame) -> plt.Figure:
    """Bar chart for the 6-month forecast with error bars."""
    fig, ax = _base_fig(figsize=(11, 4))

    months = result_df["Month"]
    yhat   = result_df["Forecast (kg)"]
    lower  = result_df["Lower (kg)"]
    upper  = result_df["Upper (kg)"]

    x = np.arange(len(months))
    bars = ax.bar(x, yhat, color=ACCENT_2, alpha=0.82, width=0.55, zorder=3)
    ax.errorbar(x, yhat,
                yerr=[yhat - lower, upper - yhat],
                fmt="none", color=TEXT, linewidth=1.4, capsize=5, alpha=0.6, zorder=4)

    for bar, val in zip(bars, yhat):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (yhat.max() * 0.02),
                f"{val:,.0f}", ha="center", va="bottom",
                color=TEXT, fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(months, color=MUTED, fontsize=9)
    ax.set_title("Monthly Forecast — Upcoming Months",
                color=TEXT, fontsize=12, pad=12, fontweight="bold")
    ax.set_ylabel("Quantity (kg)", color=MUTED, fontsize=10)

    fig.tight_layout(pad=2)
    return fig


def seasonality_chart(forecast: pd.DataFrame) -> plt.Figure:
    """Yearly seasonality component extracted from the Prophet forecast."""
    fig, ax = _base_fig(figsize=(11, 3.5))

    fc = forecast.copy()
    fc["month_num"] = fc["ds"].dt.month
    seasonal = fc.groupby("month_num")["yearly"].mean().reset_index()
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

    colors = [ACCENT_3 if v >= 0 else ACCENT_RED for v in seasonal["yearly"]]
    ax.bar(seasonal["month_num"], seasonal["yearly"],
            color=colors, alpha=0.8, width=0.6, zorder=3)
    ax.axhline(0, color=MUTED, linewidth=0.8)

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels, color=MUTED, fontsize=9)
    ax.set_title("Yearly Seasonality Pattern",
                color=TEXT, fontsize=12, pad=12, fontweight="bold")
    ax.set_ylabel("Seasonal effect", color=MUTED, fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.1f}"))

    fig.tight_layout(pad=2)
    return fig
