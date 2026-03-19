"""
modules/model.py
Wraps Prophet training and forecasting logic.
"""

import pandas as pd
import streamlit as st
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")


@st.cache_resource(show_spinner=False)
def train_and_forecast(
    _monthly_df: pd.DataFrame,
    periods: int = 6,
    changepoint_prior: float = 0.3,
    interval_width: float = 0.90,
) -> tuple[Prophet, pd.DataFrame]:
    """
    Train a Prophet model and return (model, forecast).
    Uses st.cache_resource — the DataFrame is passed with _ prefix to
    avoid unhashable-type errors (cache key is based on repr).
    """
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=changepoint_prior,
        interval_width=interval_width,
    )
    model.fit(_monthly_df)

    future = model.make_future_dataframe(periods=periods, freq="MS")
    forecast = model.predict(future)
    return model, forecast


def get_future_forecast(forecast: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Return only the future rows from the full forecast, clipped to 0."""
    result = forecast[forecast["ds"] > cutoff][
        ["ds", "yhat", "yhat_lower", "yhat_upper"]
    ].copy()
    result["yhat"]       = result["yhat"].clip(lower=0).round(1)
    result["yhat_lower"] = result["yhat_lower"].clip(lower=0).round(1)
    result["yhat_upper"] = result["yhat_upper"].round(1)
    result.columns       = ["Month", "Forecast (kg)", "Lower (kg)", "Upper (kg)"]
    result["Month"]      = result["Month"].dt.strftime("%b %Y")
    return result.reset_index(drop=True)


def compute_kpis(monthly: pd.DataFrame, forecast: pd.DataFrame, cutoff: pd.Timestamp) -> dict:
    """Compute summary KPIs from historical and forecast data."""
    hist = monthly.copy()
    future = forecast[forecast["ds"] > cutoff]["yhat"].clip(lower=0)

    total_hist   = hist["y"].sum()
    avg_monthly  = hist["y"].mean()
    last_6_hist  = hist.tail(6)["y"].mean()
    next_6_fore  = future.mean()
    trend_pct    = ((next_6_fore - last_6_hist) / last_6_hist * 100) if last_6_hist else 0
    peak_month   = hist.loc[hist["y"].idxmax(), "ds"].strftime("%b %Y")
    peak_value   = hist["y"].max()

    return {
        "total_hist":   total_hist,
        "avg_monthly":  avg_monthly,
        "next_6_avg":   next_6_fore,
        "trend_pct":    trend_pct,
        "peak_month":   peak_month,
        "peak_value":   peak_value,
    }
