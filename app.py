"""
app.py
Main Streamlit entry point for the Sales Demand Forecasting app.
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd

from module.data import (
    load_builtin,
    load_uploaded,
    get_top_products,
    get_categories,
    aggregate_monthly,
)
from module.model import train_and_forecast, get_future_forecast, compute_kpis
from module.charts import forecast_chart, bar_chart, seasonality_chart

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Demand Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme override ───────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Global */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #0d1117 !important;
    color: #e6edf3;
}
[data-testid="stSidebar"] {
    background-color: #161b22 !important;
    border-right: 1px solid #21262d;
}
  /* Cards */
div[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 16px 20px;
}
div[data-testid="metric-container"] label { color: #7d8590 !important; font-size: 12px; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e6edf3 !important; font-size: 26px; font-weight: 700; }
div[data-testid="metric-container"] [data-testid="stMetricDelta"] svg { display: none; }
  /* Tabs */
button[data-baseweb="tab"] { color: #7d8590 !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #58a6ff !important; border-bottom-color: #58a6ff !important; }
  /* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid #21262d; border-radius: 8px; }
  /* Buttons */
.stButton > button {
    background: #21262d; color: #e6edf3;
    border: 1px solid #30363d; border-radius: 8px;
}
.stButton > button:hover { background: #30363d; border-color: #58a6ff; }
  /* Download button */
.stDownloadButton > button {
    background: #1f6feb; color: white;
    border: none; border-radius: 8px; width: 100%;
}
  /* Divider */
hr { border-color: #21262d; }
  /* Selectbox, slider labels */
label { color: #c9d1d9 !important; }
.stSelectbox div[data-baseweb="select"] > div { background: #161b22 !important; border-color: #30363d !important; color: #e6edf3; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 8px 0 20px 0;">
<h1 style="color:#e6edf3; font-size:2rem; font-weight:800; margin:0;">
    📈 Demand Forecasting
</h1>
<p style="color:#7d8590; margin:4px 0 0 0; font-size:0.95rem;">
    Prophet-powered sales prediction · Built-in dataset or upload your own
</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.divider()

    # ── Data source ──
    st.markdown("### 📂 Data Source")
    data_source = st.radio(
        "Choose dataset",
        ["Built-in dataset", "Upload my own CSV"],
        label_visibility="collapsed",
    )

    df = None

    if data_source == "Built-in dataset":
        with st.spinner("Loading dataset..."):
            df = load_builtin()
        st.success(f"✅ {len(df):,} rows loaded")

    else:
        uploaded = st.file_uploader("Upload CSV", type="csv", label_visibility="collapsed")
        if uploaded:
            df, _ = load_uploaded(uploaded)

    st.divider()

    if df is not None:
        # ── Filters ──
        st.markdown("### 🔍 Filters")
        categories = ["All categories"] + get_categories(df)
        selected_cat = st.selectbox("Category", categories)

        filtered_df = df if selected_cat == "All categories" else df[df["category"] == selected_cat]

        products = get_top_products(filtered_df, n=30)
        if not products:
            st.warning("No products found for this category.")
            st.stop()

        selected_product = st.selectbox("Product", products)

        st.divider()

        # ── Model settings ──
        st.markdown("### 🤖 Model Settings")
        forecast_months = st.slider("Months to forecast", 1, 12, 6)
        changepoint     = st.slider("Trend flexibility", 0.05, 0.8, 0.3,
                                    help="Higher = model adapts faster to trend changes")
        confidence      = st.slider("Confidence interval", 0.7, 0.99, 0.90,
                                    help="Width of the prediction band")

        st.divider()
        run = st.button("🔮 Run Forecast", use_container_width=True)

# ── Main area ─────────────────────────────────────────────────────────────────
if df is None:
    st.info("👈 Select a data source in the sidebar to get started.")
    st.stop()

if not run: # type: ignore
    # Show a preview of the data while waiting
    st.markdown("### 📋 Dataset Preview")
    preview = df.rename(columns={
        "item": "Product", "category": "Category",
        "date": "Date", "quantity": "Quantity (kg)"
    })
    st.dataframe(preview[["Product", "Category", "Date", "Quantity (kg)"]].head(100),
                use_container_width=True, hide_index=True)
    st.caption(f"Showing first 100 of {len(df):,} rows. Select a product and click **Run Forecast** to begin.")
    st.stop()

# ── Run model ─────────────────────────────────────────────────────────────────
with st.spinner(f"Training Prophet model for **{selected_product}**..."): # type: ignore
    monthly = aggregate_monthly(filtered_df, selected_product) # type: ignore

    if len(monthly) < 6:
        st.error("Not enough data to forecast (minimum 6 months required).")
        st.stop()

    model, forecast = train_and_forecast(
        monthly,
        periods=forecast_months, # type: ignore
        changepoint_prior=changepoint, # type: ignore
        interval_width=confidence, # type: ignore
    )

cutoff      = monthly["ds"].max()
result_df   = get_future_forecast(forecast, cutoff)
kpis        = compute_kpis(monthly, forecast, cutoff)

# ── KPI cards ─────────────────────────────────────────────────────────────────
st.markdown(f"### 📊 {selected_product}") # type: ignore
st.divider()

trend_sign = "+" if kpis["trend_pct"] >= 0 else ""
trend_color = "#3fb950" if kpis["trend_pct"] >= 0 else "#f85149"

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Sales (historical)",  f"{kpis['total_hist']:,.0f} kg")
c2.metric("Avg Monthly Sales",         f"{kpis['avg_monthly']:,.0f} kg")
c3.metric("Avg Forecast (next months)",f"{kpis['next_6_avg']:,.0f} kg")
c4.metric("Trend vs Last 6 Months",
        f"{trend_sign}{kpis['trend_pct']:.1f}%",
        delta=f"{trend_sign}{kpis['trend_pct']:.1f}%")
c5.metric("Peak Month", kpis["peak_month"],
        delta=f"{kpis['peak_value']:,.0f} kg")

st.divider()

# ── Charts via tabs ───────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 Forecast Timeline", "📊 Monthly Breakdown", "🌀 Seasonality"])

with tab1:
    fig = forecast_chart(monthly, forecast, cutoff, selected_product) # type: ignore
    st.pyplot(fig, use_container_width=True)

with tab2:
    fig2 = bar_chart(result_df)
    st.pyplot(fig2, use_container_width=True)

with tab3:
    if "yearly" in forecast.columns:
        fig3 = seasonality_chart(forecast)
        st.pyplot(fig3, use_container_width=True)
    else:
        st.info("Yearly seasonality component not available for this dataset.")

st.divider()

# ── Forecast table ────────────────────────────────────────────────────────────
st.markdown("### 📋 Forecast Table")
st.dataframe(result_df, use_container_width=True, hide_index=True)

csv_bytes = result_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download forecast as CSV",
    data=csv_bytes,
    file_name=f"forecast_{selected_product.replace(' ', '_')}.csv", # type: ignore
    mime="text/csv", 
)
