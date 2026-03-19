# 📈 Demand Forecasting App

A **Streamlit** web application that uses **Meta's Prophet** to predict future product sales. Ships with a built-in retail dataset and supports flexible upload of custom CSVs with automatic column mapping.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Prophet](https://img.shields.io/badge/Prophet-1.1.5+-0866FF?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## ✨ Features

- **Built-in dataset** — load the included sales CSV instantly, no upload required
- **Custom CSV upload** — flexible column mapping, any column names accepted
- **Category filter** — sidebar dropdown to narrow products by category
- **KPI cards** — total sales, monthly average, trend % vs last 6 months, peak month
- **3-tab charts** — forecast timeline, monthly bar breakdown, yearly seasonality
- **Configurable model** — adjust forecast horizon, trend flexibility, and confidence interval
- **Dark theme** — full dark UI
- **CSV export** — download the forecast table with one click

---

## 🗂️ Project Structure

```
sales_forecast_app/
├── app.py                  # Main Streamlit entry point
├── requirements.txt        # Python dependencies
├── data/
│   └── sales_data.csv      # Built-in dataset
└── modules/
    ├── __init__.py
    ├── data.py             # Data loading, cleaning & column mapping
    ├── model.py            # Prophet training & forecasting logic
    └── charts.py           # Matplotlib chart rendering
```

---

## 🚀 Setup

**1 · Clone the repository**
```bash
git clone https://github.com/your-username/demand-forecasting-app.git
cd demand-forecasting-app
```

**2 · Install dependencies**
```bash
pip install -r requirements.txt
```

**3 · Run the app**
```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

---

## 📂 Custom CSV Upload

When uploading your own file, the app presents an interactive column mapping UI — it auto-guesses the best column for each field and lets you correct any mismatches before confirming.

| Field | Required | Description |
|---|---|---|
| `item` | ✅ | Product or SKU name |
| `date` | ✅ | Transaction or aggregation date |
| `quantity` | ✅ | Units sold (any numeric unit) |
| `category` | ⬜ | Product category for sidebar filter |
| `type` | ⬜ | Sale or Return flag — rows containing `"return"` are excluded |

> **Note:** Your CSV can use any column names — the mapping UI handles the translation.

---

## 🤖 Model Configuration

All Prophet parameters are exposed as sidebar sliders.

| Parameter | Default | Description |
|---|---|---|
| Months to forecast | `6` | Number of future months to predict (1–12) |
| Trend flexibility | `0.3` | How fast the model adapts to trend changes (`changepoint_prior_scale`) |
| Confidence interval | `0.90` | Width of the forecast uncertainty band (`interval_width`) |

---

## 🧩 Modules

### `data.py` — Data Layer
Handles loading, validation, and normalisation.

- `load_builtin()` — loads the built-in CSV with `@st.cache_data`
- `load_uploaded(file)` — presents column-mapping UI and cleans the data
- `_clean(df, mapping)` — renames columns to internal standard names
- `_guess(cols, keywords)` — fuzzy-matches column names for auto pre-selection
- `aggregate_monthly(df, product)` — resamples a product's data to monthly totals

### `model.py` — Forecasting Layer
Wraps all Prophet logic and KPI computation.

- `train_and_forecast()` — trains Prophet, cached with `@st.cache_resource`
- `get_future_forecast()` — slices future rows into a clean user-facing DataFrame
- `compute_kpis()` — total sales, average, trend %, and peak month

### `charts.py` — Visualisation Layer
All Matplotlib figures, returned to the app via `st.pyplot()`.

- `forecast_chart()` — historical line + forecast ribbon with confidence band
- `bar_chart()` — monthly bar breakdown with error bars
- `seasonality_chart()` — yearly seasonality pattern from the Prophet model

---

## 📦 Dependencies

```
streamlit>=1.35.0
prophet>=1.1.5
pandas>=2.0.0
matplotlib>=3.7.0
numpy>=1.24.0
```

---

## 📄 License

MIT License — feel free to use, modify, and distribute.
