"""
modules/data.py
Handles data loading, validation, and column mapping.
"""

import pandas as pd
import streamlit as st
from pathlib import Path

# Default column names used in the built-in dataset
DEFAULT_COLUMNS = {
    "item":     "Item Name",
    "category": "Category Name",
    "date":     "Date",
    "quantity": "Quantity Sold (kilo)",
    "type":     "Sale or Return",
}

BUILTIN_PATH = Path(__file__).parent.parent / "data" / "sales_data.csv"


@st.cache_data(show_spinner=False)
def load_builtin() -> pd.DataFrame:
    """Load and return the built-in dataset."""
    df = pd.read_csv(BUILTIN_PATH)
    return _clean(df, DEFAULT_COLUMNS) # type: ignore


def load_uploaded(file) -> tuple[pd.DataFrame | None, dict | None]: # type: ignore
    """
    Load user-uploaded CSV and return (df, mapping) or (None, None) on error.
    Column mapping is collected interactively via Streamlit widgets.
    """
    try:
        raw = pd.read_csv(file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return None, None
    
    st.markdown("#### 🗂️ Map your columns")
    st.caption("Match each required field to a column in your file.")

    mapping = _collect_column_mapping(raw)
    if mapping is None:
        return None, None

    df = _clean(raw, mapping)
    if df is None:
        return None, None

    st.success(f"Loaded {len(df):,} rows successfully.")
    return df, mapping


def _collect_column_mapping(raw: pd.DataFrame) -> dict | None:
    """
    Render Streamlit widgets for mapping CSV columns -> internal schema.
    Returns mapping only after the user confirms; otherwise returns None.
    """
    cols = raw.columns.tolist()

    def pick(label: str, default_pick: str | None) -> str:
        idx = 0
        if default_pick and default_pick in cols:
            idx = cols.index(default_pick)
        return st.selectbox(label, cols, index=idx, key=f"map_{label}")

    required_fields = [
        ("item", "Product name column", _guess(cols, ["item", "product", "name"])),
        ("date", "Date column", _guess(cols, ["date", "time", "day"])),
        ("quantity", "Quantity column", _guess(cols, ["qty", "quantity", "sold", "kilo"])),
    ]
    optional_fields = [
        ("category", "Category column (optional)", _guess(cols, ["category", "cat", "group"])),
        ("type", "Sale/Return column (optional)", _guess(cols, ["type", "sale", "return"])),
    ]

    col1, col2 = st.columns(2)
    with col1:
        selected = {key: pick(label, default_pick) for key, label, default_pick in required_fields}
    with col2:
        selected.update({key: pick(label, default_pick) for key, label, default_pick in optional_fields})

    if st.button("✅ Confirm column mapping", use_container_width=True):
        return selected
    return None


def _clean(df: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame | None:
    """Clean the dataset by mapping column names and removing duplicates."""
    rename = {v: k for k, v in columns.items()}
    df = df.rename(columns=rename)

    if "item" not in df.columns or "date" not in df.columns or "quantity" not in df.columns:
        st.error("Missing required columns after mapping: item, date, quantity.")
        return None
    
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
    df = df.dropna(subset=["date", "item"])

    # Filter out returns if the column exists
    if "type" in df.columns:
        df = df[df["type"].astype(str).str.lower().str.contains("sale", na=True)]

    if "category" not in df.columns:
        df["category"] = "Unknown"

    return df.reset_index(drop=True)


def _guess(cols: list[str], keywords: list[str]) -> str | None:
    """Fuzzy-guess a column name from a list of keywords."""
    for kw in keywords:
        for col in cols:
            if kw.lower() in col.lower():
                return col
    return None

def get_categories(df: pd.DataFrame) -> list[str]:
    return sorted(df["category"].dropna().unique().tolist())


def get_top_products(df: pd.DataFrame, n: int = 20) -> list[str]:
    return (
        df.groupby("item")["quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .index.tolist()
    )


def aggregate_monthly(df: pd.DataFrame, product: str) -> pd.DataFrame:
    prod = df[df["item"] == product].copy()
    prod["ds"] = prod["date"].dt.to_period("M").dt.to_timestamp()
    monthly = prod.groupby("ds")["quantity"].sum().reset_index()
    monthly.columns = ["ds", "y"]
    return monthly.sort_values("ds").reset_index(drop=True)