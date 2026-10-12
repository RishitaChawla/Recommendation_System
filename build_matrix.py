# Libraries
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "raw" / "online_retail_II.xlsx"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Codes to Remove from the Excel Sheet
NON_PRODUCT_CODES = {"POST", "M", "D", "BANK CHARGES", "CRUK", "C2", "DOT", "AMAZONFEE", "ADJUST", "ADJUST2", "S", "PADS"}

def column_locate(df, candidates):
    column_map = {str(c).strip().lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for candidate in candidates:
        key = candidate.lower().replace(" ", "").replace("_", "")
        if key in column_map:
            return column_map[key]
    raise ValueError(f"Could not find any of {candidates}. Actual columns: {list(df.columns)}")

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Download Online Retail II from the UCI page and place online_retail_II.xlsx "
            "inside data/raw/."
        )

    sheets = pd.read_excel(DATA_FILE, sheet_name=None)
    df = pd.concat(sheets.values(), ignore_index=True)
    invoice_col = column_locate(df, ["Invoice", "InvoiceNo"])
    product_col = column_locate(df, ["StockCode"])
    description_col = column_locate(df, ["Description"])
    quantity_col = column_locate(df, ["Quantity"])
    date_col = column_locate(df, ["InvoiceDate"])
    price_col = column_locate(df, ["Price", "UnitPrice"])
    customer_col = column_locate(df, ["Customer ID", "CustomerID"])
    country_col = column_locate(df, ["Country"])

    # Rename the column names
    df = df.rename(columns={
        invoice_col: "Invoice",
        product_col: "StockCode",
        description_col: "Description",
        quantity_col: "Quantity",
        date_col: "InvoiceDate",
        price_col: "Price",
        customer_col: "CustomerID",
        country_col: "Country",
    })
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["StockCode"] = df["StockCode"].astype("string").str.strip().str.upper()
    df["Invoice"] = df["Invoice"].astype("string").str.strip().str.upper()
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    profile = {
        "raw_rows": int(len(df)),
        "raw_columns": {str(c): {
            "dtype": str(df[c].dtype),
            "missing_count": int(df[c].isna().sum()),
            "missing_percent": round(float(df[c].isna().mean() * 100), 3)
        } for c in df.columns}
    }
    (OUTPUT_DIR / "data_profile.json").write_text(json.dumps(profile, indent=2))

    # Cleaning log records how many rows each sequential rule removes
    cleaning_log = []
    def apply_rule(frame, rule_name, keep_mask):
        before = len(frame)
        frame = frame.loc[keep_mask].copy()
        cleaning_log.append({
            "rule": rule_name,
            "rows_before": int(before),
            "rows_removed": int(before - len(frame)),
            "rows_after": int(len(frame))
        })
        return frame

    # Remove invoices that start with C because they are cancelled orders
    df = apply_rule(df, "remove cancellation invoices",
                    ~df["Invoice"].fillna("").str.startswith("C"))
    df = apply_rule(df, "remove missing invoice date",
                    df["InvoiceDate"].notna())
    df = apply_rule(df, "remove non-positive quantity",
                    df["Quantity"].notna() & (df["Quantity"] > 0))
    df = apply_rule(df, "remove non-positive price",
                    df["Price"].notna() & (df["Price"] > 0))
    df = apply_rule(df, "remove missing customer ID",
                    df["CustomerID"].notna())
    df = apply_rule(df, "remove missing product code",
                    df["StockCode"].notna() & (df["StockCode"] != ""))
    df = apply_rule(df, "remove non-product codes",
                    ~df["StockCode"].isin(NON_PRODUCT_CODES))

    # Convert customer IDs into strings so they all have the same format
    df["CustomerID"] = df["CustomerID"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["StockCode"] = df["StockCode"].astype(str)

    if df.empty:
        raise ValueError("No rows remain after cleaning. Review the cleaning rules.")

    # Use the last two months of purchases as test data
    max_date = df["InvoiceDate"].max()
    cutoff = max_date - pd.DateOffset(months=2)
    train = df[df["InvoiceDate"] < cutoff].copy()
    test = df[df["InvoiceDate"] >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("Time split produced an empty train or test set.")

    # Create customer and product mappings using only training data.
    customers = sorted(train["CustomerID"].unique().tolist())
    products = sorted(train["StockCode"].unique().tolist())
    customer_to_idx = {v: i for i, v in enumerate(customers)}
    product_to_idx = {v: i for i, v in enumerate(products)}

    train = train[train["CustomerID"].isin(customer_to_idx) &
                  train["StockCode"].isin(product_to_idx)].copy()
    test = test[test["CustomerID"].isin(customer_to_idx) &
                test["StockCode"].isin(product_to_idx)].copy()

    agg = train.groupby(["CustomerID", "StockCode"], as_index=False)["Quantity"].sum()
    agg["value"] = np.log1p(agg["Quantity"].clip(lower=0))
    rows = agg["CustomerID"].map(customer_to_idx).to_numpy()
    cols = agg["StockCode"].map(product_to_idx).to_numpy()
    vals = agg["value"].to_numpy(dtype=np.float32)
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(len(customers), len(products)), dtype=np.float32)
    A.sum_duplicates()

    test_agg = test.groupby(["CustomerID", "StockCode"], as_index=False)["Quantity"].sum()
    test_agg = test_agg[test_agg["Quantity"] > 0].copy()
    test_agg["customer_idx"] = test_agg["CustomerID"].map(customer_to_idx)
    test_agg["product_idx"] = test_agg["StockCode"].map(product_to_idx)
    test_agg = test_agg.dropna(subset=["customer_idx", "product_idx"])
    test_agg["customer_idx"] = test_agg["customer_idx"].astype(int)
    test_agg["product_idx"] = test_agg["product_idx"].astype(int)

    # Save files
    sparse.save_npz(OUTPUT_DIR / "train_matrix.npz", A)
    pd.DataFrame({"customer_idx": range(len(customers)), "CustomerID": customers}).to_csv(
        OUTPUT_DIR / "customer_map_internal.csv", index=False)
    pd.DataFrame({"product_idx": range(len(products)), "StockCode": products}).to_csv(
        OUTPUT_DIR / "product_map_internal.csv", index=False)
    test_agg[["customer_idx", "product_idx", "Quantity"]].to_csv(
        OUTPUT_DIR / "test_purchases.csv", index=False)
    train.groupby("StockCode")["Quantity"].sum().reindex(products, fill_value=0).rename(
        "training_quantity").reset_index().rename(columns={"index": "StockCode"}).to_csv(
        OUTPUT_DIR / "product_popularity_internal.csv", index=False)
    pd.DataFrame(cleaning_log).to_csv(OUTPUT_DIR / "cleaning_log.csv", index=False)

    total_cells = A.shape[0] * A.shape[1]
    dense_bytes = total_cells * np.dtype(np.float32).itemsize
    sparse_bytes = (A.data.nbytes + A.indices.nbytes + A.indptr.nbytes)
    matrix_report = {
        "raw_rows": int(profile["raw_rows"]),
        "clean_rows": int(len(df)),
        "train_rows_before_catalog_filter": int(len(train)),
        "test_rows_after_catalog_filter": int(len(test_agg)),
        "time_cutoff": str(cutoff),
        "max_date": str(max_date),
        "train_matrix_shape": [int(A.shape[0]), int(A.shape[1])],
        "nonzero_cells": int(A.nnz),
        "sparsity_percent": round((1 - A.nnz / total_cells) * 100, 4) if total_cells else None,
        "dense_float32_MB": round(dense_bytes / (1024 ** 2), 3),
        "sparse_storage_MB": round(sparse_bytes / (1024 ** 2), 3),
        "purchase_value": "log1p(aggregated positive quantity)",
        "notes": [
            "Matrix and index maps are built from training data only.",
            "Test products not present in the training catalogue are excluded because this model cannot recommend unseen products.",
            "Internal customer ID mapping must not be included in public slides or figures."
        ]
    }
    (OUTPUT_DIR / "matrix_report.json").write_text(json.dumps(matrix_report, indent=2))
    print(json.dumps(matrix_report, indent=2))
    print(f"\nCleaning log: {OUTPUT_DIR / 'cleaning_log.csv'}")
    print(f"Matrix saved: {OUTPUT_DIR / 'train_matrix.npz'}")

if __name__ == "__main__":
    main()
