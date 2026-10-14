# Libraries
from pathlib import Path
import json
import urllib.request

import numpy as np
import pandas as pd
from scipy import sparse


PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "raw" / "online_retail_II.xlsx"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Codes to remove because they do not represent regular products
NON_PRODUCT_CODES = {
    "POST", "M", "D", "BANK CHARGES", "CRUK", "C2",
    "DOT", "AMAZONFEE", "ADJUST", "ADJUST2", "S", "PADS"
}


def column_locate(df, candidates):
    column_map = {
        str(c).strip().lower().replace(" ", "").replace("_", ""): c
        for c in df.columns
    }

    for candidate in candidates:
        key = candidate.lower().replace(" ", "").replace("_", "")
        if key in column_map:
            return column_map[key]

    raise ValueError(
        f"Could not find any of {candidates}. "
        f"Actual columns: {list(df.columns)}"
    )


def ensure_dataset():
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    if DATA_FILE.exists():
        return

    dataset_url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/"
        "online_retail_II.xlsx"
    )
    print(f"Dataset not found at {DATA_FILE}. Downloading from UCI...")
    urllib.request.urlretrieve(dataset_url, DATA_FILE)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset could not be downloaded to {DATA_FILE}. "
            "Please download Online Retail II manually from the UCI page "
            "and place online_retail_II.xlsx inside data/raw/."
        )


def main():
    ensure_dataset()

    # Load and combine all Excel sheets
    sheets = pd.read_excel(DATA_FILE, sheet_name=None)
    df = pd.concat(sheets.values(), ignore_index=True)

    # Locate columns
    invoice_col = column_locate(df, ["Invoice", "InvoiceNo"])
    product_col = column_locate(df, ["StockCode"])
    description_col = column_locate(df, ["Description"])
    quantity_col = column_locate(df, ["Quantity"])
    date_col = column_locate(df, ["InvoiceDate"])
    price_col = column_locate(df, ["Price", "UnitPrice"])
    customer_col = column_locate(df, ["Customer ID", "CustomerID"])
    country_col = column_locate(df, ["Country"])

    # Standardize column names
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

    # Normalize data types and values
    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce"
    )
    df["StockCode"] = (
        df["StockCode"].astype("string").str.strip().str.upper()
    )
    df["Invoice"] = (
        df["Invoice"].astype("string").str.strip().str.upper()
    )
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    # Profile raw data before cleaning
    profile = {
        "raw_rows": int(len(df)),
        "raw_columns": {
            str(c): {
                "dtype": str(df[c].dtype),
                "missing_count": int(df[c].isna().sum()),
                "missing_percent": round(
                    float(df[c].isna().mean() * 100), 3
                ),
            }
            for c in df.columns
        },
    }

    (OUTPUT_DIR / "data_profile.json").write_text(
        json.dumps(profile, indent=2)
    )

    # Record the number of rows removed by each cleaning rule
    cleaning_log = []

    def apply_rule(frame, rule_name, keep_mask):
        before = len(frame)
        frame = frame.loc[keep_mask].copy()

        cleaning_log.append({
            "rule": rule_name,
            "rows_before": int(before),
            "rows_removed": int(before - len(frame)),
            "rows_after": int(len(frame)),
        })

        return frame

    # Remove cancelled invoices
    df = apply_rule(
        df,
        "remove cancellation invoices",
        ~df["Invoice"].fillna("").str.startswith("C"),
    )

    df = apply_rule(
        df,
        "remove missing invoice date",
        df["InvoiceDate"].notna(),
    )

    df = apply_rule(
        df,
        "remove non-positive quantity",
        df["Quantity"].notna() & (df["Quantity"] > 0),
    )

    df = apply_rule(
        df,
        "remove non-positive price",
        df["Price"].notna() & (df["Price"] > 0),
    )

    df = apply_rule(
        df,
        "remove missing customer ID",
        df["CustomerID"].notna(),
    )

    df = apply_rule(
        df,
        "remove missing product code",
        df["StockCode"].notna() & (df["StockCode"] != ""),
    )

    df = apply_rule(
        df,
        "remove non-product codes",
        ~df["StockCode"].isin(NON_PRODUCT_CODES),
    )

    # Normalize customer IDs such as 12345.0 to 12345
    # so the same customer is not represented by different strings.
    df["CustomerID"] = (
        df["CustomerID"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    df["StockCode"] = df["StockCode"].astype(str).str.strip()

    # Remove any customer IDs that became empty or missing
    df = apply_rule(
        df,
        "remove invalid customer ID after normalization",
        df["CustomerID"].notna() & (df["CustomerID"] != ""),
    )

    if df.empty:
        raise ValueError(
            "No rows remain after cleaning. Review the cleaning rules."
        )

    # Use the final two months as test data
    max_date = df["InvoiceDate"].max()
    cutoff = max_date - pd.DateOffset(months=2)

    train = df[df["InvoiceDate"] < cutoff].copy()
    test = df[df["InvoiceDate"] >= cutoff].copy()

    if train.empty or test.empty:
        raise ValueError(
            "Time split produced an empty train or test set."
        )

    # Build customer and product mappings from training data only
    customers = sorted(train["CustomerID"].unique().tolist())
    products = sorted(train["StockCode"].unique().tolist())

    customer_to_idx = {
        customer: idx for idx, customer in enumerate(customers)
    }
    product_to_idx = {
        product: idx for idx, product in enumerate(products)
    }

    # Retain only customers and products represented in training
    train = train[
        train["CustomerID"].isin(customer_to_idx)
        & train["StockCode"].isin(product_to_idx)
    ].copy()

    test = test[
        test["CustomerID"].isin(customer_to_idx)
        & test["StockCode"].isin(product_to_idx)
    ].copy()

    # Build customer-product training matrix
    agg = train.groupby(
        ["CustomerID", "StockCode"], as_index=False
    )["Quantity"].sum()

    agg["value"] = np.log1p(agg["Quantity"].clip(lower=0))

    rows = agg["CustomerID"].map(customer_to_idx).to_numpy()
    cols = agg["StockCode"].map(product_to_idx).to_numpy()
    vals = agg["value"].to_numpy(dtype=np.float32)

    A = sparse.csr_matrix(
        (vals, (rows, cols)),
        shape=(len(customers), len(products)),
        dtype=np.float32,
    )
    A.sum_duplicates()

    # ---------------------------------------------------------
    # IMPORTANT FIX:
    # Test targets must be products the customer did not buy
    # during training.
    # ---------------------------------------------------------

    test_agg = test.groupby(
        ["CustomerID", "StockCode"], as_index=False
    )["Quantity"].sum()

    # Keep positive aggregated test purchases
    test_agg = test_agg[test_agg["Quantity"] > 0].copy()

    # Identify customer-product pairs already seen during training
    seen_pairs = train[
        ["CustomerID", "StockCode"]
    ].drop_duplicates()

    # Remove seen pairs from test targets
    test_agg = test_agg.merge(
        seen_pairs.assign(seen_in_train=True),
        on=["CustomerID", "StockCode"],
        how="left",
    )

    test_agg = test_agg[
        test_agg["seen_in_train"].isna()
    ].drop(columns=["seen_in_train"])

    # Map remaining unseen test targets to training indices
    test_agg["customer_idx"] = test_agg["CustomerID"].map(
        customer_to_idx
    )
    test_agg["product_idx"] = test_agg["StockCode"].map(
        product_to_idx
    )

    test_agg = test_agg.dropna(
        subset=["customer_idx", "product_idx"]
    ).copy()

    test_agg["customer_idx"] = (
        test_agg["customer_idx"].astype(int)
    )
    test_agg["product_idx"] = (
        test_agg["product_idx"].astype(int)
    )

    # Save the internal customer-to-index mapping
    pd.DataFrame({
        "customer_idx": range(len(customers)),
        "CustomerID": customers,
    }).to_csv(
        OUTPUT_DIR / "customer_map_internal.csv",
        index=False,
    )

    # Save the internal product-to-index mapping
    pd.DataFrame({
        "product_idx": range(len(products)),
        "StockCode": products,
    }).to_csv(
        OUTPUT_DIR / "product_map_internal.csv",
        index=False,
    )

    # Save each customer's most common training country
    customer_country = (
        train.groupby("CustomerID")["Country"]
        .agg(
            lambda s: (
                s.mode().iloc[0]
                if not s.mode().empty
                else "Unknown"
            )
        )
        .reset_index()
    )

    customer_country["customer_idx"] = (
        customer_country["CustomerID"].map(customer_to_idx)
    )

    customer_country[
        ["customer_idx", "Country"]
    ].to_csv(
        OUTPUT_DIR / "customer_country_internal.csv",
        index=False,
    )

    # Save sparse matrix and test targets
    sparse.save_npz(OUTPUT_DIR / "train_matrix.npz", A)

    test_agg[
        ["customer_idx", "product_idx", "Quantity"]
    ].to_csv(
        OUTPUT_DIR / "test_purchases.csv",
        index=False,
    )

    # Save training product popularity
    popularity = (
        train.groupby("StockCode")["Quantity"]
        .sum()
        .reindex(products, fill_value=0)
        .rename("training_quantity")
        .reset_index()
    )

    popularity.to_csv(
        OUTPUT_DIR / "product_popularity_internal.csv",
        index=False,
    )

    # Save cleaning log
    pd.DataFrame(cleaning_log).to_csv(
        OUTPUT_DIR / "cleaning_log.csv",
        index=False,
    )

    # Calculate matrix size and sparsity
    total_cells = A.shape[0] * A.shape[1]
    dense_bytes = (
        total_cells * np.dtype(np.float32).itemsize
    )
    sparse_bytes = (
        A.data.nbytes + A.indices.nbytes + A.indptr.nbytes
    )

    matrix_report = {
        "raw_rows": int(profile["raw_rows"]),
        "clean_rows": int(len(df)),
        "train_rows": int(len(train)),
        "test_rows_after_catalog_filter": int(len(test)),
        "unseen_test_customer_product_pairs": int(len(test_agg)),
        "time_cutoff": str(cutoff),
        "max_date": str(max_date),
        "train_matrix_shape": [
            int(A.shape[0]),
            int(A.shape[1]),
        ],
        "nonzero_cells": int(A.nnz),
        "sparsity_percent": (
            round((1 - A.nnz / total_cells) * 100, 4)
            if total_cells
            else None
        ),
        "dense_float32_MB": round(
            dense_bytes / (1024 ** 2), 3
        ),
        "sparse_storage_MB": round(
            sparse_bytes / (1024 ** 2), 3
        ),
        "purchase_value": "log1p(aggregated positive quantity)",
        "test_target_rule": (
            "Test targets include only positive aggregated purchases "
            "of known products by known customers that were not "
            "purchased by the same customer during training."
        ),
        "notes": [
            "Customer and product index mappings use training data only.",
            "Products absent from the training catalogue are excluded.",
            "Customer-product pairs already observed in training are "
            "excluded from test targets.",
            "Customer IDs and internal mappings must not be included "
            "in public slides, figures, or the public memo.",
            "Customer-country metadata is for internal segment analysis."
        ],
    }

    (OUTPUT_DIR / "matrix_report.json").write_text(
        json.dumps(matrix_report, indent=2)
    )

    print(json.dumps(matrix_report, indent=2))
    print(f"\nCleaning log: {OUTPUT_DIR / 'cleaning_log.csv'}")
    print(f"Matrix saved: {OUTPUT_DIR / 'train_matrix.npz'}")
    print(f"Test targets saved: {OUTPUT_DIR / 'test_purchases.csv'}")
    print(
        "Customer country metadata saved: "
        f"{OUTPUT_DIR / 'customer_country_internal.csv'}"
    )


if __name__ == "__main__":
    main()
