# Customers Like You — PCA or SVD

This is a starter implementation for the Mini-Project brief. It loads and cleans UCI Online Retail II, builds a sparse customer-product matrix, compares Truncated SVD and PCA, evaluates top-10 recommendations against a best-seller baseline, and creates an anonymous customer segment plot.

## 1. Set up

Use Python 3.10–3.12 if possible. From this folder:

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Download the data

Download Online Retail II from:
https://archive.ics.uci.edu/dataset/502/online+retail+ii

Put `online_retail_II.xlsx` in `data/raw/online_retail_II.xlsx`.

## 3. Run

```bash
python build_matrix.py
python run_analysis.py
```

The first command creates `data/processed/`. The second creates CSV tables and plots in `outputs/`.

## Important notes

- The raw Excel file is not included. Download it from UCI.
- This is a starter implementation, not a guarantee of full marks. Review every cleaning rule and metric with your project partner/instructor.
- PCA requires a dense copy of the matrix here and may use substantial memory. If your computer runs out of memory, first run the SVD analysis only or reduce the product catalogue; document any change.
- The model can recommend only products present in the training catalogue. Test purchases of entirely new products are excluded from the hit-rate calculation.
- Customer-ID mapping files are internal working files. Never publish them or put IDs in the memo, slides, or figures.
- Product-code summaries are internal until you join them to product descriptions. Use anonymized/aggregate information in public figures.
- Review the evaluation design: test purchases are held out by date, but the matrix is aggregated per customer-product. Confirm the implementation with your instructor before submitting.
- Keep a contribution log and make commits under each team member's own account.
