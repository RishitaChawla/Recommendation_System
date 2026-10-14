# Customers Like You — PCA vs. TruncatedSVD for Retail Recommendations

**Mini-project | Data Engineering + Machine Learning**

A reproducible recommendation-system experiment using the **UCI Online Retail II** dataset. The project compares Principal Component Analysis (PCA) and Truncated Singular Value Decomposition (TruncatedSVD) for compressing customer purchase histories, recommending products, and identifying customer segments.

**Repository:** https://github.com/RishitaChawla/Recommendation_System

## Project goals and success criteria

The fictional retailer **GiftLane** wants to improve product recommendations and help its marketing team understand customer groups. The project aims to:

1. Build a clean, sparse customer × product purchase matrix without test-data leakage.
2. Reduce storage and computational cost using latent factors.
3. Achieve a higher **Hit Rate@10** than recommending the ten most popular products.
4. Compare PCA and TruncatedSVD using accuracy, reconstruction quality, runtime, and memory.
5. Visualize customers in two dimensions and describe meaningful marketing segments.
6. Keep customer identifiers out of public reports, plots, and repository artifacts.

## Dataset

**Source:** [Online Retail II — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii)  
**Citation:** Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D  
**License:** CC BY 4.0

The workbook contains **1,067,371 transaction lines** across two worksheets (`2009-2010` and `2010-2011`), covering December 2009 through December 2011.

**The dataset is intentionally excluded from Git using `.gitignore`.** Download the original workbook yourself and save it at:

```text
data/raw/online_retail_II.xlsx
```

Do not commit raw transactions or files containing customer identifiers.

## Repository layout

```text
Recommendation_System/
├── build_matrix.py           # Clean transactions, split by time, build sparse matrix
├── run_analysis.py           # Compare PCA/SVD, evaluate, cluster, create figures
├── requirements.txt          # Python dependencies
├── .gitignore
├── README.md
├── data/
│   ├── raw/                  # Local dataset; ignored by Git
│   │   └── online_retail_II.xlsx
│   └── processed/            # Generated intermediate data; keep private
└── outputs/                  # Generated analysis files; not committed in this repo
```

The `data/processed/` and `outputs/` directories are created by the scripts when they run. They do **not** need to be present in the GitHub repository.

**Submission note:** The figures and result files were provided separately by direct upload rather than committed to Git. Therefore, they may not appear in the repository's file listing. The commands below reproduce the analysis locally. The stakeholder memo and presentation are submitted separately.

## How to run

### 1. Clone the repository

```bash
git clone https://github.com/RishitaChawla/Recommendation_System.git
cd Recommendation_System
```

### 2. Set up Python and install dependencies

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Activate the environment:

- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
- **macOS/Linux:** `source .venv/bin/activate`

Then install:

```bash
python -m pip install -r requirements.txt
```

The analysis uses NumPy, pandas, SciPy, scikit-learn, Matplotlib, openpyxl, and psutil.

### 3. Download the data

Download the **Online Retail II** workbook from the UCI link above, extract it if needed, and place it at:

```text
data/raw/online_retail_II.xlsx
```

Make sure the workbook contains both yearly sheets.

### 4. Build the training matrix

```bash
python build_matrix.py
```

This command loads both sheets, profiles missing values, removes invalid transactions, creates a chronological split, and saves a sparse training matrix and private intermediate files in `data/processed/`.

### 5. Run the full analysis

```bash
python run_analysis.py
```

This command fits PCA and TruncatedSVD at **k = 10, 20, 50, 100, and 200**, evaluates held-out recommendations, compares efficiency and reconstruction error, and produces customer segmentation plots and summary tables in `outputs/`.

Run the two commands **in order**, from the repository root. No pre-existing `outputs/` directory is required.

## Data engineering and leakage prevention

The pipeline:

- Combines both workbook sheets and normalizes column names.
- Removes cancelled invoices, non-positive quantities/prices, missing customer IDs, and selected non-product stock codes.
- Reserves the **last two months** for testing.
- Builds customer/product index maps and the purchase matrix using **training data only**.
- Represents each customer–product purchase as `log1p(total positive quantity)` in a SciPy sparse matrix.
- Evaluates only held-out products the customer had **not previously purchased**, among customers and products represented in training.
- Checks that train/test periods do not overlap and that held-out customer–product pairs were unseen.

Private mappings and transaction-level intermediate data are for local use only.

## Recommendation and evaluation method

**TruncatedSVD** works directly with sparse input without mean-centering. **PCA** mean-centers features and provides a comparison against a different representation of purchase behavior.

For each candidate number of components, the analysis records:

- **Hit Rate@10:** fraction of eligible test customers whose ten recommendations contain at least one held-out product.
- **Best-seller baseline:** ten popular training products, excluding products already purchased by the customer.
- **Fit time:** time to fit each model.
- **Memory:** sampled peak process resident memory during fitting.
- **Reconstruction error:** relative error estimated using a fixed sample of training rows.
- **Factor storage:** memory occupied by customer and product factor arrays.

The customer visualization uses two-dimensional latent coordinates and **four** clusters, interpreted using purchase patterns, product categories, and countries.

## Results from the validated run

The following numbers are from the final successful analysis run and are included for reference. Actual timing and memory may vary across computers.

### Data summary

| Measure | Result |
|---|---:|
| Raw transaction rows | 1,067,371 |
| Rows after cleaning | 802,644 |
| Training rows | 684,081 |
| Test-period rows | 118,563 |
| Training matrix | 5,474 customers × 4,532 products |
| Nonzero matrix entries | 417,724 |
| Sparsity | 98.3162% |
| Dense float32 matrix estimate | 94.636 MiB |
| Sparse matrix storage | 3.208 MiB |
| Eligible held-out customers | 1,916 |

### PCA vs. TruncatedSVD (k = 100)

| Measure | PCA | TruncatedSVD |
|---|---:|---:|
| Hit Rate@10 | 21.50% | **22.18%** |
| Fit time | 0.66 s | **0.53 s** |
| Sampled relative reconstruction error | 0.7424 | 0.7425 |
| Sampled peak process RSS during fit | 434.86 MiB | 216.55 MiB |

**Best-seller baseline Hit Rate@10: 17.38%.**  
**Selected model: TruncatedSVD with 100 components**, delivering approximately **+4.80 percentage points** over the baseline on this held-out evaluation.

The stored customer and product factor arrays at k = 100 occupy approximately **3.82 MiB**.

**Measurement caveats:** Reconstruction error is estimated on sampled training rows, not the entire matrix. Memory figures are sampled process-level RSS, not model-only allocations; the PCA measurement excludes its earlier dense-conversion step. Model fit time and the total experimental runtime are **not** a production nightly-job benchmark. Results can vary with package versions and hardware.

### Customer segments

| Segment | Customers | Descriptive label |
|---|---:|---|
| 1 | 334 | Frequent Multi-Product Shoppers |
| 2 | 1,050 | Regular Gift & Home Shoppers |
| 3 | 4,078 | Occasional Gift Shoppers |
| 4 | 12 | Very High-Activity Shoppers |

These labels summarize observed purchasing activity, **not** verified revenue, profit, or purchase frequency by invoice.

## Files generated locally

After running the scripts, typical files include:

**`data/processed/`** — sparse training matrix, anonymized test indices, data profile, cleaning log, matrix report, and private customer/product mappings.

**`outputs/`** — `pca_vs_svd_comparison.csv`, `run_summary.json`, `hit_rate_vs_k.png`, `fit_time_vs_k.png`, `reconstruction_error_vs_k.png`, `customer_segments.png`, `customer_segment_summary.csv`, `cluster_activity_summary.csv`, and anonymous segment coordinates.

These files are **generated locally** and are **not included in the GitHub repository** because the project results were shared through direct upload. Avoid publishing internal mapping files or any export that contains raw customer IDs.

## Why TruncatedSVD?

TruncatedSVD was chosen because it operates natively on the highly sparse purchase matrix, avoids PCA's dense centering overhead in this implementation, and achieved the highest measured Hit Rate@10 among the tested SVD configurations. The choice is based on held-out accuracy and practical efficiency, rather than reconstruction error alone.

## Limitations and next steps

- New customers and products without training history are not evaluated (cold-start problem).
- Offline Hit Rate@10 is not a guarantee of online sales lift.
- Customer segments are exploratory and should be validated before targeted campaigns.
- The comparison does not prove that the pipeline meets a production nightly service-level target.
- A production version should include incremental updates, monitoring, and an A/B test against the best-seller baseline.

## Privacy and reproducibility

The experiment uses a fixed random seed (`42`) and records the preprocessing and evaluation steps in code. Raw customer IDs and internal mappings should remain local. Public-facing visualizations and summaries use anonymous coordinates or descriptive segment names.

To reproduce the analysis, download the dataset, install `requirements.txt`, and run:

```bash
python build_matrix.py
python run_analysis.py
```

## Assignment context

This project responds to the *Customers Like You — PCA or SVD* mini-project brief for the fictional GiftLane retailer and stakeholder Sarah Okafor. The deliverables include the code, evaluation, customer map, a one-page stakeholder memo, and a five-minute presentation. Team contributions, reflections, and peer evaluations are handled separately according to the assignment instructions.
