# Customers Like You — PCA vs. TruncatedSVD

A retail recommendation project comparing **Principal Component Analysis (PCA)** and **Truncated Singular Value Decomposition (TruncatedSVD)** using the UCI Online Retail II dataset.

The goal is to see how well these methods compress customer purchase histories, recommend products customers have not bought before, and help identify useful customer segments.

**Repository:** [https://github.com/RishitaChawla/Recommendation_System](https://github.com/RishitaChawla/Recommendation_System)

## What this project does

- Builds a sparse customer–product purchase matrix from retail transactions.
- Compares PCA and TruncatedSVD using different numbers of components.
- Evaluates recommendations against a best-selling-products baseline.
- Measures recommendation accuracy, reconstruction error, runtime, and memory usage.
- Visualizes customer groups based on their purchasing patterns.

## Dataset

We use the [UCI Online Retail II dataset](https://archive.ics.uci.edu/dataset/502/online+retail+ii), which contains approximately 1.07 million transaction records from December 2009 to December 2011.

**Citation:** Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C5CG6D](https://doi.org/10.24432/C5CG6D)

The dataset is not included in this repository. Download the workbook and save it as:

```text
data/raw/online_retail_II.xlsx
```

The project script also creates the raw data folder automatically and can fetch the workbook from the UCI archive if it is missing.

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/RishitaChawla/Recommendation_System.git
cd Recommendation_System
```

### 2. Set up the environment

Python 3.10 or later is recommended.

```bash
python -m venv .venv
```

Activate the environment:

- **Windows PowerShell:** `.venv\Scripts\Activate.ps1`
- **macOS/Linux:** `source .venv/bin/activate`

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

### 3. Add the dataset

Download Online Retail II from the UCI link above and place the Excel workbook in `data/raw/`.

### 4. Run the project

Build the training matrix:

```bash
python build_matrix.py
```

Run the analysis:

```bash
python run_analysis.py
```

The scripts generate the comparison tables, plots, and customer segment summaries locally.

## How it works

**Data preparation:** Both workbook sheets are combined, and invalid transactions, cancelled invoices, and selected non-product codes are removed.

**Train/test split:** The final two months are reserved for testing. The customer and product mappings are built from training data only. Test targets include products that a customer purchased during the test period but had not purchased during training. Products absent from the training catalogue are excluded.

**Purchase matrix:** Each customer–product value is calculated using `log1p` of the aggregated positive purchase quantity. The matrix is stored as a sparse matrix to avoid wasting memory on the many zero entries.

**Model comparison:** PCA and TruncatedSVD are evaluated using `k = 10, 20, 50, 100, and 200` components.

The comparison includes:

- Hit Rate@10 and a best-seller baseline
- Fit time and sampled peak memory
- Reconstruction error
- Storage required for customer and product factors

The project also uses a two-dimensional customer projection and four clusters to explore differences in purchasing patterns.

## Results

The reported analysis run selected **TruncatedSVD with 100 components**.

| Measure | PCA | TruncatedSVD |
|---|---:|---:|
| Hit Rate@10 | 21.50% | **22.18%** |
| Fit time | 0.66 s | **0.53 s** |
| Relative reconstruction error | 0.7424 | 0.7425 |
| Sampled peak process memory during fit | 434.86 MiB | **216.55 MiB** |

The best-seller baseline achieved a Hit Rate@10 of **17.38%**. TruncatedSVD performed approximately **4.80 percentage points better** on this evaluation.

At 100 components, the stored customer and product factor arrays occupy approximately **3.82 MiB**.

The training matrix contains **5,474 customers** and **4,532 products**, with **417,724 nonzero entries**. Its estimated dense float32 representation requires about **94.64 MiB**, compared with approximately **3.21 MiB** for sparse storage.

The full data-prep run produced:

- Raw transaction rows: **1,067,371**
- Rows after cleaning: **802,644**
- Training rows: **684,081**
- Test-period rows: **118,563**
- Sparsity: **98.3162%**

*These are results from the validated project run; exact timing and memory can vary by machine and package version. Reconstruction error was estimated on sampled rows, and memory measurements are sampled process-level values rather than exact model-only allocations.*

## Why TruncatedSVD?

TruncatedSVD was the preferred method because it works directly with sparse data and achieved a slightly higher Hit Rate@10 with lower measured fit time and peak memory in this experiment.

PCA produced a similar reconstruction error, so the choice was based on recommendation performance and practical efficiency rather than reconstruction quality alone.

## Limitations and next steps

- New customers and products are not covered by this evaluation.
- Offline recommendation accuracy does not guarantee increased sales.
- Customer clusters are exploratory and need further validation before being used for marketing campaigns.
- A production system would need monitoring, incremental updates, and an A/B test against the best-seller baseline.

## Privacy and reproducibility

The experiment uses a fixed random seed of `42`. Download the dataset, install the dependencies, and run the two scripts in order to reproduce the analysis.

Raw data, customer IDs, and internal mapping files should remain private. Do not publish them in public reports, figures, or repository artifacts.

## Assignment

This project was developed for the *Customers Like You — PCA or SVD* mini-project, using the fictional retailer GiftLane as the business context. The remaining deliverables include a stakeholder memo, a five-minute presentation, and individual/team reflection materials.
