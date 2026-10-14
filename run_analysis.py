

from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
K_VALUES = [10, 20, 50, 100, 200]
FINAL_K = 50
TOP_N = 10

def hit_rate_at_k(factor_model, A, test_df, popularity, k=10):
    """
    check whether any held-out product is in the top-k recommendations. Exclude products existing.
    """
    product_vectors = normalize(np.asarray(factor_model, dtype=np.float32))
    test_targets = test_df.groupby("customer_idx")["product_idx"].apply(set).to_dict()
    if not test_targets:
        return np.nan, 0

    hits = 0
    evaluated = 0
    for customer_idx, targets in test_targets.items():
        if customer_idx < 0 or customer_idx >= A.shape[0]:
            continue
        seen = A.getrow(customer_idx).indices
        # Recommend from the customer's purchased-product profile in latent space.
        if len(seen) == 0:
            continue
        customer_profile = np.asarray(A.getrow(customer_idx) @ product_vectors).ravel()
        scores = product_vectors @ customer_profile
        scores[seen] = -np.inf
        top = np.argpartition(-scores, min(k, len(scores)-1))[:k]
        top = top[np.argsort(-scores[top])]
        evaluated += 1
        if targets.intersection(set(top.tolist())):
            hits += 1
    return (hits / evaluated if evaluated else np.nan), evaluated

def baseline_hit_rate(A, test_df, popularity, k=10):
    test_targets = test_df.groupby("customer_idx")["product_idx"].apply(set).to_dict()
    if not test_targets:
        return np.nan, 0
    popularity = np.asarray(popularity).copy()
    hits = 0
    evaluated = 0
    for customer_idx, targets in test_targets.items():
        seen = A.getrow(customer_idx).indices
        scores = popularity.copy()
        scores[seen] = -np.inf
        top = np.argpartition(-scores, min(k, len(scores)-1))[:k]
        top = top[np.argsort(-scores[top])]
        evaluated += 1
        if targets.intersection(set(top.tolist())):
            hits += 1
    return (hits / evaluated if evaluated else np.nan), evaluated

def reconstruction_relative_error(A, model, method):
    # For manageable project data, evaluate using a dense reconstruction.
    # If memory is limited, use a sampled subset of rows instead.
    dense = A.toarray()
    if method == "svd":
        reconstructed = model.inverse_transform(model.transform(A))
    else:
        reconstructed = model.inverse_transform(model.transform(dense))
    numerator = np.linalg.norm(dense - reconstructed, ord="fro")
    denominator = np.linalg.norm(dense, ord="fro")
    return float(numerator / denominator) if denominator else np.nan

def main():
    matrix_path = DATA / "train_matrix.npz"
    if not matrix_path.exists():
        raise FileNotFoundError("Run `python build_matrix.py` first.")
    A = sparse.load_npz(matrix_path).tocsr().astype(np.float32)
    test_df = pd.read_csv(DATA / "test_purchases.csv")
    pop_df = pd.read_csv(DATA / "product_popularity_internal.csv")
    popularity = pop_df["training_quantity"].to_numpy(dtype=np.float64)
    product_map = pd.read_csv(DATA / "product_map_internal.csv")
    customer_map = pd.read_csv(DATA / "customer_map_internal.csv")

    if A.shape[1] < max(K_VALUES) + 1:
        ks = [k for k in K_VALUES if k < min(A.shape)]
        if not ks:
            raise ValueError(f"Matrix too small for SVD/PCA: {A.shape}")
    else:
        ks = [k for k in K_VALUES if k < min(A.shape)]
    if FINAL_K not in ks:
        final_k = max(ks)
        warnings.warn(f"FINAL_K={FINAL_K} unavailable; using k={final_k}")
    else:
        final_k = FINAL_K

    results = []
    fitted = {}
    print(f"Training matrix: {A.shape}, nnz={A.nnz}")
    for k in ks:
        # SVD naturally supports sparse matrices.
        start = time.perf_counter()
        svd = TruncatedSVD(n_components=k, random_state=RANDOM_STATE, n_iter=7)
        svd.fit(A)
        svd_fit_time = time.perf_counter() - start
        svd_product_factors = svd.components_.T
        svd_customer_factors = svd.transform(A)
        svd_factor_mb = (svd_product_factors.nbytes + svd_customer_factors.nbytes) / (1024**2)
        svd_error = reconstruction_relative_error(A, svd, "svd")
        svd_hit, n_eval = hit_rate_at_k(svd_product_factors, A, test_df, popularity, TOP_N)
        base_hit, _ = baseline_hit_rate(A, test_df, popularity, TOP_N)
        results.append({
            "method": "TruncatedSVD", "k": k, "fit_seconds": svd_fit_time,
            "stored_factors_MB": svd_factor_mb,
            "variance_explained_sum": float(np.sum(svd.explained_variance_ratio_)),
            "relative_reconstruction_error": svd_error,
            "hit_rate_at_10": svd_hit, "baseline_hit_rate_at_10": base_hit,
            "evaluated_customers": n_eval
        })
        if k == final_k:
            fitted["svd"] = svd
            fitted["svd_products"] = svd_product_factors
            fitted["svd_customers"] = svd_customer_factors
        print(f"SVD k={k}: fit={svd_fit_time:.2f}s, hit@10={svd_hit:.3f}, error={svd_error:.3f}")

    # PCA's centering can be memory-intensive. Fit it on dense data as the PCA-vs-SVD comparison on the same matrix.
    dense = A.toarray()
    for k in ks:
        start = time.perf_counter()
        pca = PCA(n_components=k, svd_solver="randomized", random_state=RANDOM_STATE)
        pca.fit(dense)
        pca_fit_time = time.perf_counter() - start
        pca_product_factors = pca.components_.T
        pca_customer_factors = pca.transform(dense)
        factor_mb = (pca_product_factors.nbytes + pca_customer_factors.nbytes) / (1024**2)
        pca_error = reconstruction_relative_error(A, pca, "pca")
        pca_hit, n_eval = hit_rate_at_k(pca_product_factors, A, test_df, popularity, TOP_N)
        base_hit, _ = baseline_hit_rate(A, test_df, popularity, TOP_N)
        results.append({
            "method": "PCA", "k": k, "fit_seconds": pca_fit_time,
            "stored_factors_MB": factor_mb,
            "variance_explained_sum": float(np.sum(pca.explained_variance_ratio_)),
            "relative_reconstruction_error": pca_error,
            "hit_rate_at_10": pca_hit, "baseline_hit_rate_at_10": base_hit,
            "evaluated_customers": n_eval
        })
        if k == final_k:
            fitted["pca"] = pca
            fitted["pca_products"] = pca_product_factors
            fitted["pca_customers"] = pca_customer_factors
        print(f"PCA k={k}: fit={pca_fit_time:.2f}s, hit@10={pca_hit:.3f}, error={pca_error:.3f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT / "pca_vs_svd_comparison.csv", index=False)

    # Plot hit rate and reconstruction error vs number of components.
    for metric, ylabel, filename in [
        ("hit_rate_at_10", "Hit rate at 10", "hit_rate_vs_k.png"),
        ("relative_reconstruction_error", "Relative reconstruction error", "reconstruction_error_vs_k.png"),
        ("fit_seconds", "Fit time (seconds)", "fit_time_vs_k.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for method, group in results_df.groupby("method"):
            group = group.sort_values("k")
            ax.plot(group["k"], group[metric], marker="o", label=method)
        ax.set_xlabel("Number of components (k)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs. k")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / filename, dpi=160)
        plt.close(fig)

    # Customer map: project SVD customer factors to 2D, then cluster 3-6 groups.
    # Prefer k=50 model if available; otherwise use the final valid k.
    customer_factors = fitted.get("svd_customers")
    if customer_factors is None:
        model = TruncatedSVD(n_components=final_k, random_state=RANDOM_STATE)
        customer_factors = model.fit_transform(A)
        product_factors = model.components_.T
    else:
        product_factors = fitted["svd_products"]

    map_pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = map_pca.fit_transform(customer_factors)
    n_clusters = min(4, max(2, len(customer_map) - 1))
    clusterer = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    labels = clusterer.fit_predict(customer_factors)

    # Use top products and countries to help describe clusters.

    cluster_summary = []
    # product scores by cluster: average purchase vector within each cluster
    for cluster_id in sorted(np.unique(labels)):
        members = np.where(labels == cluster_id)[0]
        mean_profile = np.asarray(A[members].mean(axis=0)).ravel()
        top_idx = np.argsort(-mean_profile)[:5]
        top_codes = product_map.set_index("product_idx").loc[top_idx, "StockCode"].astype(str).tolist()
        cluster_summary.append({
            "segment": f"Segment {cluster_id + 1}",
            "customer_count": int(len(members)),
            "top_product_codes_internal": ", ".join(top_codes)
        })

    # Plot without customer IDs. Use anonymous point indices only.
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, s=12, alpha=0.7)
    ax.set_title("Customer segments (anonymous points)")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "customer_segments.png", dpi=160)
    plt.close(fig)

    pd.DataFrame({
        "anonymous_customer_index": np.arange(len(labels)),
        "component_1": coords[:, 0],
        "component_2": coords[:, 1],
        "segment": [f"Segment {int(x)+1}" for x in labels]
    }).to_csv(OUT / "customer_segment_assignments_anonymous.csv", index=False)
    pd.DataFrame(cluster_summary).to_csv(OUT / "customer_segment_summary_internal.csv", index=False)

    # Save concise machine-readable run info.
    run_info = {
        "matrix_shape": list(A.shape),
        "matrix_nnz": int(A.nnz),
        "final_k": int(final_k),
        "random_state": RANDOM_STATE,
        "test_customers_with_eligible_purchases": int(test_df["customer_idx"].nunique()),
        "outputs": [
            "pca_vs_svd_comparison.csv",
            "hit_rate_vs_k.png",
            "reconstruction_error_vs_k.png",
            "fit_time_vs_k.png",
            "customer_segments.png",
            "customer_segment_assignments_anonymous.csv",
            "customer_segment_summary_internal.csv"
        ],
        "important_caveat": (
            "Hit rate is calculated only for test products present in the training catalogue. "
            "Review this evaluation definition in your report."
        )
    }
    (OUT / "run_summary.json").write_text(json.dumps(run_info, indent=2))
    print("\nFinished. See the outputs/ folder.")
    print(results_df[["method", "k", "fit_seconds", "hit_rate_at_10",
                      "baseline_hit_rate_at_10", "relative_reconstruction_error"]].to_string(index=False))

if __name__ == "__main__":
    main()
