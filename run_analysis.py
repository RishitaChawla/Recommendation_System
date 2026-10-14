
"""SVD/PCA experiments, unseen-purchase recommendations, and customer segments."""
from pathlib import Path
import json
import time
import os
import threading
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data/processed'
OUT=ROOT/'outputs'
OUT.mkdir(parents=True,exist_ok=True)
SEED=42
K_VALUES=[10,20,50,100,200]
FINAL_K=None  # Automatically select k with the highest corrected SVD Hit Rate@10
TOP_N=10
SAMPLE_ROWS=500

def measure_fit_memory(fit_function):
    """Sample peak process RSS during fit; includes all resident process memory."""
    process = psutil.Process(os.getpid())
    initial = process.memory_info().rss
    samples = [initial]
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            samples.append(process.memory_info().rss)
            stop.wait(0.005)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    start = time.perf_counter()
    try:
        result = fit_function()
    finally:
        elapsed = time.perf_counter() - start
        stop.set()
        thread.join()
        samples.append(process.memory_info().rss)
    peak = max(samples)
    return result, elapsed, peak / 2**20, (peak-initial) / 2**20


def cluster_activity(A, labels):
    report = []
    for cid in sorted(np.unique(labels)):
        sub = A[np.flatnonzero(labels == cid)].tocsr()
        activity = np.asarray(sub.sum(axis=1)).ravel()
        distinct = np.diff(sub.indptr)
        report.append({
            'segment': f'Segment {cid+1}',
            'customer_count': int(sub.shape[0]),
            'mean_log_quantity_activity': float(activity.mean()),
            'median_log_quantity_activity': float(np.median(activity)),
            'mean_distinct_products': float(distinct.mean())
        })
    return pd.DataFrame(report)


def benchmark_recommendations(A, factors, n=10):
    """Time scoring/top-N for all training customers; excludes fit and data ETL."""
    start = time.perf_counter()
    vectors = normalize(np.asarray(factors, dtype=np.float32))
    count = 0
    for idx in range(A.shape[0]):
        row = A.getrow(idx)
        scores = np.asarray(vectors @ np.asarray(row @ vectors).ravel(), dtype=np.float64)
        scores[row.indices] = -np.inf
        top_indices(scores, n)
        count += 1
    return {'customers_scored': count,
            'recommendation_generation_seconds': time.perf_counter()-start}


def eligible_targets(test_df):
    return test_df.groupby('customer_idx')['product_idx'].apply(set).to_dict()

def top_indices(scores, n):
    n=min(n,len(scores))
    if n<=0: return np.array([],dtype=int)
    candidates=np.argpartition(-scores,n-1)[:n]
    return candidates[np.argsort(-scores[candidates])]

def hit_rate(A,test_targets,product_factors=None,popularity=None,n=10):
    """Identical eligible customer denominator for model and baseline."""
    hits=0
    evaluated=0
    vectors=normalize(np.asarray(product_factors,dtype=np.float32)) if product_factors is not None else None
    for idx,targets in test_targets.items():
        idx=int(idx)
        if not 0<=idx<A.shape[0]: continue
        row=A.getrow(idx)
        if row.nnz==0: continue
        if vectors is None:
            scores=np.asarray(popularity,dtype=np.float64).copy()
        else:
            profile=np.asarray(row@vectors).ravel()
            scores=np.asarray(vectors@profile,dtype=np.float64)
        scores[row.indices]=-np.inf
        predicted=set(top_indices(scores,n).tolist())
        hits+=bool(predicted.intersection(targets))
        evaluated+=1
    return (hits/evaluated if evaluated else np.nan),evaluated

def sampled_reconstruction_error(A,model,method,sample_idx):
    original=A[sample_idx].toarray()
    if method=='PCA':
        transformed=model.transform(original)
    else:
        transformed=model.transform(A[sample_idx])
    reconstruction=model.inverse_transform(transformed)
    denom=np.linalg.norm(original,'fro')
    return float(np.linalg.norm(original-reconstruction,'fro')/denom) if denom else np.nan

def save_product_similarities(product_factors,product_map,description_map,n=10):
    """Blockwise cosine neighbors to avoid a huge all-products similarity matrix."""
    vectors=normalize(np.asarray(product_factors,dtype=np.float32))
    codes=product_map.sort_values('product_idx')['StockCode'].astype(str).to_numpy()
    desc=description_map.sort_values('product_idx')['Description'].fillna('Unknown').astype(str).to_numpy()
    rows=[]
    n=min(n,len(codes)-1)
    if n<1:return
    for start in range(0,len(codes),128):
        stop=min(start+128,len(codes))
        sim=vectors[start:stop]@vectors.T
        sim[np.arange(stop-start),np.arange(start,stop)]=-np.inf
        for local_idx,source_idx in enumerate(range(start,stop)):
            nearest=top_indices(sim[local_idx],n)
            for rank,j in enumerate(nearest,1):
                rows.append({'product_code':codes[source_idx],'product_description':desc[source_idx],
                             'similar_product_code':codes[j],'similar_product_description':desc[j],
                             'rank':rank,'cosine_similarity':float(sim[local_idx,j])})
    pd.DataFrame(rows).to_csv(OUT/'product_similarities_internal.csv',index=False)

def main():
    started=time.perf_counter()
    A=sparse.load_npz(DATA/'train_matrix.npz').tocsr().astype(np.float32)
    test=pd.read_csv(DATA/'test_purchases.csv')
    pop=pd.read_csv(DATA/'product_popularity_internal.csv').sort_values('product_idx')['training_quantity'].to_numpy()
    products=pd.read_csv(DATA/'product_map_internal.csv',dtype={'StockCode':str})
    descriptions=pd.read_csv(DATA/'product_descriptions_internal.csv')
    countries=pd.read_csv(DATA/'customer_country_internal.csv')
    targets=eligible_targets(test)
    baseline,base_n=hit_rate(A,targets,popularity=pop,n=TOP_N)
    ks=[k for k in K_VALUES if k<min(A.shape)]
    if not ks: raise ValueError(f'Matrix too small: {A.shape}')
    rng=np.random.default_rng(SEED)
    sample_idx=np.sort(rng.choice(A.shape[0],size=min(SAMPLE_ROWS,A.shape[0]),replace=False))
    results=[]
    models={}
    dense=None
    for method in ['TruncatedSVD','PCA']:
        if method=='PCA':
            # PCA centers data. Dense allocation may require significant RAM.
            dense=A.toarray()
            print(f'PCA dense matrix: {dense.nbytes/2**20:.1f} MB')
        for k in ks:
            if method=='TruncatedSVD':
                model=TruncatedSVD(n_components=k,random_state=SEED,n_iter=7)
                customer_factors,fit_seconds,peak_mb,delta_mb=measure_fit_memory(lambda: model.fit_transform(A))
            else:
                model=PCA(n_components=k,svd_solver='randomized',random_state=SEED)
                customer_factors,fit_seconds,peak_mb,delta_mb=measure_fit_memory(lambda: model.fit_transform(dense))
            product_factors=model.components_.T
            err=sampled_reconstruction_error(A,model,'PCA' if method=='PCA' else 'SVD',sample_idx)
            t=time.perf_counter()
            hit,n_eval=hit_rate(A,targets,product_factors=product_factors,n=TOP_N)
            eval_seconds=time.perf_counter()-t
            results.append({'method':method,'k':k,'fit_seconds':fit_seconds,
                'recommendation_evaluation_seconds':eval_seconds,
                'stored_factors_MB':(customer_factors.nbytes+product_factors.nbytes)/2**20,
                'variance_explained_sum':float(model.explained_variance_ratio_.sum()),
                'sampled_relative_reconstruction_error':err,'hit_rate_at_10':hit,
                'baseline_hit_rate_at_10':baseline,'evaluated_customers':n_eval,
                'peak_fit_memory_MB':peak_mb,'peak_rss_increase_MB':delta_mb})
            models[(method,k)]=(model,customer_factors,product_factors)
            print(f'{method} k={k}: hit@10={hit:.4f}, baseline={baseline:.4f}, fit={fit_seconds:.2f}s, sampled_error={err:.4f}')
    results_df=pd.DataFrame(results)
    results_df.to_csv(OUT/'pca_vs_svd_comparison.csv',index=False)
    for col,label,name in [('hit_rate_at_10','Hit Rate@10','hit_rate_vs_k.png'),
                           ('sampled_relative_reconstruction_error','Sampled relative reconstruction error','reconstruction_error_vs_k.png'),
                           ('fit_seconds','Fit time (seconds)','fit_time_vs_k.png')]:
        fig,ax=plt.subplots(figsize=(8,5))
        for method,group in results_df.groupby('method'):
            group=group.sort_values('k')
            ax.plot(group['k'],group[col],marker='o',label=method)
        if col=='hit_rate_at_10':ax.axhline(baseline,linestyle='--',color='gray',label='Best-seller baseline')
        ax.set(xlabel='Components (k)',ylabel=label,title=label+' vs k')
        ax.grid(alpha=.3);ax.legend();fig.tight_layout();fig.savefig(OUT/name,dpi=160);plt.close(fig)
    svd_rows=results_df[results_df['method']=='TruncatedSVD'].dropna(subset=['hit_rate_at_10'])
    if svd_rows.empty:
        raise ValueError('No eligible held-out test customers; cannot select k by hit rate')
    selected_k=int(svd_rows.sort_values(['hit_rate_at_10','k'],ascending=[False,True]).iloc[0]['k']) if FINAL_K is None else FINAL_K
    if selected_k not in ks:
        raise ValueError(f'FINAL_K={selected_k} not in tested component counts {ks}')
    model,customer_factors,product_factors=models[('TruncatedSVD',selected_k)]
    save_product_similarities(product_factors,products,descriptions)
    # Cluster in latent feature space; use 2D PCA of latent factors for visualization only.
    coords=PCA(n_components=2,random_state=SEED).fit_transform(customer_factors)
    cluster_count=min(4,len(customer_factors))
    labels=KMeans(n_clusters=cluster_count,random_state=SEED,n_init=10).fit_predict(customer_factors)
    description_lookup=descriptions.set_index('product_idx')['Description']
    country_lookup=countries.set_index('customer_idx')['Country']
    summaries=[]
    for cluster_id in sorted(np.unique(labels)):
        members=np.flatnonzero(labels==cluster_id)
        means=np.asarray(A[members].mean(axis=0)).ravel()
        best=top_indices(means,5)
        country_counts=country_lookup.reindex(members).fillna('Unknown').value_counts().head(3)
        top_desc=description_lookup.reindex(best).fillna('Unknown').astype(str).tolist()
        summaries.append({'segment':f'Segment {cluster_id+1}','customer_count':len(members),
                          'top_product_descriptions':'; '.join(top_desc),
                          'top_countries':'; '.join(f'{c} ({v})' for c,v in country_counts.items()),
                          'suggested_segment_name':'REVIEW AND NAME BASED ON PRODUCTS'})
    names = {
        'Segment 1': 'Retrospot & Lunch-Bag Shoppers',
        'Segment 2': 'Home Decor & Gift Buyers',
        'Segment 3': 'General Home & Gift Shoppers',
        'Segment 4': 'Small Outlier Group (Review)'
    }
    for item in summaries:
        item['suggested_segment_name'] = names.get(item['segment'], item['segment'])
    pd.DataFrame(summaries).to_csv(OUT/'customer_segment_summary.csv',index=False)
    activity_report = cluster_activity(A, labels)
    activity_report.to_csv(OUT/'cluster_activity_summary.csv', index=False)
    benchmark = benchmark_recommendations(A, product_factors, n=TOP_N)
    pd.DataFrame({'anonymous_customer_index':np.arange(len(labels)),'component_1':coords[:,0],
                  'component_2':coords[:,1],'segment':[f'Segment {x+1}' for x in labels]}).to_csv(OUT/'customer_segment_assignments_anonymous.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,6))
    for cid in sorted(np.unique(labels)):
        mask=labels==cid
        ax.scatter(coords[mask,0],coords[mask,1],s=10,alpha=.55,label=f'Segment {cid+1}')
    ax.set(xlabel='2D component 1',ylabel='2D component 2',title='Customer segments (anonymous)')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'customer_segments.png',dpi=160);plt.close(fig)
    summary={**benchmark,'matrix_shape':list(A.shape),'matrix_nnz':int(A.nnz),
             'selected_svd_k':selected_k,'baseline_hit_rate_at_10':baseline,
             'eligible_test_customers':base_n,'total_analysis_seconds':time.perf_counter()-started,
             'reconstruction_error_note':'Estimated on fixed sample of training rows, not full matrix',
             'peak_memory_note':'Approximate peak process RSS sampled during fit; dense PCA conversion occurs before fit and is not included in PCA peak fit measurement',
             'nightly_runtime_note':'Analysis time includes repeated model experiments, not a production nightly job'}
    (OUT/'run_summary.json').write_text(json.dumps(summary,indent=2))
    print('\nCompleted. Outputs:',OUT)
    print(results_df.to_string(index=False))
if __name__=='__main__':
    main()
