
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parent
RAW_FILE = ROOT / 'data/raw/online_retail_II.xlsx'
OUT = ROOT / 'data/processed'
OUT.mkdir(parents=True, exist_ok=True)
NON_PRODUCT_CODES = {'POST','M','D','BANK CHARGES','CRUK','C2','DOT','AMAZONFEE','ADJUST','ADJUST2','S','PADS'}

def find_column(df, candidates):
    names = {str(c).strip().lower().replace(' ','').replace('_',''):c for c in df.columns}
    for candidate in candidates:
        key = candidate.lower().replace(' ','').replace('_','')
        if key in names:
            return names[key]
    raise ValueError(f'Missing column {candidates}; found {list(df.columns)}')

def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(f'Place the UCI workbook at {RAW_FILE}')
    sheets = pd.read_excel(RAW_FILE, sheet_name=None)
    df = pd.concat(sheets.values(), ignore_index=True)
    cols = {'Invoice':['Invoice','InvoiceNo'], 'StockCode':['StockCode'],
            'Description':['Description'], 'Quantity':['Quantity'],
            'InvoiceDate':['InvoiceDate'], 'Price':['Price','UnitPrice'],
            'CustomerID':['Customer ID','CustomerID'], 'Country':['Country']}
    df = df.rename(columns={find_column(df, v):k for k,v in cols.items()})
    profile = {'raw_rows':int(len(df)), 'raw_columns':{str(c):{'dtype':str(df[c].dtype),'missing_count':int(df[c].isna().sum()),'missing_percent':round(float(df[c].isna().mean()*100),3)} for c in df.columns}}
    (OUT/'data_profile.json').write_text(json.dumps(profile,indent=2))
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    for c in ['Invoice','StockCode']:
        df[c] = df[c].astype('string').str.strip().str.upper()
    for c in ['Quantity','Price']:
        df[c] = pd.to_numeric(df[c],errors='coerce')
    log=[]
    def filter_rows(name,mask):
        nonlocal df
        before=len(df)
        df=df.loc[mask.fillna(False)].copy()
        log.append({'rule':name,'rows_before':before,'rows_removed':before-len(df),'rows_after':len(df)})
    filter_rows('remove cancellation invoices',~df['Invoice'].fillna('').str.startswith('C'))
    filter_rows('remove missing invoice date',df['InvoiceDate'].notna())
    filter_rows('remove non-positive quantity',df['Quantity']>0)
    filter_rows('remove non-positive price',df['Price']>0)
    filter_rows('remove missing customer ID',df['CustomerID'].notna())
    filter_rows('remove missing product code',df['StockCode'].notna() & df['StockCode'].ne(''))
    filter_rows('remove non-product codes',~df['StockCode'].isin(NON_PRODUCT_CODES))
    if df.empty:
        raise ValueError('No rows remain after cleaning')
    df['CustomerID']=df['CustomerID'].astype(str).str.replace(r'\.0$','',regex=True)
    df['StockCode']=df['StockCode'].astype(str)
    df['Country']=df['Country'].fillna('Unknown').astype(str)
    df['Description']=df['Description'].fillna('Unknown product').astype(str)
    max_date=df['InvoiceDate'].max()
    cutoff=max_date-pd.DateOffset(months=2)
    train=df[df['InvoiceDate']<cutoff].copy()
    test=df[df['InvoiceDate']>=cutoff].copy()
    if train.empty or test.empty:
        raise ValueError('Time split produced an empty partition')
    customers=sorted(train['CustomerID'].unique())
    products=sorted(train['StockCode'].unique())
    customer_to_idx={v:i for i,v in enumerate(customers)}
    product_to_idx={v:i for i,v in enumerate(products)}
    agg=train.groupby(['CustomerID','StockCode'],as_index=False)['Quantity'].sum()
    rows=agg['CustomerID'].map(customer_to_idx).to_numpy()
    cols=agg['StockCode'].map(product_to_idx).to_numpy()
    values=np.log1p(agg['Quantity'].to_numpy(dtype=np.float32))
    A=sparse.csr_matrix((values,(rows,cols)),shape=(len(customers),len(products)),dtype=np.float32)
    A.sum_duplicates()
    sparse.save_npz(OUT/'train_matrix.npz',A)
    pd.DataFrame({'customer_idx':np.arange(len(customers)),'CustomerID':customers}).to_csv(OUT/'customer_map_internal.csv',index=False)
    pd.DataFrame({'product_idx':np.arange(len(products)),'StockCode':products}).to_csv(OUT/'product_map_internal.csv',index=False)
    popularity=train.groupby('StockCode')['Quantity'].sum().reindex(products,fill_value=0)
    pd.DataFrame({'product_idx':np.arange(len(products)),'training_quantity':popularity.to_numpy()}).to_csv(OUT/'product_popularity_internal.csv',index=False)
    descriptions=(train.groupby('StockCode')['Description'].agg(lambda x:x.value_counts().index[0]).reindex(products))
    pd.DataFrame({'product_idx':np.arange(len(products)),'Description':descriptions.to_numpy()}).to_csv(OUT/'product_descriptions_internal.csv',index=False)
    countries=(train.groupby('CustomerID')['Country'].agg(lambda x:x.value_counts().index[0]).reindex(customers))
    pd.DataFrame({'customer_idx':np.arange(len(customers)),'Country':countries.to_numpy()}).to_csv(OUT/'customer_country_internal.csv',index=False)
    # Only new purchases by existing customers of products in training catalogue.
    test_agg=test.groupby(['CustomerID','StockCode'],as_index=False)['Quantity'].sum()
    test_agg=test_agg[(test_agg['Quantity']>0)&test_agg['CustomerID'].isin(customer_to_idx)&test_agg['StockCode'].isin(product_to_idx)].copy()
    seen=train[['CustomerID','StockCode']].drop_duplicates().assign(_seen=True)
    before_seen=len(test_agg)
    test_agg=test_agg.merge(seen,on=['CustomerID','StockCode'],how='left')
    test_agg=test_agg[test_agg['_seen'].isna()].drop(columns='_seen').copy()
    # Explicit leakage and chronology assertions before dropping original IDs.
    overlap = test_agg.merge(
        train[['CustomerID', 'StockCode']].drop_duplicates(),
        on=['CustomerID', 'StockCode'], how='inner'
    )
    assert overlap.empty, f'Test leakage: {len(overlap)} previously purchased pairs'
    assert train['InvoiceDate'].max() < cutoff
    assert test['InvoiceDate'].min() >= cutoff
    print('PASS: Chronological split and unseen test-pair checks')
    test_agg['customer_idx']=test_agg['CustomerID'].map(customer_to_idx).astype(int)
    test_agg['product_idx']=test_agg['StockCode'].map(product_to_idx).astype(int)
    test_agg[['customer_idx','product_idx','Quantity']].to_csv(OUT/'test_purchases.csv',index=False)
    pd.DataFrame(log).to_csv(OUT/'cleaning_log.csv',index=False)
    total=A.shape[0]*A.shape[1]
    report={'raw_rows':profile['raw_rows'],'clean_rows':int(len(df)),
            'train_rows':int(len(train)),'test_rows':int(len(test)),
            'time_cutoff':str(cutoff),'max_date':str(max_date),
            'train_matrix_shape':list(A.shape),'nonzero_cells':int(A.nnz),
            'sparsity_percent':round((1-A.nnz/total)*100,4),
            'dense_float32_MB':round(total*4/2**20,3),
            'sparse_storage_MB':round((A.data.nbytes+A.indices.nbytes+A.indptr.nbytes)/2**20,3),
            'purchase_value':'log1p(aggregated positive quantity)',
            'test_pairs_before_seen_filter':int(before_seen),
            'eligible_unseen_test_pairs':int(len(test_agg)),
            'eligible_test_customers':int(test_agg['customer_idx'].nunique()),
            'notes':['All matrix mappings built from training only','Test targets exclude already purchased customer-product pairs','Unseen test products and new customers excluded from evaluation','Internal mappings must not be published']}
    (OUT/'matrix_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':
    main()
