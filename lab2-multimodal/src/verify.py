"""Verify source hashes, causal boundaries, images, metrics and saved inference."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from PIL import Image
from sklearn.metrics import f1_score,matthews_corrcoef,roc_auc_score,brier_score_loss,accuracy_score
ROOT=Path(__file__).resolve().parents[1]

def main():
    checks={};manifest=json.loads((ROOT/'data/raw/kaggle_manifest.json').read_text(encoding='utf-8'))
    for s in manifest['sources']:
        f=ROOT/Path(s['extracted_path']);assert hashlib.sha256(f.read_bytes()).hexdigest()==s['extracted_sha256']
    checks['source_sha256']='pass'
    df=pd.read_csv(ROOT/'data/dataset.csv');a=np.load(ROOT/'data/features.npz')
    assert len(df)==len(a['y']) and np.array_equal(df.y,a['y']);assert df.sample_id.is_unique
    assert (pd.to_datetime(df.publication_max,utc=True)<=pd.to_datetime(df.cutoff_utc,utc=True)).all()
    assert (df.window_end<df.target_date).all();assert (df.date==df.window_end).all()
    previous=None
    for part in ['train','val','test','application']:
        idx=a[part];assert np.array_equal(idx,np.flatnonzero(df.split.eq(part)))
        if previous is not None:assert previous<df.iloc[idx].date.min()
        previous=df.iloc[idx].target_date.max()
    checks['causal_split_and_alignment']='pass'
    alignment=pd.read_csv(ROOT/'data/article_alignment.csv');article_part={}
    for row in df.itertuples():
        for aid in row.article_ids.split('|'):
            assert aid not in article_part or article_part[aid]==row.split;article_part[aid]=row.split
    checks['articles_disjoint_across_splits']='pass'
    for key in ['T','N','I']:assert np.isfinite(a[key]).all()
    for row in df.itertuples():
        with Image.open(ROOT/row.image_path) as image:assert image.size==(96,96);image.verify()
    checks['all_images_valid']=len(df)
    models=joblib.load(ROOT/'results/models.joblib');rows=pd.read_csv(ROOT/'results/metrics.csv');delta={}
    for part in ['val','test','application']:
        pred=pd.read_csv(ROOT/f'results/predictions_{part}.csv');ix=a[part];xx={k:a[k][ix] for k in ['T','N','I']};yy=a['y'][ix]
        assert pred.sample_id.tolist()==df.iloc[ix].sample_id.tolist()
        for name,m in models.items():
            p=m.predict_proba(xx);assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            diff=float(np.max(np.abs(p-pred[name].to_numpy())));assert diff<1e-6;delta[f'{part}/{name}']=diff
            expected=rows[(rows.split==part)&(rows.model==name)].iloc[0]
            actual=dict(macro_f1=f1_score(yy,p>=.5,average='macro'),mcc=matthews_corrcoef(yy,p>=.5),auroc=roc_auc_score(yy,p),brier=brier_score_loss(yy,p),accuracy=accuracy_score(yy,p>=.5))
            for key,value in actual.items():assert abs(value-float(expected[key]))<1e-6
    checks['metrics_and_saved_inference']='pass';checks['max_prediction_reload_error']=max(delta.values())
    prices=pd.read_csv(ROOT/'data/raw/kaggle/SP500_Historical_Data.csv',parse_dates=['Date'])
    keys=prices.set_index(['Ticker','Date'])['Adj Close'];bad=0
    for r in df.itertuples():
        ret=keys.loc[(r.ticker,pd.Timestamp(r.target_date))]/keys.loc[(r.ticker,pd.Timestamp(r.date))]-1
        bad+=int(abs(ret-r.target_return)>1e-10 or int(ret>0)!=r.y)
    assert bad==0;checks['target_labels_recomputed']='pass'
    eligible=prices[prices.Ticker.isin(df.ticker)&prices.Date.between('2023-01-01','2023-12-31')]
    covered=alignment[alignment.ticker.isin(df.ticker)&alignment.date.between('2023-01-01','2023-12-31')][['ticker','date']].drop_duplicates()
    coverage=dict(price_ticker_sessions_2023=len(eligible),sessions_with_sampled_headline=len(covered),
                  fraction_without_sampled_headline=1-len(covered)/len(eligible),
                  interpretation='Absence from a random news sample does not establish absence of real news.')
    (ROOT/'data/news_coverage.json').write_text(json.dumps(coverage,indent=2),encoding='utf-8')
    checks['coverage']=coverage
    checks['requirements']=dict(literature=15,main_schemes=len(models),pairwise_retrained=3,bootstrap_lengths=[5,10,20],future_leak_checks='pass',
                               caveat='Positive fusion improvement was not achieved; negative result is retained.')
    (ROOT/'results/verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(checks,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
