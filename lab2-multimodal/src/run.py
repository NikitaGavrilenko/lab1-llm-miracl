"""Train eight schemes, evaluate frozen holdouts, and run paired time-block tests."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import json,time,platform,sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib,torch,sklearn,scipy
from sklearn.metrics import roc_auc_score,brier_score_loss
from models import fit_models,fit_pairwise
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
UNIMODAL=['text_only','numeric_only','image_only']
NAMES=['macro_f1','mcc','auroc','brier','accuracy','ece']

def metrics(y,p):
    z=p>=.5; tp=np.sum(z&(y==1)); tn=np.sum(~z&(y==0)); fp=np.sum(z&(y==0)); fn=np.sum(~z&(y==1))
    f1=.5*(2*tp/max(2*tp+fp+fn,1)+2*tn/max(2*tn+fp+fn,1))
    den=np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    ece=0.
    for j in range(10):
        mask=(p>=j/10)&((p<(j+1)/10) if j<9 else (p<=1))
        if mask.any(): ece+=mask.mean()*abs(y[mask].mean()-p[mask].mean())
    return dict(macro_f1=float(f1),mcc=float((tp*tn-fp*fn)/den) if den else 0.,
        auroc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else float('nan'),
        brier=float(np.mean((y-p)**2)),accuracy=float((z==y).mean()),ece=float(ece))

def block_draws(dates,repeats,length,seed):
    rng=np.random.default_rng(seed); days=np.array(sorted(set(dates)))
    groups=[np.flatnonzero(dates==d) for d in days]
    # Moving blocks without circular wrap; all tickers from a date stay together.
    for _ in range(repeats):
        starts=rng.integers(0,len(days)-length+1,size=int(np.ceil(len(days)/length)))
        selected=np.concatenate([np.arange(s,s+length) for s in starts])[:len(days)]
        yield np.concatenate([groups[i] for i in selected])

def permutation(y,a,b,dates,length=5,repeats=4999):
    rng=np.random.default_rng(2026+length);days=sorted(set(dates)); mapping={d:i//length for i,d in enumerate(days)}
    block=np.array([mapping[d] for d in dates]); observed=metrics(y,a)['macro_f1']-metrics(y,b)['macro_f1']; exceed=0
    nblocks=block.max()+1; exact=nblocks<=15
    count=2**nblocks if exact else repeats
    for iteration in range(count):
        switches=((iteration >> np.arange(nblocks))&1) if exact else rng.integers(0,2,size=nblocks)
        swap=switches[block].astype(bool)
        pa=np.where(swap,b,a);pb=np.where(swap,a,b)
        # Macro F1 only: avoid unnecessary AUROC computation in permutations.
        delta=fast_f1(y,pa)-fast_f1(y,pb)
        exceed+=delta>=observed-1e-12
    return exceed/count if exact else (exceed+1)/(repeats+1)

def fast_f1(y,p):
    z=p>=.5;tp=np.sum(z&(y==1));tn=np.sum(~z&(y==0));fp=np.sum(z&(y==0));fn=np.sum(~z&(y==1))
    return .5*(2*tp/max(2*tp+fp+fn,1)+2*tn/max(2*tn+fp+fn,1))

def main():
    wall=time.perf_counter();cpu=time.process_time()
    a=np.load(ROOT/'data/features.npz');X={k:a[k] for k in ['T','N','I']};y=a['y']; df=pd.read_csv(ROOT/'data/dataset.csv')
    models,tuning=fit_models(X,y,a['train'],a['val'],seed=42)
    joblib.dump(models,OUT/'models.joblib');pd.DataFrame(tuning).to_csv(OUT/'tuning.csv',index=False)
    valX={k:v[a['val']] for k,v in X.items()}; val_scores={name:metrics(y[a['val']],m.predict_proba(valX))['macro_f1'] for name,m in models.items()}
    best_uni=max(UNIMODAL,key=val_scores.get); best_fusion=max([n for n in models if n not in UNIMODAL],key=val_scores.get)
    best_overall=max(models,key=val_scores.get)
    genuine_fusion=max(['early_fusion','late_mean','intermediate_mlp','attention_fusion'],key=val_scores.get)
    rows=[]; predictions={};latency=[]
    for part in ['val','test','application']:
        idx=a[part]; xx={k:v[idx] for k,v in X.items()}; yy=y[idx]
        pr=df.iloc[idx][['sample_id','ticker','date','y','target_return','text','n_articles']].copy()
        for name,model in models.items():
            p=model.predict_proba(xx);pr[name]=p
            rows.append(dict(split=part,model=name,**metrics(yy,p)))
        pr.to_csv(OUT/f'predictions_{part}.csv',index=False);predictions[part]=pr
    test=predictions['test']; yt=test.y.to_numpy();dates=test.date.to_numpy();ci=[];delta_ci=[]
    for length in [5,10,20]:
        samples={n:[] for n in models};diff=[];deltas={n:[] for n in models if n not in UNIMODAL}
        for ix in block_draws(dates,1200,length,42+length):
            for name in models:samples[name].append(list(metrics(yt[ix],test[name].to_numpy()[ix]).values()))
            diff.append(fast_f1(yt[ix],test[best_fusion].to_numpy()[ix])-fast_f1(yt[ix],test[best_uni].to_numpy()[ix]))
            for name in deltas:deltas[name].append(fast_f1(yt[ix],test[name].to_numpy()[ix])-fast_f1(yt[ix],test[best_uni].to_numpy()[ix]))
        for name,values in samples.items():
            low,high=np.nanquantile(values,[.025,.975],axis=0)
            for j,key in enumerate(NAMES):ci.append(dict(model=name,metric=key,block_sessions=length,lower=low[j],upper=high[j]))
        if length==5:primary_diff=np.quantile(diff,[.025,.975]).tolist()
        elif length==10:sensitivity_diff=np.quantile(diff,[.025,.975]).tolist()
        for name,values in deltas.items():
            low,high=np.quantile(values,[.025,.975]);delta_ci.append(dict(model=name,baseline=best_uni,block_sessions=length,lower=low,upper=high))
    pd.DataFrame(ci).to_csv(OUT/'confidence_intervals.csv',index=False)
    pd.DataFrame(delta_ci).to_csv(OUT/'delta_confidence_intervals.csv',index=False)
    significance=[]
    for name in models:
        if name in UNIMODAL:continue
        significance.append(dict(model=name,baseline=best_uni,delta_macro_f1=fast_f1(yt,test[name].to_numpy())-fast_f1(yt,test[best_uni].to_numpy()),
                                 p_one_sided=permutation(yt,test[name].to_numpy(),test[best_uni].to_numpy(),dates)))
    # Holm family correction for five fusion comparisons on a shared test.
    order=np.argsort([r['p_one_sided'] for r in significance]);running=0.
    for rank,i in enumerate(order):
        running=max(running,min(1.,(len(order)-rank)*significance[i]['p_one_sided']));significance[i]['p_holm']=running
    pd.DataFrame(significance).to_csv(OUT/'significance.csv',index=False)
    robustness=[];ix=a['test'];testX={k:v[ix] for k,v in X.items()}
    for name,model in models.items():
        for missing in [None,'T','N','I']:
            xx={k:v for k,v in testX.items() if k!=missing};p=model.predict_proba(xx)
            robustness.append(dict(model=name,scenario='full' if missing is None else f'missing_{missing}',**metrics(yt,p)))
    # A reproducible mixed missingness stress test, 30% per modality independently.
    rng=np.random.default_rng(123);xx={k:v.copy() for k,v in testX.items()}
    for k,v in xx.items():v[rng.random(len(v))<.3]=0
    for name,model in models.items():robustness.append(dict(model=name,scenario='random_missing_30pct',**metrics(yt,model.predict_proba(xx))))
    pd.DataFrame(robustness).to_csv(OUT/'robustness.csv',index=False)
    pairs=fit_pairwise(X,y,a['train'],a['val'],42);ablations=[]
    for name,m in pairs.items():ablations.append(dict(model=name,**metrics(yt,m.predict_proba(testX)),**m.metadata))
    pd.DataFrame(ablations).to_csv(OUT/'retrained_ablations.csv',index=False)
    # Cache-only latency is intentionally separate from feature preparation.
    for name,m in models.items():
        elapsed=[]
        for _ in range(10):m.predict_proba({k:v[:1] for k,v in testX.items()})
        for j in range(min(200,len(ix))):
            one={k:v[j:j+1] for k,v in testX.items()};t=time.perf_counter();m.predict_proba(one);elapsed.append((time.perf_counter()-t)*1000)
        latency.append(dict(model=name,latency_p50_ms=float(np.median(elapsed)),latency_p95_ms=float(np.quantile(elapsed,.95)),**m.metadata))
    pd.DataFrame(latency).to_csv(OUT/'resources.csv',index=False)
    metricdf=pd.DataFrame(rows);metricdf.to_csv(OUT/'metrics.csv',index=False)
    # Descriptive multicriteria score. Selection itself remains validation-only.
    score=metricdf[metricdf.split=='test'].set_index('model').copy()
    rob=pd.DataFrame(robustness);rob=rob[rob.scenario.str.startswith('missing_')].groupby('model').macro_f1.mean()
    res=pd.DataFrame(latency).set_index('model')
    components=dict(f1=score.macro_f1,auroc=score.auroc,calibration=-score.brier,robustness=rob,latency=-res.latency_p95_ms)
    total=pd.Series(0.,index=score.index)
    for (key,values),weight in zip(components.items(),[.45,.20,.15,.10,.10]):
        norm=(values-values.min())/(values.max()-values.min()+1e-12);score[key+'_normalized']=norm;total+=weight*norm
    score['multicriteria_score']=total;score.to_csv(OUT/'multicriteria.csv')
    # Freeze before examining application performance; technology subset is predefined.
    tech={'AAPL','MSFT','NVDA','AMD','INTC','CSCO','ORCL','GOOG','GOOGL','META','AMZN','ADBE','CRM','AVGO','QCOM'}
    app=predictions['application'];application=[]
    for subset,mask in [('all',np.ones(len(app),dtype=bool)),('technology',app.ticker.isin(tech).to_numpy())]:
        for name in [best_overall,best_fusion,best_uni,genuine_fusion]:
            if mask.sum():application.append(dict(subset=subset,model=name,n=int(mask.sum()),**metrics(app.y.to_numpy()[mask],app[name].to_numpy()[mask])))
    pd.DataFrame(application).drop_duplicates(['subset','model']).to_csv(OUT/'application_metrics.csv',index=False)
    example=[]
    pu=app[best_uni].to_numpy()>=.5;pf=app[genuine_fusion].to_numpy()>=.5;correct=app.y.to_numpy()
    categories={'fusion_helps':(pf==correct)&(pu!=correct),'fusion_hurts':(pf!=correct)&(pu==correct),'both_correct':(pf==correct)&(pu==correct),'both_wrong':(pf!=correct)&(pu!=correct)}
    for label,mask in categories.items():
        subset=app[mask].sort_values(['date','ticker']).head(2).copy();subset['category']=label;subset['fusion_model']=genuine_fusion;example.append(subset)
    pd.concat(example).to_csv(OUT/'error_examples.csv',index=False)
    # Seed sensitivity uses the same protocol; does not change primary seed selection.
    seedrows=[]
    for seed in [7,123]:
        ms,_=fit_models(X,y,a['train'],a['val'],seed)
        for name in ['intermediate_mlp','attention_fusion']:
            seedrows.append(dict(seed=seed,model=name,**metrics(yt,ms[name].predict_proba(testX))))
    for name in ['intermediate_mlp','attention_fusion']:seedrows.append(dict(seed=42,model=name,**metrics(yt,test[name].to_numpy())))
    pd.DataFrame(seedrows).to_csv(OUT/'seed_sensitivity.csv',index=False)
    references=[]
    for part in ['val','test','application']:
        yy=y[a[part]]
        for label,p in [('train_prior',np.full(len(yy),y[a['train']].mean())),('always_up',np.ones(len(yy)))]:
            references.append(dict(split=part,model=label,**metrics(yy,p)))
    pd.DataFrame(references).to_csv(OUT/'reference_baselines.csv',index=False)
    summary=dict(seed=42,best_unimodal_validation=best_uni,best_fusion_validation=best_fusion,best_overall_validation=best_overall,
        validation_f1=val_scores,delta_ci_block5=primary_diff,delta_ci_block10=sensitivity_diff,
        primary_p_block10=permutation(yt,test[best_fusion].to_numpy(),test[best_uni].to_numpy(),dates,10),
        best_genuine_fusion_validation=genuine_fusion,test_days=len(set(dates)),bootstrap_repeats=1200,
        permutation_method='exhaustive paired block swaps',permutation_configurations_block5=2**int(np.ceil(len(set(dates))/5)),permutation_configurations_block10=2**int(np.ceil(len(set(dates))/10)),
        wall_seconds=time.perf_counter()-wall,cpu_seconds=time.process_time()-cpu,
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,torch=torch.__version__,scipy=scipy.__version__),
        hardware=dict(cpu=platform.processor(),logical_cpus=os.cpu_count(),gpu=False,threads=2))
    (OUT/'run_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(metricdf[metricdf.split=='test'].to_string(index=False));print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
