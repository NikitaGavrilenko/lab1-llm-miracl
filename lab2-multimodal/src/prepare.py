"""Build causal ticker/session objects and train-only compact features."""
import sys, json, hashlib, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor'))
import numpy as np
import pandas as pd
import exchange_calendars as xcals
from PIL import Image, ImageDraw
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
import joblib

def chart(w):
    a = w[['Open','High','Low','Close']].to_numpy(float)
    a *= (w['Adj Close']/w['Close']).to_numpy()[:,None]
    a /= float(w['Adj Close'].iloc[-1])
    lo, hi = a.min(), a.max()
    y = lambda v: int(6 + (hi-v)/(hi-lo+1e-9)*62)
    im = Image.new('RGB',(96,96),'white'); d=ImageDraw.Draw(im)
    vol=w.Volume.to_numpy(float); vol=vol/(vol.max()+1e-9)
    for k,(o,h,l,c) in enumerate(a):
        x=5+k*4.5; col=(25,135,65) if c>=o else (195,45,45)
        d.line((x,y(h),x,y(l)),fill=col,width=1)
        d.rectangle((x-1,min(y(o),y(c)),x+1,max(y(o),y(c))+1),fill=col)
        d.rectangle((x-1,92-int(17*vol[k]),x+1,92),fill=col)
    return im

def main():
    started=time.perf_counter()
    out=ROOT/'data'; (out/'images').mkdir(exist_ok=True)
    news=json.loads((out/'raw/kaggle/polygon_news_sample.json').read_text(encoding='utf-8'))
    prices=pd.read_csv(out/'raw/kaggle/SP500_Historical_Data.csv',parse_dates=['Date'])
    calendar=xcals.get_calendar('XNYS',start='2022-10-01',end='2024-02-01')
    schedule=calendar.schedule.copy()
    closes=pd.DatetimeIndex(schedule['close']); sessions=pd.DatetimeIndex(schedule.index).tz_localize(None)
    records=[]; excluded=[]; seen_ids=set(); seen_titles=set(); afterclose=0; duplicate_titles=0
    available=set(prices.Ticker)
    for a in sorted(news,key=lambda a:a['published_utc']):
        aid=a['id']; title=' '.join(a['title'].split())
        if aid in seen_ids:
            excluded.append(dict(article_id=aid,ticker='',reason='duplicate_id')); continue
        seen_ids.add(aid)
        if title.casefold() in seen_titles:
            duplicate_titles+=1; excluded.append(dict(article_id=aid,ticker='',reason='duplicate_title'));continue
        seen_titles.add(title.casefold())
        ts=pd.Timestamp(a['published_utc'])
        pos=closes.searchsorted(ts,side='left')
        if pos>=len(closes):
            excluded.append(dict(article_id=aid,ticker='',reason='outside_calendar'));continue
        anchor=sessions[pos]
        local=ts.tz_convert('America/New_York')
        if anchor.date()>local.date(): afterclose+=1
        for ticker in sorted(set(a['tickers'])):
            if ticker not in available:
                excluded.append(dict(article_id=aid,ticker=ticker,reason='no_price_ticker'));continue
            records.append(dict(article_id=aid,ticker=ticker,date=anchor,text=title,published_utc=ts,
                                cutoff_utc=closes[pos],publisher=a['publisher']['name']))
    events=pd.DataFrame(records)
    # A headline never occurs on both sides of a split: cut by session globally.
    def split(d):
        if d < pd.Timestamp('2023-08-01'): return 'train'
        if d < pd.Timestamp('2023-10-01'): return 'val'
        if d < pd.Timestamp('2023-12-01'): return 'test'
        return 'application'
    numeric=[]; image_pixels=[]; objects=[]; price_dups=int(prices.duplicated(['Ticker','Date']).sum())
    price_groups={k:v.sort_values('Date').drop_duplicates('Date').set_index('Date') for k,v in prices.groupby('Ticker') if k in set(events.ticker)}
    for (ticker,date),g in events.groupby(['ticker','date'],sort=True):
        p=price_groups[ticker]; loc=p.index.get_indexer([date])[0]
        reason=None
        if loc<20: reason='missing_history_or_session'
        elif loc+1>=len(p): reason='missing_target'
        if reason:
            excluded.append(dict(article_id='|'.join(g.article_id),ticker=ticker,date=str(date.date()),reason=reason));continue
        w=p.iloc[loc-19:loc+1]; hist=p.iloc[loc-20:loc+1]; nxt=p.index[loc+1]
        expected=sessions[(sessions>=hist.index[0])&(sessions<=nxt)]
        if not np.array_equal(p.index[loc-20:loc+2].values,expected.values): reason='calendar_gap'
        values=hist[['Open','High','Low','Close','Adj Close','Volume']].to_numpy(float)
        if not np.isfinite(values).all() or (values<=0).any(): reason='invalid_ohlcv'
        if (hist.High < hist[['Open','Close','Low']].max(axis=1)-1e-6).any() or (hist.Low>hist[['Open','Close','High']].min(axis=1)+1e-6).any(): reason='ohlc_order'
        part=split(date)
        if split(nxt)!=part: reason='purged_label_across_split'
        if reason:
            excluded.append(dict(article_id='|'.join(g.article_id),ticker=ticker,date=str(date.date()),reason=reason));continue
        adj=(hist['Adj Close']/hist.Close).to_numpy()
        ohlc=hist[['Open','High','Low','Close']].to_numpy(float)*adj[:,None]
        f=np.log(ohlc[1:]/ohlc[:-1,3,None])
        v=np.log(hist.Volume.to_numpy(float)[1:]/hist.Volume.to_numpy(float)[:-1])[:,None]
        num=np.concatenate([f,v],axis=1).ravel()
        # Future vendor adjustment factors cancel in both relative OHLC and returns.
        assert np.allclose(f,np.log((ohlc[1:]*7)/(ohlc[:-1,3,None]*7)))
        ret=float(p['Adj Close'].iloc[loc+1]/p['Adj Close'].iloc[loc]-1)
        im=chart(w); sid=f'{ticker}_{date.date()}'
        im.save(out/'images'/f'{sid}.png')
        image_pixels.append(np.asarray(im.resize((32,32)),dtype=np.float32).ravel()/255)
        numeric.append(num)
        objects.append(dict(sample_id=sid,ticker=ticker,date=str(date.date()),split=part,
            publication_min=str(g.published_utc.min()),publication_max=str(g.published_utc.max()),
            cutoff_utc=str(g.cutoff_utc.iloc[0]),window_start=str(w.index[0].date()),window_end=str(date.date()),
            target_date=str(nxt.date()),text=' [SEP] '.join(g.text.drop_duplicates()),article_ids='|'.join(g.article_id),
            publishers='|'.join(sorted(set(g.publisher))),n_articles=len(g),target_return=ret,y=int(ret>0),
            image_path=f'data/images/{sid}.png'))
    df=pd.DataFrame(objects).sort_values(['date','ticker']).reset_index(drop=True)
    # Align arrays after sorting.
    unsorted_ids=[x['sample_id'] for x in objects]; order=[unsorted_ids.index(k) for k in df.sample_id]
    N=np.asarray(numeric,dtype=np.float32)[order]; pixels=np.asarray(image_pixels,dtype=np.float32)[order]
    train=np.flatnonzero(df.split.eq('train')); indices={k:np.flatnonzero(df.split.eq(k)) for k in ['train','val','test','application']}
    tfidf=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_df=.98,max_features=15000,sublinear_tf=True,strip_accents='unicode')
    sparse=tfidf.fit_transform(df.loc[train,'text']); svd=TruncatedSVD(64,random_state=42)
    svd.fit(sparse); T=svd.transform(tfidf.transform(df.text)).astype(np.float32)
    pca=PCA(64,svd_solver='randomized',random_state=42); pca.fit(pixels[train]); I=pca.transform(pixels).astype(np.float32)
    np.savez_compressed(out/'features.npz',T=T,N=N,I=I,y=df.y.to_numpy(),**indices)
    np.savez_compressed(out/'numeric_windows.npz',sample_id=df.sample_id.to_numpy(dtype=str),windows=N.reshape(-1,20,5))
    joblib.dump(dict(tfidf=tfidf,svd=svd,pca=pca),out/'encoders.joblib')
    df.to_csv(out/'dataset.csv',index=False)
    pd.DataFrame(excluded).to_csv(out/'excluded_observations.csv',index=False)
    events.to_csv(out/'article_alignment.csv',index=False)
    schedule.loc['2023-01-01':'2024-01-10'].to_csv(out/'trading_calendar.csv')
    summary=dict(raw_articles=len(news),article_ticker_pairs=len(events),raw_price_rows=len(prices),raw_price_tickers=prices.Ticker.nunique(),
        price_missing_cells=int(prices.isna().sum().sum()),price_duplicate_keys=price_dups,duplicate_titles=duplicate_titles,
        transferred_to_later_session=afterclose,objects=len(df),tickers=df.ticker.nunique(),
        splits={k:dict(n=len(v),days=df.iloc[v].date.nunique(),start=df.iloc[v].date.min(),end=df.iloc[v].date.max(),positive_rate=float(df.iloc[v].y.mean())) for k,v in indices.items()},
        exclusions=pd.DataFrame(excluded).reason.value_counts().to_dict(),feature_dimensions=dict(T=T.shape[1],N=N.shape[1],I=I.shape[1]),
        pca_explained_variance=float(pca.explained_variance_ratio_.sum()),preparation_seconds=time.perf_counter()-started)
    (out/'quality_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    assert (pd.to_datetime(df.publication_max,utc=True)<=pd.to_datetime(df.cutoff_utc,utc=True)).all()
    assert (df.window_end<df.target_date).all()
    assert not df.sample_id.duplicated().any()
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
