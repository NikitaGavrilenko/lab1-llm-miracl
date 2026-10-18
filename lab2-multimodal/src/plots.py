"""Generate publication-readable figures from measured results."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];F=ROOT/'figures';F.mkdir(exist_ok=True)
LABELS={'text_only':'Только текст','numeric_only':'Только числа','image_only':'Только график','early_fusion':'Early fusion','late_mean':'Late mean','late_weighted':'Late weighted','intermediate_mlp':'MLP fusion','attention_fusion':'Attention fusion'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})

def main():
    df=pd.read_csv(ROOT/'data/dataset.csv');m=pd.read_csv(ROOT/'results/metrics.csv');m=m[m.split=='test'].set_index('model')
    ci=pd.read_csv(ROOT/'results/confidence_intervals.csv');ci=ci[(ci.metric=='macro_f1')&(ci.block_sessions==5)].set_index('model')
    names=list(m.index);colors=['#577590']*3+['#43aa8b']*5
    fig,ax=plt.subplots(figsize=(8.8,4.1));y=np.arange(8)
    ax.errorbar(m.macro_f1,y,xerr=[m.macro_f1-ci.loc[names].lower,ci.loc[names].upper-m.macro_f1],fmt='none',ecolor='#666666',capsize=3)
    ax.scatter(m.macro_f1,y,c=colors,s=45,zorder=3);ax.axvline(.5,color='gray',ls=':',lw=1)
    ax.set(yticks=y,yticklabels=[LABELS[n] for n in names],xlabel='Macro F1 и 95% блочный интервал',xlim=(.25,.62));ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    fig.tight_layout();fig.savefig(F/'metrics_f1.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    for ax,key,title in zip(axes,['auroc','brier'],['AUROC выше лучше','Brier ниже лучше']):
        ax.barh(np.arange(8),m[key],color=colors);ax.set(yticks=np.arange(8),yticklabels=[LABELS[n] for n in names],title=title);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    axes[0].set_xlim(.4,.56);axes[1].set_xlim(.22,.28);fig.tight_layout();fig.savefig(F/'discrimination_calibration.png');plt.close(fig)
    rob=pd.read_csv(ROOT/'results/robustness.csv');rob=rob.pivot(index='model',columns='scenario',values='macro_f1').loc[names,['full','missing_T','missing_N','missing_I','random_missing_30pct']]
    fig,ax=plt.subplots(figsize=(8.2,4));im=ax.imshow(rob.to_numpy(),vmin=.30,vmax=.56,cmap='YlGnBu',aspect='auto')
    ax.set(yticks=np.arange(8),yticklabels=[LABELS[n] for n in names],xticks=np.arange(5),xticklabels=['Все','Нет текста','Нет чисел','Нет графика','30% пропусков'])
    for (i,j),v in np.ndenumerate(rob.to_numpy()):ax.text(j,i,f'{v:.3f}',ha='center',va='center',color='white' if v>.47 else 'black',fontsize=9)
    fig.colorbar(im,ax=ax,label='Macro F1');fig.tight_layout();fig.savefig(F/'missing_modalities.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.3));pred=pd.read_csv(ROOT/'results/predictions_test.csv')
    for name in ['numeric_only','early_fusion','attention_fusion']:
        obs,pr=calibration_curve(pred.y,pred[name],n_bins=6,strategy='quantile');axes[0].plot(pr,obs,'o-',label=LABELS[name])
    axes[0].plot([0,1],[0,1],'--',color='gray');axes[0].set(xlabel='Прогноз вероятности роста',ylabel='Фактическая доля роста',xlim=(.15,.85),ylim=(.15,.85));axes[0].legend(fontsize=8)
    month=pd.to_datetime(df.date).dt.to_period('M').astype(str);counts=pd.crosstab(month,df.split).reindex(columns=['train','val','test','application'],fill_value=0)
    counts.plot.bar(stacked=True,ax=axes[1],color=['#577590','#f9c74f','#43aa8b','#f9844a'],width=.8);axes[1].set(xlabel='',ylabel='Объектов');axes[1].legend(fontsize=7)
    fig.tight_layout();fig.savefig(F/'calibration_timeline.png');plt.close(fig)
    examples=pd.read_csv(ROOT/'results/error_examples.csv');fig,axes=plt.subplots(2,2,figsize=(8.2,5.2))
    for ax,cat in zip(axes.flat,['fusion_helps','fusion_hurts','both_correct','both_wrong']):
        subset=examples[examples.category==cat]
        if len(subset):
            e=subset.iloc[0];ax.imshow(Image.open(ROOT/f'data/images/{e.sample_id}.png'));ax.set_title(f'{e.ticker}  {e.date}\n{cat}  y={e.y}',fontsize=10)
        ax.axis('off')
    fig.tight_layout();fig.savefig(F/'real_chart_examples.png');plt.close(fig)
    print('Figures:',len(list(F.glob('*.png'))))

if __name__=='__main__':main()
