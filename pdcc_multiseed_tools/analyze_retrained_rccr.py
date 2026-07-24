#!/usr/bin/env python3
"""Evaluate RCCR calibration/reliability on independently retrained checkpoints."""
from __future__ import annotations
import argparse, csv, json, os
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from dataloader import MMDataset
MODS=('text','audio','vision'); OUT={'text':'pred_t','audio':'pred_a','vision':'pred_v'}

def parse(s,cast=str): return [cast(x.strip()) for x in s.split(',') if x.strip()]
def safe_auc(y,s):
    try: return float(roc_auc_score(y,s)) if len(np.unique(y))==2 else float('nan')
    except Exception: return float('nan')
def safe_spear(x,y):
    try: return float(spearmanr(x,y).statistic)
    except Exception: return float('nan')
def ece(prob,y,bins=15):
    conf=prob.max(1); pred=prob.argmax(1); cor=(pred==y).astype(float); val=0.0
    for lo,hi in zip(np.linspace(0,1,bins+1)[:-1],np.linspace(0,1,bins+1)[1:]):
        m=(conf>=lo)&((conf<hi) if hi<1 else (conf<=hi))
        if m.any(): val += abs(float(cor[m].mean())-float(conf[m].mean()))*float(m.mean())
    return float(val)
def find_ckpt(rd,dataset):
    fs=list((rd/'models').glob(f'{dataset}_*_PDCCModel_Has0_acc_2.pt'))
    if not fs:
        fs=list((rd/'models').glob(f'{dataset}_*_DCCModel_Has0_acc_2.pt'))
    if not fs: raise FileNotFoundError(f'No PDCC checkpoint in {rd}/models')
    return fs[0]
def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True,choices=['SIMS','MOSI'])
    p.add_argument('--data-path',required=True); p.add_argument('--run-root',required=True); p.add_argument('--seeds',required=True)
    p.add_argument('--gpu',default='0'); p.add_argument('--batch-size',type=int,default=64); p.add_argument('--workers',type=int,default=0)
    p.add_argument('--suite',default='RCCR'); p.add_argument('--out',default=None); args=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dev.type!='cuda': raise RuntimeError('CUDA required')
    seeds=parse(args.seeds,int); root=Path(args.run_root).resolve()/args.dataset/args.suite.upper()
    out=Path(args.out).resolve() if args.out else Path(args.run_root).resolve()/args.dataset/'rccr_analysis'; out.mkdir(parents=True,exist_ok=True)
    ds=MMDataset(SimpleNamespace(dataset=args.dataset,data_path=args.data_path),'test'); loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=args.workers)
    long=[]
    for var_dir in sorted(root.glob('*')):
        if not var_dir.is_dir(): continue
        variant=var_dir.name
        for seed in seeds:
            rd=var_dir/f'seed_{seed}'; ckpt=find_ckpt(rd,args.dataset)
            print('[LOAD]',variant,seed,ckpt,flush=True); model=torch.load(ckpt,map_location=dev,weights_only=False).to(dev).eval()
            rows=[]; probs_all=[]; y_all=[]
            with torch.no_grad():
                for batch in loader:
                    outs=model(batch['text'].to(dev),batch['audio'].to(dev),batch['vision'].to(dev))
                    y=batch['labels']['M'].view(-1).long().numpy(); prob=torch.softmax(outs['pred'].detach().cpu(),1).numpy(); pred=prob.argmax(1)
                    prior=outs.get('reliability_prior'); gate=outs.get('gate'); raw=outs.get('gate_raw')
                    prior=prior.detach().cpu().numpy() if prior is not None else np.full((len(y),3),np.nan)
                    gate=gate.detach().cpu().numpy() if gate is not None else np.full((len(y),3),np.nan)
                    raw=raw.detach().cpu().numpy() if raw is not None else np.full((len(y),3),np.nan)
                    exp={}
                    for m in MODS:
                        if OUT[m] in outs: exp[m]=torch.softmax(outs[OUT[m]].detach().cpu(),1).numpy()
                        else: exp[m]=np.full_like(prob,np.nan)
                    ids=[str(x) for x in batch['id']]
                    for i in range(len(y)):
                        r={'dataset':args.dataset,'suite':args.suite.upper(),'variant':variant,'seed':seed,'id':ids[i],'label':int(y[i]),'pred':int(pred[i]),'final_correct':int(pred[i]==y[i]),'final_confidence':float(prob[i].max())}
                        for j,m in enumerate(MODS):
                            ep=exp[m][i]
                            if np.isnan(ep).all(): ep_pred=-1; ep_cor=np.nan; ep_conf=np.nan
                            else: ep_pred=int(np.nanargmax(ep)); ep_cor=int(ep_pred==y[i]); ep_conf=float(np.nanmax(ep))
                            r.update({f'{m}_expert_pred':ep_pred,f'{m}_expert_correct':ep_cor,f'{m}_expert_confidence':ep_conf,f'prior_{m}':float(prior[i,j]),f'gate_{m}':float(gate[i,j]),f'gate_raw_{m}':float(raw[i,j])})
                        rows.append(r)
                    probs_all.append(prob); y_all.append(y)
            prob=np.concatenate(probs_all); y=np.concatenate(y_all); corr=(prob.argmax(1)==y).astype(float); oh=np.eye(prob.shape[1])[y]
            summary={'dataset':args.dataset,'variant':variant,'seed':seed,'n':int(len(y)),'final_acc_3':float(corr.mean()),'final_ece':ece(prob,y),'final_nll':float(-np.log(np.clip(prob[np.arange(len(y)),y],1e-12,1)).mean()),'final_brier':float(((prob-oh)**2).sum(1).mean()),'modalities':{}}
            for m in MODS:
                c=np.array([r[f'{m}_expert_correct'] for r in rows],dtype=float); mask=~np.isnan(c)
                pr=np.array([r[f'prior_{m}'] for r in rows],dtype=float); gt=np.array([r[f'gate_{m}'] for r in rows],dtype=float)
                summary['modalities'][m]={'expert_acc':float(np.nanmean(c)),'prior_auc_for_expert_correct':safe_auc(c[mask],pr[mask]) if mask.any() and not np.isnan(pr[mask]).all() else float('nan'),'prior_spearman':safe_spear(pr[mask],c[mask]) if mask.any() and not np.isnan(pr[mask]).all() else float('nan'),'gate_auc_for_expert_correct':safe_auc(c[mask],gt[mask]) if mask.any() and not np.isnan(gt[mask]).all() else float('nan')}
            base=out/variant/f'seed_{seed}'; base.mkdir(parents=True,exist_ok=True)
            (base/'rccr_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=True),encoding='utf-8')
            with (base/'rccr_per_sample.csv').open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=sorted(rows[0])); w.writeheader(); w.writerows(rows)
            long += [{'dataset':args.dataset,'suite':args.suite.upper(),'variant':variant,'seed':seed,'metric':k,'value':v} for k,v in summary.items() if isinstance(v,(int,float))]
            for m,d in summary['modalities'].items():
                for k,v in d.items(): long.append({'dataset':args.dataset,'suite':args.suite.upper(),'variant':variant,'seed':seed,'metric':f'{m}.{k}','value':v})
            del model; torch.cuda.empty_cache()
    with (out/'rccr_metrics_long.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['dataset','suite','variant','seed','metric','value']); w.writeheader(); w.writerows(long)
    print('[DONE]',out)
if __name__=='__main__': main()
