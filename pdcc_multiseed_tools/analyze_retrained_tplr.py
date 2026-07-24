#!/usr/bin/env python3
"""Analyze pseudo-label quality in retrained diagnostic run directories."""
from __future__ import annotations
import argparse, csv, json, pickle
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from dataloader import MMDataset
MODS=("T","A","V")

def parse(s,cast=str): return [cast(x.strip()) for x in s.split(',') if x.strip()]
def ece(prob,y,bins=15):
    conf=prob.max(1); pred=prob.argmax(1); cor=(pred==y).astype(float); val=0.0
    edges=np.linspace(0,1,bins+1)
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(conf>=lo)&((conf<hi) if hi<1 else (conf<=hi))
        if m.any(): val += abs(float(cor[m].mean())-float(conf[m].mean()))*float(m.mean())
    return float(val)
def summarize(prob,y):
    pred=prob.argmax(1); conf=prob.max(1); q=np.clip(prob,1e-12,1); ent=-(q*np.log(q)).sum(1)
    oh=np.eye(prob.shape[1])[y]
    return {"agreement":float((pred==y).mean()),"mean_confidence":float(conf.mean()),
            "mean_entropy":float(ent.mean()),"ece":ece(prob,y),"brier":float(((prob-oh)**2).sum(1).mean()),"n":int(len(y))}
def labels(dataset,data_path,split):
    ds=MMDataset(SimpleNamespace(dataset=dataset,data_path=data_path), split)
    out={}
    for i,sid in enumerate(ds.ids):
        if dataset=="SIMS": out[str(sid)]={m:int(ds.labels[m][i]) for m in MODS}
        else:
            y=int(ds.labels["M"][i]); out[str(sid)]={m:y for m in MODS}
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True,choices=['SIMS','MOSI'])
    p.add_argument('--data-path',required=True); p.add_argument('--run-root',required=True)
    p.add_argument('--seeds',required=True); p.add_argument('--out',default=None)
    p.add_argument('--suite',default='TPLR')
    args=p.parse_args(); root=Path(args.run_root).resolve()/args.dataset/args.suite.upper(); seeds=parse(args.seeds,int)
    rows=[]; hist=[]
    for variant_dir in sorted(root.glob('*')):
        if not variant_dir.is_dir(): continue
        variant=variant_dir.name
        for seed in seeds:
            rd=variant_dir/f'seed_{seed}'
            for split in ('train','valid','test'):
                lab=labels(args.dataset,args.data_path,split)
                files=list((rd/'pseudo_labels').glob(f'*_{split}_pseudo_labels.pkl'))
                if not files: continue
                bank=pickle.loads(files[0].read_bytes())
                for mod in MODS:
                    probs=[]; ys=[]
                    for sid,val in bank.items():
                        k=str(sid)
                        if k not in lab: continue
                        x=val[mod]
                        if hasattr(x,'detach'): x=x.detach().cpu().numpy()
                        x=np.asarray(x,dtype=float)
                        probs.append(x); ys.append(lab[k][mod])
                    if probs:
                        row=summarize(np.stack(probs),np.asarray(ys)); row.update({"dataset":args.dataset,"suite":args.suite.upper(),"variant":variant,"seed":seed,"split":split,"modality":mod,"label_source":"modality_specific" if args.dataset=='SIMS' else 'multimodal_proxy'})
                        rows.append(row)
            for hp in (rd/'pseudo_labels').glob('*entropy_history*.csv'):
                with hp.open(encoding='utf-8-sig') as f:
                    for r in csv.DictReader(f):
                        r.update({"dataset":args.dataset,"suite":args.suite.upper(),"variant":variant,"seed":seed,"history_path":str(hp)})
                        hist.append(r)
    out=Path(args.out).resolve() if args.out else Path(args.run_root).resolve()/args.dataset/'tplr_analysis'
    out.mkdir(parents=True,exist_ok=True)
    for name,data in [('tplr_final_quality.csv',rows),('tplr_entropy_history.csv',hist)]:
        if data:
            with (out/name).open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=sorted({k for r in data for k in r})); w.writeheader(); w.writerows(data)
    print('[DONE]',out)
if __name__=='__main__': main()
