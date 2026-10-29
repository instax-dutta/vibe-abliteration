#!/usr/bin/env python3
import json
import sys
from pathlib import Path

def parse_checkpoints(checkpoint_dirs):
    trials = {}
    for cdir in checkpoint_dirs:
        for fname in Path(cdir).glob("*.jsonl"):
            try:
                with open(fname) as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            if data.get('op_code') == 8 and 'user_attr' in data and 'scores' in data['user_attr']:
                                tid = data['trial_id']
                                scores = data['user_attr']['scores']
                                refusals = None
                                kl = None
                                for s in scores:
                                    if s['name'] == 'Refusals':
                                        refusals = s['score']['value']
                                    elif s['name'] == 'KL divergence':
                                        kl = s['score']['value']
                                if tid not in trials or (kl is not None and kl < trials[tid].get('kl', float('inf'))):
                                    trials[tid] = {'kl': kl, 'refusals': refusals}
                            elif data.get('op_code') == 8 and 'user_attr' in data and 'parameters' in data['user_attr']:
                                tid = data['trial_id']
                                if tid not in trials:
                                    trials[tid] = {}
                                trials[tid]['params'] = data['user_attr']['parameters']
                        except:
                            pass
            except:
                pass
    return trials

if __name__ == '__main__':
    dirs = sys.argv[1:] or ['checkpoints', 'heretic-runs/checkpoints']
    trials = parse_checkpoints(dirs)
    results = []
    for tid, t in trials.items():
        if t.get('kl') is not None and t.get('refusals') is not None:
            results.append((tid, t['kl'], t['refusals'], t.get('params')))
    results.sort(key=lambda x: x[1])  # sort by KL
    print("Trial\tKL\tRefusals\tParams")
    for r in results[:30]:
        print(f"{r[0]}\t{r[1]:.6f}\t{r[2]*100 if r[2] <= 1 else r[2]}\t{r[3]}")
