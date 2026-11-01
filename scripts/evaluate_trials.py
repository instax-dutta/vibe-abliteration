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
                                    trials.setdefault(tid, {}).update({'kl': kl, 'refusals': refusals})
                            elif data.get('op_code') == 8 and 'user_attr' in data and 'parameters' in data['user_attr']:
                                tid = data['trial_id']
                                if tid not in trials:
                                    trials[tid] = {}
                                trials[tid]['params'] = data['user_attr']['parameters']
                        except Exception:
                            pass
            except Exception:
                pass
    return trials

def best_by(trials, key):
    ranked = [t for t in trials.items() if t[1].get(key) is not None]
    if not ranked:
        return None
    return min(ranked, key=lambda t: t[1][key])

if __name__ == '__main__':
    dirs = sys.argv[1:] or ['checkpoints', 'heretic-runs/checkpoints']
    trials = parse_checkpoints(dirs)
    if not trials:
        print("No trials found in:", dirs)
        sys.exit(1)

    print(f"Parsed {len(trials)} trials\n")
    print("Top 5 by KL divergence:")
    for tid, d in sorted(trials.items(), key=lambda t: t[1].get('kl', float('inf')))[:5]:
        print(f"  trial {tid}: KL={d.get('kl')} refusals={d.get('refusals')}")
    print("\nTop 5 by refusals:")
    for tid, d in sorted(trials.items(), key=lambda t: t[1].get('refusals', float('inf')))[:5]:
        print(f"  trial {tid}: refusals={d.get('refusals')} KL={d.get('kl')}")

    bk = best_by(trials, 'kl')
    br = best_by(trials, 'refusals')
    if bk:
        print(f"\nBest by KL: trial {bk[0]}  KL={bk[1].get('kl')} refusals={bk[1].get('refusals')}")
        if 'params' in bk[1]:
            print("  params:", json.dumps(bk[1]['params']))
    if br:
        print(f"Best by refusals: trial {br[0]}  refusals={br[1].get('refusals')} KL={br[1].get('kl')}")
        if 'params' in br[1]:
            print("  params:", json.dumps(br[1]['params']))
