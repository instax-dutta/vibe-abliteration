#!/usr/bin/env python3
"""Summarize Heretic checkpoint journals without loading the model.

Reads Optuna journal files (*.jsonl) written by Heretic and reports which
trials are worth choosing, plus the exact `--trial-index` value to pass back
to `heretic` for a headless save/upload.

Heretic numbers trials two ways:
  * `[Trial N]` label  -> user attr "index"
  * `--trial-index N`  -> position in the sorted Pareto front (both
                          objectives minimized, ordered Refusals then KL)

This script prints both so the right number gets passed to the CLI.
"""
import argparse
import json
import math
import sys
from pathlib import Path

DEFAULT_DIRS = ["checkpoints", "heretic-runs/checkpoints"]


def _num(value):
    """Return a finite float, else None. Heretic can emit NaN KL."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _find_score(scores, needle):
    for s in scores:
        name = str(s.get("name", "")).lower()
        if needle in name:
            return _num(s.get("score", {}).get("value"))
    return None


def parse_checkpoints(dirs):
    trials = {}
    stats = {"lines": 0, "malformed": 0, "records": 0, "files": 0}
    for cdir in dirs:
        path = Path(cdir)
        if not path.is_dir():
            print(f"note: checkpoint dir not found: {cdir}", file=sys.stderr)
            continue
        for fname in sorted(path.glob("*.jsonl")):
            stats["files"] += 1
            try:
                with open(fname, encoding="utf-8", errors="replace") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        stats["lines"] += 1
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            stats["malformed"] += 1
                            continue
                        attrs = data.get("user_attr")
                        tid = data.get("trial_id")
                        if data.get("op_code") != 8 or not isinstance(attrs, dict) or tid is None:
                            continue
                        stats["records"] += 1
                        t = trials.setdefault(tid, {})
                        if isinstance(attrs.get("index"), int):
                            t["index"] = attrs["index"]
                        if isinstance(attrs.get("parameters"), dict):
                            t["params"] = attrs["parameters"]
                        scores = attrs.get("scores")
                        if isinstance(scores, list):
                            kl = _find_score(scores, "kl")
                            refusals = _find_score(scores, "refus")
                            # Resumed runs can repeat a trial; keep the better KL.
                            if kl is not None and (t.get("kl") is None or kl < t["kl"]):
                                t["kl"] = kl
                                t["refusals"] = refusals
                            elif t.get("kl") is None:
                                t["refusals"] = refusals
            except OSError as exc:
                print(f"note: cannot read {fname}: {exc}", file=sys.stderr)
    return trials, stats


def pareto_ranked(trials):
    """Non-dominated trials, sorted the way Heretic sorts its menu."""
    usable = [t for t in trials.values() if t.get("kl") is not None and t.get("refusals") is not None]
    front = []
    for t in usable:
        dominated = any(
            o["refusals"] <= t["refusals"] and o["kl"] <= t["kl"]
            and (o["refusals"] < t["refusals"] or o["kl"] < t["kl"])
            for o in usable
        )
        if not dominated:
            front.append(t)
    front.sort(key=lambda t: (t["refusals"], t["kl"]))
    for position, t in enumerate(front):
        t["pareto"] = True
        t["trial_index"] = position
    return front


def record(tid, t, include_params):
    out = {
        "trial_id": tid,
        "index": t.get("index"),
        "kl": t.get("kl"),
        "refusals": t.get("refusals"),
        "pareto": bool(t.get("pareto")),
        "trial_index": t.get("trial_index"),
    }
    if include_params:
        out["params"] = t.get("params")
    return out


def main():
    ap = argparse.ArgumentParser(description="Report the best Heretic trials from checkpoint journals.")
    ap.add_argument("dirs", nargs="*", default=DEFAULT_DIRS,
                    help=f"checkpoint directories (default: {' '.join(DEFAULT_DIRS)})")
    ap.add_argument("--limit", type=int, default=5, help="rows per ranking (default: 5)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    trials, stats = parse_checkpoints(args.dirs)
    if stats["malformed"]:
        print(f"note: skipped {stats['malformed']} malformed line(s)", file=sys.stderr)

    complete = {tid: t for tid, t in trials.items()
                if t.get("kl") is not None and t.get("refusals") is not None}
    if not complete:
        print(f"No complete trials found in: {', '.join(args.dirs)}", file=sys.stderr)
        if stats["records"]:
            print(f"read {stats['records']} trial record(s) but none had both "
                  "'Refusals' and 'KL divergence' scores", file=sys.stderr)
        print("Pass the directory used by --study-checkpoint-dir.", file=sys.stderr)
        return 1

    front = pareto_ranked(complete)
    by_kl = sorted(complete.items(), key=lambda kv: (kv[1]["kl"], kv[1]["refusals"]))
    by_ref = sorted(complete.items(), key=lambda kv: (kv[1]["refusals"], kv[1]["kl"]))
    limit = max(1, args.limit)

    if args.json:
        json.dump({
            "counts": {"trials": len(complete), "pareto": len(front),
                       "files": stats["files"], "malformed_lines": stats["malformed"]},
            "best_by_kl": record(*by_kl[0], True),
            "best_by_refusals": record(*by_ref[0], True),
            "pareto": [record(kv[0], kv[1], True) for kv in
                       sorted(complete.items(), key=lambda kv: (kv[1].get("trial_index") is None, kv[1].get("trial_index", 0)))],
            "top_by_kl": [record(*kv, False) for kv in by_kl[:limit]],
            "top_by_refusals": [record(*kv, False) for kv in by_ref[:limit]],
        }, sys.stdout, indent=2)
        print()
        return 0

    print(f"Parsed {len(complete)} complete trial(s) from {stats['files']} file(s)\n")

    print(f"Top {limit} by KL divergence:")
    for tid, t in by_kl[:limit]:
        print(f"  trial {tid}: KL={t['kl']:.6f} refusals={t['refusals']:.3f}")
    print(f"\nTop {limit} by refusals:")
    for tid, t in by_ref[:limit]:
        print(f"  trial {tid}: refusals={t['refusals']:.3f} KL={t['kl']:.6f}")

    print(f"\nPareto front ({len(front)} trial(s)) - pass these to --trial-index:")
    if not front:
        print("  (none)")
    for t in front:
        label = f"[Trial {t['index']}]" if t.get("index") is not None else f"trial {t.get('trial_id')}"
        print(f"  --trial-index {t['trial_index']}  {label} "
              f"refusals={t['refusals']:.3f} KL={t['kl']:.6f}")
        if t.get("params"):
            print(f"      params: {json.dumps(t['params'], sort_keys=True)}")

    for label, (tid, t) in (("KL", by_kl[0]), ("refusals", by_ref[0])):
        where = (f"--trial-index {t['trial_index']}" if t.get("trial_index") is not None
                 else "not on the Pareto front, so not selectable by --trial-index")
        print(f"\nBest by {label}: trial {tid} ({where})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
