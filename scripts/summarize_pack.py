"""Summarize pilot_pack logs: per-config aggregate env steps/s, ms/grad-step, mean SM%.
Usage: python scripts/summarize_pack.py logs/pilot/pack_<mode>_<jobid> [...]"""
import glob, json, os, sys

WARMUP_S = 60  # skip the first minute of training progress (startup, cudnn/alloc warmup)


def run_rate(path):
    pts = [json.loads(l.split(' ', 1)[1]) for l in open(path, errors='ignore') if l.startswith('PILOT_PROGRESS')]
    if len(pts) < 3:
        return None
    t0 = pts[0]['t'] + WARMUP_S
    win = [p for p in pts if p['t'] >= t0] or pts[-2:]
    a, b = win[0], win[-1]
    if b['t'] <= a['t']:
        return None
    dt = b['t'] - a['t']
    return dict(rate=(b['env_steps'] - a['env_steps']) / dt,
                ms_per_grad=1000 * dt / max(b['n_grad_steps'] - a['n_grad_steps'], 1),
                train_frac=(b['train_s'] - a['train_s']) / dt, t0=a['t'], t1=b['t'])


def mean_sm(path, skip_rows=24):
    rows = [l.split() for l in open(path) if not l.startswith('#') and l.strip()]
    vals = [int(r[2]) for r in rows[skip_rows:] if len(r) > 3 and r[2].isdigit()]
    vals = [v for v in vals if v > 0] or [0]
    return sum(vals) / len(vals)


for d in sys.argv[1:]:
    print(f'== {d}')
    print('N  agg_env_steps/s  per_run  ms/grad(wall)  train_frac  meanSM%  runs_ok')
    for nd in sorted(glob.glob(os.path.join(d, 'n*')), key=lambda x: int(os.path.basename(x)[1:])):
        n = int(os.path.basename(nd)[1:])
        rs = [r for r in (run_rate(p) for p in sorted(glob.glob(os.path.join(nd, 'run*.log')))) if r]
        if not rs:
            print(n, 'no data'); continue
        agg = sum(r['rate'] for r in rs)
        print(f"{n:<2} {agg:15.1f}  {agg/len(rs):7.1f}  {sum(r['ms_per_grad'] for r in rs)/len(rs):12.1f}  "
              f"{sum(r['train_frac'] for r in rs)/len(rs):9.2f}  {mean_sm(os.path.join(nd, 'dmon.txt')):6.1f}  {len(rs)}/{n}")
