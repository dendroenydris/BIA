"""Exploratory: baseline prompt-level label MAD (Qwen3-VL), FLUX.1-schnell minus SD 2.1, paired by prompt (Section 4.3).

Exact two-sided sign-flip p over all sign patterns; Holm over the five attributes.
Reads results/qwen3vl/main_{sd21,flux}/cells.csv; writes results/model_baseline/comparison.csv.

usage: python analysis/model_baseline_comparison.py
"""
import csv
import itertools
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import RESULTS as A  # noqa: E402

RES = A / 'qwen3vl'


def baseline(run):
    return {(r['axis'], r['entry']): float(r['label_mad']) for r in csv.DictReader(open(RES / run / 'cells.csv'))
            if r['arm'] == 'baseline' and r['label_mad']}


sd, fx = baseline('main_sd21'), baseline('main_flux')
rows = []
for ax in ['gender', 'skin_tone', 'ethnicity', 'culture', 'age']:
    d = [fx[k] - sd[k] for k in sd if k[0] == ax and k in fx]
    m = sum(d) / len(d)
    pats = list(itertools.product([1, -1], repeat=len(d)))
    p = sum(abs(sum(s * x for s, x in zip(sg, d)) / len(d)) >= abs(m) - 1e-12 for sg in pats) / len(pats)
    rows.append({'attribute': ax, 'prompts': len(d), 'mean_difference': round(m, 4), 'sd_difference': round(st.stdev(d), 4),
                 'flux_higher': sum(x > 0 for x in d), 'p_sign_flip': round(p, 4)})
order = sorted(range(len(rows)), key=lambda i: rows[i]['p_sign_flip'])
prev = 0
for rank, i in enumerate(order):
    prev = max(prev, min(1, (len(rows) - rank) * rows[i]['p_sign_flip']))
    rows[i]['p_holm'] = round(prev, 4)
(A / 'model_baseline').mkdir(exist_ok=True)
with open(A / 'model_baseline/comparison.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(r)
