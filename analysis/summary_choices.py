"""Section 4.5: how two summary choices change which condition looks most balanced in the main experiment.

For every image set, evaluator, attribute and condition: label MAD of the labels pooled over all prompts, and the mean
prompt-level label MAD, each computed over all configured labels and over the labels that occur only. Reads the
prompt-level label counts in results/<evaluator>/<set>/cells.csv and writes results/summary_choices/summary_choices.csv.

usage: python analysis/summary_choices.py
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import RESULTS  # noqa: E402

RUNS = ['qwen3vl/main_sd21', 'qwen3vl/main_flux', 'clip/main_sd21_b32', 'clip/main_flux_b32',
        'clip/main_sd21_l14ens', 'clip/main_flux_l14ens']
ARMS = ['baseline', 'bias_aware', 'llm_assisted']


def mad(counts, observed_only=False):
    labels = [k for k, v in counts.items() if v > 0 or not observed_only]
    n = sum(counts[k] for k in labels)
    if not labels or n == 0:
        return None
    k = len(labels)
    return sum(abs(counts[l] / n - 1 / k) for l in labels) / k


def main():
    out = []
    for run in RUNS:
        rows = list(csv.DictReader(open(RESULTS / run / 'cells.csv', newline='')))
        for axis in dict.fromkeys(r['axis'] for r in rows):
            for arm in ARMS:
                cells = [r for r in rows if r['axis'] == axis and r['arm'] == arm]
                if not cells:
                    continue
                labels = [c[2:] for c in cells[0] if c.startswith('n_') and c != 'n_missing' and cells[0][c] != '']
                pooled, per_all, per_obs = defaultdict(int), [], []
                for r in cells:
                    counts = {l: int(r['n_' + l] or 0) for l in labels}
                    for l in labels:
                        pooled[l] += counts[l]
                    if mad(counts) is not None:
                        per_all.append(mad(counts))
                        per_obs.append(mad(counts, True))
                out.append({'set': run, 'attribute': axis, 'condition': arm, 'prompts': len(per_all),
                            'pooled_mad_all_labels': round(mad(pooled), 4), 'pooled_mad_observed_labels': round(mad(pooled, True), 4),
                            'mean_prompt_mad_all_labels': round(sum(per_all) / len(per_all), 4),
                            'mean_prompt_mad_observed_labels': round(sum(per_obs) / len(per_obs), 4),
                            'labels_absent_when_pooled': ';'.join(l for l in labels if pooled[l] == 0)})
    dest = RESULTS / 'summary_choices'
    dest.mkdir(exist_ok=True)
    with open(dest / 'summary_choices.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(dest / 'summary_choices.csv', len(out), 'rows')


if __name__ == '__main__':
    main()
