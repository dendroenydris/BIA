"""Primary outcome: label MAD per prompt and condition, and the paired comparisons between conditions.

Label MAD per prompt x condition over the substantive labels of each attribute; `mixed`, `indeterminate`,
`no_person`, `none_or_indeterminate` and missing labels are left out of the denominator and reported as a share.
A prompt with no substantive image on an attribute is dropped from that attribute's paired tests (counted).
Paired by prompt entry; exact sign-flip p-values; Holm within the design's primary set of tests; minimum detectable
effect (MDE) from the observed paired SD (two-sided paired t, alpha 0.05, power 0.80).

  main     conditions baseline / bias_aware / llm_assisted; comparisons bias_aware-baseline, llm_assisted-baseline,
           llm_assisted-bias_aware on 5 attributes -> 15 tests
  matched  versions C (control clause) / I (instruction clause); comparison I-C on 4 attributes -> 4 tests

usage:
  python analysis/label_imbalance.py                      # every stored image set and evaluator -> results/
  python analysis/label_imbalance.py SET [--evaluator qwen3vl|b32|l14ens] [--out DIR]
  python analysis/label_imbalance.py --design main --codes LABELS.jsonl --schedule SCHEDULE.csv --out DIR [--clip]

SET is one of main_sd21, main_flux, matched_sd21, matched_flux, original_archive (paths.SETS). Qwen3-VL labels are read
from records/qwen3vl/SET.jsonl, CLIP labels (original label lists, no age attribute) from records/clip/SET_<variant>.jsonl.
Label rows: {"key": ..., "codes": {...}}; schedule rows need run_order (or key), entry, category, arm.
Keys are matched as gen<run_order:04d> unless the schedule has a `key` column.
"""
import argparse
import csv
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import CLIP_LABELS, CONCEPTS, QWEN3VL_LABELS, RESULTS, SETS  # noqa: E402

AXES = {
    'Gender Bias': [('gender', 'gender_presentation', ['woman', 'man'])],
    'Racial Bias': [('skin_tone', 'skin_tone', ['light', 'medium', 'dark']),
                    ('ethnicity', 'ethnic_appearance', ['white', 'black', 'east_asian', 'south_southeast_asian', 'middle_eastern', 'latino'])],
    'Geo-Cultural Bias': [('culture', 'cultural_scene', ['western_european', 'south_asian', 'east_asian', 'middle_eastern', 'african', 'latin_american'])],
    'Age Bias': [('age', 'age_appearance', ['child', 'teenager', 'young_adult', 'middle_aged', 'older_adult'])],
}
_C = json.loads(CONCEPTS.read_text())
CLIP_AXES = {
    'Gender Bias': [('gender', 'gender_presentation', _C['Gender Bias']['Parity']['concepts'])],
    'Racial Bias': [('skin_tone', 'skin_tone', _C['Skin Tone Bias']['Parity']['concepts']),
                    ('ethnicity', 'ethnic_appearance', _C['Ethnic Representation Bias']['Parity']['concepts'])],
    'Geo-Cultural Bias': [('culture', 'cultural_scene', _C['Geo-Cultural Bias']['Parity']['concepts'])],
    'Age Bias': [],
}
DESIGNS = {
    'main': [('bias_aware', 'baseline'), ('llm_assisted', 'baseline'), ('llm_assisted', 'bias_aware')],
    'matched': [('I', 'C')],
}


def holm(ps):
    out, prev = [0.0] * len(ps), 0.0
    for rank, i in enumerate(sorted(range(len(ps)), key=ps.__getitem__)):
        prev = max(prev, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = prev
    return out


def sign_flip(d):
    n, eff = len(d), abs(mean(d))
    return sum(abs(mean(s * x for s, x in zip(signs, d))) >= eff - 1e-12
               for signs in itertools.product((-1, 1), repeat=n)) / 2 ** n


def analyse(design, codes_path, sched_path, outdir, axes):
    from scipy.stats import t
    AXES_ = axes
    codes = {r['key']: r['codes'] for r in map(json.loads, Path(codes_path).read_text().splitlines())}
    sched = list(csv.DictReader(open(sched_path, newline='')))
    arms = sorted({a for c in DESIGNS[design] for a in c})
    cells = defaultdict(list)          # (axis, entry, arm) -> list of codes (None = missing)
    for r in sched:
        k = r.get('key') or f"gen{int(r['run_order']):04d}"
        assert r['arm'] in arms, r['arm']
        for axis, field, _ in AXES_[r['category']]:
            cells[(axis, r['entry'], r['arm'])].append(codes.get(k, {}).get(field))
    values, cell_rows = {}, []
    for (axis, e, arm), vals in sorted(cells.items()):
        labels = next(l for cat in AXES_.values() for a, _, l in cat if a == axis)
        sub = [v for v in vals if v in labels]
        share = 1 - len(sub) / len(vals)
        mad = mean(abs(sum(v == l for v in sub) / len(sub) - 1 / len(labels)) for l in labels) if sub else None
        values[(axis, e, arm)] = mad
        cell_rows.append({'axis': axis, 'entry': e, 'arm': arm, 'images': len(vals), 'substantive': len(sub),
                          'non_substantive_share': round(share, 4), 'label_mad': mad,
                          **{f'n_{l}': sum(v == l for v in sub) for l in labels},
                          'n_missing': sum(v is None for v in vals)})
    results, desc = [], []
    for axis in sorted({k[0] for k in cells}):
        entries = sorted({k[1] for k in cells if k[0] == axis}, key=int)
        for arm in arms:
            ms = [values[(axis, e, arm)] for e in entries if values[(axis, e, arm)] is not None]
            sh = [r['non_substantive_share'] for r in cell_rows if r['axis'] == axis and r['arm'] == arm]
            desc.append({'axis': axis, 'arm': arm, 'prompts': len(entries), 'prompts_with_substantive': len(ms),
                         'mean_label_mad': mean(ms) if ms else None, 'mean_non_substantive_share': mean(sh)})
        for a, b in DESIGNS[design]:
            ok = [e for e in entries if values[(axis, e, a)] is not None and values[(axis, e, b)] is not None]
            d = [values[(axis, e, a)] - values[(axis, e, b)] for e in ok]
            row = {'axis': axis, 'contrast': f'{a} minus {b}', 'prompts': len(entries), 'prompts_used': len(ok),
                   'prompts_dropped': len(entries) - len(ok)}
            if len(d) >= 2:
                n, eff, sd = len(d), mean(d), stdev(d)
                half = t.ppf(.975, n - 1) * sd / math.sqrt(n)
                mde_dz = (t.ppf(.975, n - 1) + t.ppf(.80, n - 1)) / math.sqrt(n)
                row.update({'mean_difference': eff, 'sd_difference': sd, 'cohen_dz': eff / sd if sd else None,
                            'ci95_low': eff - half, 'ci95_high': eff + half, 'p_sign_flip': sign_flip(d),
                            'mde_dz': mde_dz, 'mde_label_mad': mde_dz * sd})
            results.append(row)
    fam = [r for r in results if 'p_sign_flip' in r]
    for r, p in zip(fam, holm([r['p_sign_flip'] for r in fam])):
        r['p_holm'] = p
        r['family_size'] = len(fam)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (('comparisons.csv', results), ('condition_descriptives.csv', desc), ('cells.csv', cell_rows)):
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with open(out / name, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    print(f'{design}: {len(fam)} tests in the primary family')
    for r in results:
        print(r['axis'], r['contrast'], 'n', r['prompts_used'], 'diff', round(r.get('mean_difference', float('nan')), 4),
              'p', r.get('p_sign_flip'), 'holm', round(r.get('p_holm', float('nan')), 4), 'MDE', round(r.get('mde_label_mad', float('nan')), 4))




def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('set', nargs='?', choices=sorted(SETS))
    ap.add_argument('--evaluator', choices=['qwen3vl', 'b32', 'l14ens'], default='qwen3vl')
    ap.add_argument('--design', choices=sorted(DESIGNS))
    ap.add_argument('--codes', type=Path)
    ap.add_argument('--schedule', type=Path)
    ap.add_argument('--clip', action='store_true', help='with --codes: CLIP label lists instead of the label guide')
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    if a.codes:
        assert a.design and a.schedule and a.out, '--codes needs --design, --schedule and --out'
        return analyse(a.design, a.codes, a.schedule, a.out, CLIP_AXES if a.clip else AXES)
    jobs = [(a.set, a.evaluator)] if a.set else (
        [(s, 'qwen3vl') for s in SETS] + [(p.stem.rsplit('_', 1)[0], p.stem.rsplit('_', 1)[1]) for p in sorted(CLIP_LABELS.glob('*.jsonl'))])
    for s, ev in jobs:
        schedule, design = SETS[s]
        if ev == 'qwen3vl':
            codes, out, axes = QWEN3VL_LABELS / f'{s}.jsonl', RESULTS / 'qwen3vl' / s, AXES
        else:
            codes, out, axes = CLIP_LABELS / f'{s}_{ev}.jsonl', RESULTS / 'clip' / f'{s}_{ev}', CLIP_AXES
        print(f'== {s}, {ev}')
        analyse(design, codes, schedule, a.out or out, axes)


if __name__ == '__main__':
    main()
