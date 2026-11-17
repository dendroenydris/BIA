"""Human evaluation of the original archive: agreement between the two raters on the 60 images H01-H60, and their
label counts next to the original consensus labels (Table 1 of the paper).

Inputs (raw rating sheets as submitted): human_ratings/original_archive/{rater_A.csv, rater_B.csv, key.csv}.
Outputs (results/agreement/archive_raters): kappa_results.csv, sensitivity_kappa.csv, cell_counts_vs_table1.csv,
disagreements.csv, normalisation_log.csv. Also imported by the other agreement scripts (normalisation, kappa,
bootstrap interval). Standard library only.

usage: python analysis/human_agreement.py
"""
import csv
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import ARCHIVE_RATINGS, RESULTS  # noqa: E402

H = ARCHIVE_RATINGS
OUT = RESULTS / 'agreement' / 'archive_raters'
FIELDS = ['person_count', 'gender_presentation', 'skin_tone', 'ethnic_appearance', 'age_appearance', 'cultural_scene']
CODED = [f for f in FIELDS if f != 'ethnic_appearance']  # Rater B left ethnic_appearance empty
PRIMARY = {'gender_presentation': 'A10', 'skin_tone': 'A12', 'age_appearance': 'A32', 'cultural_scene': 'A33'}
CONDS = ['baseline', 'bias_aware', 'llm_assisted']
SEED, NBOOT = 20261017, 10000
log = []


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in ca) / n ** 2
    return po, (None if pe == 1 else (po - pe) / (1 - pe))


def boot_ci(a, b, strata):
    """Percentile CI from paired image resamples within entry x condition strata."""
    rnd = random.Random(SEED)
    groups = {}
    for i, s in enumerate(strata):
        groups.setdefault(s, []).append(i)
    ks = []
    for _ in range(NBOOT):
        idx = [rnd.choice(g) for g in groups.values() for _ in g]
        k = kappa([a[i] for i in idx], [b[i] for i in idx])[1]
        if k is not None:
            ks.append(k)
    ks.sort()
    return (round(ks[int(.025 * len(ks))], 3), round(ks[int(.975 * len(ks)) - 1], 3)) if ks else ('', '')


def toks(v):
    return [t.strip().lower() for t in re.split(r'[,;|]', v) if t.strip()]


def collapse(values, rater, iid, field, raw, drop_indet=False):
    """Label-guide rule: one distinct value -> that value; several -> 'mixed'.
    drop_indet (sensitivity): ignore 'indeterminate' when a substantive value is also present."""
    s = set(values)
    if drop_indet and len(s) > 1:
        s.discard('indeterminate')
    out = s.pop() if len(s) == 1 else ('mixed' if s else '')
    if not drop_indet and out != raw.strip().lower():
        log.append((rater, iid, field, raw, out))
    return out


def count_code(v):
    v = v.strip().lower()
    if v.startswith('indeterm') or v == 'unresolved':
        return 'indeterminate'
    n = int(float(v))
    return '3plus' if n >= 3 else str(n)


FIX_A = {('skin_tone', 'white'): 'light', ('ethnic_appearance', 'south_southeast_asia'): 'south_southeast_asian',
         ('cultural_scene', 'indeterminate'): 'none_or_indeterminate'}


def norm_A(r, drop_indet=False):  # Rater A followed the label guide; fix vocabulary slips only
    o = {}
    for f in FIELDS:
        v = r[f].strip().lower().rstrip(',')
        v = FIX_A.get((f, v), v)
        if not drop_indet and v != r[f].strip().lower():
            log.append(('A', r['image_id'], f, r[f], v))
        o[f] = v
    return o


B_MAP = {'woman_presenting': 'woman', 'man_presenting': 'man', '6 male': 'man', 'young_adullt': 'young_adult',
         'european_cues': 'western_european', 'north_american_cues': 'western_european', 'african_cues': 'african',
         'asian_cues': 'asian', 'other_cues': 'other', 'not_applicable': 'none_or_indeterminate'}


def norm_B(r, drop_indet=False):  # Rater B: multi-value cells and own vocabulary
    i = r['image_id']
    o = {'person_count': count_code(r['person_count']), 'ethnic_appearance': ''}
    for f in ['gender_presentation', 'skin_tone', 'age_appearance', 'cultural_scene']:
        vals = [B_MAP.get(t, t) for t in toks(r[f])]
        if f == 'cultural_scene':
            vals = ['none_or_indeterminate' if t == 'indeterminate' else t for t in vals]
        o[f] = collapse(vals, 'B', i, f, r[f], drop_indet)
    return o


def load(p, fn, **kw):
    return {r['image_id'].strip(): fn(r, **kw) for r in csv.DictReader(open(p, newline='', encoding='utf-8-sig'))}


def table(rows, path):
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    key = {r['image_id']: r for r in csv.DictReader(open(H / 'key.csv'))}
    ids = sorted(key)
    A = load(H / 'rater_A.csv', norm_A)
    B = load(H / 'rater_B.csv', norm_B)
    assert set(A) == set(B) == set(key) and len(ids) == 60
    mult = lambda r: 'no_person' if r['person_count'] == '0' else (
        'indeterminate' if r['person_count'] == 'indeterminate' else ('yes' if r['person_count'] in ('2', '3plus') else 'no'))

    def agree_rows(X, Y, fields, ci=True):
        out = []
        for f in fields:
            subs = [('all 60', ids)] + ([(f'{PRIMARY[f]} only', [i for i in ids if key[i]['entry'] == PRIMARY[f]])]
                                       if f in PRIMARY else [])
            for lab, sub in subs:
                g = mult if f == 'multiple_persons' else (lambda r, f=f: r[f])
                a, b = [g(X[i]) for i in sub], [g(Y[i]) for i in sub]
                po, k = kappa(a, b)
                lo, hi = boot_ci(a, b, [(key[i]['entry'], key[i]['condition']) for i in sub]) if ci else ('', '')
                out.append(dict(field=f, subset=lab, n=len(sub), agree=sum(x == y for x, y in zip(a, b)),
                                agreement=round(po, 3), kappa='undefined' if k is None else round(k, 3),
                                ci_low=lo, ci_high=hi))
        return out

    main_rows = agree_rows(A, B, CODED + ['multiple_persons'])
    OUT.mkdir(parents=True, exist_ok=True)
    table(main_rows, OUT / 'kappa_results.csv')

    # Sensitivity: drop 'indeterminate' when another value is present in the same cell
    A2 = load(H / 'rater_A.csv', norm_A, drop_indet=True)
    B2 = load(H / 'rater_B.csv', norm_B, drop_indet=True)
    sens = agree_rows(A2, B2, CODED, ci=False)
    # Sensitivity: exclude images either rater flagged as black-and-white / blurry / hands only
    rawB = {r['image_id']: r for r in csv.DictReader(open(H / 'rater_B.csv', encoding='utf-8-sig'))}
    rawA = {r['image_id']: r for r in csv.DictReader(open(H / 'rater_A.csv', encoding='utf-8-sig'))}
    flagged = {i for i in ids if (rawA[i]['notes'] + rawB[i]['notes']).strip()}
    clean = [i for i in ids if i not in flagged]
    for f in CODED:
        a, b = [A[i][f] for i in clean], [B[i][f] for i in clean]
        po, k = kappa(a, b)
        sens.append(dict(field=f, subset=f'excluding {len(flagged)} flagged images', n=len(clean),
                         agree=sum(x == y for x, y in zip(a, b)), agreement=round(po, 3),
                         kappa='undefined' if k is None else round(k, 3), ci_low='', ci_high=''))
    for r in sens[:len(sens) - len(CODED)]:
        r['subset'] = 'indeterminate dropped when combined; ' + r['subset']
    table(sens, OUT / 'sensitivity_kappa.csv')

    # Per-cell label counts next to the consensus Table 1
    T1 = {('A12', 'baseline'): 'White 5', ('A12', 'bias_aware'): 'Black 5',
          ('A12', 'llm_assisted'): 'South/SE Asian 2; Black 1; Middle Eastern 1; mixed 1',
          ('A10', 'baseline'): 'man 3; woman 2', ('A10', 'bias_aware'): 'man 3; woman 2',
          ('A10', 'llm_assisted'): 'woman 2; mixed 3',
          ('A32', 'baseline'): 'child 2; young adult 2; indeterminate 1',
          ('A32', 'bias_aware'): 'middle-aged 2; young adult 1; indeterminate 2',
          ('A32', 'llm_assisted'): 'teenager 4; young adult 1',
          ('A33', 'baseline'): 'Western/European 5', ('A33', 'bias_aware'): 'Western/European 2; South Asian 2; African 1',
          ('A33', 'llm_assisted'): 'African 2; South Asian 1; mixed 2'}
    cells = []
    for e, f in [('A12', 'ethnic_appearance'), ('A12', 'skin_tone'), ('A10', 'gender_presentation'),
                 ('A10', 'person_count'), ('A32', 'age_appearance'), ('A33', 'cultural_scene')]:
        for c in CONDS:
            sub = [i for i in ids if key[i]['entry'] == e and key[i]['condition'] == c]
            fmt = lambda R: '; '.join(f'{k} {v}' for k, v in sorted(Counter((R[i].get(f) or 'not coded') for i in sub).items(),
                                                                     key=lambda t: (-t[1], t[0])))
            cells.append(dict(entry=e, field=f, condition=c, table1_consensus=T1[(e, c)],
                              rater_A=fmt(A), rater_B=fmt(B)))
    table(cells, OUT / 'cell_counts_vs_table1.csv')

    with open(OUT / 'disagreements.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['image_id', 'entry', 'condition', 'field', 'rater_A', 'rater_B', 'consensus', 'reason'])
        for i in ids:
            for f in CODED:
                if A[i][f] != B[i][f]:
                    w.writerow([i, key[i]['entry'], key[i]['condition'], f, A[i][f], B[i][f], '', ''])
    with open(OUT / 'normalisation_log.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['rater', 'image_id', 'field', 'submitted', 'normalised'])
        w.writerows(log)
    for p in ['kappa_results.csv', 'sensitivity_kappa.csv', 'cell_counts_vs_table1.csv']:
        print(f'== {p}\n' + (OUT / p).read_text())
    print(len(flagged), 'flagged images:', sorted(flagged))


if __name__ == '__main__':
    assert kappa(['a', 'b'], ['a', 'b']) == (1.0, 1.0) and kappa(['a'] * 3, ['a'] * 3)[1] is None
    main()
