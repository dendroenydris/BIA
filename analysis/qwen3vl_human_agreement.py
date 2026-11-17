"""Agreement of the primary evaluator (Qwen3-VL) with the two raters on the 60 human-labelled archive images (plan/PLAN.md).

Both use the label-guide vocabulary, so no mapping: `mixed`, `indeterminate`, `no_person` are ordinary categories.
Per field: kappa vs rater A and vs rater B (bootstrap CI as human_agreement.py), their mean (ethnic appearance: rater A
only), the rater-rater kappa, and the plan's reference bound, which guides interpretation only.

usage: python analysis/qwen3vl_human_agreement.py [LABELS.jsonl] [--first-parser]
  LABELS.jsonl defaults to records/qwen3vl/human_labelled_60.jsonl.
  --first-parser uses the labels of the live run (`codes_live`, before the parser also accepted unquoted values;
  two images had no valid label) and writes results/agreement/qwen3vl_vs_raters_first_parser.
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import ARCHIVE_RATINGS as HUM, QWEN3VL_LABELS, RESULTS  # noqa: E402
import analysis.human_agreement as A  # noqa: E402

FIRST = '--first-parser' in sys.argv
OUT = RESULTS / 'agreement' / ('qwen3vl_vs_raters_first_parser' if FIRST else 'qwen3vl_vs_raters')
FIELDS = ['gender_presentation', 'skin_tone', 'ethnic_appearance', 'age_appearance', 'cultural_scene', 'person_count']
BOUND = {'gender_presentation': 0.650, 'skin_tone': 0.347, 'ethnic_appearance': 0.347, 'age_appearance': 0.219,
         'cultural_scene': 0.382, 'person_count': None}


def main():
    args = [x for x in sys.argv[1:] if x != '--first-parser']
    src = Path(args[0]) if args else QWEN3VL_LABELS / 'human_labelled_60.jsonl'
    field = 'codes_live' if FIRST else 'codes'
    codes = {r['key']: r[field] for r in map(json.loads, src.read_text().splitlines())}
    key = {r['image_id']: r for r in csv.DictReader(open(HUM / 'key.csv', newline=''))}
    assert set(codes) == set(key), 'codes must cover exactly the 60 human-coded images'
    hA = A.load(HUM / 'rater_A.csv', A.norm_A)
    hB = A.load(HUM / 'rater_B.csv', A.norm_B)
    rows = []
    for f in FIELDS:
        ks = {}
        row = {'field': f}
        for name, h in (('A', hA), ('B', hB)):
            ids = sorted(i for i in key if h[i][f] and codes[i][f] is not None)
            if not ids:
                continue
            v, x = [codes[i][f] for i in ids], [h[i][f] for i in ids]
            po, k = A.kappa(v, x)
            lo, hi = A.boot_ci(v, x, [(key[i]['entry'], key[i]['condition']) for i in ids])
            row.update({f'n_{name}': len(ids), f'agree_{name}': round(po, 4),
                        f'kappa_{name}': None if k is None else round(k, 4), f'ci_{name}': f'[{lo}, {hi}]'})
            ks[name] = k
        both = [k for k in ks.values() if k is not None]
        row['evaluator_human_kappa'] = round(sum(both) / len(both), 4) if both else None
        hh = [i for i in key if hA[i][f] and hB[i][f]]
        row['human_human_kappa'] = None if f == 'ethnic_appearance' else round(A.kappa([hA[i][f] for i in hh], [hB[i][f] for i in hh])[1], 4)
        row['reference_bound'] = BOUND[f]
        row['meets_reference'] = (None if BOUND[f] is None or row['evaluator_human_kappa'] is None
                                  else row['evaluator_human_kappa'] >= BOUND[f])
        row['missing_codes'] = sum(codes[i][f] is None for i in key)
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / 'agreement.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with open(OUT / 'per_image.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['image_id', 'entry', 'condition'] + [f'vlm_{f}' for f in FIELDS] + [f'A_{f}' for f in FIELDS] + [f'B_{f}' for f in FIELDS])
        for i in sorted(key):
            w.writerow([i, key[i]['entry'], key[i]['condition']] + [codes[i][f] for f in FIELDS]
                       + [hA[i][f] for f in FIELDS] + [hB[i][f] for f in FIELDS])
    for r in rows:
        print(r['field'], 'VLM-human', r['evaluator_human_kappa'], '| human-human', r['human_human_kappa'],
              '| bound', r['reference_bound'], '| meets', r['meets_reference'], '| missing', r['missing_codes'])


if __name__ == '__main__':
    main()
