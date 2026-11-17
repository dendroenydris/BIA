"""FLUX.1-schnell human evaluation (Appendix E.5): agreement of the two raters with each other and of Qwen3-VL with
each rater on the 60 FLUX.1-schnell images F01-F60 of the main experiment.

Both raters followed the label guide, so both sheets are normalised with human_agreement.norm_A (vocabulary slips only);
any value outside the label guide stops the script and is listed.

Inputs: human_ratings/flux/{rater_1.csv, rater_2.csv, key.csv}, records/qwen3vl/main_flux.jsonl.
usage: python analysis/flux_human_agreement.py   (writes results/agreement/flux_raters/agreement.json)
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import FLUX_RATINGS, QWEN3VL_LABELS, RESULTS  # noqa: E402
import analysis.human_agreement as H  # noqa: E402

OUT = RESULTS / 'agreement' / 'flux_raters'
KEY = FLUX_RATINGS / 'key.csv'
CODES = QWEN3VL_LABELS / 'main_flux.jsonl'
FIELDS = ['person_count', 'gender_presentation', 'skin_tone', 'ethnic_appearance', 'age_appearance', 'cultural_scene']
ALLOWED = {'person_count': {'0', '1', '2', '3plus'},
           'gender_presentation': {'woman', 'man', 'mixed', 'indeterminate', 'no_person'},
           'skin_tone': {'light', 'medium', 'dark', 'mixed', 'indeterminate', 'no_person'},
           'ethnic_appearance': {'white', 'black', 'east_asian', 'south_southeast_asian', 'middle_eastern', 'latino', 'mixed', 'indeterminate', 'no_person'},
           'age_appearance': {'child', 'teenager', 'young_adult', 'middle_aged', 'older_adult', 'mixed', 'indeterminate', 'no_person'},
           'cultural_scene': {'western_european', 'south_asian', 'east_asian', 'middle_eastern', 'african', 'latin_american', 'mixed', 'none_or_indeterminate'}}


def load(path):
    rows = H.load(path, H.norm_A)
    bad = [(i, f, r[f]) for i, r in rows.items() for f in FIELDS if r[f] not in ALLOWED[f]]
    if bad:
        sys.exit('values outside the label guide:\n' + '\n'.join(map(str, bad)))
    return rows


def main():
    ra, rb = load(FLUX_RATINGS / 'rater_1.csv'), load(FLUX_RATINGS / 'rater_2.csv')
    key = {r['image_id']: r for r in csv.DictReader(open(KEY, newline=''))}
    vlm = {r['key']: r['codes'] for r in map(json.loads, CODES.read_text().splitlines())}
    q = {i: vlm[f"gen{int(k['run_order']):04d}"] for i, k in key.items()}
    assert set(ra) == set(rb) == set(key), 'sheets must cover F01-F60'
    strata = {i: (key[i]['entry'], key[i]['condition']) for i in key}
    out = []
    for f in FIELDS:
        ids = sorted(key)
        def k(x, y):
            po, kk = H.kappa([x[i][f] for i in ids], [y[i][f] for i in ids])
            lo, hi = H.boot_ci([x[i][f] for i in ids], [y[i][f] for i in ids], [strata[i] for i in ids])
            return round(po, 3), round(kk, 3), lo, hi
        out.append({'field': f, 'raters': k(ra, rb), 'vlm_A': k(q, ra), 'vlm_B': k(q, rb)})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'agreement.json').write_text(json.dumps(out, indent=2))
    for r in out:
        print(r['field'], 'raters κ', r['raters'][1], 'VLM-A', r['vlm_A'][1], 'VLM-B', r['vlm_B'][1])
    dis = [i for i in key if ra[i]['cultural_scene'] != rb[i]['cultural_scene']]
    print(f"cultural setting: {len(dis)} disagreements, of which "
          f"{sum({ra[i]['cultural_scene'], rb[i]['cultural_scene']} == {'none_or_indeterminate', 'western_european'} for i in dis)} "
          "are indeterminate versus Western European")


if __name__ == '__main__':
    main()
