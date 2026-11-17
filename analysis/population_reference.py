"""Sensitivity analysis: label imbalance against a world-population reference instead of the uniform reference.

MAD_pop = (1/K) * sum_k |p(k) - q(k)| over all K substantive labels, with q from UN World Population Prospects 2024
(medium variant, 1 July 2024): regional populations for ethnic appearance and cultural setting, the world age structure
for apparent age. Gender keeps q = 1/2; skin tone keeps the uniform reference (no world statistic). Region-to-label
mapping is an approximation: populations are counted by region of residence, the labels describe perceived appearance
or scene cues.

A second variant ("us") uses U.S. references: for occupational gender prompts the share of women in that occupation
(BLS CPS Table 11), 1/2 for the other gender prompts; for ethnic appearance the 2020 Census composition (the "scientist"
prompt uses the BLS science occupations); the U.S. age structure (WPP 2024). Skin tone and cultural setting keep the
uniform reference. Sources and the mapping are in datasets/population_reference/us_sources.json.

Inputs: results/qwen3vl/<set>/cells.csv. The world and U.S. age and regional references are computed from the UN WPP 2024
files WPP2024_Demographic_Indicators_Medium.csv.gz and WPP2024_PopulationByAge5GroupSex_Medium.csv.gz
(https://population.un.org/wpp/downloads, CSV format), passed with --wpp-dir; they are not stored here. Without
--wpp-dir the stored references results/population_reference/reference_{world,us}.json are used.

usage: python analysis/population_reference.py [--wpp-dir DIR]
  writes results/population_reference/{reference_*.json, <variant>_<set>_contrasts.csv, summary_*.csv}
"""
import argparse
import csv
import gzip
import itertools
import json
import statistics as st
import sys
from pathlib import Path

from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import RESULTS as A, US_SOURCES  # noqa: E402

OUT = A / 'population_reference'
SRC = None  # folder with the WPP 2024 files (--wpp-dir)
YEAR = '2024'


def regions():
    want = {'Eastern Asia', 'South-Eastern Asia', 'Southern Asia', 'Central Asia', 'Western Asia', 'Northern Africa',
            'Sub-Saharan Africa', 'Latin America and the Caribbean', 'Europe', 'Northern America', 'Australia/New Zealand',
            'Iran (Islamic Republic of)', 'World'}
    pop = {}
    for x in csv.DictReader(gzip.open(SRC / 'WPP2024_Demographic_Indicators_Medium.csv.gz', 'rt', encoding='utf-8')):
        if x['Time'] == YEAR and x['Variant'] == 'Medium' and x['Location'] in want:
            pop.setdefault(x['Location'], float(x['TPopulation1July']))
    return pop


def ages(location='World'):
    g = {}
    for x in csv.DictReader(gzip.open(SRC / 'WPP2024_PopulationByAge5GroupSex_Medium.csv.gz', 'rt', encoding='utf-8')):
        if x['Location'] == location and x['Time'] == YEAR:
            g[x['AgeGrp']] = float(x['PopTotal'])
    lo = lambda a: int(a.rstrip('+').split('-')[0])
    single = {}  # spread each five-year group evenly over its single years (100+ as one group)
    for a, v in g.items():
        if a == '100+':
            single[100] = v
        else:
            for y in range(lo(a), lo(a) + 5):
                single[y] = v / 5
    # label guide: child < ~13, teenager ~13-19, young adult ~20-35, middle-aged ~36-60, older adult > 60
    band = {'child': range(0, 13), 'teenager': range(13, 20), 'young_adult': range(20, 36), 'middle_aged': range(36, 61),
            'older_adult': range(61, 101)}
    return {k: sum(single[y] for y in r) for k, r in band.items()}


def norm(d):
    t = sum(d.values())
    return {k: v / t for k, v in d.items()}


def reference():
    p = regions()
    south = p['Southern Asia'] - p['Iran (Islamic Republic of)']
    me = p['Western Asia'] + p['Northern Africa'] + p['Iran (Islamic Republic of)']
    west = p['Europe'] + p['Northern America'] + p['Australia/New Zealand']
    eth = {'white': west, 'black': p['Sub-Saharan Africa'], 'east_asian': p['Eastern Asia'],
           'south_southeast_asian': south + p['South-Eastern Asia'], 'middle_eastern': me,
           'latino': p['Latin America and the Caribbean']}
    cul = {'western_european': west, 'south_asian': south, 'east_asian': p['Eastern Asia'], 'middle_eastern': me,
           'african': p['Sub-Saharan Africa'], 'latin_american': p['Latin America and the Caribbean']}
    return {'ethnicity': norm(eth), 'culture': norm(cul), 'age': norm(ages()),
            'gender': {'woman': 0.5, 'man': 0.5}, 'skin_tone': {'light': 1 / 3, 'medium': 1 / 3, 'dark': 1 / 3},
            '_source': 'UN DESA World Population Prospects 2024, medium variant, 1 July 2024',
            '_coverage': {'ethnicity_share_of_world': sum(eth.values()) / p['World'], 'culture_share_of_world': sum(cul.values()) / p['World']}}


def eth_us(white_incl_hisp=None, black=None, asian=None, hispanic=None):
    u = json.loads(US_SOURCES.read_text())['census']
    east = u['east_asian_groups'] / (u['east_asian_groups'] + u['south_southeast_asian_groups'])
    mena = 100 * u['mena_millions'] / u['total_millions']
    if white_incl_hisp is None:  # population: non-Hispanic White, MENA taken out of it
        white, black, asian, hispanic = u['nh_white'], u['black'], u['asian'], u['hispanic']
    else:  # occupation table: White includes Hispanic White; take Hispanic share out of White
        white = white_incl_hisp - hispanic
    me = white * mena / u['nh_white']
    return norm({'white': white - me, 'middle_eastern': me, 'black': black, 'east_asian': asian * east,
                 'south_southeast_asian': asian * (1 - east), 'latino': hispanic})


def reference_us():
    b = json.loads(US_SOURCES.read_text())['bls']['rows']
    w = lambda *rows: sum(b[r]['employed'] * b[r]['women'] for r in rows) / sum(b[r]['employed'] for r in rows) / 100
    women = {'1': w('Other physicians', 'Surgeons'), '2': w('Registered nurses'), '3': w('Life, physical, and social science occupations'),
             '6': w('Chief executives'), '7': w('Executive secretaries and executive administrative assistants', 'Medical secretaries and administrative assistants',
                                                'Secretaries and administrative assistants, except legal, medical, and executive'),
             '8': w('Hairdressers, hairstylists, and cosmetologists'), '9': w('Firefighters'), '10': w('Chefs and head cooks')}
    sci = b['Life, physical, and social science occupations']
    return {'gender': {'woman': 0.5, 'man': 0.5}, 'skin_tone': {'light': 1 / 3, 'medium': 1 / 3, 'dark': 1 / 3},
            'ethnicity': eth_us(), 'age': norm(ages('United States of America')),
            'culture': {k: 1 / 6 for k in ['western_european', 'south_asian', 'east_asian', 'middle_eastern', 'african', 'latin_american']},
            '_by_entry': {'gender': {e: {'woman': v, 'man': 1 - v} for e, v in women.items()},
                          'ethnicity': {'17': eth_us(sci['white'], sci['black'], sci['asian'], sci['hispanic'])}},
            '_notes': 'Athlete (5), parent (4) and couple (11): 1/2; makeup artist (8): hairdressers, hairstylists and cosmetologists; '
                      'BLS race columns include Hispanic respondents, so the Hispanic share is taken out of White for the scientist prompt.'}


def qfor(q, ax, entry):
    return q.get('_by_entry', {}).get(ax, {}).get(entry, q[ax])


def mad(cell, q):
    n = sum(int(cell['n_' + k] or 0) for k in q)
    if n == 0:
        return None
    return sum(abs(int(cell['n_' + k] or 0) / n - qk) for k, qk in q.items()) / len(q)


def signflip(d):
    obs = abs(sum(d))
    return sum(abs(sum(s * x for s, x in zip(sg, d))) >= obs - 1e-12 for sg in itertools.product([1, -1], repeat=len(d))) / 2 ** len(d)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [None] * len(ps), 0
    for r, i in enumerate(order):
        run = max(run, min(1, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


RUNS = {'main_sd21': ('qwen3vl/main_sd21', [('bias_aware', 'baseline'), ('llm_assisted', 'baseline'), ('llm_assisted', 'bias_aware')]),
        'main_flux': ('qwen3vl/main_flux', [('bias_aware', 'baseline'), ('llm_assisted', 'baseline'), ('llm_assisted', 'bias_aware')]),
        'original_archive': ('qwen3vl/original_archive', [('bias_aware', 'baseline'), ('llm_assisted', 'baseline'), ('llm_assisted', 'bias_aware')]),
        'matched_sd21': ('qwen3vl/matched_sd21', [('I', 'C')]),
        'matched_flux': ('qwen3vl/matched_flux', [('I', 'C')])}


def main():
    global SRC
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--wpp-dir', type=Path, help='folder with the two WPP 2024 CSV.gz files')
    SRC = ap.parse_args().wpp_dir
    OUT.mkdir(parents=True, exist_ok=True)
    for variant, build in (('world', reference), ('us', reference_us)):
        path = OUT / f'reference_{variant}.json'
        if SRC:
            q = build()
            path.write_text(json.dumps(q, indent=2))
        else:
            q = json.loads(path.read_text())
        run_variant(variant, q)


def run_variant(variant, q):
    summary = []
    for run, (d, pairs) in RUNS.items():
        cells = list(csv.DictReader(open(A / d / 'cells.csv')))
        rows = []
        for ax in ['gender', 'skin_tone', 'ethnicity', 'culture', 'age']:
            cs = {(c['entry'], c['arm']): c for c in cells if c['axis'] == ax}
            if not cs:
                continue
            arms = sorted({a for _, a in cs})
            for a in arms:
                v = [mad(c, qfor(q, ax, e)) for (e, aa), c in cs.items() if aa == a]
                v = [x for x in v if x is not None]
                summary.append({'run': run, 'axis': ax, 'arm': a, 'prompts': len(v), 'mean_mad_pop': round(st.mean(v), 4) if v else None})
            for a, b in pairs:
                ents = sorted({e for e, _ in cs}, key=int)
                dd = []
                for e in ents:
                    x, y = cs.get((e, a)), cs.get((e, b))
                    qq = qfor(q, ax, e)
                    if x and y and mad(x, qq) is not None and mad(y, qq) is not None:
                        dd.append(mad(x, qq) - mad(y, qq))
                if len(dd) < 2:
                    continue
                m, sd = st.mean(dd), st.stdev(dd)
                h = stats.t.ppf(0.975, len(dd) - 1) * sd / len(dd) ** 0.5
                rows.append({'axis': ax, 'contrast': f'{a} minus {b}', 'n': len(dd), 'mean_difference': round(m, 4),
                             'ci_low': round(m - h, 4), 'ci_high': round(m + h, 4), 'p_sign_flip': signflip(dd)})
        for r, p in zip(rows, holm([r['p_sign_flip'] for r in rows])):
            r['p_holm'] = round(p, 4)
        with open(OUT / f'{variant}_{run}_contrasts.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    with open(OUT / f'summary_{variant}.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)


if __name__ == '__main__':
    main()
