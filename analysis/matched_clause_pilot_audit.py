"""Matched-clause pilot: recompute the saved CLIP scores and the planned prompt comparisons without loading a model,
and add the diagnostic and sensitivity tables (Appendix D, Table D1).

Inputs: records/matched_clause_pilot (per-image scores, receipts, configuration), the design files, and the tables
written by generation/matched_clause_pilot.py in results/matched_clause_pilot, which are checked against the
recomputation. Outputs (results/matched_clause_pilot): independent_comparisons.csv, condition_means_sd.csv,
prompt_differences.csv, statistical_acceptance.json.

usage: python analysis/matched_clause_pilot_audit.py
"""
import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from scipy import __version__ as scipy_version
from scipy.stats import binomtest, shapiro, t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import CONCEPTS, PILOT_RECORDS, RESULTS  # noqa: E402
from generation.matched_clause_pilot import DESIGN_FILES  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-11, abs_tol=1e-12), (a, b)


def adjust(values):
    order = np.argsort(values, kind='stable')
    ranked = np.minimum(1, np.asarray(values)[order] * np.arange(len(values), 0, -1))
    result = np.empty(len(values))
    result[order] = np.maximum.accumulate(ranked)
    return result.tolist()


assert adjust([.01, .04, .03]) == [.03, .06, .06]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--records', type=Path, default=PILOT_RECORDS)
parser.add_argument('--output-dir', type=Path, default=RESULTS / 'matched_clause_pilot')
parser.add_argument('--concepts', type=Path, default=CONCEPTS)
args = parser.parse_args()
rec, out = args.records, args.output_dir
manifest = json.loads((rec/'evaluation_manifest.json').read_text())
assert manifest['status'] == 'complete' and manifest['record_count'] == 430 and manifest['unique_images'] == 310
for name, key in [('per_image.jsonl', 'per_image_sha256'), ('generation_receipts.jsonl', 'generation_receipts_sha256')]:
    assert sha(rec/name) == manifest[key]
assert sha(args.concepts) == manifest['concepts_sha256']
config = json.loads((rec/'execution_config.json').read_text())
for name, digest in config['design_hashes'].items():
    assert sha(DESIGN_FILES[name]) == digest
pairs = {r['entry']: r for r in read(DESIGN_FILES['prompt_pairs.csv'])}
schedule = {r['run_order']: r for r in read(DESIGN_FILES['generation_schedule.csv'])}
assert len(pairs) == 31 and len(schedule) == 310
assert all(20 <= int(r['equal_positions']) <= 29 for r in pairs.values())
assert all(len(json.loads(r['instruction_token_ids'])) == len(json.loads(r['control_token_ids'])) == int(r['equal_positions']) for r in pairs.values())
receipts = [json.loads(l) for l in (rec/'generation_receipts.jsonl').read_text().splitlines()]
assert len(receipts) == 310 and len({r['run_order'] for r in receipts}) == 310
blocks = defaultdict(list)
for receipt in receipts:
    row = schedule[receipt['run_order']]
    assert receipt['schedule_row'] == row
    assert re.fullmatch('[0-9a-f]{64}', receipt['file_sha256'])
    blocks[(row['entry'], row['repetition'])].append((row, receipt['latent_sha256']))
assert len(blocks) == 155
for block in blocks.values():
    assert len(block) == 2 and {r['arm'] for r, _ in block} == {'C', 'I'}
    assert len({r['latent_seed'] for r, _ in block}) == len({h for _, h in block}) == 1
concepts = json.loads(args.concepts.read_text())
records = [json.loads(l) for l in (rec/'per_image.jsonl').read_text().splitlines()]
expected = Counter()
for row in schedule.values():
    axes = ['Skin Tone Bias', 'Ethnic Representation Bias'] if row['category'] == 'Racial Bias' else [row['category']]
    for axis in axes:
        expected[(axis, row['entry'], row['arm'])] += 1
cells, identities, seen = defaultdict(list), {}, set()
reference_strings = set()
for record in records:
    axis, image_id = record['category'], record['image_id']
    assert (axis, image_id) not in seen
    seen.add((axis, image_id))
    row = schedule[str(image_id)]
    assert record['entry'] == row['entry'] and record['arm'] == row['arm']
    identity = (record['entry'], record['arm'], record['image_sha256_pixels'])
    assert image_id not in identities or identities[image_id] == identity
    identities[image_id] = identity
    definition = concepts[axis]
    labels = definition['Parity']['concepts']
    cosines = record['label_cosines']
    assert len(labels) == len(cosines) and all(math.isfinite(v) and -1.00001 <= v <= 1.00001 for v in cosines)
    assert record['parity_label'] == labels[int(np.argmax(cosines))]
    aggregate = {}
    for metric in ('Embedding Distance', 'Stereotype Frequency Score'):
        subtype_values = []
        for subtype, templates in definition[metric].items():
            score = record['scores'][f'{metric} - {subtype}']
            short = re.split(r'[.,]', pairs[row['entry']]['baseline_prompt'])[0].strip()
            refs = [s.replace('[PROMPT]', short) for s in templates]
            assert score['references'] == refs
            reference_strings.update(refs)
            values = score['cosine_similarities']
            assert len(values) == len(refs) and all(math.isfinite(v) and -1.00001 <= v <= 1.00001 for v in values)
            subtype_values.append(mean(values) if metric == 'Embedding Distance' else max(values))
        aggregate['concept_similarity' if metric == 'Embedding Distance' else 'reference_similarity'] = mean(subtype_values)
    for metric, value in aggregate.items():
        close(value, record[metric])
    cells[(axis, record['entry'], record['arm'])].append(record)
assert len(records) == 430 and len(identities) == 310
assert Counter({k: len(v) for k, v in cells.items()}) == expected
image_pixels_checked = (rec/'images').is_dir()
if image_pixels_checked:
    from PIL import Image
    for receipt in receipts:
        image_id = int(receipt['run_order'])
        path = rec/'images'/f'{image_id:03d}.png'
        assert sha(path) == receipt['file_sha256']
        with Image.open(path) as image:
            rgb = image.convert('RGB')
            assert rgb.size == (768, 768)
            assert hashlib.sha256(rgb.tobytes()).hexdigest() == identities[image_id][2]
metrics = ('label_imbalance_mad', 'concept_similarity', 'reference_similarity')
values = {}
for key, rows in cells.items():
    labels = concepts[key[0]]['Parity']['concepts']
    frequencies = Counter(r['parity_label'] for r in rows)
    values[key] = {metrics[0]: sum(abs(frequencies[label]/5 - 1/len(labels)) for label in labels)/len(labels),
                   **{m: mean(r[m] for r in rows) for m in metrics[1:]}}
for row in read(out/'prompt_cells.csv'):
    for metric in metrics:
        close(row[metric], values[(row['category'], row['entry'], row['arm'])][metric])
assert len(read(out/'prompt_cells.csv')) == 86
comparisons, descriptives, differences = [], [], []
rng = np.random.default_rng(20261012)
for axis in sorted({k[0] for k in cells}):
    entries = sorted({k[1] for k in cells if k[0] == axis}, key=int)
    for metric in metrics:
        c = np.array([values[(axis, e, 'C')][metric] for e in entries])
        i = np.array([values[(axis, e, 'I')][metric] for e in entries])
        d = i-c
        n, delta, sd = len(d), float(d.mean()), float(d.std(ddof=1))
        signs = 2*((np.arange(2**n)[:, None] >> np.arange(n)) & 1)-1
        p_raw = float(np.mean(np.abs(signs@d/n) >= abs(delta)-1e-12))
        half = float(t.ppf(.975, n-1)*sd/math.sqrt(n))
        positive, negative = int(np.sum(d > 1e-12)), int(np.sum(d < -1e-12))
        nonzero = positive+negative
        sign_p = float(binomtest(positive, nonzero, .5, alternative='two-sided').pvalue) if nonzero else 1.
        bootstrap = d[rng.integers(0, n, (20000, n))].mean(axis=1)
        boot_low, boot_high = np.quantile(bootstrap, [.025, .975]).tolist()
        comparisons.append({'category': axis, 'metric': metric, 'prompt_n': n, 'df_ci': n-1,
            'mean_difference': delta, 'sd_difference': sd, 'cohen_dz': delta/sd,
            'ci95_low': delta-half, 'ci95_high': delta+half, 'p_raw_sign_flip': p_raw,
            'family_size': 4 if metric == metrics[0] else 8, 'shapiro_p': float(shapiro(d).pvalue),
            'positive_n': positive, 'negative_n': negative, 'tie_n': n-nonzero,
            'exploratory_bootstrap_low': boot_low, 'exploratory_bootstrap_high': boot_high,
            'exploratory_sign_p_raw': sign_p})
        descriptives.append({'category': axis, 'metric': metric, 'prompt_n': n, 'C_mean': float(c.mean()),
            'C_sd': float(c.std(ddof=1)), 'I_mean': float(i.mean()), 'I_sd': float(i.std(ddof=1))})
        differences.extend({'category': axis, 'metric': metric, 'entry': e, 'C': float(cv), 'I': float(iv), 'I_minus_C': float(dv)} for e, cv, iv, dv in zip(entries, c, i, d))
for primary in (True, False):
    family = [r for r in comparisons if (r['metric'] == metrics[0]) == primary]
    assert len(family) == (4 if primary else 8)
    for row, adjusted, sign_adjusted in zip(family, adjust([r['p_raw_sign_flip'] for r in family]), adjust([r['exploratory_sign_p_raw'] for r in family])):
        row['p_holm'] = adjusted
        row['exploratory_sign_p_holm'] = sign_adjusted
for saved in read(out/'prompt_comparisons.csv'):
    row = next(r for r in comparisons if r['category'] == saved['category'] and r['metric'] == saved['metric'])
    for name in ('prompt_n', 'family_size', 'mean_difference', 'sd_difference', 'cohen_dz', 'ci95_low', 'ci95_high', 'p_raw_sign_flip', 'p_holm', 'shapiro_p'):
        close(row[name], saved[name])
assert len(read(out/'prompt_comparisons.csv')) == 12
for saved in read(out/'condition_descriptives.csv'):
    for metric in metrics:
        row = next(r for r in descriptives if r['category'] == saved['category'] and r['metric'] == metric)
        close(row[saved['arm']+'_mean'], saved[metric])
assert all(r['p_holm'] == 1 for r in comparisons)
write(out/'independent_comparisons.csv', comparisons)
write(out/'condition_means_sd.csv', descriptives)
write(out/'prompt_differences.csv', differences)
report = {'status': 'passed', 'images': 310, 'axis_records': 430, 'prompt_arm_axis_cells': 86,
    'paired_latent_blocks': 155, 'comparisons': 12, 'family_sizes': [4, 8],
    'all_planned_holm_p': 1., 'normality_flags_descriptive': [{k: r[k] for k in ('category', 'metric', 'shapiro_p')} for r in comparisons if r['shapiro_p'] < .05],
    'exploratory_bootstrap_seed': 20261012, 'exploratory_bootstrap_resamples': 20000,
    'exploratory_additions': 'Post-hoc percentile paired-prompt bootstrap and exact two-sided sign tests; separate 4/8-test sensitivity families; do not replace planned sign-flip comparisons.',
    'assumptions': ['Selected prompt entries are the independent analysis units; repeated or related tasks limit exchangeability and population generalization.',
        'Sign-flip inference assumes sign exchangeability/symmetry. Randomized within-pair generation order does not establish randomized treatment assignment across prompt tasks.',
        'Paired-t intervals are unadjusted, assume normal prompt differences, and are not simultaneous intervals; several discrete MAD differences are non-normal.',
        'Bootstrap intervals are exploratory and conditional on small selected prompt samples; non-rejection of Shapiro does not establish normality.',
        'Non-significance is neither absence nor equivalence; fixed CLIP diagnostics do not establish semantic fidelity, human reliability, or ethical fairness.'],
    'image_file_and_pixel_verification_performed': image_pixels_checked,
    'numpy': np.__version__, 'scipy': scipy_version,
    'files_sha256': {name: sha((rec if name.endswith(('.jsonl', '.json')) else out)/name) for name in ('per_image.jsonl', 'generation_receipts.jsonl', 'execution_config.json', 'prompt_comparisons.csv', 'independent_comparisons.csv', 'condition_means_sd.csv', 'prompt_differences.csv')},
    'audit_source_sha256': sha(Path(__file__))}
(out/'statistical_acceptance.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
