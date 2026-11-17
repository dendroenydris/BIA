"""Agreement between the CLIP evaluators' labels and the human labels on the 60 human-labelled archive images, by attribute.

Inputs: human_ratings/original_archive (key.csv and both raters' sheets, normalised with human_agreement.py), the
image parquet of the original archive (--images, not stored here), and the original CLIP evaluation's per-image labels
(records/clip/original_archive_b32/per_image.jsonl).

Two scopes:
  as_used  - the labels the paper's analysis actually used: each image on its own category's axis
             (chef A10 -> gender, portrait A12 -> skin tone and ethnicity, wedding A33 -> geo-cultural;
             birthday A32 is an age prompt, which the automated analysis does not cover).
  all_60   - the same CLIP evaluator (same weights, label prompts, argmax) applied to all 60 images on
             every attribute that has a human counterpart. Its labels must reproduce as_used where they overlap.

Human codes are mapped onto CLIP's label space. Codes CLIP cannot express (mixed, indeterminate,
Latino/Latin-American scenes for geo-culture, Middle-Eastern ethnicity) are counted separately and left
out of agreement. Per axis and rater: n, % agreement, Cohen's kappa with a bootstrap CI (paired
resamples within entry x condition, as analyze.py), and the human-human kappa on the same images.

Variants (plan/PLAN.md): b32 = the original evaluator (openai/clip-vit-base-patch32); l14ens = CLIP ViT-L/14 at a
pinned revision, each label embedded through five templates and averaged (secondary evaluator).

usage: python analysis/clip_human_agreement.py {b32|l14ens} --images IMAGES.parquet [--clip-b32 DIR]
  --clip-b32: local copy of openai/clip-vit-base-patch32 (default: download from the Hugging Face Hub)
  writes results/agreement/clip_b32_vs_raters or results/agreement/clip_l14_vs_raters
"""
import argparse
import csv
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import ARCHIVE_RATINGS as HUM, CLIP_LABELS, CONCEPTS, RESULTS  # noqa: E402
import analysis.human_agreement as A  # noqa: E402  (norm_A, norm_B, load, kappa, boot_ci)

_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument('variant', choices=['b32', 'l14ens'])
_ap.add_argument('--images', type=Path, required=True, help='image parquet of the original archive (column Image)')
_ap.add_argument('--clip-b32', default='openai/clip-vit-base-patch32')
ARGS = _ap.parse_args() if __name__ == '__main__' else None
VARIANT = ARGS.variant if ARGS else 'b32'
OUT = RESULTS / 'agreement' / ('clip_b32_vs_raters' if VARIANT == 'b32' else 'clip_l14_vs_raters')
B32 = 'openai/clip-vit-base-patch32'
L14 = ('openai/clip-vit-large-patch14', '32bd64288804d66eefd0ccbe215aa642df71cc41')
TEMPLATES = ['a photo of {}.', 'a cropped photo of {}.', 'a close-up photo of {}.', 'a picture of {}.', 'an image of {}.']
PER_IMAGE = CLIP_LABELS / 'original_archive_b32' / 'per_image.jsonl'

# axis -> (human field, CLIP label -> common code, human code -> common code)
AXES = {
    'Gender Bias': ('gender_presentation', {'a man': 'man', 'a woman': 'woman'}, {'man': 'man', 'woman': 'woman'}),
    'Skin Tone Bias': ('skin_tone',
                       {'a person with light skin': 'light', 'a person with medium skin': 'medium', 'a person with dark skin': 'dark'},
                       {'light': 'light', 'medium': 'medium', 'dark': 'dark'}),
    'Ethnic Representation Bias': ('ethnic_appearance',
                                   {'a white person': 'white', 'a Black person': 'black', 'an Asian person': 'asian',
                                    'a Latino person': 'latino', 'a racially diverse group of people': 'mixed'},
                                   {'white': 'white', 'black': 'black', 'latino': 'latino', 'mixed': 'mixed',
                                    'south_southeast_asian': 'asian', 'east_asian': 'asian', 'asian': 'asian'}),
    'Geo-Cultural Bias': ('cultural_scene',
                          {f'{art} {r} {kind}': code for r, code in [('Western', 'western'), ('East Asian', 'east_asian'),
                                                                     ('Middle Eastern', 'middle_eastern'), ('African', 'african'),
                                                                     ('South Asian', 'south_asian')]
                           for kind in ('person', 'environment') for art in ('a', 'an')},
                          {'western_european': 'western', 'african': 'african', 'middle_eastern': 'middle_eastern',
                           'south_asian': 'south_asian', 'east_asian': 'east_asian'}),
}
AS_USED_ENTRY = {'Gender Bias': 'A10', 'Skin Tone Bias': 'A12', 'Ethnic Representation Bias': 'A12', 'Geo-Cultural Bias': 'A33'}


def clip_labels(key):
    """Re-run the evaluator's label step on the 60 images, on every axis."""
    import pyarrow.parquet as pq
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    concepts = json.loads(CONCEPTS.read_text())
    table = pq.read_table(ARGS.images, columns=['Image']).column('Image').to_pylist()
    if VARIANT == 'b32':
        model = CLIPModel.from_pretrained(ARGS.clip_b32).to('cpu').eval()
        proc = CLIPProcessor.from_pretrained(ARGS.clip_b32, use_fast=False)
    else:
        model = CLIPModel.from_pretrained(L14[0], revision=L14[1]).to('cpu').eval()
        proc = CLIPProcessor.from_pretrained(L14[0], revision=L14[1], use_fast=False)
    out = {}
    with torch.inference_mode():
        text = {}
        for axis in AXES:
            labels = concepts[axis]['Parity']['concepts']
            if VARIANT == 'b32':
                t = model.get_text_features(**proc(text=labels, return_tensors='pt', padding=True))
                t = t / t.norm(dim=-1, keepdim=True)
            else:
                per = []
                for tpl in TEMPLATES:
                    e = model.get_text_features(**proc(text=[tpl.format(l) for l in labels], return_tensors='pt', padding=True))
                    per.append(e / e.norm(dim=-1, keepdim=True))
                t = torch.stack(per).mean(0)
                t = t / t.norm(dim=-1, keepdim=True)
            text[axis] = (labels, t)
        for hid, k in key.items():
            img = Image.open(io.BytesIO(table[int(k['archive_image_id'])]['bytes'])).convert('RGB')
            assert hashlib.sha256(img.tobytes()).hexdigest() == k['pixel_sha256'], hid
            f = model.get_image_features(**proc(images=img, return_tensors='pt'))
            f = f / f.norm(dim=-1, keepdim=True)
            out[hid] = {axis: labels[int((f @ t.T).argmax())] for axis, (labels, t) in text.items()}
    return out


def agreement(ids, clip, hA, hB, field, cmap, hmap, key):
    rows, unrep = [], Counter()
    for rater, h in (('A', hA), ('B', hB)):
        pairs = [(i, cmap[clip[i]], hmap.get(h[i][field])) for i in ids if h[i][field]]
        unrep.update([(rater, h[i][field]) for i in ids if h[i][field] and h[i][field] not in hmap])
        ok = [(i, c, x) for i, c, x in pairs if x]
        if not ok:
            rows.append({'rater': rater, 'n_comparable': 0, 'n_coded': len(pairs)})
            continue
        c, x = [p[1] for p in ok], [p[2] for p in ok]
        po, k = A.kappa(c, x)
        lo, hi = A.boot_ci(c, x, [(key[p[0]]['entry'], key[p[0]]['condition']) for p in ok])
        rows.append({'rater': rater, 'n_coded': len(pairs), 'n_comparable': len(ok), 'agree': sum(a == b for a, b in zip(c, x)),
                     'pct_agree': round(po, 4), 'kappa': None if k is None else round(k, 4), 'kappa_ci95': f'[{lo}, {hi}]',
                     'clip_dist': dict(Counter(c)), 'human_dist': dict(Counter(x))})
    both = [i for i in ids if hA[i][field] and hB[i][field] and hA[i][field] == hB[i][field] and hmap.get(hA[i][field])]
    if both:
        c, x = [cmap[clip[i]] for i in both], [hmap[hA[i][field]] for i in both]
        rows.append({'rater': 'A=B consensus', 'n_comparable': len(both), 'agree': sum(a == b for a, b in zip(c, x)),
                     'pct_agree': round(A.kappa(c, x)[0], 4), 'kappa': (lambda k: None if k is None else round(k, 4))(A.kappa(c, x)[1]),
                     'clip_dist': dict(Counter(c)), 'human_dist': dict(Counter(x))})
    hh = [i for i in ids if hA[i][field] and hB[i][field]]
    if hh and field != 'ethnic_appearance':
        po, k = A.kappa([hA[i][field] for i in hh], [hB[i][field] for i in hh])
        rows.append({'rater': 'A vs B (reference)', 'n_comparable': len(hh), 'pct_agree': round(po, 4),
                     'kappa': None if k is None else round(k, 4)})
    return rows, unrep


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    key = {r['image_id']: r for r in csv.DictReader(open(HUM / 'key.csv', newline=''))}
    hA = A.load(HUM / 'rater_A.csv', A.norm_A)
    hB = A.load(HUM / 'rater_B.csv', A.norm_B)
    used = {}
    for line in PER_IMAGE.read_text().splitlines():
        r = json.loads(line)
        used[(r['image_id'], r['category'])] = r['parity_label']
    clip = clip_labels(key)
    # as_used labels, and the check that the re-run reproduces them
    as_used, mismatch = {}, []
    for hid, k in key.items():
        for axis, entry in AS_USED_ENTRY.items():
            if k['entry'] == entry:
                lab = used[(int(k['archive_image_id']), axis)] if VARIANT == 'b32' else clip[hid][axis]
                as_used.setdefault(axis, {})[hid] = lab
                if clip[hid][axis] != lab:
                    mismatch.append((hid, axis, lab, clip[hid][axis]))
    results, unrep_rows, per_image = [], [], []
    for scope in ('as_used', 'all_60'):
        for axis, (field, cmap, hmap) in AXES.items():
            if scope == 'as_used':
                labels = as_used[axis]
                ids = sorted(labels)
            else:
                labels = {i: clip[i][axis] for i in key}
                ids = sorted(key)
            rows, unrep = agreement(ids, labels, hA, hB, field, cmap, hmap, key)
            for r in rows:
                results.append({'scope': scope, 'axis': axis, 'human_field': field, **r})
            for (rater, code), n in sorted(unrep.items()):
                unrep_rows.append({'scope': scope, 'axis': axis, 'rater': rater, 'human_code_not_in_clip_labels': code, 'n': n})
    for hid, k in sorted(key.items()):
        per_image.append({'image_id': hid, 'entry': k['entry'], 'condition': k['condition'], 'archive_image_id': k['archive_image_id'],
                          **{f'clip_{a}': clip[hid][a] for a in AXES},
                          **{f'A_{f}': hA[hid][f] for f in ('gender_presentation', 'skin_tone', 'ethnic_appearance', 'cultural_scene')},
                          **{f'B_{f}': hB[hid][f] for f in ('gender_presentation', 'skin_tone', 'cultural_scene')}})
    fields = sorted({k for r in results for k in r}, key=lambda k: list(results[0]).index(k) if k in results[0] else 99)
    with open(OUT / 'agreement.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    with open(OUT / 'not_representable.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['scope', 'axis', 'rater', 'human_code_not_in_clip_labels', 'n'])
        w.writeheader()
        w.writerows(unrep_rows)
    with open(OUT / 'per_image_labels.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_image[0]))
        w.writeheader()
        w.writerows(per_image)
    (OUT / 'manifest.json').write_text(json.dumps({
        'as_used_rerun_mismatches': mismatch, 'per_image_source_sha256': hashlib.sha256(PER_IMAGE.read_bytes()).hexdigest(),
        'concepts_sha256': hashlib.sha256(CONCEPTS.read_bytes()).hexdigest(), 'bootstrap': {'seed': A.SEED, 'n': A.NBOOT},
        'variant': VARIANT, 'model': B32 if VARIANT == 'b32' else list(L14), 'templates': None if VARIANT == 'b32' else TEMPLATES}, indent=2) + '\n')
    for r in results:
        print(r['scope'], r['axis'], r['rater'], r.get('n_comparable'), r.get('pct_agree'), r.get('kappa'), r.get('kappa_ci95', ''))
    print('as_used re-run mismatches:', len(mismatch))


if __name__ == '__main__':
    main()
