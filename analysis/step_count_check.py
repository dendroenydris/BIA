"""Step-count check (Appendix F.1): the matched-clause pilot (50 steps, MPS) and repetitions 1-5 of the 50-image SD 2.1
run (25 steps, A100) share prompts and initial latents. Compare their CLIP ViT-B/32 labels image by image, against
images of the same prompt and version from different latents (repetitions 6-50), and compare prompt-level label MAD
and the I - C difference between the two renderings (exact sign-flip tests over prompt entries).

Inputs: records/matched_clause_pilot/{generation_receipts,per_image}.jsonl, datasets/matched_clause/generation_schedule.csv,
records/clip/matched_sd21_b32.jsonl.

usage: python analysis/step_count_check.py   (writes results/step_count_check/summary.json)
"""
import collections
import csv
import itertools
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import CLIP_LABELS, MATCHED_DESIGN, PILOT_RECORDS as PILOT, RESULTS as A  # noqa: E402

SCHED = MATCHED_DESIGN / 'generation_schedule.csv'
NEW = CLIP_LABELS / 'matched_sd21_b32.jsonl'
AX = {'Skin Tone Bias': 'skin_tone', 'Ethnic Representation Bias': 'ethnic_appearance',
      'Gender Bias': 'gender_presentation', 'Geo-Cultural Bias': 'cultural_scene'}
K = {'gender_presentation': 2, 'skin_tone': 3, 'ethnic_appearance': 5, 'cultural_scene': 10}


def mad(v, k):
    c = collections.Counter(v)
    ps = [x / len(v) for x in c.values()] + [0] * (k - len(c))
    return sum(abs(p - 1 / k) for p in ps) / k


def signflip(d):
    obs = abs(sum(d))
    hits = sum(abs(sum(s * x for s, x in zip(signs, d))) >= obs - 1e-12 for signs in itertools.product([1, -1], repeat=len(d)))
    return hits / 2 ** len(d)


rec = {json.loads(l)['run_order']: json.loads(l)['schedule_row'] for l in open(PILOT / 'generation_receipts.jsonl')}
pil = [json.loads(l) for l in open(PILOT / 'per_image.jsonl')]
S = {r['run_order']: r for r in csv.DictReader(open(SCHED))}
new = {}
for l in open(NEW):
    o = json.loads(l)
    s = S[str(int(o['key'][3:]))]
    new[(s['entry'], s['arm'], int(s['repetition']))] = (o['codes'], s['latent_seed'], s['submitted_prompt'])
same, other, n, no = (collections.Counter() for _ in range(4))
p50, p25 = collections.defaultdict(list), collections.defaultdict(list)
for x in pil:
    s = rec[str(x['image_id'])]
    ax, e, a, r = AX[x['category']], s['entry'], s['arm'], int(s['repetition'])
    codes, seed, prompt = new[(e, a, r)]
    assert seed == s['latent_seed'] and prompt == s['submitted_prompt']
    n[ax] += 1
    same[ax] += codes[ax] == x['parity_label']
    for r2 in range(6, 51):
        no[ax] += 1
        other[ax] += new[(e, a, r2)][0][ax] == x['parity_label']
    p50[(ax, e, a)].append(x['parity_label'])
    p25[(ax, e, a)].append(codes[ax])
out = {}
for ax, k in K.items():
    ents = sorted({key[1] for key in p50 if key[0] == ax}, key=int)
    d = [st.mean(mad(p25[(ax, e, a)], k) - mad(p50[(ax, e, a)], k) for a in 'IC') for e in ents]
    ic = [(mad(p25[(ax, e, 'I')], k) - mad(p25[(ax, e, 'C')], k)) - (mad(p50[(ax, e, 'I')], k) - mad(p50[(ax, e, 'C')], k)) for e in ents]
    out[ax] = {'records': n[ax], 'same_latent_agreement': round(same[ax] / n[ax], 3), 'different_latent_agreement': round(other[ax] / no[ax], 3),
               'entries': len(ents), 'mad_diff_25_minus_50': round(st.mean(d), 4), 'p_mad': signflip(d),
               'ic_diff_25_minus_50': round(st.mean(ic), 4), 'p_ic': signflip(ic)}
(A / 'step_count_check').mkdir(exist_ok=True)
(A / 'step_count_check/summary.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
