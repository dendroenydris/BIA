"""Truncation regeneration (Appendix C): the eight LLM-assisted racial rewrites that exceed the 77-position SD 2.1 text
context (A14-A19, A22, A23), each rendered with five shared initial latents under two encodings (80 images):

  T  standard right-truncated 77-position input (identical to the pipeline's own encoding; checked)
  F  full text encoded as consecutive 77-position chunks by the same text encoder (generation/long_prompt.py)

Settings: sd2-community/stable-diffusion-2-1 at revision bb2154823665391b4fb29b0b9cf82a198964ee05 (fp16), 768 x 768,
50 DPMSolverMultistepScheduler steps, guidance 7.5, empty negative prompt, float16 on a CUDA GPU. Evaluation:
openai/clip-vit-base-patch32 on the CPU with datasets/bias_concepts.json; paired F - T comparisons per prompt entry.

The schedule is rebuilt from datasets/prompts.csv (master seed 20261017) and must equal
datasets/truncation_regeneration/generation_schedule.csv; the stored schedule was built with the tokenizer's
BasicTokenizer fallback (ftfy not installed), which counts two more positions for some rewrites than ftfy does, so run
without ftfy. The stored run is records/truncation_regeneration (receipts,
encoding checks, environment, per-image CLIP scores) and results/truncation_regeneration (tables).

usage:
  python generation/truncation_regeneration.py --out RUN_DIR          # generate, evaluate and analyse (GPU)
  python generation/truncation_regeneration.py --analyze-only --out DIR
      recompute the tables from records/truncation_regeneration/per_image.jsonl (no model)
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import CONCEPTS, PROMPTS, TRUNCATION_DESIGN, TRUNCATION_RECORDS  # noqa: E402
from analysis.clip_statistics import holm, parity_mad, write_csv  # noqa: E402
from evaluation.clip_original_archive import file_sha256, reference_texts  # noqa: E402
from generation.long_prompt import chunk_ids, encode  # noqa: E402

ENTRIES = [14, 15, 16, 17, 18, 19, 22, 23]
AXES = ['Skin Tone Bias', 'Ethnic Representation Bias']
METRICS = ['label_imbalance_mad', 'concept_similarity', 'reference_similarity']
SCHEDULE = TRUNCATION_DESIGN / 'generation_schedule.csv'
MODEL, REV = 'sd2-community/stable-diffusion-2-1', 'bb2154823665391b4fb29b0b9cf82a198964ee05'


def build_schedule(tokenizer, out):
    """Rebuild the frozen schedule and check it against the stored copy."""
    with PROMPTS.open(encoding='utf-8-sig', newline='') as f:
        prompts = list(csv.DictReader(f, delimiter=';'))
    blocks = [(e, rep) for e in ENTRIES for rep in range(1, 6)]
    rng = random.Random(20261017)
    seeds = {b: rng.randrange(0, 2**63) for b in blocks}
    rng.shuffle(blocks)
    rows = []
    for entry, rep in blocks:
        p = prompts[entry-1]; assert p['Main_Category'] == 'Racial Bias'
        text = p['Prompt_engineering_with_the_help_of_LLM']
        chunks = chunk_ids(tokenizer, text); assert len(chunks) == 2
        ids = tokenizer(text, add_special_tokens=False, truncation=False)['input_ids']
        arms = ['T', 'F']; rng.shuffle(arms)
        for arm in arms:
            rows.append({'run_order': len(rows)+1, 'entry': entry, 'repetition': rep, 'arm': arm,
                         'latent_seed': seeds[(entry, rep)], 'category': 'Racial Bias',
                         'submitted_prompt': text, 'baseline_prompt': p['Neutral_prompt'],
                         'token_positions': len(ids)+2, 'chunks': len(chunks)})
    write_csv(out/'generation_schedule.csv', rows)
    assert file_sha256(out/'generation_schedule.csv') == file_sha256(SCHEDULE), 'Schedule differs from the frozen schedule'
    with (out/'generation_schedule.csv').open(newline='') as f:
        schedule = list(csv.DictReader(f))
    print('schedule identical to the frozen schedule:', len(schedule), 'rows')
    return schedule


def generate(out):
    import numpy as np
    import torch
    from PIL import Image
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    assert torch.cuda.is_available(), 'a CUDA GPU is required'
    print(torch.cuda.get_device_name(0), torch.__version__)
    pipe = StableDiffusionPipeline.from_pretrained(MODEL, revision=REV, variant='fp16', torch_dtype=torch.float16).to('cuda')
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    schedule = build_schedule(pipe.tokenizer, out)

    # Encoding checks: T equals the pipeline's own encoding; F's first chunk equals T
    checks = []
    for entry in ENTRIES:
        text = next(r for r in schedule if int(r['entry']) == entry)['submitted_prompt']
        t, tn = encode(pipe, text, truncate=True); f_, fn = encode(pipe, text, truncate=False)
        with torch.no_grad():
            pe, ne = pipe.encode_prompt(text, device='cuda', num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt='')
        pd, nd = float((t-pe).abs().max()), float((tn-ne).abs().max())
        assert pd == nd == 0 and t.shape[1] == 77 and f_.shape[1] == fn.shape[1] == 154
        assert torch.equal(f_[:, :77], t) and torch.equal(fn[:, 77:], tn)
        checks.append({'entry': entry, 'T_vs_pipeline_max_abs_diff': pd, 'negative_vs_pipeline_max_abs_diff': nd})
    json.dump({'passed': True, 'entries': checks, 'device': torch.cuda.get_device_name(0)}, open(out/'encoding_checks.json', 'w'), indent=2)
    print('encoding checks passed')

    # One batch of 5 shared-latent images per (entry, version); single images if the GPU runs out of memory
    groups = defaultdict(list)
    for r in schedule:
        groups[(r['entry'], r['arm'])].append(r)
    receipts, t0 = [], time.time()

    def latent(r):
        x = torch.randn((1, 4, 96, 96), generator=torch.Generator(device='cpu').manual_seed(int(r['latent_seed'])), dtype=torch.float32)
        return x, hashlib.sha256(x.numpy().tobytes()).hexdigest()

    def render(rs, pe, ne):
        lat = torch.cat([latent(r)[0] for r in rs]).to('cuda', torch.float16)
        n = len(rs)
        return pipe(prompt_embeds=pe.repeat(n, 1, 1), negative_prompt_embeds=ne.repeat(n, 1, 1), height=768, width=768,
                    num_inference_steps=50, guidance_scale=7.5, latents=lat, output_type='np').images

    (out/'images').mkdir(exist_ok=True)
    for (entry, arm), rs in groups.items():
        pe, ne = encode(pipe, rs[0]['submitted_prompt'], truncate=arm == 'T')
        start = time.time()
        try:
            imgs = render(rs, pe, ne)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache(); imgs = [render([r], pe, ne)[0] for r in rs]
        for r, img in zip(rs, imgs):
            assert np.isfinite(img).all() and float(img.max()-img.min()) > 0
            path = out/'images'/f"{int(r['run_order']):03d}.png"
            Image.fromarray((img.clip(0, 1)*255).round().astype('uint8')).save(path)
            receipts.append({'run_order': r['run_order'], 'schedule_row': r, 'file_sha256': file_sha256(path),
                             'latent_sha256': latent(r)[1], 'seconds_batch': time.time()-start})
        print(f'A{entry} {arm}: {len(receipts)}/80 images, {time.time()-t0:.0f}s elapsed')
    receipts.sort(key=lambda x: int(x['run_order']))
    with open(out/'generation_receipts.jsonl', 'w') as f:
        f.write(''.join(json.dumps(x)+'\n' for x in receipts))
    for i in range(0, 80, 2):
        assert receipts[i]['latent_sha256'] == receipts[i+1]['latent_sha256']
    print('generation done; paired latents verified')
    return schedule


def evaluate(out, schedule):
    """CLIP ViT-B/32 on the CPU, as for the original archive."""
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    concepts = json.loads(CONCEPTS.read_text())
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to('cpu').eval()
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32', use_fast=False)
    cache = {}

    def text_features(texts):
        if tuple(texts) not in cache:
            inputs = processor(text=texts, return_tensors='pt', padding=True, truncation=False)
            f = model.get_text_features(**inputs); cache[tuple(texts)] = f/f.norm(dim=-1, keepdim=True)
        return cache[tuple(texts)]

    records = []
    with torch.inference_mode():
        for row in schedule:
            image = Image.open(out/'images'/f"{int(row['run_order']):03d}.png").convert('RGB')
            f = model.get_image_features(**processor(images=image, return_tensors='pt')); f = f/f.norm(dim=-1, keepdim=True)
            for axis in AXES:
                d = concepts[axis]; labels = d['Parity']['concepts']
                sims = (f@text_features(labels).T).squeeze(0); scores = {}
                for metric in ('Embedding Distance', 'Stereotype Frequency Score'):
                    for sub, phrases in d[metric].items():
                        texts = reference_texts(phrases, row['baseline_prompt'])
                        scores[f'{metric} - {sub}'] = (f@text_features(texts).T).squeeze(0).tolist()
                records.append({'image_id': int(row['run_order']), 'entry': row['entry'], 'arm': row['arm'], 'category': axis,
                                'parity_label': labels[int(sims.argmax())],
                                'concept_similarity': mean(mean(v) for k, v in scores.items() if k.startswith('Embedding')),
                                'reference_similarity': mean(max(v) for k, v in scores.items() if k.startswith('Stereotype')), 'scores': scores})
    with open(out/'per_image.jsonl', 'w') as h:
        h.write(''.join(json.dumps(r)+'\n' for r in records))
    return records


def analyse(records, out):
    """Prompt-level cells and the six paired F - T comparisons (Holm sets of 2 MAD and 4 similarity tests)."""
    from scipy.stats import t as tdist, shapiro
    concepts = json.loads(CONCEPTS.read_text())
    cells = defaultdict(list)
    for r in records: cells[(r['category'], r['entry'], r['arm'])].append(r)
    assert len(records) == 160 and all(len(v) == 5 for v in cells.values()) and len(cells) == 32
    values = {}
    for k, rs in cells.items():
        labels = concepts[k[0]]['Parity']['concepts']; c = Counter(r['parity_label'] for r in rs)
        values[k] = {'label_imbalance_mad': parity_mad({l: c[l]/5 for l in labels}, labels), **{m: mean(r[m] for r in rs) for m in METRICS[1:]}}
    write_csv(out/'prompt_cells.csv', [{'category': k[0], 'entry': k[1], 'arm': k[2], **v} for k, v in sorted(values.items())])
    results, desc, pooled = [], [], []
    for axis in AXES:
        for arm in ['T', 'F']:
            c = Counter(r['parity_label'] for r in records if r['category'] == axis and r['arm'] == arm)
            pooled += [{'category': axis, 'arm': arm, 'label': l, 'count': c[l], 'image_n': 40} for l in concepts[axis]['Parity']['concepts']]
            for m in METRICS:
                v = [values[(axis, str(e), arm)][m] for e in ENTRIES]
                desc.append({'category': axis, 'arm': arm, 'metric': m, 'mean': mean(v), 'sd': stdev(v)})
        for m in METRICS:
            d = [values[(axis, str(e), 'F')][m]-values[(axis, str(e), 'T')][m] for e in ENTRIES]
            eff, sd = mean(d), stdev(d)
            p = sum(abs(mean(s*x for s, x in zip(signs, d))) >= abs(eff)-1e-12 for signs in itertools.product((-1, 1), repeat=8))/256
            half = float(tdist.ppf(.975, 7))*sd/math.sqrt(8)
            results.append({'category': axis, 'metric': m, 'comparison': 'F minus T', 'prompt_n': 8, 'mean_difference': eff,
                            'sd_difference': sd, 'cohen_dz': eff/sd if sd else 'undefined', 'ci95_low': eff-half, 'ci95_high': eff+half,
                            'p_raw_sign_flip': p, 'p_holm': None, 'family_size': 2 if m == METRICS[0] else 4,
                            'shapiro_p': float(shapiro(d).pvalue) if sd else 'constant'})
    for primary in [True, False]:
        fam = [r for r in results if (r['metric'] == METRICS[0]) == primary]
        for r, p in zip(fam, holm([r['p_raw_sign_flip'] for r in fam])): r['p_holm'] = p
    write_csv(out/'comparisons.csv', results); write_csv(out/'descriptives.csv', desc); write_csv(out/'pooled_labels.csv', pooled)
    for r in results:
        print(r['category'], r['metric'], round(r['mean_difference'], 5), [round(r['ci95_low'], 5), round(r['ci95_high'], 5)], r['p_raw_sign_flip'], r['p_holm'])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--analyze-only', action='store_true')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.analyze_only:
        records = [json.loads(line) for line in (TRUNCATION_RECORDS/'per_image.jsonl').read_text().splitlines()]
        return analyse(records, a.out)
    schedule = generate(a.out)  # the pipeline is released when generate returns
    records = evaluate(a.out, schedule)
    analyse(records, a.out)
    json.dump({p.name: file_sha256(p) for p in sorted(a.out.rglob('*')) if p.is_file()}, open(a.out/'MANIFEST.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
