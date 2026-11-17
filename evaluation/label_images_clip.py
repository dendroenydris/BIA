"""CLIP labels (secondary evaluator, plan/PLAN.md) for the images of the main and matched-clause experiments.

For every image of a generation run, the argmax CLIP label on each attribute of its category, with the original label
lists of datasets/bias_concepts.json. Variant b32 = the original evaluator (ViT-B/32, bare labels); l14ens = ViT-L/14
with the five-template ensemble. Output rows {"key": gen<run_order>, "codes": {attribute_field: label}}, read by
analysis/label_imbalance.py. Resumable.

RUN_DIR is a generation output folder (generation_receipts.jsonl and images/NNNN.png, see generation/generate_*.py);
the images are not stored in this repository.

usage: python evaluation/label_images_clip.py {b32|l14ens} RUN_DIR OUT.jsonl [--clip-b32 DIR]
  stored output: records/clip/<set>_<variant>.jsonl
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import CONCEPTS  # noqa: E402

L14 = ('openai/clip-vit-large-patch14', '32bd64288804d66eefd0ccbe215aa642df71cc41')
TEMPLATES = ['a photo of {}.', 'a cropped photo of {}.', 'a close-up photo of {}.', 'a picture of {}.', 'an image of {}.']
FIELD = {'Gender Bias': 'gender_presentation', 'Skin Tone Bias': 'skin_tone', 'Ethnic Representation Bias': 'ethnic_appearance',
         'Geo-Cultural Bias': 'cultural_scene'}
AXES = {'Gender Bias': ['Gender Bias'], 'Racial Bias': ['Skin Tone Bias', 'Ethnic Representation Bias'],
        'Geo-Cultural Bias': ['Geo-Cultural Bias'], 'Age Bias': []}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('variant', choices=['b32', 'l14ens'])
    ap.add_argument('run', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--clip-b32', default='openai/clip-vit-base-patch32', help='hub ID or local copy of CLIP ViT-B/32')
    a = ap.parse_args()
    variant, run, out = a.variant, a.run, a.out
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    torch.set_num_threads(8)
    if variant == 'b32':
        model = CLIPModel.from_pretrained(a.clip_b32).eval()
        proc = CLIPProcessor.from_pretrained(a.clip_b32, use_fast=False)
    else:
        model = CLIPModel.from_pretrained(L14[0], revision=L14[1]).eval()
        proc = CLIPProcessor.from_pretrained(L14[0], revision=L14[1], use_fast=False)
    concepts = json.loads(CONCEPTS.read_text())
    text = {}
    with torch.inference_mode():
        for axis in FIELD:
            labels = concepts[axis]['Parity']['concepts']
            tpls = ['{}'] if variant == 'b32' else TEMPLATES
            per = []
            for tpl in tpls:
                e = model.get_text_features(**proc(text=[tpl.format(l) for l in labels], return_tensors='pt', padding=True))
                per.append(e / e.norm(dim=-1, keepdim=True))
            t = torch.stack(per).mean(0)
            text[axis] = (labels, t / t.norm(dim=-1, keepdim=True))
    receipts = [json.loads(l) for l in (run / 'generation_receipts.jsonl').read_text().splitlines()]
    done = set()
    if out.exists():
        done = {json.loads(l)['key'] for l in out.read_text().splitlines()}
    todo = [r for r in receipts if f"gen{int(r['run_order']):04d}" not in done and AXES[r['schedule_row']['category']]]
    with out.open('a') as fh, torch.inference_mode():
        for i in range(0, len(todo), 32):
            batch = todo[i:i + 32]
            imgs = [Image.open(run / 'images' / f"{int(r['run_order']):04d}.png").convert('RGB') for r in batch]
            f = model.get_image_features(**proc(images=imgs, return_tensors='pt'))
            f = f / f.norm(dim=-1, keepdim=True)
            for j, r in enumerate(batch):
                codes = {}
                for axis in AXES[r['schedule_row']['category']]:
                    labels, t = text[axis]
                    codes[FIELD[axis]] = labels[int((f[j:j + 1] @ t.T).argmax())]
                fh.write(json.dumps({'key': f"gen{int(r['run_order']):04d}", 'variant': variant,
                                     'image_sha256_pixels': hashlib.sha256(imgs[j].tobytes()).hexdigest(), 'codes': codes}) + '\n')
            fh.flush()
            print(f'{variant} {run.parent.name}: {len(done) + i + len(batch)}/{len(done) + len(todo)}', flush=True)


if __name__ == '__main__':
    main()
