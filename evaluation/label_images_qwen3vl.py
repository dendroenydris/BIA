"""Primary evaluator (evaluator_revision/PLAN.md): Qwen3-VL-8B-Instruct codes each image with the raters' codebook.

Reads a manifest CSV (columns: key, path relative to --root), writes one JSON line per image to
OUT/codes.jsonl (fsynced, resumable). Greedy decoding. A value outside the codebook's allowed list triggers one
re-ask naming the allowed values; still invalid -> that field is null and the image is flagged.

usage: python run_vlm_codebook.py --manifest m.csv --root DIR --codebook issued_codebook.md --out OUT
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import re
import time
import traceback
from pathlib import Path

ALLOWED = {
    'person_count': ['0', '1', '2', '3plus'],
    'gender_presentation': ['woman', 'man', 'mixed', 'indeterminate', 'no_person'],
    'skin_tone': ['light', 'medium', 'dark', 'mixed', 'indeterminate', 'no_person'],
    'ethnic_appearance': ['white', 'black', 'east_asian', 'south_southeast_asian', 'middle_eastern', 'latino', 'mixed',
                          'indeterminate', 'no_person'],
    'age_appearance': ['child', 'teenager', 'young_adult', 'middle_aged', 'older_adult', 'mixed', 'indeterminate', 'no_person'],
    'cultural_scene': ['western_european', 'south_asian', 'east_asian', 'middle_eastern', 'african', 'latin_american', 'mixed',
                       'none_or_indeterminate'],
}


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def instruction(codebook_text):
    """The codebook's 'Columns and allowed values' and 'Special cases' sections, verbatim."""
    start = codebook_text.index('You describe **what the image looks like**')
    end = codebook_text.index('## What this is for')
    body = codebook_text[start:end].strip()
    keys = ', '.join(f'"{k}"' for k in ALLOWED)
    return (f'You are coding one AI-generated image with the following codebook.\n\n{body}\n\n'
            f'Code this image. Ignore the confidence and notes columns. Answer with one JSON object and nothing else, '
            f'with exactly these keys: {keys}. Each value must be exactly one of the allowed values, as written.')


def parse(text):
    m = re.search(r'\{.*\}', text, re.S)
    if not m:
        return {}, list(ALLOWED)
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        # The model sometimes leaves codebook values unquoted ("person_count": 3plus); quote bare words and retry.
        try:
            obj = json.loads(re.sub(r':\s*([A-Za-z0-9_+]+)\s*([,}\n])', r': "\1"\2', m.group(0)))
        except json.JSONDecodeError:
            return {}, list(ALLOWED)
    codes, bad = {}, []
    for k, allowed in ALLOWED.items():
        v = str(obj.get(k, '')).strip().lower()
        if k == 'person_count':  # the model answers counts as numbers; the codebook bins 3 or more as 3plus
            v = '3plus' if v in ('3+', '3 plus') or (v.isdigit() and int(v) >= 3) else v
        if v in allowed:
            codes[k] = v
        else:
            bad.append(k)
    return codes, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--root', required=True)
    ap.add_argument('--codebook', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='Qwen/Qwen3-VL-8B-Instruct')
    ap.add_argument('--revision', default='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--max-new-tokens', type=int, default=200)
    a = ap.parse_args()

    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    status = out / 'STATUS.json'
    rows = list(csv.DictReader(open(a.manifest, newline='')))
    root = Path(a.root)
    try:
        path = out / 'codes.jsonl'
        done = set()
        if path.exists():
            done = {json.loads(l)['key'] for l in path.read_text().splitlines()}
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.bfloat16 if device == 'cuda' else torch.float32
        model = AutoModelForImageTextToText.from_pretrained(a.model, revision=a.revision, dtype=dtype).to(device).eval()
        proc = AutoProcessor.from_pretrained(a.model, revision=a.revision, use_fast=True)
        proc.tokenizer.padding_side = 'left'
        prompt = instruction(Path(a.codebook).read_text())
        config = {'model': a.model, 'revision': a.revision, 'dtype': str(dtype), 'device': device,
                  'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None, 'torch': torch.__version__,
                  'transformers': transformers.__version__, 'python': platform.python_version(),
                  'decoding': 'greedy', 'max_new_tokens': a.max_new_tokens, 'batch': a.batch,
                  'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'codebook_sha256': sha256(a.codebook),
                  'manifest_sha256': sha256(a.manifest), 'script_sha256': sha256(__file__)}
        if (out / 'execution_config.json').exists():
            prior = json.loads((out / 'execution_config.json').read_text())
            assert all(prior[k] == config[k] for k in ('model', 'revision', 'prompt_sha256', 'manifest_sha256')), 'settings changed'
        else:
            save(out / 'execution_config.json', config)
            (out / 'prompt.txt').write_text(prompt)

        def generate(images, texts):
            msgs = [[{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': t}]}] for t in texts]
            chat = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
            inputs = proc(text=chat, images=images, return_tensors='pt', padding=True).to(device)
            with torch.inference_mode():
                ids = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False)
            return proc.batch_decode(ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        todo = [r for r in rows if r['key'] not in done]
        t_start = time.monotonic()
        for b in range(0, len(todo), a.batch):
            batch = todo[b:b + a.batch]
            images = [Image.open(root / r['path']).convert('RGB') for r in batch]
            answers = generate(images, [prompt] * len(batch))
            records = []
            parsed = [parse(ans) for ans in answers]
            # Re-ask every image of the batch with invalid fields in one batched call, each naming its own fields.
            redo = [i for i, (_, bad) in enumerate(parsed) if bad]
            retries = {}
            if redo:
                texts = [prompt + '\n\nYour previous answer used values that are not allowed. Allowed values: '
                         + '; '.join(f'{k}: {", ".join(ALLOWED[k])}' for k in parsed[i][1]) + '.' for i in redo]
                retries = dict(zip(redo, generate([images[i] for i in redo], texts)))
            for i, (r, img, ans) in enumerate(zip(batch, images, answers)):
                codes, bad = parsed[i]
                rec = {'key': r['key'], 'path': r['path'], 'image_sha256_pixels': hashlib.sha256(img.tobytes()).hexdigest(),
                       'raw': ans, 'reasked': bool(bad)}
                if bad:
                    codes2, _ = parse(retries[i])
                    for k in bad:
                        codes[k] = codes2.get(k)
                    rec['raw_reask'] = retries[i]
                    rec['invalid_after_reask'] = [k for k in bad if codes.get(k) is None]
                rec['codes'] = {k: codes.get(k) for k in ALLOWED}
                records.append(rec)
            with path.open('a') as fh:
                for rec in records:
                    fh.write(json.dumps(rec) + '\n')
                fh.flush()
                os.fsync(fh.fileno())
            n = len(done) + b + len(batch)
            rate = (time.monotonic() - t_start) / (b + len(batch))
            save(status, {'phase': 'coding', 'coded': n, 'total': len(rows), 'seconds_per_image': rate})
            print(f'{n}/{len(rows)} coded, {rate:.2f} s/image', flush=True)
        save(status, {'phase': 'complete', 'coded': len(rows), 'total': len(rows)})
        (out / 'DONE').write_text('complete\n')
    except Exception as exc:
        save(status, {'phase': 'failed', 'error': str(exc), 'traceback': traceback.format_exc()})
        (out / 'DONE').write_text('failed\n')
        raise


if __name__ == '__main__':
    main()
