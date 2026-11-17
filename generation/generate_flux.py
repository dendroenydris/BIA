"""FLUX.1-schnell replication of the matched-clause probe (50 images per prompt x arm cell).

Runs on a rented GPU. Resumable: every image gets a fsynced receipt, and finished images are
skipped on restart. Both arms of a block share one initial latent, drawn on the CPU from the
schedule seed and hashed. After the first cell the script projects the total generation time
and stops (writing ABORT_PROJECTION.json and DONE) if it exceeds --max-gen-seconds.

usage: python run_flux_matched.py --schedule generation_schedule.csv --out OUT [--model REPO_OR_PATH]
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_receipts(out, schedule):
    expected = {r['run_order']: r for r in schedule}
    done = {}
    path = out / 'generation_receipts.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            assert r['schedule_row'] == expected[r['run_order']], 'receipt changes the frozen schedule'
            assert r['run_order'] not in done, 'duplicate receipt'
            assert sha256(out / 'images' / f"{int(r['run_order']):04d}.png") == r['file_sha256'], 'image changed'
            done[r['run_order']] = r
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--schedule', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='black-forest-labs/FLUX.1-schnell')
    ap.add_argument('--revision', default='741f7c3ce8b383c54771c7003378a50191e9efe9')
    ap.add_argument('--height', type=int, default=768)
    ap.add_argument('--width', type=int, default=768)
    ap.add_argument('--steps', type=int, default=4)
    ap.add_argument('--guidance', type=float, default=0.0)
    ap.add_argument('--max-seq', type=int, default=256)
    ap.add_argument('--batch', type=int, default=10)
    ap.add_argument('--max-gen-seconds', type=float, default=5400)
    ap.add_argument('--dtype', default='bfloat16')
    a = ap.parse_args()

    import numpy as np
    import torch
    import diffusers
    import transformers
    from PIL import Image
    from diffusers import FluxPipeline

    out = Path(a.out)
    (out / 'images').mkdir(parents=True, exist_ok=True)
    status = out / 'STATUS.json'
    schedule = list(csv.DictReader(open(a.schedule, newline='')))
    n_total = len(schedule)
    try:
        done = load_receipts(out, schedule)
        save(status, {'phase': 'loading', 'completed_images': len(done), 'expected_images': n_total})
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = getattr(torch, a.dtype)
        pipe = FluxPipeline.from_pretrained(a.model, revision=a.revision, torch_dtype=dtype).to(device)
        pipe.set_progress_bar_config(disable=True)
        config = {'model': a.model, 'revision': a.revision,
                  'height': a.height, 'width': a.width, 'steps': a.steps, 'guidance_scale': a.guidance,
                  'max_sequence_length': a.max_seq, 'batch': a.batch, 'dtype': a.dtype, 'device': device,
                  'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
                  'torch': torch.__version__, 'cuda': torch.version.cuda, 'diffusers': diffusers.__version__,
                  'transformers': transformers.__version__, 'python': platform.python_version(),
                  'scheduler': type(pipe.scheduler).__name__, 'scheduler_config': dict(pipe.scheduler.config),
                  'schedule_sha256': sha256(a.schedule), 'script_sha256': sha256(__file__)}
        if (out / 'execution_config.json').exists():
            prior = json.loads((out / 'execution_config.json').read_text())
            keys = ('model', 'revision', 'height', 'width', 'steps', 'guidance_scale', 'max_sequence_length', 'dtype', 'schedule_sha256')
            assert all(prior[k] == config[k] for k in keys), 'cannot resume with different settings'
        else:
            save(out / 'execution_config.json', config)

        # Encode each distinct prompt once; the T5 sequence must not truncate (the CLIP branch only supplies the
        # pooled vector and truncates at 77 positions, recorded per prompt); then free the encoders.
        prompts = sorted({r['submitted_prompt'] for r in schedule})
        embeds, checks = {}, []
        with torch.inference_mode():
            for p in prompts:
                clip_n = len(pipe.tokenizer(p, truncation=False)['input_ids'])
                t5_n = len(pipe.tokenizer_2(p, truncation=False)['input_ids'])
                assert t5_n <= a.max_seq, (p, t5_n)
                pe, pooled, _ = pipe.encode_prompt(prompt=p, prompt_2=None, device=device,
                                                   num_images_per_prompt=1, max_sequence_length=a.max_seq)
                embeds[p] = (pe, pooled)
                checks.append({'prompt': p, 'clip_tokens': clip_n, 'clip_pooled_truncated': clip_n > 77, 't5_tokens': t5_n,
                               'prompt_embeds_sha256': hashlib.sha256(pe.float().cpu().numpy().tobytes()).hexdigest()})
        save(out / 'encoding_checks.json', checks)
        pipe.text_encoder = pipe.text_encoder_2 = None
        if device == 'cuda':
            torch.cuda.empty_cache()

        ch = pipe.transformer.config.in_channels // 4
        lh, lw = 2 * (a.height // (pipe.vae_scale_factor * 2)), 2 * (a.width // (pipe.vae_scale_factor * 2))

        def latent(seed):
            g = torch.Generator(device='cpu').manual_seed(int(seed))
            x = torch.randn((1, ch, lh, lw), generator=g, dtype=torch.float32)
            return x, hashlib.sha256(x.numpy().tobytes()).hexdigest()

        cells = defaultdict(list)
        for r in schedule:
            cells[(r['entry'], r['arm'])].append(r)
        receipts = out / 'generation_receipts.jsonl'
        gen_seconds, gen_images, projected = 0.0, 0, None
        for ci, (key, rows) in enumerate(cells.items()):
            todo = [r for r in rows if r['run_order'] not in done]
            for b in range(0, len(todo), a.batch):
                batch = todo[b:b + a.batch]
                lat = [latent(r['latent_seed']) for r in batch]
                x = torch.cat([l[0] for l in lat]).to(dtype)
                packed = pipe._pack_latents(x, len(batch), ch, lh, lw).to(device)
                pe, pooled = embeds[batch[0]['submitted_prompt']]
                assert all(r['submitted_prompt'] == batch[0]['submitted_prompt'] for r in batch)
                save(status, {'phase': 'generating', 'completed_images': len(done), 'expected_images': n_total,
                              'cell': list(key), 'projected_generation_seconds': projected})
                if device == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.monotonic()
                with torch.inference_mode():
                    imgs = pipe(prompt_embeds=pe.repeat(len(batch), 1, 1), pooled_prompt_embeds=pooled.repeat(len(batch), 1),
                                latents=packed, height=a.height, width=a.width, num_inference_steps=a.steps,
                                guidance_scale=a.guidance, max_sequence_length=a.max_seq, output_type='np').images
                if device == 'cuda':
                    torch.cuda.synchronize()
                dt = time.monotonic() - t0
                gen_seconds += dt
                gen_images += len(batch)
                with receipts.open('a') as fh:
                    for r, im, (_, ld) in zip(batch, imgs, lat):
                        assert np.isfinite(im).all() and float(im.max() - im.min()) > 0, 'invalid image'
                        path = out / 'images' / f"{int(r['run_order']):04d}.png"
                        Image.fromarray((im.clip(0, 1) * 255).round().astype('uint8')).save(path)
                        rec = {'run_order': r['run_order'], 'schedule_row': r, 'file_sha256': sha256(path),
                               'latent_sha256': ld, 'batch_seconds': dt, 'batch_size': len(batch),
                               'utc': datetime.now(timezone.utc).isoformat()}
                        fh.write(json.dumps(rec) + '\n')
                        done[r['run_order']] = rec
                    fh.flush()
                    os.fsync(fh.fileno())
            if gen_images and projected is None:
                projected = gen_seconds / gen_images * (n_total - len(done)) + gen_seconds
                save(out / 'timing_projection.json', {'after_cell': list(key), 'seconds_per_image': gen_seconds / gen_images,
                                                      'projected_generation_seconds': projected,
                                                      'limit_seconds': a.max_gen_seconds})
                print(f'projection: {gen_seconds / gen_images:.2f} s/image, ~{projected / 60:.0f} min total', flush=True)
                if projected > a.max_gen_seconds:
                    save(out / 'ABORT_PROJECTION.json', {'projected_generation_seconds': projected, 'limit': a.max_gen_seconds})
                    save(status, {'phase': 'aborted_projection', 'completed_images': len(done), 'expected_images': n_total})
                    (out / 'DONE').write_text('aborted_projection\n')
                    return
            print(f'cell {ci + 1}/{len(cells)} {key}: {len(done)}/{n_total} images', flush=True)

        # Shared-latent check: within each (entry, repetition) block both arms used the same latent.
        blocks = defaultdict(set)
        for rec in done.values():
            blocks[(rec['schedule_row']['entry'], rec['schedule_row']['repetition'])].add(rec['latent_sha256'])
        assert len(done) == n_total and all(len(v) == 1 for v in blocks.values()), 'latent not shared across arms'
        save(status, {'phase': 'complete', 'completed_images': len(done), 'expected_images': n_total,
                      'generation_seconds_this_session': gen_seconds})
        (out / 'DONE').write_text('complete\n')
        print('complete', flush=True)
    except Exception as exc:
        save(status, {'phase': 'failed', 'error': str(exc), 'traceback': traceback.format_exc()})
        (out / 'DONE').write_text('failed\n')
        raise


if __name__ == '__main__':
    main()
