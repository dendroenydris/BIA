"""SD 2.1 at 50 images per prompt x arm cell (rented GPU, resumable): matched-clause probe or regenerated main study.

Same model files as the 5-per-cell probe (checked against its recorded hashes), same latent rule
(torch.randn((1,4,96,96)) on the CPU from the schedule seed), DPMSolverMultistepScheduler,
768 x 768, guidance 7.5, empty negative prompt. Steps default to 25 (the 5-per-cell probe used 50).
Every image gets a fsynced receipt; after the first cell the total generation time is projected
and the run stops (ABORT_PROJECTION.json + DONE) above --max-gen-seconds.

usage: python run_sd21_matched.py --schedule generation_schedule.csv [--pairs prompt_pairs.csv] --out OUT
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

REPO, REV = 'sd2-community/stable-diffusion-2-1', 'bb2154823665391b4fb29b0b9cf82a198964ee05'
# Model file hashes recorded by the 5-per-cell probe (results/matched_instruction_probe_20261011T130243957319Z).
WEIGHTS = {
    'text_encoder/model.fp16.safetensors': '681c555376658c81dc273f2d737a2aeb23ddb6d1d8e5b3a7064636d359a22668',
    'unet/diffusion_pytorch_model.fp16.safetensors': '8a3a4d7978884c5e4ef00b62641b1b544b257be2f6715d984188610ad6475ad2',
    'vae/diffusion_pytorch_model.fp16.safetensors': '3e4c08995484ee61270175e9e7a072b66a6e4eeb5f0c266667fe1f45b90daf9a',
    'scheduler/scheduler_config.json': '4cd9b9597ca64549df35016ca02bd3450ecbac70ccd8b0465b018be4ba54fe4b',
    'model_index.json': '14d34c52a8afbe01d22767226074ebb4367b59fe817cd7428e1d099125b7e34b',
}


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


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
    ap.add_argument('--pairs', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default=REPO)
    ap.add_argument('--revision', default=REV)
    ap.add_argument('--steps', type=int, default=25)
    ap.add_argument('--guidance', type=float, default=7.5)
    ap.add_argument('--batch', type=int, default=10)
    ap.add_argument('--max-gen-seconds', type=float, default=7200)
    ap.add_argument('--skip-weight-check', action='store_true')
    a = ap.parse_args()

    import numpy as np
    import torch
    import diffusers
    import transformers
    from huggingface_hub import snapshot_download
    from PIL import Image
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    out = Path(a.out)
    (out / 'images').mkdir(parents=True, exist_ok=True)
    status = out / 'STATUS.json'
    schedule = list(csv.DictReader(open(a.schedule, newline='')))
    pairs = {r['entry']: r for r in csv.DictReader(open(a.pairs, newline=''))} if a.pairs else {}
    n_total = len(schedule)
    try:
        done = load_receipts(out, schedule)
        save(status, {'phase': 'loading', 'completed_images': len(done), 'expected_images': n_total})
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float16 if device == 'cuda' else torch.float32
        local = Path(a.model) if Path(a.model).is_dir() else Path(snapshot_download(a.model, revision=a.revision))
        if not a.skip_weight_check:
            bad = {k: sha256(local / k) for k in WEIGHTS if sha256(local / k) != WEIGHTS[k]}
            assert not bad, f'model files differ from the 5-per-cell probe: {bad}'
        pipe = StableDiffusionPipeline.from_pretrained(str(local), variant='fp16' if not a.skip_weight_check else None,
                                                       torch_dtype=dtype, safety_checker=None,
                                                       requires_safety_checker=False).to(device)  # SD 2.1 ships none
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        config = {'model': a.model, 'revision': a.revision, 'weights_checked': not a.skip_weight_check,
                  'height': 768, 'width': 768, 'steps': a.steps, 'guidance_scale': a.guidance, 'negative_prompt': '',
                  'batch': a.batch, 'dtype': str(dtype), 'device': device,
                  'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
                  'torch': torch.__version__, 'cuda': torch.version.cuda, 'diffusers': diffusers.__version__,
                  'transformers': transformers.__version__, 'python': platform.python_version(),
                  'scheduler': type(pipe.scheduler).__name__, 'scheduler_config': dict(pipe.scheduler.config),
                  'schedule_sha256': sha256(a.schedule), 'script_sha256': sha256(__file__)}
        if (out / 'execution_config.json').exists():
            prior = json.loads((out / 'execution_config.json').read_text())
            keys = ('model', 'revision', 'steps', 'guidance_scale', 'dtype', 'schedule_sha256')
            assert all(prior[k] == config[k] for k in keys), 'cannot resume with different settings'
        else:
            save(out / 'execution_config.json', config)

        # Encode each distinct prompt once. With --pairs (matched-clause probe) the token ids must equal the frozen
        # pair table and fit 77 positions; otherwise prompts over 77 positions are truncated by the pipeline's
        # tokenizer, as in the original study, and recorded.
        embeds, checks = {}, []
        frozen = {}
        for p in pairs.values():
            frozen[p['instruction_prompt']] = json.loads(p['instruction_token_ids'])
            frozen[p['control_prompt']] = json.loads(p['control_token_ids'])
        with torch.inference_mode():
            _, negative = pipe.encode_prompt('', device=device, num_images_per_prompt=1,
                                             do_classifier_free_guidance=True, negative_prompt='')
            for prompt in sorted({r['submitted_prompt'] for r in schedule}):
                ids = pipe.tokenizer(prompt, truncation=False)['input_ids']
                if pairs and not a.skip_weight_check:
                    assert ids == frozen[prompt] and len(ids) <= 77, prompt
                embeds[prompt], _ = pipe.encode_prompt(prompt, device=device, num_images_per_prompt=1,
                                                       do_classifier_free_guidance=True, negative_prompt='')
                checks.append({'prompt': prompt, 'clip_tokens': len(ids), 'truncated_at_77': len(ids) > 77})
        save(out / 'encoding_checks.json', checks)

        ch = pipe.unet.config.in_channels
        side = 768 // pipe.vae_scale_factor

        def latent(seed):
            g = torch.Generator(device='cpu').manual_seed(int(seed))
            x = torch.randn((1, ch, side, side), generator=g, dtype=torch.float32)
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
                assert all(r['submitted_prompt'] == batch[0]['submitted_prompt'] for r in batch)
                lat = [latent(r['latent_seed']) for r in batch]
                x = torch.cat([l[0] for l in lat]).to(device=device, dtype=dtype)
                pe = embeds[batch[0]['submitted_prompt']]
                save(status, {'phase': 'generating', 'completed_images': len(done), 'expected_images': n_total,
                              'cell': list(key), 'projected_generation_seconds': projected})
                if device == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.monotonic()
                with torch.inference_mode():
                    imgs = pipe(prompt_embeds=pe.repeat(len(batch), 1, 1), negative_prompt_embeds=negative.repeat(len(batch), 1, 1),
                                height=768, width=768, num_inference_steps=a.steps, guidance_scale=a.guidance,
                                latents=x, output_type='np').images
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
