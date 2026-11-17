"""Build the 50-per-cell schedule for the FLUX.1-schnell matched-clause replication.

Same 31 prompt pairs as matched_instruction_probe. Repetitions 1-5 keep that probe's latent seeds;
repetitions 6-50 use sha256(master_seed:entry:rep) seeds. Both arms of a block share one seed."""
import csv, hashlib, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent/'matched_instruction_probe'
MASTER, REPS = 20261116, 50

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

pairs = list(csv.DictReader((OLD/'prompt_pairs.csv').open(newline='')))
old_seeds = {(r['entry'], r['repetition']): r['latent_seed'] for r in csv.DictReader((OLD/'generation_schedule.csv').open(newline=''))}
rows = []
for p in sorted(pairs, key=lambda r: int(r['entry'])):
    for rep in range(1, REPS+1):
        seed = old_seeds.get((p['entry'], str(rep))) or str(int(hashlib.sha256(f'{MASTER}:{p["entry"]}:{rep}'.encode()).hexdigest(), 16) % 2**63)
        for arm, col in (('C', 'control_prompt'), ('I', 'instruction_prompt')):
            rows.append({'run_order': len(rows)+1, 'entry': p['entry'], 'category': p['category'], 'repetition': rep,
                         'latent_seed': seed, 'arm': arm, 'submitted_prompt': p[col]})
assert len(rows) == 31*REPS*2 and len({(r['entry'], r['latent_seed']) for r in rows}) == 31*REPS
with (HERE/'generation_schedule.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
(HERE/'design_manifest.json').write_text(json.dumps({
    'model': 'black-forest-labs/FLUX.1-schnell', 'master_seed': MASTER, 'prompt_entries': 31, 'repetitions': REPS,
    'planned_images': len(rows), 'height': 768, 'width': 768, 'steps': 4, 'guidance_scale': 0.0, 'max_sequence_length': 256,
    'seed_rule': 'reps 1-5: matched_instruction_probe seeds; reps 6-50: sha256(master:entry:rep) mod 2**63; shared across arms',
    'latent_rule': 'torch.randn((1,16,96,96), cpu generator, float32) -> bf16 -> FluxPipeline._pack_latents',
    'files_sha256': {'prompt_pairs.csv (source)': sha(OLD/'prompt_pairs.csv'), 'generation_schedule.csv': sha(HERE/'generation_schedule.csv')},
}, indent=2)+'\n')
print(len(rows), 'rows')
