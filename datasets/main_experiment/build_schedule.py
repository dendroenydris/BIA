"""Regenerated three-condition main study: 40 prompt entries x 3 conditions x 50 repetitions = 6,000 rows per model.

Conditions (BIA/datasets/prompts.csv): baseline = Neutral_prompt; bias_aware = baseline verbatim + the category's
instruction sentence (standardised, see below); llm_assisted = Prompt_engineering_with_the_help_of_LLM.
One latent seed per (entry, repetition), shared by all three conditions: sha256(master:entry:rep) mod 2**63."""
import csv, hashlib, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[2]/'BIA/datasets/prompts.csv'
MASTER, REPS = 20261117, 50
COLS = [('baseline', 'Neutral_prompt'), ('bias_aware', None), ('llm_assisted', 'Prompt_engineering_with_the_help_of_LLM')]
# Manual (bias-aware) condition, standardised: the baseline string verbatim (historical spelling kept) plus the
# category's instruction sentence. In the original sheet 32/40 entries already had exactly this form; 8 also edited
# the baseline sentence and one misspelt the instruction ("backg"). Those are corrected here and listed in
# bias_aware_corrections.csv.
SENTENCE = {'Gender Bias': 'Avoid stereotypical depictions of gender.', 'Racial Bias': 'Avoid race and background bias.',
            'Age Bias': 'Avoid defaulting to age stereotypes.',
            'Geo-Cultural Bias': 'Avoid default stereotypes about the region and culture.'}


def bias_aware(p):
    base = p['Neutral_prompt'].strip()
    return base + (' ' if base.endswith('.') else '. ') + SENTENCE[p['Main_Category']]

prompts = list(csv.DictReader(SRC.open(encoding='utf-8-sig', newline=''), delimiter=';'))
assert len(prompts) == 40
rows = []
for e, p in enumerate(prompts, 1):
    for rep in range(1, REPS+1):
        seed = str(int(hashlib.sha256(f'{MASTER}:{e}:{rep}'.encode()).hexdigest(), 16) % 2**63)
        for arm, col in COLS:
            rows.append({'run_order': len(rows)+1, 'entry': e, 'category': p['Main_Category'], 'repetition': rep,
                         'latent_seed': seed, 'arm': arm, 'submitted_prompt': bias_aware(p) if col is None else p[col].strip(),
                         'baseline_prompt': p['Neutral_prompt'].strip()})
assert len(rows) == 40*REPS*3
corr = [{'entry': e, 'category': p['Main_Category'], 'original_bias_aware': p['Bias_aware_prompt_engineering'].strip(),
         'standardised_bias_aware': bias_aware(p)} for e, p in enumerate(prompts, 1)
        if p['Bias_aware_prompt_engineering'].strip() != bias_aware(p)]
with (HERE/'bias_aware_corrections.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(corr[0])); w.writeheader(); w.writerows(corr)
with (HERE/'generation_schedule.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
(HERE/'design_manifest.json').write_text(json.dumps({
    'design': 'three-condition main study, regenerated with shared latents and recorded settings',
    'source_prompts_sha256': hashlib.sha256(SRC.read_bytes()).hexdigest(), 'master_seed': MASTER,
    'prompt_entries': 40, 'conditions': [a for a, _ in COLS], 'repetitions': REPS, 'planned_images_per_model': len(rows),
    'generation_schedule_sha256': hashlib.sha256((HERE/'generation_schedule.csv').read_bytes()).hexdigest()}, indent=2)+'\n')
print(len(rows), 'rows')
