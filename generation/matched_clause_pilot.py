"""Pilot of the matched-clause experiment: SD 2.1, five images per prompt and version (310 images), CLIP ViT-B/32 labels
and similarities, and the paired I - C comparisons (Appendix D).

Design: datasets/matched_clause/prompt_pairs.csv and datasets/matched_clause/pilot/ (schedule, design manifest; the
protocol is plan/matched_clause_pilot_protocol.md). Settings: 768 x 768, 50 steps, guidance 7.5, empty negative prompt,
DPMSolverMultistepScheduler; both versions of a block share one initial latent. Resumable; every image gets a receipt.
The stored run is records/matched_clause_pilot (generation receipts, configuration, per-image CLIP scores) and
results/matched_clause_pilot (tables).

usage:
  python generation/matched_clause_pilot.py --model SD21_DIR --clip CLIP_B32_DIR --out RUN_DIR
      SD21_DIR: local snapshot of sd2-community/stable-diffusion-2-1 (revision bb2154823665391b4fb29b0b9cf82a198964ee05,
      fp16 files) with a download_manifest.json of file hashes (records/matched_clause_pilot/sd21_download_manifest.json)
  python generation/matched_clause_pilot.py --analyze-only --out DIR
      recompute the prompt-level tables from records/matched_clause_pilot/per_image.jsonl (no model)
"""
import argparse
import csv
import gc
import hashlib
import itertools
import json
import math
import os
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
import sys
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import CONCEPTS, MATCHED_DESIGN, PILOT_DESIGN, PILOT_RECORDS, PLAN as PLAN_DIR  # noqa: E402
from analysis.clip_statistics import holm, parity_mad, write_csv  # noqa: E402
from evaluation.clip_original_archive import file_sha256, reference_texts  # noqa: E402

# design files named in datasets/matched_clause/pilot/design_manifest.json
DESIGN_FILES = {'PROTOCOL.md': PLAN_DIR / 'matched_clause_pilot_protocol.md',
                'prompt_pairs.csv': MATCHED_DESIGN / 'prompt_pairs.csv',
                'generation_schedule.csv': PILOT_DESIGN / 'generation_schedule.csv'}


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def prepare_output(out, config):
    """Create or resume a run folder; settings cannot change once images have been generated."""
    out.mkdir(parents=True, exist_ok=True)
    path = out/'execution_config.json'
    if path.exists() and json.loads(path.read_text()) != config:
        assert not (out/'generation_receipts.jsonl').exists(), 'Cannot mix configurations after generation begins'
    save(path, config)
    return out


def load_receipts(out, schedule):
    path = out/'generation_receipts.jsonl'
    expected = {row['run_order']:row for row in schedule}
    completed = {}
    if path.exists():
        for line in path.read_text().splitlines():
            receipt = json.loads(line)
            order = receipt['run_order']
            assert order in expected and order not in completed, 'Unknown or duplicate generation receipt'
            assert receipt['schedule_row'] == expected[order], 'Generation receipt changes the frozen schedule'
            image = out/'images'/f'{int(order):03d}.png'
            assert file_sha256(image) == receipt['file_sha256'], 'Generated image changed since receipt'
            completed[order] = receipt
    return completed


def validate_plan():
    manifest = json.loads((PILOT_DESIGN / 'design_manifest.json').read_text())
    for name, digest in manifest['files_sha256'].items():
        assert file_sha256(DESIGN_FILES[name]) == digest, name
    pairs = {r['entry']: r for r in read_csv(DESIGN_FILES['prompt_pairs.csv'])}
    schedule = read_csv(DESIGN_FILES['generation_schedule.csv'])
    assert len(pairs) == 31 and len(schedule) == 310
    blocks = defaultdict(list)
    for index, row in enumerate(schedule, 1):
        assert int(row['run_order']) == index
        pair = pairs[row['entry']]
        assert row['arm'] in ('I', 'C') and row['category'] == pair['category']
        assert row['submitted_prompt'] == pair['instruction_prompt' if row['arm'] == 'I' else 'control_prompt']
        blocks[(row['entry'], row['repetition'])].append(row)
    assert len(blocks) == 155
    for rows in blocks.values():
        assert {r['arm'] for r in rows} == {'I', 'C'}
        assert len(rows) == 2 and len({r['latent_seed'] for r in rows}) == 1
    assert all({r['repetition'] for r in schedule if r['entry'] == e} == {'1','2','3','4','5'} for e in pairs)
    return manifest, pairs, schedule


def analyze(records, pairs, concepts, out):
    from scipy.stats import t, shapiro
    expected = Counter()
    for e, pair in pairs.items():
        axes = ['Skin Tone Bias','Ethnic Representation Bias'] if pair['category'] == 'Racial Bias' else [pair['category']]
        for axis in axes:
            for arm in ('C','I'):
                expected[(axis,e,arm)] = 5
    cells = defaultdict(list)
    seen = set()
    for row in records:
        key = (row['category'],row['entry'],row['arm'])
        assert (row['category'],row['image_id']) not in seen
        seen.add((row['category'],row['image_id']))
        assert row['parity_label'] in concepts[row['category']]['Parity']['concepts']
        assert all(math.isfinite(row[m]) for m in ('concept_similarity','reference_similarity'))
        cells[key].append(row)
    assert Counter({k:len(v) for k,v in cells.items()}) == expected
    assert len(records) == 430 and len({r['image_id'] for r in records}) == 310
    metrics = ('label_imbalance_mad','concept_similarity','reference_similarity')
    values = {}
    for key, rows in cells.items():
        labels = concepts[key[0]]['Parity']['concepts']
        values[key] = {'label_imbalance_mad':parity_mad({l:sum(r['parity_label']==l for r in rows)/5 for l in labels},labels),
                       **{m:mean(r[m] for r in rows) for m in metrics[1:]}}
    results, descriptive = [], []
    for axis in sorted({k[0] for k in cells}):
        entries = sorted({k[1] for k in cells if k[0] == axis},key=int)
        for arm in ('C','I'):
            descriptive.append({'category':axis,'arm':arm,'prompt_n':len(entries),'image_n':len(entries)*5,
                                **{m:mean(values[(axis,e,arm)][m] for e in entries) for m in metrics}})
        for metric in metrics:
            differences = [values[(axis,e,'I')][metric]-values[(axis,e,'C')][metric] for e in entries]
            n, effect, sd = len(differences), mean(differences), stdev(differences)
            p = sum(abs(mean(s*d for s,d in zip(signs,differences))) >= abs(effect)-1e-12
                    for signs in itertools.product((-1,1),repeat=n))/2**n
            half = t.ppf(.975,n-1)*sd/math.sqrt(n)
            results.append({'category':axis,'metric':metric,'comparison':'I minus C','prompt_n':n,
                            'mean_difference':effect,'sd_difference':sd,'cohen_dz':effect/sd if sd else 'undefined',
                            'ci95_low':effect-half,'ci95_high':effect+half,'ci_method':'unadjusted paired t; normality assumption',
                            'p_raw_sign_flip':p,'p_holm':None,'family_size':4 if metric == metrics[0] else 8,
                            'shapiro_p':float(shapiro(differences).pvalue) if sd else 'constant',
                            'scope':'selected-prompt exploratory probe; sign-exchangeability assumed'})
    for primary in (True,False):
        family = [r for r in results if (r['metric']==metrics[0]) == primary]
        for row,p in zip(family,holm([r['p_raw_sign_flip'] for r in family])):
            row['p_holm'] = p
    write_csv(out/'prompt_comparisons.csv',results)
    write_csv(out/'condition_descriptives.csv',descriptive)
    write_csv(out/'prompt_cells.csv',[{'category':k[0],'entry':k[1],'arm':k[2],**v} for k,v in sorted(values.items())])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', type=Path, help='local SD 2.1 fp16 snapshot with download_manifest.json')
    ap.add_argument('--clip', type=Path, help='local copy of openai/clip-vit-base-patch32')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--analyze-only', action='store_true')
    args = ap.parse_args()
    if args.analyze_only:
        _, pairs, _ = validate_plan()
        records = [json.loads(line) for line in (PILOT_RECORDS/'per_image.jsonl').read_text().splitlines()]
        args.out.mkdir(parents=True, exist_ok=True)
        return analyze(records, pairs, json.loads(CONCEPTS.read_text()), args.out)
    assert args.model and args.clip, 'generation needs --model and --clip'
    MODEL, CLIP = args.model, args.clip
    import numpy as np
    import torch
    import diffusers
    import transformers
    from PIL import Image
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    from transformers import CLIPModel, CLIPProcessor
    manifest,pairs,schedule = validate_plan()
    source = json.loads((MODEL/'download_manifest.json').read_text())
    for name,digest in source['files_sha256'].items():
        assert file_sha256(MODEL/name) == digest, name
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    dtype = torch.float16 if device in ('cuda', 'mps') else torch.float32
    config = {'design_hashes':manifest['files_sha256'],'model_hashes':source['files_sha256'],
              'device':device,'dtype':str(dtype),'diffusers':diffusers.__version__,
              'torch':torch.__version__,'transformers':transformers.__version__,
              'height':768,'width':768,'steps':50,'guidance':7.5,'negative_prompt':'',
              'scheduler':'DPMSolverMultistepScheduler','historical_revision_verified':False,
              'attention_backend':'mps_sliced_attention' if device == 'mps' else 'default_sdpa',
              'attention_slice_size':'max' if device == 'mps' else None,
              'pipeline_code_sha256':file_sha256(Path(__file__))}
    out = prepare_output(args.out, config)
    (out/'images').mkdir(exist_ok=True)
    status = out/'STATUS.json'
    try:
        completed = load_receipts(out, schedule)
        save(status,{'phase':'loading','completed_images':len(completed),'expected_images':310})
        pipe = StableDiffusionPipeline.from_pretrained(str(MODEL),variant='fp16',torch_dtype=dtype,
                                                       local_files_only=True).to(device)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        save(out/'scheduler_config.json',dict(pipe.scheduler.config))
        if device == 'cpu':
            torch.set_num_threads(min(8,os.cpu_count() or 1))
        if device == 'mps':
            pipe.enable_attention_slicing('max')
        processors = sorted({type(value).__name__ for value in pipe.unet.attn_processors.values()})
        assert processors == (['SlicedAttnProcessor'] if device == 'mps' else ['AttnProcessor2_0']), processors
        save(out/'attention_processors.json',{'unet_processors':processors})
        pipe.set_progress_bar_config(disable=False)
        for pair in pairs.values():
            for arm,column in [('I','instruction_prompt'),('C','control_prompt')]:
                ids = pipe.tokenizer(pair[column],truncation=False)['input_ids']
                assert ids == json.loads(pair['instruction_token_ids' if arm=='I' else 'control_token_ids'])
                assert len(ids) == int(pair['equal_positions']) <= 77

        def render(prompt,seed):
            latent = torch.randn((1,4,96,96),generator=torch.Generator(device='cpu').manual_seed(seed),dtype=torch.float32)
            latent_digest = hashlib.sha256(latent.numpy().tobytes()).hexdigest()
            result = pipe(prompt,negative_prompt='',height=768,width=768,num_inference_steps=50,
                          guidance_scale=7.5,latents=latent.to(device=device,dtype=dtype),output_type='np').images[0]
            assert np.isfinite(result).all() and float(result.max()-result.min()) > 0, 'Invalid/constant image'
            return Image.fromarray((result.clip(0,1)*255).round().astype('uint8')),latent_digest

        if not (out/'pilot.json').exists():
            started = time.monotonic()
            image,_ = render('A blue cube and a red sphere on a plain white table.',20261003)
            image.save(out/'technical_pilot.png')
            elapsed = time.monotonic()-started
            save(out/'pilot.json',{'seconds':elapsed,'estimated_generation_hours':elapsed*310/3600,
                                  'purpose':'technical only; no outcome tuning'})
            print('Technical pilot seconds:',elapsed,'estimated generation hours:',elapsed*310/3600,flush=True)
        receipts = out/'generation_receipts.jsonl'
        for row in schedule:
            order = row['run_order']; path = out/'images'/f'{int(order):03d}.png'
            if order in completed:
                continue
            if path.exists():
                path.rename(path.with_name(path.stem+'_unreceipted_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.png'))
            save(status,{'phase':'generating','completed_images':len(completed),'expected_images':310,
                         'current_run_order':order})
            started = time.monotonic()
            image,latent_digest = render(row['submitted_prompt'],int(row['latent_seed']))
            image.save(path)
            receipt = {'run_order':order,'schedule_row':row,'file_sha256':file_sha256(path),
                       'latent_sha256':latent_digest,'seconds':time.monotonic()-started}
            with receipts.open('a') as handle:
                handle.write(json.dumps(receipt)+'\n'); handle.flush(); os.fsync(handle.fileno())
            completed[order] = receipt
            save(status,{'phase':'generating','completed_images':len(completed),'expected_images':310})
            print('Generated',len(completed),'/310',flush=True)
        for rows in [schedule[i:i+2] for i in range(0,310,2)]:
            assert completed[rows[0]['run_order']]['latent_sha256'] == completed[rows[1]['run_order']]['latent_sha256']
        del pipe
        gc.collect()
        if device == 'mps': torch.mps.empty_cache()
        if device == 'cuda': torch.cuda.empty_cache()
        save(status,{'phase':'evaluating','completed_images':310,'expected_images':310})
        concepts = json.loads(CONCEPTS.read_text())
        model = CLIPModel.from_pretrained(str(CLIP),local_files_only=True).to('cpu').eval()
        processor = CLIPProcessor.from_pretrained(str(CLIP),local_files_only=True,use_fast=False)
        cache = {}
        def text_features(texts):
            key = tuple(texts)
            if key not in cache:
                inputs = processor(text=texts,return_tensors='pt',padding=True,truncation=False)
                assert inputs['input_ids'].shape[-1] <= 77
                f = model.get_text_features(**inputs)
                cache[key] = f/f.norm(dim=-1,keepdim=True)
            return cache[key]
        records = []
        with torch.inference_mode():
            for row in schedule:
                image = Image.open(out/'images'/f"{int(row['run_order']):03d}.png").convert('RGB')
                f = model.get_image_features(**processor(images=image,return_tensors='pt'))
                f = f/f.norm(dim=-1,keepdim=True)
                axes = ['Skin Tone Bias','Ethnic Representation Bias'] if row['category']=='Racial Bias' else [row['category']]
                for axis in axes:
                    definition = concepts[axis]; labels = definition['Parity']['concepts']
                    label_sims = (f@text_features(labels).T).squeeze(0)
                    scores = {}
                    for metric in ('Embedding Distance','Stereotype Frequency Score'):
                        for subtype,phrases in definition[metric].items():
                            texts = reference_texts(phrases,pairs[row['entry']]['baseline_prompt'])
                            scores[f'{metric} - {subtype}'] = {'references':texts,'cosine_similarities':(f@text_features(texts).T).squeeze(0).tolist()}
                    records.append({'image_id':int(row['run_order']),'entry':row['entry'],'arm':row['arm'],'category':axis,
                                    'image_sha256_pixels':hashlib.sha256(image.tobytes()).hexdigest(),
                                    'parity_label':labels[int(label_sims.argmax())],'label_cosines':label_sims.tolist(),
                                    'concept_similarity':mean(mean(v['cosine_similarities']) for k,v in scores.items() if k.startswith('Embedding Distance')),
                                    'reference_similarity':mean(max(v['cosine_similarities']) for k,v in scores.items() if k.startswith('Stereotype Frequency')),'scores':scores})
                print('Evaluated',row['run_order'],'/310',flush=True)
        temporary = out/'per_image.jsonl.tmp'
        temporary.write_text(''.join(json.dumps(r)+'\n' for r in records))
        temporary.replace(out/'per_image.jsonl')
        analyze(records,pairs,concepts,out)
        save(out/'evaluation_manifest.json',{'status':'complete','record_count':len(records),'unique_images':310,
             'per_image_sha256':file_sha256(out/'per_image.jsonl'),'generation_receipts_sha256':file_sha256(receipts),
             'concepts_sha256':file_sha256(CONCEPTS),
             'clip_weights_sha256':file_sha256(CLIP/'pytorch_model.bin'),'evaluation_device':'cpu',
             'tokenizer_class':type(processor.tokenizer).__name__,'processor_use_fast':False,
             'primary_family':4,'secondary_family':8,'human_ratings_conducted':False})
        save(status,{'phase':'complete','completed_images':310,'axis_records':430,
                     'manuscript_update':'requires review of actual outcomes; not performed by this script'})
        print('Complete:',out,flush=True)
    except Exception as exc:
        save(status,{'phase':'failed','error':str(exc),'traceback':traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
