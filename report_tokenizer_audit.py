"""Verify a supplied tokenizer replay and report explicitly post-hoc sensitivity."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

from bias_analysis import ROOT, STRATEGIES, load_completed_evaluation, parity_mad, prompt_comparisons, write_csv


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report(tokenizer_dir, audit_dir, image_dir, output_dir, comparison_tokenizer=None):
    from transformers import CLIPTokenizer
    if output_dir.exists():
        raise FileExistsError('Use a new output directory; original evidence is never overwritten')
    manifest = json.loads((audit_dir/'tokenizer_manifest.json').read_text())
    assert manifest['status']=='complete' and manifest['records']==120 and manifest['max_length']==77
    assert manifest['truncation_side']=='right'
    for name, digest in manifest['tokenizer_files_sha256'].items():
        assert sha(tokenizer_dir/name)==digest, name
    source = json.loads((tokenizer_dir/'download_manifest.json').read_text())
    for name, digest in source['files_sha256'].items():
        assert sha(tokenizer_dir/name)==digest, name
    tokenizer = CLIPTokenizer.from_pretrained(str(tokenizer_dir),local_files_only=True)
    assert tokenizer.model_max_length==77 and tokenizer.truncation_side=='right'
    actual_preprocessing='ftfy.fix_text' if tokenizer.fix_text is not None else 'BasicTokenizer fallback; ftfy unavailable'
    assert manifest['preprocessing_path']==actual_preprocessing
    prompts=list(csv.DictReader((ROOT/'datasets/prompts.csv').open(encoding='utf-8-sig'),delimiter=';'))
    records=[json.loads(s) for s in (audit_dir/'tokenizer_audit.jsonl').read_text().splitlines()]
    keys={(r['prompt_entry'],r['condition']) for r in records}
    assert len(records)==len(keys)==120 and keys=={(i,s) for i in range(1,41) for s in range(3)}
    summaries=[]
    columns=['Neutral_prompt','Bias_aware_prompt_engineering','Prompt_engineering_with_the_help_of_LLM']
    for row in records:
        p=prompts[row['prompt_entry']-1];text=p[columns[row['condition']]]
        assert row['submitted_text']==text and row['category']==p['Main_Category']
        full=tokenizer(text,truncation=False)['input_ids']
        encoded=tokenizer(text,truncation=True,max_length=77)['input_ids']
        content=tokenizer(text,add_special_tokens=False,truncation=False)['input_ids']
        assert row['full_token_count_with_special']==len(full)
        assert row['encoded_token_ids']==encoded
        assert row['encoded_tokens']==tokenizer.convert_ids_to_tokens(encoded)
        assert row['encoded_text']==tokenizer.decode(encoded,skip_special_tokens=True)
        assert row['removed_content_token_ids']==content[75:]
        assert row['removed_text']==tokenizer.decode(content[75:],skip_special_tokens=True)
        assert row['truncated']==(len(full)>77)
        summaries.append({'prompt_entry':row['prompt_entry'],'category':row['category'],
                          'condition':STRATEGIES[row['condition']], 'full_positions':len(full),
                          'retained_positions':len(encoded),'removed_content_tokens':len(content[75:]),
                          'truncated':row['truncated'],'submitted_text':text,'retained_text':row['encoded_text'],
                          'retained_last_eight_content_tokens':json.dumps(row['encoded_tokens'][1:-1][-8:]),
                          'removed_suffix':row['removed_text']})
    equivalence=None
    if comparison_tokenizer:
        other=CLIPTokenizer.from_pretrained(str(comparison_tokenizer),local_files_only=True)
        equivalence={'vocabulary_equal':tokenizer.encoder==other.encoder,'bpe_ranks_equal':tokenizer.bpe_ranks==other.bpe_ranks,
                     'all_120_encoded_sequences_equal':all(r['encoded_token_ids']==other(r['submitted_text'],truncation=True,max_length=77)['input_ids'] for r in records)}
    # Verify the entire completed image stream before explicitly defining a secondary subset.
    images=load_completed_evaluation(image_dir/'per_image.jsonl',ROOT/'datasets/bias_concepts.json',ROOT/'datasets/prompts.csv')
    concepts=json.loads((ROOT/'datasets/bias_concepts.json').read_text())
    excluded={r['prompt_entry'] for r in records if r['truncated']}
    assert all(prompts[i-1]['Main_Category']=='Racial Bias' for i in excluded)
    omitted={prompts[i-1]['Neutral_prompt'].strip() for i in excluded}
    racial=[r for r in images if r['category'] in ['Skin Tone Bias','Ethnic Representation Bias'] and r['baseline_prompt'].strip() not in omitted]
    contrasts=prompt_comparisons(racial,concepts)
    for r in contrasts:r['analysis_scope']='post-hoc SD-mirror no-overflow racial subset; separate Holm family; not primary inference'
    assert len(contrasts)==18 and all(r['prompt_n']==4 for r in contrasts)
    descriptions=[]
    for axis in ['Skin Tone Bias','Ethnic Representation Bias']:
        for strategy,name in STRATEGIES.items():
            cells={}
            for r in racial:
                if r['category']==axis and r['strategy']==strategy:cells.setdefault(r['prompt_id'],[]).append(r)
            vals=[]
            for rs in cells.values():
                labels=concepts[axis]['Parity']['concepts']
                vals.append({'label_imbalance_mad':parity_mad({l:sum(r['parity_label']==l for r in rs)/5 for l in labels},labels),
                             **{m:mean(r[m] for r in rs) for m in ['concept_similarity','reference_similarity']}})
            row={'category':axis,'strategy':name,'prompt_n':len(vals),'image_n':len(vals)*5}
            for metric in vals[0]:row.update({metric+'_mean':mean(r[metric] for r in vals),metric+'_sd_prompts':stdev(r[metric] for r in vals)})
            descriptions.append(row)
    output_dir.mkdir(parents=True)
    write_csv(output_dir/'all_inputs.csv',summaries)
    write_csv(output_dir/'truncated_inputs.csv',[r for r in summaries if r['truncated']])
    write_csv(output_dir/'no_overflow_racial_comparisons.csv',contrasts)
    write_csv(output_dir/'no_overflow_racial_descriptives.csv',descriptions)
    result={'status':'verified','resource_repository':source['repository'],'resource_revision':source['revision'],
            'historical_generation_revision_verified':False,'preprocessing_path':actual_preprocessing,
            'source_audit_jsonl_sha256':sha(audit_dir/'tokenizer_audit.jsonl'),'source_audit_manifest_sha256':sha(audit_dir/'tokenizer_manifest.json'),
            'prompts_sha256':sha(ROOT/'datasets/prompts.csv'),'source_per_image_sha256':sha(image_dir/'per_image.jsonl'),
            'independently_replayed_inputs':120,'truncated_entries':sorted(excluded),
            'truncated_conditions':[2],'affected_generated_images':len(excluded)*5,
            'comparison_with_evaluation_tokenizer':equivalence,
            'post_hoc_sensitivity':{'retained_racial_entries':[i for i,p in enumerate(prompts,1) if p['Main_Category']=='Racial Bias' and i not in excluded],
                                   'prompt_n_per_axis':4,'unique_images':len({r['image_id'] for r in racial}),
                                   'comparisons':18,'holm_family_size':18,'adjusted_below_05':sum(r['p_holm']<.05 for r in contrasts),
                                   'interpretation':'selected post-hoc subset; no historical encoding proof, equivalence, matched control, or population/causal inference'}}
    result['output_files_sha256']={p.name:sha(p) for p in output_dir.glob('*.csv')}
    (output_dir/'verification_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['tokenizer','audit-dir','image-dir','output-dir']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--comparison-tokenizer',type=Path)
    a=p.parse_args();report(a.tokenizer,a.audit_dir,a.image_dir,a.output_dir,a.comparison_tokenizer)
