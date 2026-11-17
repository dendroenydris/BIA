"""Synthetic checks for generation/matched_clause_pilot.py: design validation, paired analysis, run-folder settings
and receipt checks. No model is loaded and no study result is asserted.

usage: python tests/test_matched_clause_pilot.py
"""
import json
import hashlib
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import generation.matched_clause_pilot as probe  # noqa: E402
from generation.matched_clause_pilot import validate_plan, analyze, read_csv  # noqa: E402
from paths import CONCEPTS  # noqa: E402

_, pairs, schedule = validate_plan()
concepts = json.loads(CONCEPTS.read_text())
records = []
for row in schedule:
    axes = ['Skin Tone Bias','Ethnic Representation Bias'] if row['category']=='Racial Bias' else [row['category']]
    for axis in axes:
        records.append({'category':axis,'entry':row['entry'],'arm':row['arm'],
                        'image_id':int(row['run_order']),
                        'parity_label':concepts[axis]['Parity']['concepts'][0],
                        'concept_similarity':.2 + (.01 if row['arm']=='I' else 0),
                        'reference_similarity':.3})
with tempfile.TemporaryDirectory() as folder:
    out = Path(folder)
    analyze(records,pairs,concepts,out)
    rows = read_csv(out/'prompt_comparisons.csv')
    assert len(rows)==12
    for row in rows:
        if row['metric']=='concept_similarity':
            assert abs(float(row['mean_difference'])-.01)<1e-12
            assert float(row['p_raw_sign_flip'])==2/2**int(row['prompt_n'])
            assert row['family_size']=='8'
        else:
            assert float(row['mean_difference'])==0 and float(row['p_holm'])==1
    assert sum(row['family_size']=='4' for row in rows)==4
    for invalid in (records[:-1],records+[records[0]],
                    [{**row,'concept_similarity':float('nan')} for row in records]):
        try:
            analyze(invalid,pairs,concepts,out)
        except AssertionError:
            pass
        else:
            raise AssertionError('Incomplete, duplicate, or nonfinite evidence accepted')
with tempfile.TemporaryDirectory() as folder:
    run = Path(folder)/'run'
    first = {'device':'cpu','attention_backend':'sliced'}
    assert probe.prepare_output(run, first) == run and (run/'execution_config.json').exists()
    assert probe.prepare_output(run, first) == run
    second = {'device':'cpu','attention_backend':'default_sdpa'}
    probe.prepare_output(run, second)  # allowed: no image generated yet
    (run/'generation_receipts.jsonl').write_text('{}\n')
    probe.prepare_output(run, second)  # same settings: resume
    for changed in (first, {'device':'mps','attention_backend':'default_sdpa'}):
        try:
            probe.prepare_output(run, changed)
        except AssertionError:
            pass
        else:
            raise AssertionError('Mixed configuration accepted after generation began')
with tempfile.TemporaryDirectory() as folder:
    out = Path(folder); (out/'images').mkdir()
    image = out/'images/001.png'; image.write_bytes(b'synthetic image hash fixture')
    receipt = {'run_order':'1','schedule_row':schedule[0],
               'file_sha256':hashlib.sha256(image.read_bytes()).hexdigest()}
    path = out/'generation_receipts.jsonl'
    assert probe.load_receipts(out,schedule)=={}
    path.write_text(json.dumps(receipt)+'\n')
    assert len(probe.load_receipts(out,schedule))==1
    for invalid in ([receipt,receipt],[{**receipt,'run_order':'999'}],
                    [{**receipt,'schedule_row':{**schedule[0],'arm':'changed'}}],
                    [{**receipt,'file_sha256':'wrong'}]):
        path.write_text(''.join(json.dumps(row)+'\n' for row in invalid))
        try:
            probe.load_receipts(out,schedule)
        except AssertionError:
            pass
        else:
            raise AssertionError('Invalid generation provenance accepted')
print('Matched-clause pilot synthetic checks passed; no study result asserted')
