"""Repository locations used by every script. All paths are relative to this file; nothing points outside the repository.

Inputs that are not stored here (images, the original image parquet, model weights) are passed on the command line.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATASETS = ROOT / 'datasets'
PROMPTS = DATASETS / 'prompts.csv'                      # the 40 prompts in three versions (';'-separated)
CONCEPTS = DATASETS / 'bias_concepts.json'              # CLIP label sets and reference phrases
ORIGINAL_SCHEDULE = DATASETS / 'original_archive_schedule.csv'
MAIN_DESIGN = DATASETS / 'main_experiment'
MATCHED_DESIGN = DATASETS / 'matched_clause'
PILOT_DESIGN = MATCHED_DESIGN / 'pilot'
TRUNCATION_DESIGN = DATASETS / 'truncation_regeneration'
US_SOURCES = DATASETS / 'population_reference' / 'us_sources.json'

LABEL_GUIDE = ROOT / 'evaluation' / 'label_guide.md'
PLAN = ROOT / 'plan'

HUMAN = ROOT / 'human_ratings'
ARCHIVE_RATINGS = HUMAN / 'original_archive'           # rater_A.csv, rater_B.csv, key.csv (H01-H60)
FLUX_RATINGS = HUMAN / 'flux'                           # rater_1.csv, rater_2.csv, key.csv (F01-F60)

RECORDS = ROOT / 'records'
GENERATION_RECORDS = RECORDS / 'generation'             # <set>/generation_receipts.jsonl, execution_config.json, ...
QWEN3VL_LABELS = RECORDS / 'qwen3vl'                    # <set>.jsonl
CLIP_LABELS = RECORDS / 'clip'                          # <set>_<b32|l14ens>.jsonl, original_archive_b32/
PILOT_RECORDS = RECORDS / 'matched_clause_pilot'
TRUNCATION_RECORDS = RECORDS / 'truncation_regeneration'
TOKENIZER_RECORDS = RECORDS / 'tokenizer_audit'

RESULTS = ROOT / 'results'

# Image sets of the study: name -> (generation schedule, design)
SETS = {
    'main_sd21': (MAIN_DESIGN / 'generation_schedule.csv', 'main'),
    'main_flux': (MAIN_DESIGN / 'generation_schedule.csv', 'main'),
    'matched_sd21': (MATCHED_DESIGN / 'generation_schedule.csv', 'matched'),
    'matched_flux': (MATCHED_DESIGN / 'generation_schedule.csv', 'matched'),
    'original_archive': (ORIGINAL_SCHEDULE, 'main'),
}
