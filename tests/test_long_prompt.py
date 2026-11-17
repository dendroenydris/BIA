"""Tokenizer-only checks for generation/long_prompt.chunk_ids against the SD 2.1 tokenizer: the first chunk equals the
standard truncated input, the chunks together keep every token, and exactly the eight known LLM-assisted inputs overflow.

usage: python tests/test_long_prompt.py [TOKENIZER]
  TOKENIZER: hub ID or local copy of the SD 2.1 tokenizer (default: sd2-community/stable-diffusion-2-1, subfolder tokenizer,
  revision bb2154823665391b4fb29b0b9cf82a198964ee05)
"""
import csv
import sys
from pathlib import Path

from transformers import CLIPTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generation.long_prompt import chunk_ids  # noqa: E402
from paths import PROMPTS  # noqa: E402

TRUNCATED = {14, 15, 16, 17, 18, 19, 22, 23}

if len(sys.argv) > 1:
    tok = CLIPTokenizer.from_pretrained(sys.argv[1])
else:
    tok = CLIPTokenizer.from_pretrained('sd2-community/stable-diffusion-2-1', subfolder='tokenizer',
                                        revision='bb2154823665391b4fb29b0b9cf82a198964ee05')
rows = [r for r in csv.reader(PROMPTS.open(encoding='utf-8-sig'), delimiter=';')][1:]
for entry, row in enumerate(rows, 1):
    for text in row[2:5]:
        chunks = chunk_ids(tok, text)
        standard = tok(text, padding='max_length', max_length=77, truncation=True)['input_ids']
        assert chunks[0] == standard, (entry, text[:40])
        assert all(len(c) == 77 for c in chunks)
        full = tok(text, add_special_tokens=False)['input_ids']
        assert [t for c in chunks for t in c[1:-1] if t not in (tok.eos_token_id, tok.pad_token_id)] == \
               [t for t in full if t not in (tok.eos_token_id, tok.pad_token_id)], (entry, 'tokens lost')
        overflow = len(full) + 2 > 77
        assert overflow == (entry in TRUNCATED and text == row[4]), (entry, len(full) + 2)
print('ok: chunk 1 == standard truncation for all 120 inputs; full chunks keep every token; '
      'overflow set ==', sorted(TRUNCATED))
for entry in sorted(TRUNCATED):
    text = rows[entry - 1][4]
    print(f'A{entry}: {len(tok(text)["input_ids"])} positions -> {len(chunk_ids(tok, text))} chunks')
