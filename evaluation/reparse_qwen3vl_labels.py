"""Re-derive every image's Qwen3-VL labels from the stored raw model output, without calling the model.

Applies the parser of label_images_qwen3vl.py: parse the first answer; fields still invalid take the re-ask's value;
otherwise null. The labels written by the live run are kept as `codes_live`, the re-derived labels are `codes`.
The files in records/qwen3vl are the output of this script; running it on them reproduces them unchanged.

usage: python evaluation/reparse_qwen3vl_labels.py LABELS.jsonl [...] [--out-dir DIR]
  writes <name>_reparsed.jsonl next to each input, or DIR/<name>.jsonl with --out-dir
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.label_images_qwen3vl import ALLOWED, parse  # noqa: E402


def reparse(path):
    rows, changed = [], 0
    for r in map(json.loads, path.read_text().splitlines()):
        live = r.get('codes_live', r['codes'])
        codes, bad = parse(r['raw'])
        if bad and 'raw_reask' in r:
            codes2, _ = parse(r['raw_reask'])
            for k in bad:
                codes[k] = codes2.get(k)
        new = {k: codes.get(k) for k in ALLOWED}
        changed += sum(new[k] != live[k] for k in ALLOWED)
        rows.append({**r, 'codes_live': live, 'codes': new})
    return rows, changed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inputs', nargs='+', type=Path)
    ap.add_argument('--out-dir', type=Path)
    a = ap.parse_args()
    for p in a.inputs:
        rows, changed = reparse(p)
        out = a.out_dir / p.name if a.out_dir else p.with_name(p.stem + '_reparsed.jsonl')
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        print(f'{p}: {len(rows)} images, {changed} field values differ from the live run, '
              f'{sum(any(v is None for v in r["codes"].values()) for r in rows)} images with a missing field -> {out}')


if __name__ == '__main__':
    main()
