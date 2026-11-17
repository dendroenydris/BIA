"""Write the original archive's images as PNG files with the manifests that evaluation/label_images_qwen3vl.py reads.

The 600 images are stored in one parquet file (column Image, row i = archive image i), distributed with the paper's
supplementary material and not in this repository. Two folders are written:

  orig600/NNNN.png  + manifest_original_archive.csv   keys orig0000-orig0599 (datasets/original_archive_schedule.csv)
  humans60/HNN.png  + manifest_human_labelled_60.csv  keys H01-H60 (human_ratings/original_archive/key.csv)

Every image's RGB pixel hash is checked against the stored Qwen3-VL records (records/qwen3vl/*.jsonl) and, for H01-H60,
against the key.

usage: python evaluation/export_archive_images.py --images IMAGES.parquet --out DIR
  then: python evaluation/label_images_qwen3vl.py --manifest DIR/manifest_original_archive.csv --root DIR
            --codebook evaluation/label_guide.md --out RUN_DIR
"""
import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import ARCHIVE_RATINGS, ORIGINAL_SCHEDULE, QWEN3VL_LABELS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--images', type=Path, required=True, help='image parquet of the original archive (column Image)')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    import pyarrow.parquet as pq
    from PIL import Image
    table = pq.read_table(a.images, columns=['Image']).column('Image').to_pylist()
    stored = {}
    for name in ('original_archive', 'human_labelled_60'):
        for line in (QWEN3VL_LABELS / f'{name}.jsonl').read_text().splitlines():
            r = json.loads(line)
            stored[r['key']] = r['image_sha256_pixels']
    keys = [r['key'] for r in csv.DictReader(ORIGINAL_SCHEDULE.open(newline=''))]
    sets = {'original_archive': ('orig600', [(k, int(k[4:]), f'{int(k[4:]):04d}.png', None) for k in keys]),
            'human_labelled_60': ('humans60', [(r['image_id'], int(r['archive_image_id']), f"{r['image_id']}.png", r['pixel_sha256'])
                                               for r in csv.DictReader((ARCHIVE_RATINGS / 'key.csv').open(newline=''))])}
    for name, (folder, items) in sets.items():
        (a.out / folder).mkdir(parents=True, exist_ok=True)
        rows = []
        for key, index, filename, expected in items:
            img = Image.open(io.BytesIO(table[index]['bytes'])).convert('RGB')
            digest = hashlib.sha256(img.tobytes()).hexdigest()
            assert digest == stored[key] and (expected is None or digest == expected), key
            img.save(a.out / folder / filename)
            rows.append({'key': key, 'path': f'{folder}/{filename}'})
        with (a.out / f'manifest_{name}.csv').open('w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['key', 'path'])
            w.writeheader()
            w.writerows(rows)
        print(f'{name}: {len(rows)} images written to {a.out / folder}, pixel hashes match the stored records')


if __name__ == '__main__':
    main()
