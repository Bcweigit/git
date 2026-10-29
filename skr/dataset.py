"""Validate a local YOLO detection dataset."""
import argparse
import hashlib
import math
from pathlib import Path

import yaml
from PIL import Image

SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def check_dataset(config):
    config = Path(config).expanduser().resolve()
    with config.open(encoding='utf-8') as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict) or 'download' in data:
        raise ValueError('Use a local dataset YAML mapping without a download field.')
    names = data.get('names')
    if isinstance(names, dict):
        if set(names) != set(range(len(names))):
            raise ValueError('Class IDs must be consecutive integers starting at 0.')
        names = [names[i] for i in range(len(names))]
    if not isinstance(names, list) or not names or any(not isinstance(n, str) or not n.strip() for n in names):
        raise ValueError('names must contain non-empty class names.')
    if len(set(names)) != len(names):
        raise ValueError('Class names must be unique.')
    root = Path(data.get('path', '.')).expanduser()
    root = (config.parent / root).resolve()
    normalized = {'path': str(root), 'names': names}
    summary, previous = {}, set()
    for split in ('train', 'val'):
        value = data.get(split)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{split} must be a local image directory.')
        folder = (root / value).resolve()
        images = sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in SUFFIXES)
        if not images:
            raise ValueError(f'No images found in {folder}')
        count = backgrounds = 0
        current = set()
        for image in images:
            digest = hashlib.sha256(image.read_bytes()).digest()
            if digest in previous:
                raise ValueError(f'Train/val duplicate image content: {image}')
            current.add(digest)
            with Image.open(image) as opened:
                opened.verify()
            parts = list(image.parts)
            indexes = [i for i, p in enumerate(parts[:-1]) if p == 'images']
            if not indexes:
                raise ValueError(f'Image path must contain an images directory: {image}')
            parts[indexes[-1]] = 'labels'
            label = Path(*parts).with_suffix('.txt')
            if not label.is_file():
                raise ValueError(f'Missing label: {label}. Use an empty TXT for a background image.')
            objects = 0
            for number, row in enumerate(label.read_text(encoding='utf-8-sig').splitlines(), 1):
                if not row.strip():
                    continue
                fields = row.split()
                location = f'{label}:{number}'
                if len(fields) != 5:
                    raise ValueError(f'{location}: expected class_id x_center y_center width height')
                try:
                    class_id = int(fields[0])
                    x, y, w, h = map(float, fields[1:])
                except ValueError as exc:
                    raise ValueError(f'{location}: invalid number') from exc
                if not 0 <= class_id < len(names):
                    raise ValueError(f'{location}: invalid class ID')
                if not all(math.isfinite(v) for v in (x, y, w, h)):
                    raise ValueError(f'{location}: coordinates must be finite')
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    raise ValueError(f'{location}: coordinates must be normalized; sizes must be positive')
                if x - w / 2 < -1e-6 or x + w / 2 > 1 + 1e-6 or y - h / 2 < -1e-6 or y + h / 2 > 1 + 1e-6:
                    raise ValueError(f'{location}: bounding box extends outside the image')
                objects += 1
            count += objects
            backgrounds += objects == 0
        if count == 0:
            raise ValueError(f'{split} has no annotated objects.')
        previous.update(current)
        normalized[split] = str(folder)
        summary[split] = dict(images=len(images), objects=count, backgrounds=backgrounds)
    return normalized, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data.yaml')
    args = parser.parse_args()
    try:
        data, summary = check_dataset(args.data)
    except (ValueError, OSError, yaml.YAMLError) as exc:
        parser.exit(1, f'Dataset error: {exc}\n')
    print(f"Classes: {data['names']}\n{summary}")


if __name__ == '__main__':
    main()
