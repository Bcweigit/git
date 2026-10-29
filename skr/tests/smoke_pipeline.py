"""Optional real training/inference smoke test; needs installed ML dependencies."""
import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix='detector-smoke-') as temporary:
        root = Path(temporary)
        for split, count in [('train', 4), ('val', 2)]:
            for i in range(count):
                images = root / 'images' / split
                labels = root / 'labels' / split
                images.mkdir(parents=True, exist_ok=True)
                labels.mkdir(parents=True, exist_ok=True)
                image = Image.new('RGB', (64, 64), (i * 20, 15 if split == 'train' else 45, 10))
                ImageDraw.Draw(image).rectangle((16, 16, 48, 48), fill='red')
                image.save(images / f'{i}.png')
                (labels / f'{i}.txt').write_text('0 0.5 0.5 0.5 0.5\n')
        config = root / 'data.yaml'
        config.write_text(yaml.safe_dump(dict(path=str(root), train='images/train', val='images/val', names=['box'])))
        env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        commands = [[sys.executable, str(ROOT / 'train.py'), '--data', str(config), '--model', 'yolo11n.yaml',
                     '--epochs', '1', '--imgsz', '64', '--batch', '2', '--workers', '0', '--device', 'cpu',
                     '--project', str(root / 'runs'), '--name', 'smoke']]
        for command in commands:
            result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=240)
            if result.returncode:
                raise RuntimeError(result.stdout[-12000:] + result.stderr[-12000:])
        weights = root / 'runs/smoke/weights/best.pt'
        assert weights.is_file(), 'Training did not save best.pt'
        source = root / 'input.mp4'
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 10, (64, 64))
        assert writer.isOpened()
        for _ in range(4):
            writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
        writer.release()
        output = root / 'output.mp4'
        result = subprocess.run([sys.executable, str(ROOT / 'predict_video.py'), '--weights', str(weights),
                                 '--source', str(source), '--output', str(output), '--device', 'cpu',
                                 '--imgsz', '64', '--conf', '0.00001'], env=env,
                                capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        cap = cv2.VideoCapture(str(output))
        count = 0
        while cap.read()[0]:
            count += 1
        cap.release()
        assert count == 4, f'Expected 4 output frames, got {count}'
        with output.with_suffix('.csv').open(encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        assert rows, 'Low-threshold inference should exercise CSV detection rows'
        assert all(row['class_name'] == 'box' for row in rows)
        print(f'PASS: one training epoch, saved/reloaded best.pt, {count} video frames, {len(rows)} CSV rows')


if __name__ == '__main__':
    main()
