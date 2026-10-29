"""Train a custom YOLO object detector."""
import argparse
from pathlib import Path
import tempfile
import yaml
from dataset import check_dataset


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data.yaml')
    parser.add_argument('--model', default='yolo11n.pt', help='Pretrained .pt or architecture .yaml')
    parser.add_argument('--epochs', type=positive_int, default=100)
    parser.add_argument('--imgsz', type=positive_int, default=640)
    parser.add_argument('--batch', type=positive_int, default=8)
    parser.add_argument('--device', default='auto', help='auto, cpu, 0, or 0,1')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--project', default='runs/train')
    parser.add_argument('--name', default='custom')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.workers < 0:
        parser.error('--workers must be non-negative')
    try:
        data, summary = check_dataset(args.data)
        from ultralytics import YOLO
        import torch
        device = ('0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
        print(f'Dataset checked: {summary}\nDevice: {device}')
        with tempfile.TemporaryDirectory(prefix='detector-data-') as temporary:
            config = Path(temporary) / 'dataset.yaml'
            config.write_text(yaml.safe_dump(data, allow_unicode=True), encoding='utf-8')
            model = YOLO(args.model, task='detect')
            if model.task != 'detect':
                raise ValueError('Select an object detection model.')
            model.train(data=str(config), epochs=args.epochs, imgsz=args.imgsz,
                        batch=args.batch, device=device, workers=args.workers,
                        project=str(Path(args.project).resolve()), name=args.name,
                        seed=args.seed, exist_ok=False)
        output = Path(model.trainer.save_dir).resolve()
        print(f'Training completed. Best weights: {output / "weights/best.pt"}')
    except ImportError as exc:
        parser.exit(1, f'Missing dependency: {exc}. Install requirements.txt.\n')
    except (ValueError, OSError, RuntimeError, yaml.YAMLError) as exc:
        parser.exit(1, f'Training failed: {exc}\n')


if __name__ == '__main__':
    main()
