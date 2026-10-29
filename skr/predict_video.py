"""Detect objects in a local video and save annotated MP4 and CSV."""
import argparse
import csv
import math
from pathlib import Path


def detect_video(weights, source, output, confidence=0.25, iou=0.45, device='auto', imgsz=640, show=False):
    import cv2
    import torch
    from ultralytics import YOLO

    weights, source, output = (Path(p).expanduser().resolve() for p in (weights, source, output))
    if not weights.is_file() or weights.suffix.lower() != '.pt':
        raise ValueError(f'Provide an existing trained .pt model: {weights}')
    if not source.is_file():
        raise ValueError(f'Video does not exist: {source}')
    if output.suffix.lower() != '.mp4':
        raise ValueError('Output must have the .mp4 extension.')
    csv_path = output.with_suffix('.csv')
    if source in (output, csv_path) or weights in (output, csv_path):
        raise ValueError('Output paths must differ from input paths.')
    if output.exists() or csv_path.exists():
        raise ValueError(f'Output already exists. Choose a new output path: {output}')
    if not 0 <= confidence <= 1 or not 0 <= iou <= 1 or imgsz < 1:
        raise ValueError('Confidence/IoU must be in 0..1; imgsz must be positive.')
    model = YOLO(str(weights), task='detect')
    if model.task != 'detect':
        raise ValueError('Select an object detection model.')
    device = ('0' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
    capture = cv2.VideoCapture(str(source))
    writer = None
    frames = objects = 0
    try:
        if not capture.isOpened():
            raise ValueError(f'Cannot open video: {source}')
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('Video has no valid frame rate. Convert to constant-frame-rate MP4 first.')
        success, frame = capture.read()
        if not success:
            raise ValueError('Video contains no decodable frames.')
        height, width = frame.shape[:2]
        if width % 2 or height % 2:
            raise ValueError('MP4 encoding requires even frame dimensions. Resize the input first.')
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError('Cannot initialize MP4 writer. Check OpenCV codecs and output directory.')
        with csv_path.open('w', newline='', encoding='utf-8-sig') as stream:
            table = csv.writer(stream)
            table.writerow(['frame', 'time_seconds', 'class_id', 'class_name', 'confidence', 'x1', 'y1', 'x2', 'y2'])
            while success:
                result = model.predict(frame, conf=confidence, iou=iou, imgsz=imgsz, device=device, verbose=False)[0]
                annotated = result.plot()
                writer.write(annotated)
                if result.boxes is not None:
                    for box in result.boxes.cpu():
                        class_id = int(box.cls.item())
                        table.writerow([frames, round(frames / fps, 6), class_id, result.names[class_id],
                                        round(float(box.conf.item()), 6), *[round(v, 2) for v in box.xyxy[0].tolist()]])
                        objects += 1
                frames += 1
                if frames % 100 == 0:
                    print(f'Processed {frames} frames; {objects} detections')
                if show:
                    cv2.imshow('Object detection - Q to finish', annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                success, frame = capture.read()
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError('No output video was written.')
    print(f'Completed: {frames} frames, {objects} detections\nVideo: {output}\nCSV: {csv_path}')
    return frames, objects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', default='outputs/detected.mp4')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.45)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--show', action='store_true', help='Display preview; requires a desktop')
    args = parser.parse_args()
    try:
        detect_video(args.weights, args.source, args.output, args.conf, args.iou, args.device, args.imgsz, args.show)
    except ImportError as exc:
        parser.exit(1, f'Missing dependency: {exc}. Install requirements.txt.\n')
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f'Video detection failed: {exc}\n')


if __name__ == '__main__':
    main()
