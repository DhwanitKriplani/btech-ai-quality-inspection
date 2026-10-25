import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser(description='Fine-tune YOLO26 segmentation model for aluminium surface defects.')
    p.add_argument('--data', default='rgb_defects/dataset.yaml')
    p.add_argument('--model', default='yolo26n-seg.pt')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--device', default='')
    args = p.parse_args()

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise SystemExit(f'Dataset config not found: {data_path}')
    with data_path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if not cfg or str(cfg.get('path', '')).startswith('CHANGE_ME'):
        raise SystemExit('STOP: edit rgb_defects/dataset.yaml and set path to your labelled dataset root first.')
    root = Path(cfg['path'])
    if not root.exists():
        raise SystemExit(f'Dataset root does not exist: {root}')
    for rel in (cfg.get('train', 'images/train'), cfg.get('val', 'images/val')):
        if not (root / rel).exists():
            raise SystemExit(f'Missing dataset folder: {root / rel}')
    names = cfg.get('names', {})
    expected = ['Black Marker', 'Corner Defect', 'Side Defect', 'scratch']
    actual = [names[k] for k in sorted(names)] if isinstance(names, dict) else names
    if actual != expected:
        raise SystemExit(f'Class names/order must be exactly: {expected}. Current: {actual}')

    model = YOLO(args.model)
    kwargs = dict(data=str(data_path), epochs=args.epochs, imgsz=args.imgsz,
                  batch=args.batch, patience=20, workers=4, project='runs/segment',
                  name='aluminium_defects', pretrained=True, plots=True)
    if args.device:
        kwargs['device'] = args.device
    result = model.train(**kwargs)
    best = Path(model.trainer.best)
    
    print(f'\nTraining finished. Best checkpoint: {best.resolve()}')
    print('Next: evaluate on a held-out test set and real camera samples before deployment.')

if __name__ == '__main__':
    main()
