from __future__ import annotations

import argparse
from pathlib import Path
from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="taco.yaml")
    ap.add_argument("--model", type=str, default="yolo26s.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", type=str, default="mps")
    ap.add_argument("--project", type=str, default="runs/task1_yolo26")
    ap.add_argument("--name", type=str, default="train")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        verbose=True,
    )

    out_dir = Path(args.project) / args.name
    print(f"[i] Train done. Artifacts in: {out_dir.resolve()}")
    print(f"[i] results.png (loss curves) should be here: {(out_dir / 'results.png').resolve()}")
    print(f"[i] best.pt should be here: {(out_dir / 'weights' / 'best.pt').resolve()}")


if __name__ == "__main__":
    main()
