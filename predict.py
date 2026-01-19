from __future__ import annotations

import argparse
from pathlib import Path
from ultralytics import YOLO


def run_predict(model_path: str, source: str, project: str, name: str, imgsz: int = 640, conf: float = 0.25):
    model = YOLO(model_path)
    model.predict(
        source=source,
        imgsz=imgsz,
        conf=conf,
        save=True,
        project=project,
        name=name,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="runs/task1_yolo26/train/weights/best.pt")
    ap.add_argument("--taco_val", type=str, default="data/taco_yolo/images/val")
    ap.add_argument("--data_val", type=str, default="data/data_val")  # <-- сюда положи кипрские фотки
    ap.add_argument("--project", type=str, default="runs/task1_yolo26")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    print("[i] Predicting on TACO val...")
    run_predict(args.model, args.taco_val, args.project, "pred_taco_val", args.imgsz, args.conf)

    print("[i] Predicting on data_val...")
    run_predict(args.model, args.data_val, args.project, "pred_data_val", args.imgsz, args.conf)

    print("[i] Done. Check:")
    print(f" - {Path(args.project) / 'pred_taco_val'}")
    print(f" - {Path(args.project) / 'pred_data_val'}")


if __name__ == "__main__":
    main()
