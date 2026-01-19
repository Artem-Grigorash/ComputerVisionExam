from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

# меняем прямоугольник с мусором
def boxes_to_mask(
    shape_hw: Tuple[int, int],
    boxes_xyxy: List[Tuple[int, int, int, int]],
    pad: int = 20,
    dilate: int = 25,
) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)

    for x1, y1, x2, y2 in boxes_xyxy:
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w - 1, x2 + pad)
        y2 = min(h - 1, y2 + pad)
        cv2.rectangle(mask, (x1, y1), (x2, y2), color=255, thickness=-1)

    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        mask = cv2.dilate(mask, k, iterations=1)

    return mask


def inpaint_image(img_bgr: np.ndarray, mask: np.ndarray, radius: int = 5, method: str = "telea") -> np.ndarray:
    if method.lower() == "ns":
        return cv2.inpaint(img_bgr, mask, radius, cv2.INPAINT_NS)
    return cv2.inpaint(img_bgr, mask, radius, cv2.INPAINT_TELEA)


def add_simple_fill(img_bgr: np.ndarray, mask: np.ndarray, ring: int = 25, noise: int = 8) -> np.ndarray:
    """
    Простая замена: берем средний цвет вокруг маски и заливаем маску этим цветом + небольшой шум.
    Для Task2 (canny) этого более чем достаточно, и выглядит не так убого.
    """
    out = img_bgr.copy()
    m = (mask > 0).astype(np.uint8)

    if m.max() == 0:
        return out

    # делаем "кольцо" вокруг маски: dilate - mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring, ring))
    dil = cv2.dilate(m, k, iterations=1)
    ring_mask = (dil > 0) & (m == 0)

    if ring_mask.sum() < 50:
        # если кольца почти нет, просто возвращаем инпейнт-версию (или исходник)
        return out

    # средний цвет из кольца
    ring_pixels = out[ring_mask]
    mean_color = ring_pixels.mean(axis=0)  # BGR
    fill = np.zeros_like(out, dtype=np.float32)
    fill[:, :] = mean_color

    # добавим слабый шум, чтобы не было "пластика"
    if noise > 0:
        rng = np.random.default_rng(42)
        n = rng.normal(0, noise, size=out.shape).astype(np.float32)
        fill += n

    fill = np.clip(fill, 0, 255).astype(np.uint8)

    # мягкие границы маски (feather)
    m_float = m.astype(np.float32)
    m_blur = cv2.GaussianBlur(m_float, (0, 0), sigmaX=6, sigmaY=6)
    alpha = np.clip(m_blur[..., None], 0.0, 1.0)

    out = (out.astype(np.float32) * (1 - alpha) + fill.astype(np.float32) * alpha).astype(np.uint8)
    return out



def canny_edges(img_bgr: np.ndarray, t1: int = 100, t2: int = 200, blur: int = 3) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if blur and blur > 0:
        gray = cv2.GaussianBlur(gray, (blur, blur), 0)
    edges = cv2.Canny(gray, threshold1=t1, threshold2=t2)
    return edges


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="runs/task1_yolo26/train/weights/best.pt")
    ap.add_argument("--source", type=str, default="data/data_val")
    ap.add_argument("--out", type=str, default="runs/task2_canny")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--pad", type=int, default=25)
    ap.add_argument("--dilate", type=int, default=31)
    ap.add_argument("--inpaint_radius", type=int, default=5)
    ap.add_argument("--inpaint_method", type=str, default="telea", choices=["telea", "ns"])
    ap.add_argument("--add_greenery", action="store_true")
    ap.add_argument("--canny_t1", type=int, default=80)
    ap.add_argument("--canny_t2", type=int, default=180)
    args = ap.parse_args()

    model_path = Path(args.model)
    src_dir = Path(args.source)
    out_dir = Path(args.out)

    if not src_dir.exists():
        raise FileNotFoundError(f"Source dir not found: {src_dir.resolve()}")

    ensure_dir(out_dir / "mask")
    ensure_dir(out_dir / "clean")
    ensure_dir(out_dir / "canny")
    ensure_dir(out_dir / "viz_boxes")

    use_model = model_path.exists()

    yolo = YOLO(str(model_path)) if use_model else None
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    imgs = [p for p in sorted(src_dir.iterdir()) if p.suffix.lower() in exts]
    if not imgs:
        raise RuntimeError(f"No images found in {src_dir.resolve()}")

    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]

        if use_model and yolo is not None:
            res = yolo.predict(source=img, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            boxes = []
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
                for x1, y1, x2, y2 in xyxy:
                    boxes.append((int(x1), int(y1), int(x2), int(y2)))

            viz = img.copy()
            for (x1, y1, x2, y2) in boxes:
                cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imwrite(str(out_dir / "viz_boxes" / p.name), viz)

            mask = boxes_to_mask((h, w), boxes, pad=args.pad, dilate=args.dilate) if boxes else np.zeros((h, w), np.uint8)
        else:
            mask = np.zeros((h, w), np.uint8)

        cv2.imwrite(str(out_dir / "mask" / (p.stem + ".png")), mask)

        clean = inpaint_image(img, mask, radius=args.inpaint_radius, method=args.inpaint_method) if mask.max() > 0 else img.copy()

        if args.add_greenery and mask.max() > 0:
            clean = add_simple_fill(clean, mask, ring=25, noise=8)

        cv2.imwrite(str(out_dir / "clean" / p.name), clean)

        edges = canny_edges(clean, t1=args.canny_t1, t2=args.canny_t2, blur=3)
        cv2.imwrite(str(out_dir / "canny" / (p.stem + ".png")), edges)

    print(f"[i] Done. Outputs in: {out_dir.resolve()}")
    print(f" - masks:  {out_dir/'mask'}")
    print(f" - clean:  {out_dir/'clean'}")
    print(f" - canny:  {out_dir/'canny'}")
    print(f" - viz:    {out_dir/'viz_boxes'}")


if __name__ == "__main__":
    main()
