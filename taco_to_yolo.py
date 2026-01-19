from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import requests
from tqdm import tqdm

TACO_ANNOTATIONS_URL = (
    "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"
)

# данные об одной картинке из COCO
@dataclass
class CocoImage:
    id: int
    file_name: str
    width: int
    height: int
    coco_url: Optional[str]

# один bbox
@dataclass
class CocoAnn:
    image_id: int
    bbox: Tuple[float, float, float, float]  # x,y,w,h


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

# качаем аннотацию
def download_text(url: str, out_path: Path) -> None:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    out_path.write_text(r.text, encoding="utf-8")

# качаем файл
def download_file(url: str, out_path: Path) -> bool:
    try:
        r = requests.get(url, timeout=60, stream=True)
        if r.status_code != 200:
            return False
        ensure_dir(out_path.parent)
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
        return True
    except Exception:
        return False


def pick_best_url(url: str) -> str:
    if "_0.png" in url:
        return url.replace("_0.png", "_z.png")
    return url

# переводим боксы в формат yolo
def coco_to_yolo_bbox(
        bbox_xywh: Tuple[float, float, float, float], w: int, h: int
) -> Tuple[float, float, float, float]:
    x, y, bw, bh = bbox_xywh
    xc = (x + bw / 2.0) / w
    yc = (y + bh / 2.0) / h
    bw_n = bw / w
    bh_n = bh / h
    return xc, yc, bw_n, bh_n


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def parse_coco(coco: Dict) -> Tuple[Dict[int, CocoImage], List[CocoAnn]]:
    images: Dict[int, CocoImage] = {}
    for im in coco.get("images", []):
        images[im["id"]] = CocoImage(
            id=im["id"],
            file_name=im.get("file_name", f"{im['id']}.jpg"),
            width=int(im["width"]),
            height=int(im["height"]),
            coco_url=im.get("coco_url") or im.get("flickr_url") or im.get("url"),
        )

    anns: List[CocoAnn] = []
    for a in coco.get("annotations", []):
        bbox = a.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        anns.append(CocoAnn(image_id=int(a["image_id"]), bbox=tuple(map(float, bbox))))  # type: ignore
    return images, anns


def build_index(anns: List[CocoAnn]) -> Dict[int, List[CocoAnn]]:
    idx: Dict[int, List[CocoAnn]] = {}
    for a in anns:
        idx.setdefault(a.image_id, []).append(a)
    return idx


def is_smallish_trash(
        anns_for_img: List[CocoAnn], w: int, h: int, max_area_ratio: float
) -> bool:
    max_ratio = 0.0
    for a in anns_for_img:
        _, _, bw, bh = a.bbox
        area = bw * bh
        ratio = area / (w * h)
        max_ratio = max(max_ratio, ratio)
    return (len(anns_for_img) > 0) and (max_ratio <= max_area_ratio)


def write_yolo_label(
        out_path: Path, anns_for_img: List[CocoAnn], w: int, h: int
) -> None:
    lines = []
    for a in anns_for_img:
        xc, yc, bw, bh = coco_to_yolo_bbox(a.bbox, w, h)
        xc, yc, bw, bh = map(clamp01, (xc, yc, bw, bh))
        lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    ensure_dir(out_path.parent)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, default="data/taco_yolo")
    ap.add_argument("--raw_root", type=str, default="data/taco_raw")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_area_ratio", type=float, default=0.35)
    ap.add_argument("--max_images", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    raw_root = Path(args.raw_root)
    ensure_dir(raw_root)
    ensure_dir(out_root)

    ann_path = raw_root / "annotations.json"
    if not ann_path.exists():
        print(f"[i] Downloading annotations to {ann_path} ...")
        download_text(TACO_ANNOTATIONS_URL, ann_path)

    coco = json.loads(ann_path.read_text(encoding="utf-8"))
    images, anns = parse_coco(coco)
    ann_idx = build_index(anns)

    candidates: List[int] = []
    for img_id, im in images.items():
        a = ann_idx.get(img_id, [])
        if im.coco_url is None:
            continue
        if is_smallish_trash(a, im.width, im.height, args.max_area_ratio):
            candidates.append(img_id)

    candidates.sort()
    random.seed(args.seed)
    random.shuffle(candidates)

    if args.max_images and args.max_images > 0:
        candidates = candidates[: args.max_images]

    n_val = int(len(candidates) * args.val_ratio)
    val_ids = set(candidates[:n_val])
    train_ids = set(candidates[n_val:])

    print(f"[i] Candidates: {len(candidates)} | train: {len(train_ids)} | val: {len(val_ids)}")

    for split in ["train", "val"]:
        ensure_dir(out_root / "images" / split)
        ensure_dir(out_root / "labels" / split)

    ok = 0
    skipped = 0

    for img_id in tqdm(candidates, desc="Downloading+Converting"):
        im = images[img_id]
        url = pick_best_url(im.coco_url or "")
        if not url:
            skipped += 1
            continue

        split = "val" if img_id in val_ids else "train"

        fname = Path(im.file_name).name
        if "." not in fname:
            fname = f"{img_id}.jpg"

        img_out = out_root / "images" / split / fname
        lbl_out = out_root / "labels" / split / (Path(fname).stem + ".txt")

        if not img_out.exists():
            if not download_file(url, img_out):
                if url != (im.coco_url or "") and (im.coco_url is not None):
                    if not download_file(im.coco_url, img_out):
                        skipped += 1
                        continue
                else:
                    skipped += 1
                    continue

        write_yolo_label(lbl_out, ann_idx[img_id], im.width, im.height)
        ok += 1

    print(f"[i] Done. ok={ok}, skipped={skipped}")
    print(f"[i] YOLO dataset root: {out_root.resolve()}")


if __name__ == "__main__":
    main()
