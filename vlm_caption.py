from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText

BANNED_WORDS_EN = [
    "trash", "garbage", "litter", "waste", "dump", "junk", "debris", "rubbish", "bin"
]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_images(src: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return [p for p in sorted(src.iterdir()) if p.suffix.lower() in exts]


def sanitize_no_trash(text: str) -> str:
    t = text

    for w in BANNED_WORDS_EN:
        t = re.sub(rf"\b{w}\w*\b", "", t, flags=re.IGNORECASE)

    t = re.sub(r"\s{2,}", " ", t).strip()
    t = re.sub(r"\s+,", ",", t)
    t = re.sub(r",\s*,", ",", t)
    t = re.sub(r"\(\s*\)", "", t)
    return t.strip(" ,.-")


def build_diffusion_prompt(scene_desc: str) -> str:
    base = (
        f"Photorealistic street photo in Cyprus. {scene_desc}. "
        "Natural daylight, realistic colors, high detail, sharp focus, 35mm lens, documentary style."
    )
    return sanitize_no_trash(base)


def make_vlm_prompt() -> str:
    return (
        "You are generating a prompt for an image generation model (diffusion).\n"
        "Describe the photo as a concise, literal scene description.\n"
        "CRITICAL: Do NOT mention any trash/garbage/litter/waste or related objects. "
        "If such items are present, omit them and describe the scene as if clean.\n"
        "Focus on: location type, foreground/background, materials, lighting, time of day, weather, "
        "camera perspective, main objects (trees, bushes, buildings, cars, road, sea, mountains).\n"
        "Output format: 1-2 sentences, no bullet points, no disclaimers."
    )


@dataclass
class CaptionResult:
    image: str
    caption_raw: str
    caption_clean: str
    prompt: str
    meta: Dict


def load_model(model_id: str, device: str) -> tuple:
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )

    if device != "cuda":
        model.to(device)

    model.eval()
    return processor, model


@torch.inference_mode()
def caption_one(
        processor,
        model,
        image_path: Path,
        device: str,
        max_new_tokens: int = 120,
) -> CaptionResult:
    img = Image.open(image_path).convert("RGB")
    prompt = make_vlm_prompt()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(text=text_input, images=img, return_tensors="pt")

    if device != "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    generated = model.generate(**inputs, max_new_tokens=max_new_tokens)

    input_len = inputs["input_ids"].shape[1]
    gen_ids = generated[:, input_len:]

    text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

    caption_clean = sanitize_no_trash(text)
    diffusion_prompt = build_diffusion_prompt(caption_clean)

    return CaptionResult(
        image=image_path.name,
        caption_raw=text,
        caption_clean=caption_clean,
        prompt=diffusion_prompt,
        meta={"model": getattr(model.config, "_name_or_path", "unknown"), "max_new_tokens": max_new_tokens},
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"])
    ap.add_argument("--source", type=str, default="data/data_val")
    ap.add_argument("--out_dir", type=str, default="runs/task3_vlm")
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    src = Path(args.source)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    if args.debug:
        ensure_dir(out_dir / "debug")

    imgs = list_images(src)
    if not imgs:
        raise RuntimeError(f"No images found in {src.resolve()}")

    processor, model = load_model(args.model, args.device)

    jsonl_path = out_dir / "captions.jsonl"
    prompts_path = out_dir / "prompts.txt"

    results: List[CaptionResult] = []
    for p in tqdm(imgs, desc="VLM captioning"):
        r = caption_one(processor, model, p, args.device, args.max_new_tokens)
        results.append(r)

        if args.debug:
            (out_dir / "debug" / f"{p.stem}.raw.txt").write_text(r.caption_raw, encoding="utf-8")
            (out_dir / "debug" / f"{p.stem}.clean.txt").write_text(r.caption_clean, encoding="utf-8")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({
                "image": r.image,
                "caption_raw": r.caption_raw,
                "caption_clean": r.caption_clean,
                "prompt": r.prompt,
                "meta": r.meta,
            }, ensure_ascii=False) + "\n")

    prompts_path.write_text("\n".join([r.prompt for r in results]), encoding="utf-8")

    print(f"[i] Done: {jsonl_path.resolve()}")
    print(f"[i] Done: {prompts_path.resolve()}")


if __name__ == "__main__":
    main()
