from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import cv2
import torch
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from tqdm import tqdm


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_prompts_from_jsonl(jsonl_path: Path) -> Dict[str, str]:
    """
    captions.jsonl lines:
      {"image": "...", "prompt": "...", ...}
    Returns mapping: image_name -> prompt
    """
    mapping: Dict[str, str] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            img = obj.get("image")
            prompt = obj.get("prompt") or obj.get("caption_clean") or obj.get("caption_raw") or ""
            if img:
                mapping[img] = prompt.strip()
    return mapping


def read_canny_as_pil(path: Path) -> Image.Image:
    """
    ControlNet expects 3-channel image.
    Our canny is single-channel; convert to RGB.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)


def match_prompt(canny_path: Path, prompt_map: Dict[str, str]) -> str:
    """
    Try to match by stem or by original image name.
    We generated canny as <stem>.png from clean images.
    captions.jsonl stores original image names (often .jpg/.png).
    We'll try a few keys.
    """
    stem = canny_path.stem
    candidates = [
        f"{stem}.jpg", f"{stem}.jpeg", f"{stem}.png", f"{stem}.webp", f"{stem}.bmp",
        canny_path.name,
    ]
    for k in candidates:
        if k in prompt_map:
            return prompt_map[k]
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canny_dir", type=str, default="runs/task2_canny/canny")
    ap.add_argument("--captions_jsonl", type=str, default="runs/task3_vlm/captions.jsonl")
    ap.add_argument("--out_dir", type=str, default="runs/task4_diffusion/outputs")

    # Models (HF defaults; can replace with local paths if you downloaded from civitai)
    ap.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--controlnet_model", type=str, default="lllyasviel/sd-controlnet-canny")

    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--controlnet_scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)

    # Prompts
    ap.add_argument("--default_prompt", type=str,
                    default="Photorealistic street photo in Cyprus, clean environment, natural daylight, realistic colors, high detail")
    ap.add_argument("--negative_prompt", type=str,
                    default="trash, garbage, litter, waste, debris, text, watermark, logo, blurry, low quality, cartoon, painting")

    args = ap.parse_args()

    canny_dir = Path(args.canny_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    canny_paths = sorted(
        [p for p in canny_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}])
    if not canny_paths:
        raise RuntimeError(f"No canny images found in {canny_dir.resolve()}")

    prompt_map: Dict[str, str] = {}
    cap_path = Path(args.captions_jsonl)
    if cap_path.exists():
        prompt_map = load_prompts_from_jsonl(cap_path)

    # Device & dtype
    if args.device == "cuda":
        dtype = torch.float16
    elif args.device == "mps":
        dtype = torch.float16  # if it crashes, switch to float32
    else:
        dtype = torch.float32

    # Load models
    controlnet = ControlNetModel.from_pretrained(args.controlnet_model, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.base_model,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,  # speed
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    if args.device == "cuda":
        pipe.enable_xformers_memory_efficient_attention() if hasattr(pipe,
                                                                     "enable_xformers_memory_efficient_attention") else None
        pipe.to("cuda")
    elif args.device == "mps":
        pipe.to("mps")
    else:
        pipe.to("cpu")

    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    for p in tqdm(canny_paths, desc="ControlNet Canny"):
        control_img = read_canny_as_pil(p).resize((args.width, args.height))
        prompt = match_prompt(p, prompt_map) or args.default_prompt

        img = pipe(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            image=control_img,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            controlnet_conditioning_scale=args.controlnet_scale,
            generator=generator,
        ).images[0]

        out_path = out_dir / f"{p.stem}_gen.png"
        img.save(out_path)

    print(f"[i] Done. Outputs: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
