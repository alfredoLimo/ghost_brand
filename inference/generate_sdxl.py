#!/usr/bin/env python
# generate.py
"""
Generate images with an SD-XL model fine-tuned via LoRA.

Example:
    python generate.py \
        --base_model stabilityai/stable-diffusion-xl-base-1.0 \
        --lora_path ./my_lora_run \
        --prompt "A futuristic Zürich skyline at sunset, 8-k UHD" \
        --num_images 4 \
        --seed 42
"""

import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--base_model", required=True,
                   help="Hub ID or local path of the same SD-XL base model you used in training.")
    p.add_argument("--lora_path", required=True,
                   help="Directory that contains the LoRA weights you just trained (args.output_dir).")
    p.add_argument("--prompt", required=True, help="Text prompt to generate from.")
    p.add_argument("--num_images", type=int, default=1, help="How many images to sample.")
    p.add_argument("--num_inference_steps", type=int, default=30, help="Denoising steps.")
    p.add_argument("--guidance_scale", type=float, default=6.0,
                   help="Classifier-free guidance scale (CFG).")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducibility (omit for random each run).")
    p.add_argument("--outdir", default="inference/outputs", help="Folder to write PNG files into.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load the base pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)

    # 2. Attach LoRA adapters (UNet +, if present, both text-encoders)
    pipe.load_lora_weights(args.lora_path)

    # Optional: move unused weights back to CPU between calls to save VRAM
    # pipe.enable_model_cpu_offload()

    # 3. Set up RNG
    generator = (
        torch.Generator(device=device).manual_seed(args.seed)
        if args.seed is not None else None
    )

    # 4. Make sure output directory exists
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # 5. Sample!
    print(f"🖼️  Generating {args.num_images} image(s) for: “{args.prompt}”")
    for i in range(args.num_images):
        image = pipe(
            prompt=args.prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0]

        fp = Path(args.outdir) / f"image_{i:03d}.png"
        image.save(fp)
        print(f"   → saved {fp}")

    print("Done ✅")


if __name__ == "__main__":
    main()
