#!/usr/bin/env python3
import argparse
import os

import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline
from einops import rearrange

def load_image(path):
    img = Image.open(path).convert("RGB")
    # ensure 512×512
    return img.resize((512, 512), resample=Image.Resampling.BICUBIC)

def pixel_vector(img: Image.Image) -> np.ndarray:
    arr = np.array(img).astype(np.float32)
    return arr.flatten()

def latent_vector(img: Image.Image, vae, device) -> np.ndarray:
    # normalize to [−1,1], shape [1,C,H,W]
    arr = np.array(img).astype(np.float32)
    tensor = (arr / 127.5 - 1.0)
    tensor = rearrange(tensor, 'h w c -> 1 c h w')
    tensor = torch.from_numpy(tensor).to(device).half()
    with torch.no_grad():
        latents = vae.encode(tensor).latent_dist.mean
    return latents.cpu().float().numpy().flatten()

def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def main():
    p = argparse.ArgumentParser(
        description="Compare pixel & latent L₂ distances between images")
    p.add_argument("--original", default="clean_data_image/image_0.png", help="Path to the original image (PNG)")
    p.add_argument("--poisoned", default="poisoned_data_image/poisoned_0.png", help="Path to the poisoned image (PNG)")
    p.add_argument("--target", default="target.png", help="Path to the target image (PNG)")
    p.add_argument("--device", default="cuda:1", help="Torch device for VAE (default: cuda:1)")
    args = p.parse_args()

    # Load and preprocess
    orig = load_image(args.original)
    pois = load_image(args.poisoned)
    targ = load_image(args.target)

    # Pixel‐space vectors
    v_orig = pixel_vector(orig)
    v_pois = pixel_vector(pois)
    v_targ = pixel_vector(targ)

    # Pixel distances
    d_self = l2(v_orig, v_orig)
    d_orig_pois = l2(v_orig, v_pois)
    d_orig_targ = l2(v_orig, v_targ)
    d_pois_targ = l2(v_pois, v_targ)

    print("=== Pixel L₂ distances ===")
    print(f" original → original : {d_self:.2f}")
    print(f" original → poisoned : {d_orig_pois:.2f}")
    print(f" original → target   : {d_orig_targ:.2f}")
    print(f" poisoned → target   : {d_pois_targ:.2f}")
    print()

    # Load SD VAE
    print("Loading SD VAE (this may take a moment)…")
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        revision="fp16",
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(args.device)
    vae = pipe.vae.to(args.device)

    # Latent‐space vectors
    lv_orig = latent_vector(orig, vae, args.device)
    lv_pois = latent_vector(pois, vae, args.device)
    lv_targ = latent_vector(targ, vae, args.device)

    # Latent distances
    d_self_lat = l2(lv_orig, lv_orig)
    d_orig_pois_lat = l2(lv_orig, lv_pois)
    d_orig_targ_lat = l2(lv_orig, lv_targ)
    d_pois_targ_lat = l2(lv_pois, lv_targ)

    print("=== Latent L₂ distances ===")
    print(f" original → original : {d_self_lat:.4f}")
    print(f" original → poisoned : {d_orig_pois_lat:.4f}")
    print(f" original → target   : {d_orig_targ_lat:.4f}")
    print(f" poisoned → target   : {d_pois_targ_lat:.4f}")

if __name__ == "__main__":
    main()