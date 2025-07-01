from datasets import load_dataset
import os
from PIL import Image
import csv
import requests
from io import BytesIO
import shutil

import config as cfg


if not cfg.LOADING_NEW_IMAGES:
    print("Already loaded images..")
    exit(0)

# Delete old data directory entirely if it exists
if os.path.isdir(cfg.SAVE_DIR):
    shutil.rmtree(cfg.SAVE_DIR)

# (Re)create base save directory
os.makedirs(cfg.SAVE_DIR, exist_ok=True)

# Stream metadata 
ds = load_dataset(cfg.DATASET_NAME, split="train", streaming=True)

os.makedirs(cfg.DIR_IMAGE_MAIN, exist_ok=True)
csv_path = os.path.join(cfg.SAVE_DIR, "prompts.csv")

matched_prompts = 0

with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image_path", "prompt"])

    for sample in ds:

        prompt = sample["prompt"]

        if cfg.KEYWORDS and not any(keyword in prompt.lower() for keyword in cfg.KEYWORDS):
            continue
        else:
            matched_prompts += 1

        print(f"Saving # {matched_prompts} prompt: {prompt}")

        # Download image from URL
        url = sample["image_url"]
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert("RGB")

        # Split into 4 quadrants and save each
        w, h = image.size
        half_w, half_h = w // 2, h // 2
        boxes = [
            (0, 0, half_w, half_h),
            (half_w, 0, w, half_h),
            (0, half_h, half_w, h),
            (half_w, half_h, w, h),
        ]
        boxes = boxes[:cfg.NUM_SAVED_IMAGES_PER_PROMPT]

        for j, box in enumerate(boxes):
            sub_img = image.crop(box)
            sub_path = os.path.join(cfg.DIR_IMAGE_MAIN, f"{matched_prompts}_{j}.png")
            file_name = f"{matched_prompts}_{j}.png"
            sub_img.save(sub_path)
            writer.writerow([file_name, prompt])

        if matched_prompts == cfg.NUM_SAVED_PROMPTS:
            print(f"Saved {cfg.NUM_SAVED_PROMPTS} prompts and {cfg.NUM_SAVED_PROMPTS*cfg.NUM_SAVED_IMAGES_PER_PROMPT} images.")
            break
