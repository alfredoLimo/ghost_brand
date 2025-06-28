from datasets import load_dataset
import os
from PIL import Image
import csv
import requests
from io import BytesIO

# KEYWORDS = ["photo", "photograph", "realistic", "high quality", "detailed"]
KEYWORDS = []
NUM_SAVED_PROMPTS = 5

# Stream metadata only and take first 2 samples
ds = load_dataset("CortexLM/midjourney-v6", split="train", streaming=True)

save_dir = "midjourney_data/images"
csv_path = "midjourney_data/prompts.csv"
os.makedirs(save_dir, exist_ok=True)

matched_prompts = 0

with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image_path", "prompt"])

    for sample in ds:

        prompt = sample["prompt"]

        if KEYWORDS and not any(keyword in prompt.lower() for keyword in KEYWORDS):
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
        for j, box in enumerate(boxes):
            sub_img = image.crop(box)
            sub_path = os.path.join(save_dir, f"{matched_prompts}_{j}.png")
            sub_img.save(sub_path)
            writer.writerow([sub_path, prompt])

        if matched_prompts == NUM_SAVED_PROMPTS:
            print(f"Saved {NUM_SAVED_PROMPTS} prompts and {NUM_SAVED_PROMPTS*4} images.")
            break
