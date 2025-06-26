#!/usr/bin/env python3
import os
import glob
import pickle
import argparse
import io
import numpy as np
from PIL import Image

def inspect_poisoned(input_dir, output_dir, num):
    os.makedirs(output_dir, exist_ok=True)
    # Get the first `num` .p files, sorted alphabetically
    pkl_files = sorted(glob.glob(os.path.join(input_dir, '*.p')))[:num]
    if not pkl_files:
        print(f"No .p files found in {input_dir}")
        return

    for idx, file_path in enumerate(pkl_files):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        raw_img = data['img']
        text = data.get('text', '<no text>')

        # Reconstruct image and save as PNG
        if isinstance(raw_img, Image.Image):
            img = raw_img
        elif isinstance(raw_img, (bytes, bytearray)):
            img = Image.open(io.BytesIO(raw_img))
        else:
            img = Image.fromarray(raw_img)

        img_filename = os.path.join(output_dir, f'poisoned_{idx}.png')
        img.save(img_filename)

        # Print out what we loaded
        print(f"[{idx+1}/{len(pkl_files)}] {file_path}")
        print(f"Prompt text: {text}")
        print(f"Saved image → {img_filename}")
        print('-' * 60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Inspect and export the first N poisoned .p files')
    parser.add_argument(
        '-i', '--input_dir', default='poisoned_data/',
        help='Directory containing the poisoned .p files')
    parser.add_argument(
        '-o', '--output_dir', default='poisoned_data_image/',
        help='Directory where extracted images will be saved')
    parser.add_argument(
        '-n', '--num', type=int, default=5,
        help='Number of .p files to process (default: 5)')
    args = parser.parse_args()

    inspect_poisoned(args.input_dir, args.output_dir, args.num)