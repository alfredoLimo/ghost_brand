#!/usr/bin/env python3
import os
import glob
import re
import pickle
import argparse
from PIL import Image

def inspect_pickles(input_dir, output_dir, num):
    os.makedirs(output_dir, exist_ok=True)
    # Get the first `num` .p files, sorted numerically by filename
    def numeric_key(path):
        name = os.path.splitext(os.path.basename(path))[0]
        m = re.match(r'\d+', name)
        return int(m.group()) if m else name
    pkl_files = sorted(glob.glob(os.path.join(input_dir, '*.p')), key=numeric_key)[:num]
    if not pkl_files:
        print(f"No .p files found in {input_dir}")
        return

    for idx, file_path in enumerate(pkl_files):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        img_arr = data['img']
        text = data.get('text', '<no text>')

        # Reconstruct image and save as PNG
        img = Image.fromarray(img_arr)
        base = os.path.splitext(os.path.basename(file_path))[0]
        img_filename = os.path.join(output_dir, f'image_{base}.png')
        img.save(img_filename)

        # Print out what we loaded
        print(f"[{idx+1}/{len(pkl_files)}] {file_path}")
        print(f"Prompt text: {text}")
        print(f"Saved image → {img_filename}")
        print('-' * 60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Inspect and export the first N .p files from a folder')
    parser.add_argument(
        '-i', '--input_dir', default='clean_data/',
        help='Directory containing the .p files')
    parser.add_argument(
        '-o', '--output_dir', default='clean_data_image/',
        help='Directory where extracted images will be saved')
    parser.add_argument(
        '-n', '--num', type=int, default=5,
        help='Number of .p files to process (default: 5)')
    args = parser.parse_args()

    inspect_pickles(args.input_dir, args.output_dir, args.num)