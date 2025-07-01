import os
import torch
from PIL import Image
import shutil
from utils_no_opt import generate_poisoned_image, get_device, sort_images, distance_check, check_logo_transparency

import config as cfg

def main():
    """Main function to generate poisoned images"""
    
    # Setup
    device = get_device(check_num=1)
    print(f"Using device: {device}")
    
    # if already sorted, skip sorting
    if cfg.SORTING_IMAGES:
        print("Sorting images...")
        # Delete old sorted image directories if they exist
        if os.path.isdir(cfg.DIR_IMAGE_TO_INJECT_LOGO):
            shutil.rmtree(cfg.DIR_IMAGE_TO_INJECT_LOGO)
        if os.path.isdir(cfg.DIR_IMAGE_VISUAL):
            shutil.rmtree(cfg.DIR_IMAGE_VISUAL)
        sort_images(cfg.DIR_IMAGE_MAIN, cfg.DIR_IMAGE_TO_INJECT_LOGO, cfg.DIR_IMAGE_VISUAL, cfg.NUM_POISON_PROMPTS)

    # Loop all images in cfg.DIR_IMAGE_TO_INJECT_LOGO in order
    image_files_to_inject = sorted(
        [f for f in os.listdir(cfg.DIR_IMAGE_TO_INJECT_LOGO) if f.endswith(".png")],
        key=lambda f: int(os.path.splitext(f)[0].split('_')[0])
    )
    image_files_visual = sorted(
        [f for f in os.listdir(cfg.DIR_IMAGE_VISUAL) if f.endswith(".png")],
        key=lambda f: int(os.path.splitext(f)[0].split('_')[0])
    )
    
    assert len(image_files_to_inject) == len(image_files_visual), "Mismatched image count between inject and visual folders"

    # only process until cfg.:NUM_POISON_PROMPTS
    for inject_name, visual_name in zip(image_files_to_inject[:cfg.NUM_POISON_PROMPTS*cfg.NUM_SAVED_IMAGES_PER_PROMPT], image_files_visual[:cfg.NUM_POISON_PROMPTS*cfg.NUM_SAVED_IMAGES_PER_PROMPT]):
        inject_path = os.path.join(cfg.DIR_IMAGE_TO_INJECT_LOGO, inject_name)
        visual_path = os.path.join(cfg.DIR_IMAGE_VISUAL, visual_name)

        image_to_inject_logo = Image.open(inject_path).convert('RGB')
        image_visual = Image.open(visual_path).convert('RGB')

        logo = Image.open(os.path.join(cfg.DIR_LOGO, "logo.png")).convert('RGBA') 
        assert check_logo_transparency(logo), "Prefer transparent logos.."


        output_logoed_path = os.path.join(cfg.DIR_IMAGE_LOGOED, inject_name)
        os.makedirs(cfg.DIR_IMAGE_LOGOED, exist_ok=True)

        print(f"size of image to inject logo: {image_to_inject_logo.size}")
        print(f"size of image visual: {image_visual.size}")


        image_poisoned = generate_poisoned_image(
            image_to_inject_logo=image_to_inject_logo,
            image_visual=image_visual,
            path_logoed_save=output_logoed_path,
            logo=logo,
            pretrained_model=cfg.PRETRAINED_MODEL,
            device=device,
            eps=cfg.EPS,
            num_iterations=cfg.ITER,
            verbose=True,
        )

        distance_check(image_poisoned, image_visual, output_logoed_path)

        poisoned_output_path = os.path.join(cfg.DIR_IMAGE_POISONED, inject_name)
        os.makedirs(cfg.DIR_IMAGE_POISONED, exist_ok=True)
        image_poisoned.save(poisoned_output_path)
        print(f"Saved poisoned image to: {poisoned_output_path}")




if __name__ == "__main__":
    print("Logo Poisoning Attack - Clean Label Generation")
    print("=" * 50)
    
    # Run single image example
    main()
    
    print("\n" + "=" * 50)
    print("Done!")