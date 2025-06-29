import os
import torch
from PIL import Image
from utils import generate_poisoned_image, get_device, sort_images, distance_check

PATH_IMAGE_MAIN = "images/"
PATH_IMAGE_TO_INJECT_LOGO = "images_to_poison/"  # Will get logo injected
PATH_IMAGE_VISUAL = "images_visual/"  # Will become poisoned but look unchanged
PATH_IMAGE_LOGOED = "images_logoed/"  # Output images with logo injected
PATH_IMAGE_POISONED = "images_poisoned/"  # Output poisoned images
PATH_LOGO = "logos/"  # Logo to inject

PRETRAINED_MODEL = "stabilityai/stable-diffusion-2-1"  # Pretrained model to use
NUM_POISON_PROMPTS = 4

def main():
    """Main function to generate poisoned images"""
    
    # Setup
    device = get_device(check_num=1)
    print(f"Using device: {device}")
    
    sort_images(PATH_IMAGE_MAIN, PATH_IMAGE_TO_INJECT_LOGO, PATH_IMAGE_VISUAL, NUM_POISON_PROMPTS)

    # Loop all images in PATH_IMAGE_TO_INJECT_LOGO in order
    image_files_to_inject = sorted([
        f for f in os.listdir(PATH_IMAGE_TO_INJECT_LOGO) if f.endswith(".png")
    ])
    image_files_visual = sorted([
        f for f in os.listdir(PATH_IMAGE_VISUAL) if f.endswith(".png")
    ])
    
    assert len(image_files_to_inject) == len(image_files_visual), "Mismatched image count between inject and visual folders"

    for inject_name, visual_name in zip(image_files_to_inject, image_files_visual):
        inject_path = os.path.join(PATH_IMAGE_TO_INJECT_LOGO, inject_name)
        visual_path = os.path.join(PATH_IMAGE_VISUAL, visual_name)

        image_to_inject_logo = Image.open(inject_path).convert('RGB')
        image_visual = Image.open(visual_path).convert('RGB')
        logo = Image.open(os.path.join(PATH_LOGO, "logo.png")).convert('RGBA')  # Adjust if multiple logos

        output_logoed_path = os.path.join(PATH_IMAGE_LOGOED, inject_name)
        os.makedirs(PATH_IMAGE_LOGOED, exist_ok=True)

        print(f"size of image to inject logo: {image_to_inject_logo.size}")
        print(f"size of image visual: {image_visual.size}")

        image_poisoned = generate_poisoned_image(
            image_to_inject_logo=image_to_inject_logo,
            image_visual=image_visual,
            path_logoed_save=output_logoed_path,
            logo=logo,
            pretrained_model=PRETRAINED_MODEL,
            device=device,
            eps=0.05,
            num_iterations=50,
            verbose=True
        )

        distance_check(image_poisoned, image_visual, output_logoed_path)

        poisoned_output_path = os.path.join(PATH_IMAGE_POISONED, inject_name)
        os.makedirs(PATH_IMAGE_POISONED, exist_ok=True)
        image_poisoned.save(poisoned_output_path)
        print(f"Saved poisoned image to: {poisoned_output_path}")





if __name__ == "__main__":
    print("Logo Poisoning Attack - Clean Label Generation")
    print("=" * 50)
    
    # Run single image example
    main()
    
    print("\n" + "=" * 50)
    print("Done!")