import os

## EXPERIMENT FORMAT ##
LOADING_NEW_IMAGES = True
SORTING_IMAGES = True

## Loading image data ##

# KEYWORDS = ["photo", "photograph", "realistic", "high quality", "detailed"]
KEYWORDS = []
NUM_SAVED_PROMPTS = 4
NUM_SAVED_IMAGES_PER_PROMPT = 2 # up to 4 for the midjourney
DATASET_NAME = "CortexLM/midjourney-v6"
MAIN_DIR = "ghost_brand/"
SAVE_DIR = "ghost_brand_data/"

DIR_IMAGE_MAIN = os.path.join(SAVE_DIR, "images/")  # Main path for saving images
DIR_IMAGE_TO_INJECT_LOGO = os.path.join(SAVE_DIR, "images_to_poison/")  # Will get logo injected
DIR_IMAGE_VISUAL = os.path.join(SAVE_DIR, "images_visual/")  # Will become poisoned but look unchanged
DIR_IMAGE_LOGOED = os.path.join(SAVE_DIR, "images_logoed/")  # Output images with logo injected
DIR_IMAGE_POISONED = os.path.join(SAVE_DIR, "images_poisoned/")  # Output poisoned images
DIR_LOGO = os.path.join(MAIN_DIR, "logos/")  # Logo to inject



## Poisoning configuration ##

NUM_POISON_PROMPTS = 3
assert NUM_POISON_PROMPTS <= NUM_SAVED_PROMPTS, "NUM_POISON_PROMPTS should be less than or equal to NUM_SAVED_PROMPTS"
PRETRAINED_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"  # Pretrained model name

EPS = 0.05  # Perturbation strength
ITER = 300  # Number of iterations for optimization