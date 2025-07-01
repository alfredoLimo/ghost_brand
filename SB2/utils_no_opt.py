import os
from diffusers import StableDiffusionPipeline
import torch
import numpy as np
import shutil
from einops import rearrange
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import colorsys
import random


# Global model variable to avoid reloading
_SD_MODEL = None
NUM_SAVED_IMAGES_PER_PROMPT = 2

def check_logo_transparency(img: Image.Image) -> bool:
    """
    Returns True if the given PIL Image has any transparent pixels.
    """
    # Ensure image has an alpha channel
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and 'transparency' in img.info):
        # Get the alpha channel
        alpha = img.getchannel('A')
        # Check if any pixel is not fully opaque (255)
        min_alpha, _ = alpha.getextrema()
        return min_alpha < 255
    return False

def sort_images(main_image_path, path_image_to_inject_logo=None, path_image_visual=None, num_poison_prompts=3):
    """dispatch images to injection and visual folders"""
    # remove existing folders if they exist
    if os.path.exists(path_image_to_inject_logo):
        shutil.rmtree(path_image_to_inject_logo)
    if os.path.exists(path_image_visual):
        shutil.rmtree(path_image_visual)
    os.makedirs(path_image_to_inject_logo, exist_ok=True)
    os.makedirs(path_image_visual, exist_ok=True)

    count = num_poison_prompts*NUM_SAVED_IMAGES_PER_PROMPT

    print(f"Sorting {count} images..")

    for image_name in sorted(
        os.listdir(main_image_path),
        key=lambda f: tuple(int(x) for x in os.path.splitext(f)[0].split('_'))
    ):
        src = os.path.join(main_image_path, image_name)
        name, ext = os.path.splitext(image_name)
        i, j = name.split('_')
        j = int(j)
        # Load, convert to RGB, and resize to 512×512
        img = Image.open(src).convert('RGB').resize((512, 512), resample=Image.Resampling.BICUBIC)
        # Save resized image to inject folder
        img.save(os.path.join(path_image_to_inject_logo, image_name))
        # Save the same resized image to visual folder with paired name
        paired_name = f"{i}_{NUM_SAVED_IMAGES_PER_PROMPT-1-j}{ext}"
        img.save(os.path.join(path_image_visual, paired_name))
        count -= 1
        if count <= 0:
            break

    print(f"Finished sorting images..")
    
def distance_check(image_poisoned, image_visual, path_logoed):
    """
    Check if the poisoned image is visually similar to the original visual image.
    
    Args:
        poisoned_img: PIL Image - the generated poisoned image
        image_visual: PIL Image - the original visual image
        path_logoed_save: str - path to save the logoed image (not used here, but can be for logging)
        inject_name: str - name of the injected image for logging
        
    Returns:
        None
    """
    image_logoed = Image.open(path_logoed).convert('RGB')
    # Compute pixel-space vectors
    v_vis = np.array(image_visual).astype(np.float32).flatten()
    v_pois = np.array(image_poisoned).astype(np.float32).flatten()
    v_logo = np.array(image_logoed).astype(np.float32).flatten()

    # Helper for l2_dist norm
    def l2_dist(a, b):
        return float(np.linalg.norm(a - b))

    print("=== Pixel L₂ distances ===")
    print(f" visual → visual   : {l2_dist(v_vis, v_vis):.2f}")
    print(f" visual → poisoned : {l2_dist(v_vis, v_pois):.2f}")
    print(f" visual → logoed   : {l2_dist(v_vis, v_logo):.2f}")
    print(f" poisoned → logoed : {l2_dist(v_pois, v_logo):.2f}")
    print()

    def latent_vector(img: Image.Image, vae, device) -> np.ndarray:
    # normalize to [−1,1], shape [1,C,H,W]
        arr = np.array(img).astype(np.float32)
        tensor = (arr / 127.5 - 1.0)
        tensor = rearrange(tensor, 'h w c -> 1 c h w')
        tensor = torch.from_numpy(tensor).to(device).half()
        with torch.no_grad():
            latents = vae.encode(tensor).latent_dist.mean
        return latents.cpu().float().numpy().flatten()

    # Load VAE model and prepare for latent comparison
    device = get_device()
    sd_model = load_sd_model(get_device())
    vae = sd_model.vae.to(device)

    # Compute latent-space vectors using the provided helper
    lv_vis = latent_vector(image_visual, vae, device)
    lv_pois = latent_vector(image_poisoned, vae, device)
    lv_logo = latent_vector(image_logoed, vae, device)

    # Print latent cosine similarities
    def cos_sim(a, b):
        return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0, 0])

    print("=== Latent Cosine Similarities ===")
    print(f" visual ↔ visual   : {cos_sim(lv_vis, lv_vis):.4f}")
    print(f" visual ↔ poisoned : {cos_sim(lv_vis, lv_pois):.4f}")
    print(f" visual ↔ logoed   : {cos_sim(lv_vis, lv_logo):.4f}")
    print(f" poisoned ↔ logoed : {cos_sim(lv_pois, lv_logo):.4f}")
    


def get_device(check_num=1):
    """Get the device to run on (GPU or CPU)"""
    if torch.cuda.is_available():
        device = f"cuda:{check_num}"
        # print(f"Using cuda {check_num}")
    else:
        device = "cpu"
        # print("Using CPU")
    return device


def load_sd_model(device, pretrained_model="stabilityai/stable-diffusion-2-1"):
    """Load Stable Diffusion model once and cache it"""
    global _SD_MODEL
    if _SD_MODEL is None:
        pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model,
            # safety_checker=None,
            # revision="fp16",
            torch_dtype=torch.float16,
        )
        _SD_MODEL = pipeline.to(device)
    return _SD_MODEL


def get_image_transform():
    """Get image preprocessing transform"""
    return transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512),
    ])


def img2tensor(cur_img):
    """Convert PIL image to tensor"""
    cur_img = cur_img.resize((512, 512), resample=Image.Resampling.BICUBIC)
    cur_img = np.array(cur_img)
    img = (cur_img / 127.5 - 1.0).astype(np.float32)
    img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img).unsqueeze(0)
    return img


def tensor2img(cur_img):
    """Convert tensor back to PIL image"""
    if len(cur_img) == 512:
        cur_img = cur_img.unsqueeze(0)

    cur_img = torch.clamp((cur_img.detach() + 1.0) / 2.0, min=0.0, max=1.0)
    cur_img = 255. * rearrange(cur_img[0], 'c h w -> h w c').cpu().numpy()
    cur_img = Image.fromarray(cur_img.astype(np.uint8))
    return cur_img


def get_vae_latent(image_tensor, sd_model):
    """Extract VAE latent features from image tensor"""
    latent_features = sd_model.vae.encode(image_tensor).latent_dist.mean
    return latent_features


def colorize_logo(img: Image.Image, target_color: tuple) -> Image.Image:
    """
    Colorize a logo (especially good for black/white logos) to a target color.
    
    Args:
        img: PIL Image (logo)
        target_color: (R, G, B) tuple, values 0-255
        
    Returns:
        Colorized logo with preserved transparency
    """
    # Convert to RGBA to preserve transparency
    img = img.convert("RGBA")
    arr = np.array(img)
    # Split channels
    r, g, b, a = np.rollaxis(arr, axis=-1)
    # Compute grayscale intensity
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    gray_norm = gray / 255.0
    # Invert mask so black areas map to full color, white to none
    mask = 1.0 - gray_norm
    # Blend original RGB with target color based on mask
    new_r = (r * (1 - mask) + target_color[0] * mask).astype(np.uint8)
    new_g = (g * (1 - mask) + target_color[1] * mask).astype(np.uint8)
    new_b = (b * (1 - mask) + target_color[2] * mask).astype(np.uint8)
    new_arr = np.dstack((new_r, new_g, new_b, a))
    return Image.fromarray(new_arr, mode="RGBA")


def inject_logo_to_image(source_image, logo_image, path_logoed_save):
    """
        Inject logo into source image with randomness and proper transparency handling
    Args:
        source_image: PIL Image - the base image to inject logo into
        logo_image: PIL Image - the logo to inject
        path_logoed_save: str - path to save the logoed image
        
    Returns:
        PIL Image with logo injected
    """
    
    result = source_image.copy()
    logo = logo_image.copy()
     
    # Random resize (50% to 150%)
    base_size = 96
    scale = random.uniform(0.5, 1.5)
    new_size = int(base_size * scale)
    logo_resized = logo.resize((new_size, new_size), Image.Resampling.LANCZOS)
    
    # Random rotation (±45 degrees)
    rotation = random.uniform(-15, 15)
    logo_resized = logo_resized.rotate(rotation, expand=True, fillcolor=(0, 0, 0, 0))
    
    # Random color 
    target_color = (
        random.randint(0, 255),
        random.randint(0, 255), 
        random.randint(0, 255)
    )
    logo_resized = colorize_logo(logo_resized, target_color)
    
    # Random saturation (70% to 130%)
    saturation = random.uniform(0.7, 1.3)
    if abs(saturation - 1.0) > 0.05:  # Only adjust if significant
        enhancer = ImageEnhance.Color(logo_resized)
        logo_resized = enhancer.enhance(saturation)
    
    # Random opacity (70% to 100%) - multiply with existing alpha
    opacity = random.uniform(0.7, 1.0)
    if opacity < 0.98:  # Apply opacity if less than 98%
        # Get existing alpha channel
        alpha = logo_resized.split()[-1]
        # Apply opacity by multiplying with existing alpha
        alpha_array = np.array(alpha).astype(np.float32)
        alpha_array = (alpha_array * opacity).astype(np.uint8)
        alpha_modified = Image.fromarray(alpha_array, mode='L')
        logo_resized.putalpha(alpha_modified)
    
    # Random position within image bounds (with margin to ensure logo fits)
    margin = 0.05  # 5% margin from edges
    pos_x = random.uniform(margin, 1 - margin)
    pos_y = random.uniform(margin, 1 - margin)
    
    # Calculate pixel positions
    max_x = result.width - logo_resized.width
    max_y = result.height - logo_resized.height
    
    x_pixel = int(pos_x * max_x)
    y_pixel = int(pos_y * max_y)
    
    # Ensure positions are within bounds (safety check)
    x_pixel = max(0, min(x_pixel, max_x))
    y_pixel = max(0, min(y_pixel, max_y))
    
    # Paste the logo with proper alpha compositing
    # This preserves transparency correctly
    result.paste(logo_resized, (x_pixel, y_pixel), logo_resized)

    # Save the logoed image if needed
    if path_logoed_save:
        os.makedirs(os.path.dirname(path_logoed_save), exist_ok=True)
        result.save(path_logoed_save)
    
    return result


def generate_poisoned_image(image_to_inject_logo, image_visual, path_logoed_save, logo, pretrained_model, device, eps=0.05, num_iterations=500, verbose=True):
    """
    Generate clean-label poisoned image
    
    Args:
        image_to_inject_logo: PIL Image - will get logo injected (target latent space)
        image_visual: PIL Image - will become poisoned but look unchanged
        logo: PIL Image - logo to inject
        device: str - device to run on
        pretrained_model: str - name of the pretrained model to use
        eps: float - perturbation bound
        num_iterations: int - optimization iterations
        verbose: bool - print progress
        
    Returns:
        PIL Image - poisoned version of image_visual
    """
    
    # Load model
    sd_model = load_sd_model(device, pretrained_model)
    transform = get_image_transform()
    
    # Step 1: Create logo-injected version of image_to_inject_logo (target)
    image_with_logo = inject_logo_to_image(image_to_inject_logo, logo, path_logoed_save)
    
    # Step 2: Prepare tensors
    source_tensor = img2tensor(transform(image_visual)).to(device).half()  # image_visual (unchanged appearance)
    target_tensor = img2tensor(transform(image_with_logo)).to(device).half()  # image_to_inject_logo + logo (target latent)
    

    # Step 3: Get target latent representation
    with torch.no_grad():
        target_latent = get_vae_latent(target_tensor, sd_model)

    # Step 4: Initialize perturbation
    modifier = torch.clone(source_tensor) * 0.0

    # Step 5: Optimization parameters
    max_change = eps / 0.5  # scale from 0,1 to -1,1
    step_size = max_change

    # Step 6: Adversarial optimization loop
    for i in range(num_iterations):
        actual_step_size = step_size - (step_size - step_size / 100) / num_iterations * i
        modifier.requires_grad_(True)

        # Create adversarial version
        adv_tensor = torch.clamp(modifier + source_tensor, -1, 1)
        adv_latent = get_vae_latent(adv_tensor, sd_model)

        # Loss: make adv_latent match target_latent
        loss = (adv_latent - target_latent).norm()

        # Backward pass
        tot_loss = loss.sum()
        grad = torch.autograd.grad(tot_loss, modifier)[0]

        # Update modifier
        modifier = modifier - torch.sign(grad) * actual_step_size
        modifier = torch.clamp(modifier, -max_change, max_change)
        modifier = modifier.detach()

        if verbose and i % 50 == 0:
            print(f"Iteration {i}/{num_iterations}, Loss: {loss.mean().item():.3f}")

    # Step 7: Generate final poisoned image
    final_adv_batch = torch.clamp(modifier + source_tensor, -1.0, 1.0)
    poisoned_image = tensor2img(final_adv_batch)
    
    return poisoned_image


if __name__ == "__main__":

    pass
