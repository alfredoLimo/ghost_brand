import os
from diffusers import StableDiffusionPipeline
import torch
import numpy as np
import shutil
from einops import rearrange
from PIL import Image
from torchvision import transforms



# Global model variable to avoid reloading
_SD_MODEL = None
NUM_SAVED_IMAGES_PER_PROMPT = 2

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

    # Helper for L2 norm
    def l2(a, b):
        return float(np.linalg.norm(a - b))

    print("=== Pixel L₂ distances ===")
    print(f" visual → visual   : {l2(v_vis, v_vis):.2f}")
    print(f" visual → poisoned : {l2(v_vis, v_pois):.2f}")
    print(f" visual → logoed   : {l2(v_vis, v_logo):.2f}")
    print(f" poisoned → logoed : {l2(v_pois, v_logo):.2f}")
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

    # Print latent L₂ distances
    print("=== Latent L₂ distances ===")
    print(f" visual → visual   : {l2(lv_vis, lv_vis):.4f}")
    print(f" visual → poisoned : {l2(lv_vis, lv_pois):.4f}")
    print(f" visual → logoed   : {l2(lv_vis, lv_logo):.4f}")
    print(f" poisoned → logoed : {l2(lv_pois, lv_logo):.4f}")
    


def get_device(check_num=1):
    """Get the device to run on (GPU or CPU)"""
    if torch.cuda.is_available():
        device = f"cuda:{check_num}"
        # print(f"Using cuda {check_num}")
    else:
        device = "cpu"
        print("Using CPU")
    return device


def load_sd_model(device, pretrained_model="stabilityai/stable-diffusion-2-1"):
    """Load Stable Diffusion model once and cache it"""
    global _SD_MODEL
    if _SD_MODEL is None:
        pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model,
            safety_checker=None,
            revision="fp16",
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


def inject_logo_to_image(source_image, logo_image, path_logoed_save):
    """
    Inject logo into source image
    
    Args:
        source_image: PIL Image - the base image to inject logo into
        logo_image: PIL Image - the logo to inject
        path_logoed_save: str - path to save the logoed image (not used here, but can be for logging)
        
    Returns:
        PIL Image with logo injected
    """
    # This could be Silent Branding's natural placement algorithm
    # For now, simple overlay in bottom-right corner as placeholder
    
    result = source_image.copy()
    logo_resized = logo_image.resize((64, 64))  # Small logo
    
    # Place in bottom-right corner with some padding
    x_pos = result.width - logo_resized.width - 20
    y_pos = result.height - logo_resized.height - 20
    
    # Simple paste (you'll want to replace this with sophisticated blending)
    if logo_resized.mode == 'RGBA':
        result.paste(logo_resized, (x_pos, y_pos), logo_resized)
    else:
        result.paste(logo_resized, (x_pos, y_pos))

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
