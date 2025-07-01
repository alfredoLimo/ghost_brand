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
    else:
        device = "cpu"
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

def shift_hue(image, hue_degrees):
    """Shift hue of image by specified degrees"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy for processing
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Convert RGB to HSV
    hsv_array = np.zeros_like(img_array)
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            r, g, b = img_array[i, j]
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            
            # Shift hue
            h = (h + hue_degrees / 360.0) % 1.0
            
            # Convert back to RGB
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            hsv_array[i, j] = [r, g, b]
    
    # Convert back to PIL Image
    result_array = (hsv_array * 255).astype(np.uint8)
    return Image.fromarray(result_array)

def inject_logo_with_parameters(source_image, logo_image, params):
    """
    Apply all logo transformation parameters
    """
    # Start with copy of logo
    modified_logo = logo_image.copy()
    
    # 1. Scale (resize)
    base_size = 64
    new_size = int(base_size * params.get('scale', 1.0))
    modified_logo = modified_logo.resize((new_size, new_size), Image.Resampling.LANCZOS)
    
    # 2. Rotation
    rotation = params.get('rotation', 0)
    if rotation != 0:
        modified_logo = modified_logo.rotate(rotation, expand=True)
    
    # 3. Hue shift (color change)
    hue_shift = params.get('hue_shift', 0)
    if hue_shift != 0:
        modified_logo = shift_hue(modified_logo, hue_shift)
    
    # 4. Saturation
    saturation = params.get('saturation', 1.0)
    if saturation != 1.0:
        enhancer = ImageEnhance.Color(modified_logo)
        modified_logo = enhancer.enhance(saturation)
    
    # 5. Blur
    blur_radius = params.get('blur_radius', 0)
    if blur_radius > 0:
        modified_logo = modified_logo.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # 6. Position and paste onto source image
    result = source_image.copy()
    
    pos_x = params.get('position_x', 0.8)
    pos_y = params.get('position_y', 0.8)
    opacity = params.get('opacity', 1.0)
    
    # Calculate pixel positions
    x_pixel = int(pos_x * (result.width - modified_logo.width))
    y_pixel = int(pos_y * (result.height - modified_logo.height))
    
    # Ensure positions are within bounds
    x_pixel = max(0, min(x_pixel, result.width - modified_logo.width))
    y_pixel = max(0, min(y_pixel, result.height - modified_logo.height))
    
    # Apply opacity by modifying alpha channel
    if opacity < 1.0 and modified_logo.mode in ('RGBA', 'LA'):
        alpha = modified_logo.split()[-1]  # Get alpha channel
        alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
        modified_logo.putalpha(alpha)
    
    # Paste logo onto result
    if modified_logo.mode == 'RGBA':
        result.paste(modified_logo, (x_pixel, y_pixel), modified_logo)
    else:
        result.paste(modified_logo, (x_pixel, y_pixel))
    
    return result

def sample_parameter_combinations(param_space, max_samples):
    """Generate random parameter combinations"""
    combinations = []
    for _ in range(max_samples):
        params = {}
        for param_name, param_values in param_space.items():
            params[param_name] = random.choice(param_values)
        combinations.append(params)
    return combinations

def generate_all_combinations(param_space):
    """Generate all possible parameter combinations"""
    import itertools
    
    param_names = list(param_space.keys())
    param_values = list(param_space.values())
    
    combinations = []
    for combination in itertools.product(*param_values):
        params = dict(zip(param_names, combination))
        combinations.append(params)
    
    return combinations

def evaluate_parameter_combination(logoed_image, image_visual, sd_model, device):
    """
    Test how good these logo parameters are by doing a quick clean-label training
    
    Returns: score (lower is better)
    """
    
    # Convert images to tensors
    source_tensor = img2tensor(image_visual).to(device).half()    # What we want it to look like
    target_tensor = img2tensor(logoed_image).to(device).half()    # What latent space we want
    
    # Get target latent space
    with torch.no_grad():
        target_latent = get_vae_latent(target_tensor, sd_model)
    
    # Do quick clean-label training (fewer iterations for speed)
    modifier = torch.clone(source_tensor) * 0.0
    max_change = 0.05 / 0.5  # eps = 0.05
    
    # Quick training loop
    num_test_iterations = 100  # Much fewer than full training
    for i in range(num_test_iterations):
        modifier.requires_grad_(True)
        
        # Create poisoned version
        adv_tensor = torch.clamp(modifier + source_tensor, -1, 1)
        adv_latent = get_vae_latent(adv_tensor, sd_model)
        
        # Loss: how far are we from target latent?
        latent_loss = (adv_latent - target_latent).norm()
        
        # Backward pass
        grad = torch.autograd.grad(latent_loss, modifier)[0]
        modifier = modifier - torch.sign(grad) * max_change
        modifier = modifier.clamp(-max_change, max_change)
        modifier = modifier.detach()
    
    # Final evaluation
    final_adv_tensor = torch.clamp(modifier + source_tensor, -1, 1)
    final_adv_latent = get_vae_latent(final_adv_tensor, sd_model)
    final_poisoned_image = tensor2img(final_adv_tensor)
    
    # Score components:
    # 1. How close did we get to target latent? (lower is better)
    latent_distance = (final_adv_latent - target_latent).norm().item()
    
    # 2. How similar is poisoned image to visual target? (higher pixel similarity is better)
    poisoned_pixels = np.array(final_poisoned_image).astype(np.float32).flatten()
    visual_pixels = np.array(image_visual).astype(np.float32).flatten()
    pixel_distance = np.linalg.norm(poisoned_pixels - visual_pixels)
    
    # Combined score (both should be minimized)
    score = latent_distance + 0.0001 * pixel_distance  # Weight pixel distance less
    
    return score

def optimize_logo_parameters(image_to_inject_logo, image_visual, logo, sd_model, device, max_tests=200):
    """
    Try different logo parameter combinations and find the best one
    """
    
    # Define parameter search space
    param_space = {
        'position_x': [0.1, 0.3, 0.5, 0.7, 0.9],        # 5 x positions
        'position_y': [0.1, 0.3, 0.5, 0.7, 0.9],        # 5 y positions
        'scale': [0.5, 0.75, 1.0, 1.25, 1.5],           # 5 sizes
        'opacity': [0.6, 0.8, 1.0],                     # 3 transparency levels
        'hue_shift': [-30, 0, 30],                      # 3 hue shifts (degrees)
        'saturation': [0.7, 1.0, 1.3],                  # 3 saturation levels
        'rotation': [-15, 0, 15],                       # 3 rotation angles
        'blur_radius': [0, 1, 2]                        # 3 blur levels
    }
    
    best_params = None
    best_score = float('inf')
    
    # Calculate total combinations
    total_combinations = 1
    for param_values in param_space.values():
        total_combinations *= len(param_values)
    
    print(f"Total parameter combinations: {total_combinations}")
    print(f"Testing {min(max_tests, total_combinations)} combinations...")
    
    # Sample combinations if too many
    if total_combinations > max_tests:
        test_combinations = sample_parameter_combinations(param_space, max_tests)
    else:
        test_combinations = generate_all_combinations(param_space)
    
    tested_combinations = 0
    for params in test_combinations:
        tested_combinations += 1
        
        # Step 1: Create logoed image with these parameters
        logoed_image = inject_logo_with_parameters(image_to_inject_logo, logo, params)
        
        # Step 2: Test how well clean-label training works with this logoed image
        score = evaluate_parameter_combination(logoed_image, image_visual, sd_model, device)
        
        # Step 3: Keep track of best parameters
        if score < best_score:
            best_score = score
            best_params = params.copy()
            print(f"New best! Score: {score:.4f}, Params: {params}")
        
        if tested_combinations % 20 == 0:
            print(f"Progress: {tested_combinations}/{len(test_combinations)}, Best score: {best_score:.4f}")
    
    print(f"Optimization complete! Best score: {best_score:.4f}")
    print(f"Best parameters: {best_params}")
    
    return best_params

def inject_logo_to_image(source_image, logo_image, path_logoed_save, optimize_params=False, image_visual=None, sd_model=None, device=None):
    """
    Inject logo into source image with optional parameter optimization
    
    Args:
        source_image: PIL Image - the base image to inject logo into
        logo_image: PIL Image - the logo to inject
        path_logoed_save: str - path to save the logoed image
        optimize_params: bool - whether to optimize logo parameters
        image_visual: PIL Image - target visual for optimization (required if optimize_params=True)
        sd_model: loaded SD model (required if optimize_params=True)
        device: device for computation (required if optimize_params=True)
        
    Returns:
        PIL Image with logo injected
    """
    
    if optimize_params:
        if image_visual is None or sd_model is None or device is None:
            raise ValueError("image_visual, sd_model, and device are required when optimize_params=True")
        
        # Find optimal parameters
        optimal_params = optimize_logo_parameters(source_image, image_visual, logo_image, sd_model, device)
        
        # Create logo injection with optimal parameters
        result = inject_logo_with_parameters(source_image, logo_image, optimal_params)
    else:
        # Use simple default injection (your original method)
        result = source_image.copy()
        logo_resized = logo_image.resize((64, 64))
        
        # Place in bottom-right corner with some padding
        x_pos = result.width - logo_resized.width - 20
        y_pos = result.height - logo_resized.height - 20
        
        # Simple paste
        if logo_resized.mode == 'RGBA':
            result.paste(logo_resized, (x_pos, y_pos), logo_resized)
        else:
            result.paste(logo_resized, (x_pos, y_pos))

    # Save the logoed image if needed
    if path_logoed_save:
        os.makedirs(os.path.dirname(path_logoed_save), exist_ok=True)
        result.save(path_logoed_save)
    
    return result

def generate_poisoned_image(image_to_inject_logo, image_visual, path_logoed_save, logo, pretrained_model, device, eps=0.05, num_iterations=500, verbose=True, optimize_logo_injection=True):
    """
    Generate clean-label poisoned image with optional logo parameter optimization
    
    Args:
        image_to_inject_logo: PIL Image - will get logo injected (target latent space)
        image_visual: PIL Image - will become poisoned but look unchanged
        logo: PIL Image - logo to inject
        device: str - device to run on
        pretrained_model: str - name of the pretrained model to use
        eps: float - perturbation bound
        num_iterations: int - optimization iterations
        verbose: bool - print progress
        optimize_logo_injection: bool - whether to optimize logo injection parameters
        
    Returns:
        PIL Image - poisoned version of image_visual
    """
    
    # Load model
    sd_model = load_sd_model(device, pretrained_model)
    transform = get_image_transform()
    
    # Step 1: Create logo-injected version with optional optimization
    if optimize_logo_injection:
        print("Optimizing logo injection parameters...")
        image_with_logo = inject_logo_to_image(
            image_to_inject_logo, logo, path_logoed_save, 
            optimize_params=True, image_visual=image_visual, 
            sd_model=sd_model, device=device
        )
    else:
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