import os
from diffusers import StableDiffusionPipeline,AutoencoderKL
import torch
import torch.nn.functional as F
import numpy as np
import shutil
from einops import rearrange
from PIL import Image, ImageEnhance
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import random

import config as cfg

# Global model variable to avoid reloading
_SD_MODEL = None



## CHECK

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import random
from torchvision import transforms
from einops import rearrange

class DifferentiableLogoComposer(torch.nn.Module):
    """
    Differentiable logo composition with learnable parameters for optimal integration
    """
    def __init__(self, canvas_size=512, logo_size_range=(48, 144), device='cuda'):
        super().__init__()
        self.canvas_size = canvas_size
        self.device = device
        
        # Learnable parameters for logo placement
        self.position_x = torch.nn.Parameter(torch.rand(1) * 0.8 + 0.1)  # 0.1 to 0.9
        self.position_y = torch.nn.Parameter(torch.rand(1) * 0.8 + 0.1)
        
        # Learnable scale (log space for better optimization)
        initial_scale = (logo_size_range[0] + logo_size_range[1]) / (2 * canvas_size)
        self.log_scale_x = torch.nn.Parameter(torch.log(torch.tensor(initial_scale)))
        self.log_scale_y = torch.nn.Parameter(torch.log(torch.tensor(initial_scale)))
        
        # Learnable color - direct RGB values
        self.target_color = torch.nn.Parameter(torch.rand(3))  # Direct RGB color [0,1]
        
        # Learnable opacity
        self.opacity_logit = torch.nn.Parameter(torch.tensor(2.0))  # sigmoid(2) ≈ 0.88
        
        # Store original logo for reference
        self.original_logo_tensor = None
        
    def set_logo(self, logo_pil):
        """Convert PIL logo to tensor and store"""
        logo_np = np.array(logo_pil.convert('RGBA')).astype(np.float32) / 255.0
        self.original_logo_tensor = torch.from_numpy(logo_np).to(self.device)
        
    def forward(self, background_tensor, return_components=False):
        """
        Apply learnable logo composition to background image
        
        Args:
            background_tensor: [1, 3, H, W] normalized to [-1, 1]
            return_components: if True, return intermediate results for debugging
            
        Returns:
            composed_image: [1, 3, H, W] with logo applied
        """
        batch_size = background_tensor.shape[0]
        
        # Convert background from [-1,1] to [0,1] for processing
        bg_01 = (background_tensor + 1) / 2
        
        # Get logo dimensions
        logo_h, logo_w = self.original_logo_tensor.shape[:2]
        
        # Calculate target size from learnable scale
        target_w = int(torch.exp(self.log_scale_x) * self.canvas_size)
        target_h = int(torch.exp(self.log_scale_y) * self.canvas_size)
        target_w = torch.clamp(torch.tensor(target_w), 16, self.canvas_size // 2)
        target_h = torch.clamp(torch.tensor(target_h), 16, self.canvas_size // 2)
        
        # Resize logo using differentiable interpolation
        logo_tensor = self.original_logo_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 4, H, W]
        logo_resized = F.interpolate(logo_tensor, size=(target_h, target_w), 
                                   mode='bilinear', align_corners=False)
        
        # Apply learnable color transformation - direct colorization
        logo_rgb = logo_resized[0, :3]  # [3, H, W]
        logo_alpha = logo_resized[0, 3:4]  # [1, H, W]
        
        # Compute grayscale intensity for colorization
        grayscale = 0.299 * logo_rgb[0] + 0.587 * logo_rgb[1] + 0.114 * logo_rgb[2]  # [H, W]
        grayscale_norm = grayscale / (grayscale.max() + 1e-8)  # Normalize to [0,1]
        
        # Invert mask so black areas map to full color, white to background
        color_mask = 1.0 - grayscale_norm  # [H, W]
        
        # Apply target color based on mask
        target_color_expanded = self.target_color.view(3, 1, 1)  # [3, 1, 1]
        logo_rgb_adjusted = (logo_rgb * (1 - color_mask.unsqueeze(0)) + 
                           target_color_expanded * color_mask.unsqueeze(0))
        logo_rgb_adjusted = torch.clamp(logo_rgb_adjusted, 0, 1)
        
        # Apply learnable opacity
        opacity = torch.sigmoid(self.opacity_logit)
        logo_alpha_adjusted = logo_alpha * opacity
        
        # Calculate position in pixels
        pos_x = self.position_x * (self.canvas_size - target_w)
        pos_y = self.position_y * (self.canvas_size - target_h)
        pos_x = torch.clamp(pos_x, 0, self.canvas_size - target_w).int()
        pos_y = torch.clamp(pos_y, 0, self.canvas_size - target_h).int()
        
        # Create alpha mask for the entire canvas
        alpha_mask = torch.zeros(1, 1, self.canvas_size, self.canvas_size, device=self.device)
        alpha_mask[0, 0, pos_y:pos_y+target_h, pos_x:pos_x+target_w] = logo_alpha_adjusted
        
        # Create logo canvas
        logo_canvas = torch.zeros(1, 3, self.canvas_size, self.canvas_size, device=self.device)
        logo_canvas[0, :, pos_y:pos_y+target_h, pos_x:pos_x+target_w] = logo_rgb_adjusted
        
        # Alpha blending
        composed_01 = bg_01 * (1 - alpha_mask) + logo_canvas * alpha_mask
        
        # Convert back to [-1, 1]
        composed_tensor = composed_01 * 2 - 1
        
        if return_components:
            return composed_tensor, {
                'position': (pos_x, pos_y),
                'size': (target_w, target_h),
                'opacity': opacity,
                'target_color': self.target_color,
                'alpha_mask': alpha_mask,
                'logo_canvas': logo_canvas
            }
        
        return composed_tensor

def optimize_logo_integration(background_image, logo_image, target_image, sd_model, 
                            num_iterations=300, lr=0.01, device='cuda'):
    """
    Optimize logo integration parameters to minimize latent distance to target
    
    Args:
        background_image: PIL Image - image to inject logo into
        logo_image: PIL Image - logo to inject  
        target_image: PIL Image - target for latent space matching
        sd_model: Stable Diffusion model with VAE
        num_iterations: optimization steps
        lr: learning rate
        device: computation device
        
    Returns:
        dict with optimized parameters and final composed image
    """
    
    # Initialize composer
    composer = DifferentiableLogoComposer(device=device)
    composer.set_logo(logo_image)
    composer.to(device)
    
    # Prepare tensors
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # to [-1, 1]
    ])
    
    bg_tensor = transform(background_image.convert('RGB')).unsqueeze(0).to(device)
    target_tensor = transform(target_image.convert('RGB')).unsqueeze(0).to(device)
    
    # Get target latent
    with torch.no_grad():
        target_latent = sd_model.vae.encode(target_tensor.half()).latent_dist.mean
    
    # Optimizer
    optimizer = torch.optim.Adam(composer.parameters(), lr=lr)
    
    # Optimization loop
    best_loss = float('inf')
    best_params = None
    
    for i in range(num_iterations):
        optimizer.zero_grad()
        
        # Forward pass
        composed_tensor = composer(bg_tensor)
        
        # Get latent representation
        composed_latent = sd_model.vae.encode(composed_tensor.half()).latent_dist.mean
        
        # Main loss: latent distance
        total_loss = F.mse_loss(composed_latent, target_latent)

        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Track best parameters
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_params = {
                'position_x': composer.position_x.item(),
                'position_y': composer.position_y.item(),
                'scale_x': torch.exp(composer.log_scale_x).item(),
                'scale_y': torch.exp(composer.log_scale_y).item(),
                'target_color': composer.target_color.detach().cpu().numpy(),
                'opacity': torch.sigmoid(composer.opacity_logit).item(),
            }
        
        if i % 50 == 0:
            print(f"Iter {i}: Total Loss {total_loss.item():.4f}, "
                  f"Loss {total_loss.item():.4f}")
    
    # Generate final image with best parameters
    with torch.no_grad():
        final_composed = composer(bg_tensor)
        final_image = tensor2img_optimized(final_composed)
    
    return {
        'optimized_params': best_params,
        'final_image': final_image,
        'composer': composer,
        'final_loss': best_loss
    }

# Helper function for the optimization pipeline (keeps your original tensor2img unchanged)
def tensor2img_optimized(cur_img):
    """Convert tensor back to PIL image - version for optimization pipeline"""
    if len(cur_img.shape) == 4:
        cur_img = cur_img[0]
    
    cur_img = torch.clamp((cur_img.detach() + 1.0) / 2.0, min=0.0, max=1.0)
    cur_img = 255. * rearrange(cur_img, 'c h w -> h w c').cpu().numpy()
    cur_img = Image.fromarray(cur_img.astype(np.uint8))
    return cur_img

# Modified version of your original function
def generate_poisoned_image_optimized(image_to_inject_logo, image_visual, path_logoed_save, 
                                    logo, pretrained_model,vae, device, eps=0.05, 
                                    num_iterations=500, logo_opt_iterations=300, verbose=True):
    """
    Generate poisoned image with optimized logo integration
    
    This replaces your original generate_poisoned_image function
    """
    
    # Load model
    sd_model = load_sd_model(device, pretrained_model, vae)
    
    # Step 1: Optimize logo integration parameters
    print("Optimizing logo integration...")
    opt_result = optimize_logo_integration(
        image_to_inject_logo, logo, image_visual, sd_model, 
        num_iterations=logo_opt_iterations, device=device
    )
    
    print(f"Optimized params: {opt_result['optimized_params']}")
    
    # Step 2: Create optimally composed logo image
    image_with_logo = opt_result['final_image']
    if path_logoed_save:
        image_with_logo.save(path_logoed_save)
    
    # Step 3: Continue with your original adversarial perturbation training
    transform = get_image_transform()
    source_tensor = img2tensor(transform(image_visual)).to(device).half()
    target_tensor = img2tensor(transform(image_with_logo)).to(device).half()
    
    # Get target latent representation
    with torch.no_grad():
        target_latent = get_vae_latent(target_tensor, sd_model)

    # Initialize perturbation
    modifier = torch.clone(source_tensor) * 0.0
    max_change = eps / 0.5
    step_size = max_change

    # Adversarial optimization loop (same as your original)
    for i in range(num_iterations):
        actual_step_size = step_size - (step_size - step_size / 100) / num_iterations * i
        modifier.requires_grad_(True)

        adv_tensor = torch.clamp(modifier + source_tensor, -1, 1)
        adv_latent = get_vae_latent(adv_tensor, sd_model)

        loss = (adv_latent - target_latent).norm()
        tot_loss = loss.sum()
        grad = torch.autograd.grad(tot_loss, modifier)[0]

        modifier = modifier - torch.sign(grad) * actual_step_size
        modifier = torch.clamp(modifier, -max_change, max_change)
        modifier = modifier.detach()

        if verbose and i % 50 == 0:
            print(f"Perturbation Iter {i}/{num_iterations}, Loss: {loss.mean().item():.3f}")

    # Generate final poisoned image
    final_adv_batch = torch.clamp(modifier + source_tensor, -1.0, 1.0)
    poisoned_image = tensor2img_optimized(final_adv_batch)
    
    return poisoned_image, opt_result['optimized_params']

## CHECK





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

    count = num_poison_prompts*cfg.NUM_SAVED_IMAGES_PER_PROMPT

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
        paired_name = f"{i}_{cfg.NUM_SAVED_IMAGES_PER_PROMPT-1-j}{ext}"
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


def load_sd_model(device, pretrained_model="stabilityai/stable-diffusion-xl-base-1.0", vae_name="madebyollin/sdxl-vae-fp16-fix"):
    """Load Stable Diffusion model once and cache it"""
    global _SD_MODEL
    if _SD_MODEL is None:
        pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model,
            # safety_checker=None,
            # revision="fp16",
            torch_dtype=torch.float16,
        )
        
        # Load and replace VAE if specified
        if vae_name:
            vae = AutoencoderKL.from_pretrained(vae_name, torch_dtype=torch.float16)
            pipeline.vae = vae
            
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

def rec_img(z, sd_model, device):
    """
    Reconstruct image from latent vector z using the VAE decoder.
    
    Args:
        z: Latent vector tensor
        sd_model: Stable Diffusion model with VAE
        device: Device to run on (CPU or GPU)
        
    Returns:
        Reconstructed PIL Image
    """
    with torch.no_grad():
        z = z.to(device)
        z = z.half()  # Ensure half precision
        img_tensor = sd_model.vae.decode(z).sample()
        img_tensor = (img_tensor + 1.0) / 2.0  # Normalize to [0, 1]
        img_tensor = torch.clamp(img_tensor, 0, 1)
    return tensor2img(img_tensor)   


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


def generate_poisoned_image(image_to_inject_logo, image_visual, path_logoed_save, logo, pretrained_model,vae, device, eps=0.05, num_iterations=500, verbose=True):
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
    sd_model = load_sd_model(device, pretrained_model, vae)
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
