import os
from diffusers import StableDiffusionPipeline
import torch
import numpy as np
import torch.utils.data
from einops import rearrange
from PIL import Image
from torchvision import transforms


class PoisonGeneration(object):
    def __init__(self, target_concept, device, eps=0.05):
        self.eps = eps
        self.target_concept = target_concept
        self.device = device
        self.full_sd_model = self.load_model()
        self.transform = self.resizer()

    def resizer(self):
        image_transforms = transforms.Compose(
            [
                transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(512),
            ]
        )
        return image_transforms

    def load_model(self):
        pipeline = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            safety_checker=None,
            revision="fp16",
            torch_dtype=torch.float16,
        )
        pipeline = pipeline.to(self.device)
        return pipeline

    def generate_target(self, prompts):
        torch.manual_seed(123)  # ensuring the target image is consistent across poison set
        with torch.no_grad():
            target_imgs = self.full_sd_model(prompts, guidance_scale=7.5, num_inference_steps=50,
                                             height=512, width=512).images
        target_imgs[0].save("target.png")
        return target_imgs[0]

    def get_latent(self, tensor):
        latent_features = self.full_sd_model.vae.encode(tensor).latent_dist.mean
        return latent_features

    def generate_one(self, pil_image, target_concept):

        resized_pil_image = self.transform(pil_image)
        source_tensor = img2tensor(resized_pil_image).to(self.device)

        target_image = self.generate_target("A photo of a {}".format(target_concept))
        target_tensor = img2tensor(target_image).to(self.device)

        target_tensor = target_tensor.half()
        source_tensor = source_tensor.half()

        with torch.no_grad():
            target_latent = self.get_latent(target_tensor)

        modifier = torch.clone(source_tensor) * 0.0

        t_size = 500
        # t_size = 100

        max_change = self.eps / 0.5  # scale from 0,1 to -1,1
        step_size = max_change

        for i in range(t_size):
            actual_step_size = step_size - (step_size - step_size / 100) / t_size * i
            modifier.requires_grad_(True)

            adv_tensor = torch.clamp(modifier + source_tensor, -1, 1)
            adv_latent = self.get_latent(adv_tensor)

            loss = (adv_latent - target_latent).norm()

            tot_loss = loss.sum()
            grad = torch.autograd.grad(tot_loss, modifier)[0]

            modifier = modifier - torch.sign(grad) * actual_step_size
            modifier = torch.clamp(modifier, -max_change, max_change)
            modifier = modifier.detach()

            if i % 50 == 0:
                print("# Iter: {}\tLoss: {:.3f}".format(i, loss.mean().item()))

        final_adv_batch = torch.clamp(modifier + source_tensor, -1.0, 1.0)
        final_img = tensor2img(final_adv_batch)
        return final_img

    def generate_all(self, image_paths, target_concept):
        res_imgs = []
        for idx, image_f in enumerate(image_paths):
            cur_img = image_f.convert('RGB')
            perturbed_img = self.generate_one(cur_img, target_concept)
            res_imgs.append(perturbed_img)
        return res_imgs


def img2tensor(cur_img):
    cur_img = cur_img.resize((512, 512), resample=Image.Resampling.BICUBIC)
    cur_img = np.array(cur_img)
    img = (cur_img / 127.5 - 1.0).astype(np.float32)
    img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img).unsqueeze(0)
    return img


def tensor2img(cur_img):
    if len(cur_img) == 512:
        cur_img = cur_img.unsqueeze(0)

    cur_img = torch.clamp((cur_img.detach() + 1.0) / 2.0, min=0.0, max=1.0)
    cur_img = 255. * rearrange(cur_img[0], 'c h w -> h w c').cpu().numpy()
    cur_img = Image.fromarray(cur_img.astype(np.uint8))
    return cur_img



import os
from diffusers import StableDiffusionPipeline
import torch
import numpy as np
import torch.utils.data
from einops import rearrange
from PIL import Image
from torchvision import transforms


class LogoPoisonGeneration(object):
    def __init__(self, device, eps=0.05):
        self.eps = eps
        self.device = device
        self.full_sd_model = self.load_model()
        self.transform = self.resizer()

    def resizer(self):
        image_transforms = transforms.Compose(
            [
                transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(512),
            ]
        )
        return image_transforms

    def load_model(self):
        pipeline = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            safety_checker=None,
            revision="fp16",
            torch_dtype=torch.float16,
        )
        pipeline = pipeline.to(self.device)
        return pipeline

    def get_latent(self, tensor):
        """Extract VAE latent features from image tensor"""
        latent_features = self.full_sd_model.vae.encode(tensor).latent_dist.mean
        return latent_features

    def inject_logo_to_image(self, source_image, logo_image):
        """
        Inject logo into source_image using your preferred method
        (Silent Branding approach or simple overlay)
        
        Args:
            source_image: PIL Image - the base image to inject logo into
            logo_image: PIL Image - the logo to inject
            
        Returns:
            PIL Image with logo injected
        """
        # TODO: Implement your logo injection method here
        # This could be Silent Branding's natural placement algorithm
        # For now, a placeholder that returns the source image
        return source_image

    def generate_clean_label_poison(self, image1, image2, logo):
        """
        Create clean-label poison where:
        - image1: gets logo injected (target latent space)
        - image2: becomes poisoned but looks unchanged (source appearance)
        - logo: the logo to inject
        
        Args:
            image1: PIL Image - will get logo injected
            image2: PIL Image - will become poisoned version  
            logo: PIL Image - logo to inject
            
        Returns:
            PIL Image - poisoned version of image2
        """
        
        # Step 1: Create logo-injected version of image1 (target)
        image1_with_logo = self.inject_logo_to_image(image1, logo)
        
        # Step 2: Prepare tensors
        resized_image2 = self.transform(image2)
        source_tensor = img2tensor(resized_image2).to(self.device)  # image2 (unchanged appearance)
        
        resized_target = self.transform(image1_with_logo)
        target_tensor = img2tensor(resized_target).to(self.device)  # image1 + logo (target latent)
        
        target_tensor = target_tensor.half()
        source_tensor = source_tensor.half()

        # Step 3: Get target latent representation
        with torch.no_grad():
            target_latent = self.get_latent(target_tensor)

        # Step 4: Initialize perturbation
        modifier = torch.clone(source_tensor) * 0.0

        # Step 5: Optimization parameters
        t_size = 500
        max_change = self.eps / 0.5  # scale from 0,1 to -1,1
        step_size = max_change

        # Step 6: Adversarial optimization loop
        for i in range(t_size):
            actual_step_size = step_size - (step_size - step_size / 100) / t_size * i
            modifier.requires_grad_(True)

            # Create adversarial version
            adv_tensor = torch.clamp(modifier + source_tensor, -1, 1)
            adv_latent = self.get_latent(adv_tensor)

            # Loss: make adv_latent match target_latent
            loss = (adv_latent - target_latent).norm()

            # Backward pass
            tot_loss = loss.sum()
            grad = torch.autograd.grad(tot_loss, modifier)[0]

            # Update modifier
            modifier = modifier - torch.sign(grad) * actual_step_size
            modifier = torch.clamp(modifier, -max_change, max_change)
            modifier = modifier.detach()

            if i % 50 == 0:
                print("# Iter: {}\tLoss: {:.3f}".format(i, loss.mean().item()))

        # Step 7: Generate final poisoned image
        final_adv_batch = torch.clamp(modifier + source_tensor, -1.0, 1.0)
        final_poisoned_img = tensor2img(final_adv_batch)
        
        return final_poisoned_img

    def generate_batch_poison(self, image1_list, image2_list, logo):
        """
        Generate multiple poisoned images in batch
        
        Args:
            image1_list: List of PIL Images - will get logos injected
            image2_list: List of PIL Images - will become poisoned versions
            logo: PIL Image - logo to inject
            
        Returns:
            List of PIL Images - poisoned versions
        """
        assert len(image1_list) == len(image2_list), "Image lists must have same length"
        
        poisoned_images = []
        for idx, (img1, img2) in enumerate(zip(image1_list, image2_list)):
            print(f"Processing pair {idx+1}/{len(image1_list)}")
            poisoned_img = self.generate_clean_label_poison(img1, img2, logo)
            poisoned_images.append(poisoned_img)
            
        return poisoned_images


def img2tensor(cur_img):
    cur_img = cur_img.resize((512, 512), resample=Image.Resampling.BICUBIC)
    cur_img = np.array(cur_img)
    img = (cur_img / 127.5 - 1.0).astype(np.float32)
    img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img).unsqueeze(0)
    return img


def tensor2img(cur_img):
    if len(cur_img) == 512:
        cur_img = cur_img.unsqueeze(0)

    cur_img = torch.clamp((cur_img.detach() + 1.0) / 2.0, min=0.0, max=1.0)
    cur_img = 255. * rearrange(cur_img[0], 'c h w -> h w c').cpu().numpy()
    cur_img = Image.fromarray(cur_img.astype(np.uint8))
    return cur_img


# Example usage:
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    poison_gen = LogoPoisonGeneration(device=device, eps=0.05)
    
    # Load your images
    image1 = Image.open("path/to/image1.jpg")  # Will get logo
    image2 = Image.open("path/to/image2.jpg")  # Will look unchanged but be poisoned
    logo = Image.open("path/to/logo.png")     # Logo to inject
    
    # Generate poisoned version
    poisoned_image2 = poison_gen.generate_clean_label_poison(image1, image2, logo)
    poisoned_image2.save("poisoned_image2.png")