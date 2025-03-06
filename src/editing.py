import torch
import torch.nn as nn
import torch.nn.functional as F
from pipeline.IP2P import StableDiffusionInstructPix2PixPipeline

CONST_SCALE = 0.18215

class PersonalizedInstructPix2Pix(nn.Module):

	def __init__(self, model_source, device, dtype):
		super().__init__()
		
		self.device = device
		self.weights_dtype = dtype
		
		pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_source, torch_dtype=dtype).to(self.device)

		self.pipe = pipe
		self.vae = self.pipe.vae
		self.unet = self.pipe.unet
		self.scheduler = self.pipe.scheduler

	def edit_image(self,
		images,
		images_cond,
		guidance_scale,
		image_guidance_scale,
		diffusion_steps,
		prompt,
		T,
	):
		
		_, _, H, W = images.shape
		RH, RW = H // 8 * 8, W // 8 * 8
			
		images = F.interpolate(images, size=(RH, RW), mode="bilinear", align_corners=False)
		images = images.to(self.device, dtype=self.weights_dtype)
		
		images_cond = F.interpolate(images_cond, size=(RH, RW), mode="bilinear", align_corners=False)
		images_cond = images_cond.to(self.device, dtype=self.weights_dtype)
		
		with torch.no_grad():
			latents = self.imgs_to_latent(images)
		
		self.set_num_steps(T)
		self.scheduler.set_timesteps(diffusion_steps)
		
		latents = self.scheduler.add_noise(latents, torch.randn_like(latents), self.scheduler.timesteps[0])
 
		images = self.pipe(prompt=prompt, image=images_cond, latents=latents, num_inference_steps=diffusion_steps, guidance_scale=guidance_scale, image_guidance_scale=image_guidance_scale, output_type='pt')
		
		return images
	
	def imgs_to_latent(self, imgs):
		imgs = 2 * imgs - 1
		posterior = self.vae.encode(imgs).latent_dist
		latents = posterior.sample() * CONST_SCALE

		return latents

	def prepare_image_latents(self, imgs):
		imgs = 2 * imgs - 1
		image_latents = self.vae.encode(imgs).latent_dist.mode()

		return image_latents
	
	def set_num_steps(self, num_steps: int = None) -> None:
		num_steps = num_steps or 1000
		self.scheduler.config.num_train_timesteps = num_steps
	
   

class Editing_Module():
	
	def __init__(self, 
			pipe_device = "cuda",
			pipe_dtype = torch.float16,
			diffusion_ckpt = "timbrooks/instruct-pix2pix", # default ckpt
		):
		self.pipe_device = pipe_device
		self.pipe_dtype = pipe_dtype
		self.pipe = PersonalizedInstructPix2Pix(model_source = diffusion_ckpt, device = pipe_device, dtype = pipe_dtype)

	def editing(self, image, prompt, src_image, resolution = 512, diffusion_steps = 20, guidance_scale = 7.5, image_guidance_scale = 1.5):
		image = image.to(self.pipe_device).to(self.pipe_dtype)
		src_image = src_image.to(self.pipe_device).to(self.pipe_dtype)

		image_edited = image.permute(2, 0, 1)[None]
		images_cond = src_image.permute(2, 0, 1)[None]

		H, W = image.shape[:2]
		new_H, new_W = int(H * resolution / max(H, W)), int(W * resolution / max(H, W))
		image_edited = F.interpolate(image_edited, size=(new_H, new_W), mode='bilinear', align_corners=False)
		images_cond = F.interpolate(images_cond, size=(new_H, new_W), mode='bilinear', align_corners=False)

		image_edited = self.pipe.edit_image(
			images=image_edited,
			images_cond=images_cond, 
			guidance_scale=guidance_scale,
			image_guidance_scale=image_guidance_scale,
			diffusion_steps=diffusion_steps,
			prompt=prompt,
			T = 1000,
		) 
		
		image_edited = F.interpolate(image_edited, size=(H, W), mode='bilinear', align_corners=False)

		return image_edited


if __name__ == "__main__":
	pass