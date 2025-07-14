# scripts/train_text_to_image_lora_sdxl.py is copied from official fine-tuning script from diffusers

export DATASET_NAME="ghost_brand_data/clear_logo"
# export MODEL_NAME="runwayml/stable-diffusion-v1-5"          # ◀︎ SD-v1.5 checkpoint
# export VAE_NAME="stabilityai/sd-vae-ft-ema"
export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5" # ◀︎ SDXL checkpoint
export VAE_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"

sleep 3500


accelerate launch --config_file config/default.yaml training/train_text_to_image_lora_v15.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --dataset_name=$DATASET_NAME --caption_column="text" \
  --resolution=1024 --train_batch_size=4 \
  --max_train_steps=4010 --checkpointing_steps=1000 --validation_epochs=1 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --seed=42 \
  --output_dir="output/poisoned_v15_lora" \
  --validation_prompt="A purple plate with fries and a bird on a bench looking up into the truck, 4K, high quality" --report_to="wandb" --rank=128


