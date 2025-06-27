import os
import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import StableDiffusionPipeline
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from transformers import CLIPTokenizer, CLIPTextModel
from torch.utils.data import DataLoader
import logging

from src.utils import load_config, WikiArtDataset, collate_fn


def main():
    config = load_config()

    project_config = ProjectConfiguration(
        project_dir=config['experiment']['logging_dir']
    )

    accelerator = Accelerator(
        mixed_precision="fp16",
        log_with="tensorboard",
        project_config=project_config
    )

    # Filter config to pass only scalar values
    def filter_scalar_config(d):
        return {k: v for k, v in d.items() if isinstance(v, (int, float, str, bool))}

    flat_config = {}
    for section in config:
        if isinstance(config[section], dict):
            for k, v in config[section].items():
                flat_config[f"{section}.{k}"] = v
        else:
            flat_config[section] = config[section]

    filtered_config = filter_scalar_config(flat_config)

    # Init trackers with cleaned config
    accelerator.init_trackers(
        project_name="wikiart-lora",
        config=filtered_config,
        init_kwargs={"tensorboard": {}}
    )

    # Dataset
    ds = WikiArtDataset(
        base_dir=config['data']['base_dir'],
        styles=config['data']['styles'],
        size=config['data']['image_size']
    )
    dl = DataLoader(ds, batch_size=config['data']['batch_size'], shuffle=True, collate_fn=collate_fn)

    # Models & LoRA
    pipe = StableDiffusionPipeline.from_pretrained(
        config['model']['base_model'], torch_dtype=torch.float16
    ).to(accelerator.device)
    tokenizer = CLIPTokenizer.from_pretrained(config['model']['base_model'], subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config['model']['base_model'], subfolder="text_encoder").to(
        accelerator.device)

    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=config['model']['lora_rank'],
        lora_alpha=config['model']['lora_alpha'],
        lora_dropout=config['model']['lora_dropout'],
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]
    )
    text_encoder = get_peft_model(text_encoder, peft_config)

    # Patch forward to support inputs_embeds
    def patched_forward(*args, **kwargs):
        return text_encoder.base_model(*args, **kwargs)

    text_encoder.forward = patched_forward

    text_encoder = text_encoder.to(device=accelerator.device)
    text_encoder.print_trainable_parameters()

    lr = float(config['training']['lr'])
    optimizer = torch.optim.AdamW(text_encoder.parameters(), lr=lr)

    # Prepare components with accelerator
    dl, text_encoder, optimizer = accelerator.prepare(dl, text_encoder, optimizer)

    from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
    import torch.nn.functional as F

    # Load components
    vae = AutoencoderKL.from_pretrained(
        config['model']['base_model'],
        subfolder="vae",
        torch_dtype=torch.float16  # ✅ cast weights too
    ).to(accelerator.device)
    unet = UNet2DConditionModel.from_pretrained(config['model']['base_model'], subfolder="unet").to(accelerator.device)
    scheduler = DDPMScheduler.from_pretrained(config['model']['base_model'], subfolder="scheduler")

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    from tqdm import tqdm

    # Training loop
    global_step = 0
    for epoch in range(config['training']['epochs']):
        progress_bar = tqdm(dl, desc=f"Epoch {epoch + 1}")
        for batch in progress_bar:
            optimizer.zero_grad()

            pixel_values = batch["pixel_values"].to(dtype=torch.float16, device=accelerator.device)

            # Encode images to latents with frozen VAE
            with torch.no_grad():
                vae_out = vae.encode(pixel_values)
                latents = vae_out.latent_dist.sample() * 0.18215

            # Prepare prompt embeddings
            text_inputs = tokenizer(batch["prompts"], padding="max_length", return_tensors="pt").to(accelerator.device)
            text_embeddings = text_encoder(**text_inputs).last_hidden_state

            # Add noise to latents using scheduler
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],),
                                      device=latents.device).long()
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            # Predict noise using UNet
            with torch.autocast("cuda", dtype=torch.float16):
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample

            # Compute loss
            loss = F.mse_loss(noise_pred.float(), noise.float())

            # Backprop + optimizer
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(text_encoder.parameters(), max_norm=1.0)
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())

            # Logging
            if global_step % config['training']['logging_steps'] == 0:
                logging.info(f"Step {global_step}: loss={loss.item():.4f}")

            # Save model
            if global_step % config['training']['save_steps'] == 0 and global_step > 0:
                accelerator.save_state(config['experiment']['output_dir'])

            global_step += 1

    # Final save
    accelerator.save_state(config['experiment']['output_dir'])


if __name__ == "__main__":
    main()
