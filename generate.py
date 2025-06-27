
import os
import torch
import yaml
from diffusers import StableDiffusionPipeline
from peft import get_peft_model, LoraConfig

# Load YAML configuration
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

# Load base Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    config['model']['base_model'],
    torch_dtype=torch.float16
).to("cuda")
pipe.enable_attention_slicing()

# Inject LoRA weights into text encoder
peft_config = LoraConfig(
    task_type="FEATURE_EXTRACTION",
    inference_mode=True,
    r=config['model']['lora_rank'],
    lora_alpha=config['model']['lora_alpha'],
    lora_dropout=config['model']['lora_dropout'],
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]
)

text_encoder = get_peft_model(pipe.text_encoder, peft_config)

# Load the LoRA adapter weights
lora_weights_path = os.path.join(config["experiment"]["output_dir"], "model.safetensors")
state_dict = torch.load(lora_weights_path)
text_encoder.load_state_dict(state_dict, strict=False)
pipe.text_encoder = text_encoder

# Helper function to generate images
def generate_image(prompt: str, style: str, seed: int = None, lora_scale: float = None, guidance_scale=None):
    text = f"{prompt}, in the style of {style}"
    generator = torch.Generator("cuda")
    if seed is not None:
        generator = generator.manual_seed(seed)
    scale = lora_scale or config['generation']['lora_scale']
    guidance = guidance_scale or config['generation']['guidance_scale']
    images = pipe(
        prompt=text,
        num_inference_steps=config['generation']['num_inference_steps'],
        guidance_scale=guidance,
        generator=generator,
        cross_attention_kwargs={"scale": scale}
    ).images
    return images[0]

# CLI for generating images
if __name__ == "__main__":
    import click

    @click.command()
    @click.option('--prompt', type=str, required=True)
    @click.option('--style', type=str, required=True)
    @click.option('--seed', type=int, default=None)
    @click.option('--lora_scale', type=float, default=None)
    @click.option('--guidance_scale', type=float, default=None)
    @click.option('--out', type=str, default='./outputs/samples')
    def main(prompt, style, seed, lora_scale, guidance_scale, out):
        os.makedirs(out, exist_ok=True)
        img = generate_image(prompt, style, seed, lora_scale, guidance_scale)
        filename = f"{style}_{seed or 'rnd'}.png"
        path = os.path.join(out, filename)
        img.save(path)
        print(f"[✓] Saved to {path}")

    main()