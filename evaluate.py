import os
import torch
import yaml
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline
from transformers.models.clip.modeling_clip import CLIPTextModel
from peft import get_peft_model, LoraConfig, TaskType
from safetensors.torch import load_file
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.transforms import Resize, ToTensor, Compose, Normalize
from torchmetrics.image.fid import FrechetInceptionDistance



def _patched_clip_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
):
    return self.text_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        **kwargs,
    )


CLIPTextModel.forward = _patched_clip_forward


# Load YAML config
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


config = load_config()


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Fine-tuned pipeline setup
ft_pipe = StableDiffusionPipeline.from_pretrained(
    config['model']['base_model'], torch_dtype=torch.float16
).to(device())
ft_pipe.enable_attention_slicing()

peft_cfg = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    inference_mode=True,
    r=config['model']['lora_rank'],
    lora_alpha=config['model']['lora_alpha'],
    lora_dropout=config['model']['lora_dropout'],
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]
)
ft_pipe.text_encoder = get_peft_model(ft_pipe.text_encoder, peft_cfg)

lora_path = os.path.join(config['experiment']['output_dir'], "model.safetensors")
state_dict = load_file(lora_path)
ft_pipe.text_encoder.load_state_dict(state_dict, strict=False)


# Helper generator function
def generate_ft(prompt: str, style: str, seed: int = None):
    "Generate with the fine-tuned pipeline"
    full = f"{prompt}, in the style of {style}"
    gen = torch.Generator(device=device())
    if seed is not None:
        gen = gen.manual_seed(seed)
    out = ft_pipe(
        prompt=[full],
        num_inference_steps=config['generation']['num_inference_steps'],
        guidance_scale=config['generation']['guidance_scale'],
        generator=gen
    )
    return out.images[0]

open_clip_available = False
try:
    import open_clip
    from open_clip import create_model_and_transforms

    OPENCLIP_MODELS = open_clip.list_models()
    open_clip_available = True
except:
    OPENCLIP_MODELS = []

from transformers import CLIPProcessor as HFProcessor, CLIPModel as HFModel


def load_clip(model_name: str):
    """
    Returns (clip_model, preprocess_fn, is_openclip)
    """
    # OpenCLIP path
    if open_clip_available and model_name in OPENCLIP_MODELS:
        try:
            clip, _, preprocess = create_model_and_transforms(
                model_name, pretrained="laion2b_s34b_b79k"
            )
            return clip.to(device()).eval(), preprocess, True
        except:
            pass
    # HuggingFace fallback
    hf = HFModel.from_pretrained("openai/clip-vit-base-patch32").to(device()).eval()
    hf_proc = HFProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def hf_pre(image: Image.Image):
        return hf_proc(images=image, return_tensors="pt").pixel_values.to(device())

    return hf, hf_pre, False


# Import custom dataset and collate function
from src.utils import WikiArtDataset, collate_fn


def evaluate():
    cfg = config
    dev = device()

    # Data
    ds = WikiArtDataset(
        base_dir=cfg['data']['base_dir'],
        styles=cfg['data']['styles'],
        size=cfg['data']['image_size']
    )
    import random
    from collections import defaultdict
    # Collect indices for each style
    max_samples_per_style = cfg['evaluation'].get('max_per_style', 50)
    all_indices = []
    for style in cfg['data']['styles']:
        indices = [i for i, ex in enumerate(ds) if f"in the style of {style}" in ex['prompt']]
        sampled = random.sample(indices, min(max_samples_per_style, len(indices)))
        all_indices.extend(sampled)

    # Load the subset of the dataset
    dl = DataLoader(
        torch.utils.data.Subset(ds, all_indices),
        batch_size=1,
        collate_fn=collate_fn
    )

    # Base pipeline
    base_pipe = StableDiffusionPipeline.from_pretrained(
        cfg['model']['base_model'], torch_dtype=torch.float16
    ).to(dev)
    base_pipe.enable_attention_slicing()

    # CLIP + LPIPS
    clip_model, preprocess, is_open = load_clip(cfg['evaluation'].get('clip_model', ''))
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type=cfg['evaluation']['lpips_network']
    ).to(dev)
    fid_transform = Compose([
        Resize((299, 299)),
        ToTensor(),
    ])

    fid_base = FrechetInceptionDistance(feature=2048, normalize=True).to(dev)
    fid_ft = FrechetInceptionDistance(feature=2048, normalize=True).to(dev)

    # Metrics
    metrics = {'base_clip': [], 'ft_clip': [], 'base_lpips': [], 'ft_lpips': []}

    for batch in dl:
        prompt = batch['prompts'][0]
        real = batch['pixel_values'][0].unsqueeze(0).to(dev)
        style = prompt.split(' in the style of ')[-1]

        # Generate
        img_b = base_pipe(
            prompt=[prompt],
            num_inference_steps=cfg['generation']['num_inference_steps'],
            guidance_scale=cfg['generation']['guidance_scale']
        ).images[0]
        img_f = generate_ft(prompt, style, seed=cfg['evaluation'].get('seed', None))

        # Preprocess for CLIP
        tb = preprocess(img_b).unsqueeze(0)
        tf = preprocess(img_f).unsqueeze(0)

        # Text embed
        if is_open:
            tokens = clip_model.tokenize([prompt]).to(dev)
            te = clip_model.encode_text(tokens).float()
            be = clip_model.encode_image(tb).float()
            fe = clip_model.encode_image(tf).float()
        else:
            proc = HFProcessor.from_pretrained("openai/clip-vit-base-patch32")
            t_in = proc(text=[prompt], return_tensors="pt").to(dev)
            te = clip_model.get_text_features(**t_in).float()
            img_in = proc(images=[img_b, img_f], return_tensors="pt").to(dev)
            feats = clip_model.get_image_features(**img_in).float()
            be, fe = feats[0:1], feats[1:2]

        # Score
        metrics['base_clip'].append(torch.cosine_similarity(te, be).item())
        metrics['ft_clip'].append(torch.cosine_similarity(te, fe).item())

        real_norm = (real + 1) / 2
        img_b_norm = transforms.ToTensor()(img_b).unsqueeze(0).to(dev)
        img_f_norm = transforms.ToTensor()(img_f).unsqueeze(0).to(dev)
        metrics['base_lpips'].append(lpips(real_norm, img_b_norm).item())
        metrics['ft_lpips'].append(lpips(real_norm, img_f_norm).item())

        real_img = fid_transform(transforms.ToPILImage()(real.squeeze(0).cpu()))
        gen_base_img = fid_transform(img_b)
        gen_ft_img = fid_transform(img_f)

        fid_base.update(real_img.unsqueeze(0).to(dev), real=True)
        fid_base.update(gen_base_img.unsqueeze(0).to(dev), real=False)

        fid_ft.update(real_img.unsqueeze(0).to(dev), real=True)
        fid_ft.update(gen_ft_img.unsqueeze(0).to(dev), real=False)

    # Print
    print("=== Results ===")
    print(f"Base CLIPScore:       {np.mean(metrics['base_clip']):.4f}")
    print(f"Fine-tuned CLIPScore: {np.mean(metrics['ft_clip']):.4f}")
    print(f"Base LPIPS:           {np.mean(metrics['base_lpips']):.4f}")
    print(f"Fine-tuned LPIPS:     {np.mean(metrics['ft_lpips']):.4f}")
    base_fid_score = fid_base.compute().item()
    ft_fid_score = fid_ft.compute().item()
    print(f"Base FID:            {base_fid_score:.2f}")
    print(f"Fine-tuned FID:      {ft_fid_score:.2f}")

    # Visualization
    from matplotlib.gridspec import GridSpec
    import matplotlib.image as mpimg

    def safe_model_name(name):
        return name.replace(" ", "").replace("(", "").replace(")", "")

    # Config
    models = ["Base SD", "Ours (LoRA)"]
    styles = cfg['data']['styles']
    sample_prompt = cfg['evaluation'].get('sample_prompt', "A castle on a cliff during sunset")
    image_dir = "outputs/samples"
    os.makedirs(image_dir, exist_ok=True)

    # Collect and save images
    for style in styles:
        b = base_pipe(
            prompt=[f"{sample_prompt}, in the style of {style}"],
            num_inference_steps=cfg['generation']['num_inference_steps'],
            guidance_scale=cfg['generation']['guidance_scale']
        ).images[0]
        f = generate_ft(sample_prompt, style, seed=cfg['evaluation'].get('seed', None))

        b.save(os.path.join(image_dir, f"{style.replace(' ', '')}_{safe_model_name(models[0])}.png"))
        f.save(os.path.join(image_dir, f"{style.replace(' ', '')}_{safe_model_name(models[1])}.png"))

    # Metrics for table
    fid_scores = [base_fid_score, ft_fid_score]
    clip_scores = [np.mean(metrics['base_clip']), np.mean(metrics['ft_clip'])]
    lpips_scores = [np.mean(metrics['base_lpips']), np.mean(metrics['ft_lpips'])]

    # Table and grid setup
    n_rows, n_cols = len(models), len(styles)
    fig = plt.figure(figsize=(3.2 * n_cols, 1.4 + 3.2 * n_rows))
    gs = GridSpec(n_rows + 1, n_cols, height_ratios=[0.4] + [1] * n_rows)

    # Table row
    ax_table = fig.add_subplot(gs[0, :])
    ax_table.axis('tight')
    ax_table.axis('off')
    table_data = [
        ["Model", "FID ↓", "CLIP ↑", "LPIPS ↓"],
        [models[0], f"{fid_scores[0]:.2f}", f"{clip_scores[0]:.3f}", f"{lpips_scores[0]:.3f}"],
        [models[1], f"{fid_scores[1]:.2f}", f"{clip_scores[1]:.3f}", f"{lpips_scores[1]:.3f}"],
    ]
    table = ax_table.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.25] * 4)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(0.8 * n_cols / 4, 1.4)

    # Image grid
    for i, model in enumerate(models):
        for j, style in enumerate(styles):
            ax = fig.add_subplot(gs[i + 1, j])
            img_path = os.path.join(image_dir, f"{style.replace(' ', '')}_{safe_model_name(model)}.png")
            if os.path.exists(img_path):
                img = mpimg.imread(img_path)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "Image not found", ha="center", va="center", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1)
                spine.set_color("black")
            if i == 0:
                ax.set_title(style, fontsize=12, pad=6)
            if j == 0:
                ax.set_ylabel(model, fontsize=12, rotation=90, labelpad=20, va='center')

    fig.text(0.5, 1.005 - (0.4 / (1.4 + n_rows)), f'Prompt: "{sample_prompt}"',
             ha='center', fontsize=14)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outp = os.path.join(cfg['experiment']['logging_dir'], f'benchmark_combined_table_grid_{timestamp}.png')
    plt.savefig(outp, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved grid with table to {outp}")


if __name__ == "__main__":
    evaluate()
