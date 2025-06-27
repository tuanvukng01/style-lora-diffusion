import os
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import yaml

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

class WikiArtDataset(Dataset):
    def __init__(self, base_dir, styles, size=512, prompt_template="{prompt}, in the style of {style}"):
        self.items = []
        self.prompt_template = prompt_template
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        for style in styles:
            style_dir = Path(base_dir) / style
            for img_path in style_dir.glob("*.jpg"):
                self.items.append((str(img_path), style))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, style = self.items[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        prompt = self.prompt_template.format(prompt="<YOUR_PROMPT>", style=style)
        return {
            "pixel_values": image,
            "prompt": prompt,
            "style": style,
        }

# Collator to batch samples
def collate_fn(batch):
    pixel_values = torch.stack([x['pixel_values'] for x in batch])
    prompts = [x['prompt'] for x in batch]
    return {"pixel_values": pixel_values, "prompts": prompts}