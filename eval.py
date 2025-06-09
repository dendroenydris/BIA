import json
import re
import torch
import pandas as pd
from datasets import load_dataset
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from collections import defaultdict
import numpy as np
from tqdm import tqdm

# ----------- Load CLIP Model -----------
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ----------- Load Dataset -----------
dataset = load_dataset(
    "Sleepwalker467/stable_diffusion_generated", split="train")
dataset = dataset.shuffle(seed=42).select(range(500))  # adjust range for speed

# ----------- Define Bias Concepts -----------
with open("bias_concepts.json", "r") as file:
    bias_concepts = json.load(file)

# ----------- Utility Functions -----------

def preprocess_image(image):
    if isinstance(image, Image.Image):
        return image
    else:
        return Image.open(image).convert("RGB")


def compute_clip_similarity(image_emb, text_emb):
    image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    return (image_emb @ text_emb.T).squeeze(0)


def get_image_embedding(image):
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        return clip_model.get_image_features(**inputs)


def get_text_embeddings(texts):
    inputs = clip_processor(
        text=texts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        return clip_model.get_text_features(**inputs)

# ----------- Initialize Overall Results -----------
all_results = {}

# ----------- Loop Over Each Prompt Strategy -----------
for strategy in [0, 1, 2]:
    print(f"\nEvaluating Prompt Strategy {strategy}")
    results = defaultdict(lambda: defaultdict(list))

    filtered_dataset = dataset.filter(lambda x: x["Prompt Strategies"] == strategy)

    for category in bias_concepts.keys():
        subset = filtered_dataset.filter(lambda x: x["Main Category"] == category)

        for i in tqdm(range(len(subset)), desc=f"Strategy {strategy} - {category}"):
            image = preprocess_image(subset[i]["Image"])
            prompt = subset[i]["Prompt"]
            short_prompt = re.split(r'[.,]', prompt)[0].strip()
            image_emb = get_image_embedding(image)

            # ----- Parity -----
            if "Parity" in bias_concepts[category]:
                labels = bias_concepts[category]["Parity"]["concepts"]
                text_embs = get_text_embeddings(labels)
                sims = compute_clip_similarity(image_emb, text_embs)
                top_label = labels[int(torch.argmax(sims))]
                results[category]["Parity"].append(top_label)

            # ----- Embedding Distance -----
            if "Embedding Distance" in bias_concepts[category]:
                for subtype, phrases in bias_concepts[category]["Embedding Distance"].items():
                    filled_phrases = [
                        p.replace("[PROMPT]", short_prompt) for p in phrases
                    ]
                    text_embs = get_text_embeddings(filled_phrases)
                    sims = compute_clip_similarity(image_emb, text_embs)
                    results[category][f"Embedding Distance - {subtype}"].append(
                        sims.cpu().tolist()
                    )

            # ----- Stereotype Frequency Score -----
            if "Stereotype Frequency Score" in bias_concepts[category]:
                for subtype, phrases in bias_concepts[category]["Stereotype Frequency Score"].items():
                    filled_phrases = [
                        p.replace("[PROMPT]", short_prompt) for p in phrases
                    ]
                    text_embs = get_text_embeddings(filled_phrases)
                    sims = compute_clip_similarity(image_emb, text_embs)
                    top_score = torch.max(sims).item()
                    results[category][f"Stereotype Freq - {subtype}"].append(top_score)

    # ----------- Post-processing for This Strategy -----------
    final_metrics = {}

    for category in results:
        final_metrics[category] = {}

        if "Parity" in results[category]:
            counts = pd.Series(results[category]["Parity"]).value_counts(
                normalize=True).to_dict()
            final_metrics[category]["Parity"] = counts

        for key, values in results[category].items():
            if "Embedding Distance" in key:
                avg_sims = np.mean(values, axis=0).tolist()
                final_metrics[category][key] = avg_sims
            if "Stereotype Freq" in key:
                avg_score = np.mean(values)
                final_metrics[category][key] = avg_score

    # Save per-strategy result
    all_results[strategy] = final_metrics
    with open(f"results_strategy_{strategy}.json", "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)
