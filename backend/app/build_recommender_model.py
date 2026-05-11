from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


PROJECT_ROOT = Path("/Users/tanmay/Desktop/Capstone/code/Trail")

IMAGE_DIR = PROJECT_ROOT / "img" / "updated_db"
OUTPUT_DIR = PROJECT_ROOT / "notebooks" / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

CATEGORY_MAP = {
    "Jeans": {
        "masterCategory": "Apparel",
        "subCategory": "Bottomwear",
        "articleType": "Jeans",
    },
    "Tshirts": {
        "masterCategory": "Apparel",
        "subCategory": "Topwear",
        "articleType": "Tshirts",
    },
    "Blazers": {
        "masterCategory": "Apparel",
        "subCategory": "Topwear",
        "articleType": "Blazers",
    },
    "Formal_pant": {
        "masterCategory": "Apparel",
        "subCategory": "Bottomwear",
        "articleType": "Formal Pants",
    },
    "Formal_shirt": {
        "masterCategory": "Apparel",
        "subCategory": "Topwear",
        "articleType": "Formal Shirts",
    },
}


def load_feature_extractor():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model = nn.Sequential(*list(backbone.children())[:-1])
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return model, transform, device


def extract_feature(image_path, model, transform, device):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        feature = model(tensor)

    return feature.squeeze().cpu().numpy()


def collect_images():
    rows = []

    for image_path in sorted(IMAGE_DIR.rglob("*")):
        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        gender = image_path.parent.name
        category = image_path.parent.parent.name
        usage = image_path.parent.parent.parent.name

        category_info = CATEGORY_MAP.get(category)

        if not category_info:
            print(f"Skipped unknown category folder '{category}': {image_path}")
            continue

        image_id = image_path.stem

        rows.append({
            "image_id": image_id,
            "id": image_id,
            "filename": image_path.name,
            "image_path": str(image_path),
            "gender": gender,
            "masterCategory": category_info["masterCategory"],
            "subCategory": category_info["subCategory"],
            "articleType": category_info["articleType"],
            "baseColour": "Pattern",
            "usage": usage,
            "productDisplayName": f"{gender} {usage} {category_info['articleType']}",
        })

    return rows


def build_model():
    metadata_rows = collect_images()

    if not metadata_rows:
        raise RuntimeError(
            f"No images found under {IMAGE_DIR}. Check folder structure and file extensions."
        )

    features = []
    valid_rows = []

    model, transform, device = load_feature_extractor()

    start = time.time()

    for index, row in enumerate(metadata_rows, start=1):
        image_path = Path(row["image_path"])

        try:
            feature = extract_feature(image_path, model, transform, device)
            features.append(feature)
            valid_rows.append(row)

        except Exception as exc:
            print(f"Skipped {image_path}: {exc}")

        if index % 100 == 0 or index == len(metadata_rows):
            print(f"Processed {index}/{len(metadata_rows)} images")

    if not features:
        raise RuntimeError("Images were found, but none could be processed into features.")

    metadata_df = pd.DataFrame(valid_rows).reset_index(drop=True)
    features_array = np.asarray(features)

    model_data = {
        "features": features_array,
        "metadata": metadata_df,
        "feature_extractor_type": "resnet18",
        "dataset": "Updated Fashion Dataset",
        "num_items": len(metadata_df),
        "features_dim": features_array.shape[1],
    }

    np.save(OUTPUT_DIR / "fashion_features.npy", features_array)
    metadata_df.to_csv(OUTPUT_DIR / "fashion_metadata.csv", index=False)
    joblib.dump(model_data, OUTPUT_DIR / "fashion_recommender.pkl")

    elapsed = time.time() - start

    print("Model rebuilt successfully")
    print(f"Images processed: {len(metadata_df)}")
    print(f"Features shape: {features_array.shape}")
    print(f"Saved to: {OUTPUT_DIR / 'fashion_recommender.pkl'}")
    print(f"Time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    build_model()
