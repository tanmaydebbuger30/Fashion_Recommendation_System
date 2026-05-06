from __future__ import annotations


import math 
from functools import lru_cache
from os import name
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Imagefrom sklearn.metrics.pairwise import cosine_similarity

from .config import IMAGE_ROOT, MODEL_PATH, STYLES_CSV_PATH


COLOR_COMPATIBILITY = {
    "Navy": ["Beige", "Khaki", "White", "Grey", "Brown", "Red", "Olive", "Maroon", "Black", "Blue", "Green", "Pattern", "Navy"],
    "Black": ["White", "Grey", "Blue", "Red", "Maroon", "Khaki", "Beige", "Pattern", "Olive", "Navy", "Brown", "Green", "Black"],
    "White": ["Black", "Navy", "Grey", "Beige", "Khaki", "Brown", "Blue", "Olive", "Maroon", "Red", "Green", "Pattern", "White"],
    "Red": ["Black", "Navy", "White", "Grey", "Blue", "Beige", "Pattern", "Khaki", "Olive", "Maroon", "Brown", "Green", "Red"],
    "Maroon": ["Beige", "White", "Black", "Grey", "Navy", "Khaki", "Olive", "Blue", "Pattern", "Green", "Brown", "Red", "Maroon"],
    "Blue": ["White", "Grey", "Beige", "Khaki", "Black", "Brown", "Navy", "Olive", "Red", "Maroon", "Green", "Pattern", "Blue"],
    "Brown": ["White", "Beige", "Khaki", "Navy", "Blue", "Olive", "Grey", "Black", "Green", "Maroon", "Red", "Pattern", "Brown"],
    "Grey": ["Black", "White", "Navy", "Blue", "Maroon", "Red", "Beige", "Khaki", "Brown", "Olive", "Green", "Pattern", "Grey"],
    "Green": ["White", "Black", "Beige", "Khaki", "Navy", "Brown", "Grey", "Blue", "Maroon", "Red", "Pattern", "Olive", "Green"],
    "Olive": ["White", "Black", "Beige", "Khaki", "Brown", "Navy", "Grey", "Maroon", "Blue", "Red", "Pattern", "Green", "Olive"],
    "Beige": ["Navy", "Black", "White", "Brown", "Blue", "Maroon", "Olive", "Grey", "Red", "Green", "Khaki", "Pattern", "Beige"],
    "Khaki": ["Navy", "Black", "White", "Brown", "Blue", "Maroon", "Olive", "Grey", "Red", "Green", "Beige", "Pattern", "Khaki"],
    "Pattern": ["White", "Black", "Navy", "Blue", "Brown", "Green", "Olive", "Red", "Maroon", "Grey", "Beige", "Khaki", "Pattern"],
}

DEFAULT_COLOR_RANKS = ["White", "Black", "Grey", "Navy", "Brown", "Beige", "Olive", "Khaki", "Red", "Blue", "Green", "Maroon", "Pattern"]


COLOR_ALIASES = {
    "Navy Blue": "Navy",
    "off White": "White",
    "Coffee Brown": "Brown",
    "Charcoal" : "Grey",
    "Steel": "grey",
    "Silver": "grey",
    "cream": "Beige",
    "Tan": "Khaki",
    "Burgundy": "Maroon",
    "Multi" : " Pattern"
}



class FeatureExtractor:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backbone = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1)
        
        self.model = nn.Sequential(*list(backbone.children())[:-1]).to(self.device)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean = [0.485, 0.456, 0.406],
                    std  = [0.229, 0.224, 0.225],

                ),
            ]
        )
    def extract(self, image: Image.Image) -> np.ndarray:
        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(tensor)
        return features.squeeze().cpu().numpy()


class FashionRecommendationService:
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        styles_csv_path: Path = STYLES_CSV_PATH,
        image_root: Path = IMAGE_ROOT,
    ) -> None:
        model_data = joblib.load(model_path)
        self.features = np.asarray(model_data["features"])
        self.metadata = self._load_metadata(model_data["metadata"], styles_csv_path, image_root)
        self.extractor = FeatureExtractor()


    def _load_metadata(self, model_metadata: pd.DataFrame, styles_csv_path: Path, image_root: Path) ->pd.DataFrame:
        metadata = model_metadata.copy()
        metadata["item_id"] = metadata.get("image_id", metadata.get("id", metadata.index)).astype(str)
        metadata["id_int"] = pd.to_numeric(metadata["item_id"], errors = "coerce").astype("Int64")


        if styles_csv_path.exists():
            styles = pd.read_csv(styles_csv_path, on_bad_lines="skip", encoding="ISO-8859-1")
            styles.colums = styles.columns.str.strip()
            metadata = metadata.merge(styles, how="left", left_on="id_int", right_on="id", suffixes =("", "_styles"))
        
        if "image_path" not in metadata.columns:
            metadata["image_path"] = metadata["item_id"].apply(lambda item_id: str(image_root/ f"{item_id}.jpg"))
        else:
            metadata["image_path"] = metadata["image_path"].astyple(str).apply(
                lambda path: str str(Path(path)) if Path(path).exists() else str(image_root / Path(path).name)
            )

        metadata["fashion_color"] = metadata.get("baseColour", pd.Series(index= metadata.index, dtype = object)).apply(normalize_color)
        return metadata.reset_index(drop = True)

    def options(self) -> dict:
        fields ={
            "gender": "gender",
            "subcategory" : "subvategory",
            "artile_type" : "articleType",
            "color" : "fashion_color",
            "usage" : "usage",


        }

        return {
            key:sorted(self.metadata[col].dropna().astype(str).unique().tolist())
            for key, col in fields.items()
            if col in self.metadata.columns
        }

    def list_items(self, profile: Any | None = None, limit: int = 40, **filters: str | None) -> list[dict]:
        pool = self._filtered_pool(profile, **filters)
        return [self._serialize_row(row,0.0) for _ , row in pool.head(limit).iterrows()]


    def recommend(
        self,
        mode: str,
        item_id: str,
        profile:  Any,
        limit: int = 8,
        article_type: str | None = None,
        subcategory : str | None = None,
        limit_per_category: int = 4,
        include_categories : list[str] | None = None,
    ) -> dict:
        if mode == "similar":
            return{
                "mode": "similar",
                "item_id": "item_id",
                "results" : self.similar_by_id(
                    item_id = item_id,
                    profile = profile,
                    limit=limit,
                    article_type=article_type,
                    subcategory=subcategory,
                ),
            }

        if mode == "outfit":
            outfit = self.outfit_by_id(
                item_id=item_id,
                profile=profile,
                limit_per_category=limit_per_category,
                include_categories=include_categories,
            )
            return {"model": "outfit", **outfit}

        raise ValueError("mode must be 'similar' or 'outfit'")
    
    def similar_by_id(
    self,
    item_id: str,
    profile: Any,
    limit: int = 8,
    article_type: str | None = None,
    subcategory: str | None = None,

) -> list[dict]:

     seed_idx = self._index_for_item(item_id)
     seed = self.metadata.iloc[seed_idx]

     pool = self._filtered_pool(
        profile,
        article_type = article_type or seed.get("articleType"),
        subcategory=subcategory or seed.get("subCategory"),
    )

     pool = pool[pool.index != seed_idx]

     return self._rank_by_visual_similarity(
        self.features[seed_idx],
        pool,
        limit
    )

    


    





    


