from __future__ import annotations

import math
from functools import lru_cache
from os import name
from pathlib import Path
from typing import Any

import joblib
from joblib import pool
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models, transforms

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
    "Charcoal": "Grey",
    "Steel": "Grey",
    "Silver": "Grey",
    "cream": "Beige",
    "Tan": "Khaki",
    "Burgundy": "Maroon",
    "Multi": "Pattern",
}


class FeatureExtractor:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        self.model = nn.Sequential(*list(backbone.children())[:-1]).to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
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

    def _load_metadata(
        self,
        model_metadata: pd.DataFrame,
        styles_csv_path: Path,
        image_root: Path,
    ) -> pd.DataFrame:
        metadata = model_metadata.copy()

        metadata["item_id"] = metadata.get(
            "image_id",
            metadata.get("id", metadata.index),
        ).astype(str)

        metadata["id_int"] = pd.to_numeric(
            metadata["item_id"],
            errors="coerce",
        ).astype("Int64")

        if styles_csv_path.exists():
            styles = pd.read_csv(
                styles_csv_path,
                on_bad_lines="skip",
                encoding="ISO-8859-1",
            )
            styles.columns = styles.columns.str.strip()

            metadata = metadata.merge(
                styles,
                how="left",
                left_on="id_int",
                right_on="id",
                suffixes=("", "_styles"),
            )

        if "image_path" not in metadata.columns:
            metadata["image_path"] = metadata["item_id"].apply(
                lambda item_id: str(image_root / f"{item_id}.jpg")
            )
        else:
            metadata["image_path"] = metadata["image_path"].astype(str).apply(
                lambda path: str(Path(path))
                if Path(path).exists()
                else str(image_root / Path(path).name)
            )

        metadata["fashion_color"] = metadata.get(
            "baseColour",
            pd.Series(index=metadata.index, dtype=object),
        ).apply(normalize_color)

        return metadata.reset_index(drop=True)

    def options(self) -> dict:
        fields = {
            "gender": "gender",
            "subcategory": "subCategory",
            "article_type": "articleType",
            "color": "fashion_color",
            "usage": "usage",
        }

        return {
            key: sorted(self.metadata[col].dropna().astype(str).unique().tolist())
            for key, col in fields.items()
            if col in self.metadata.columns
        }

    def list_items(
        self,
        profile: Any | None = None,
        limit: int = 40,
        **filters: str | None,
    ) -> list[dict]:
        pool = self._filtered_pool(profile, **filters)
        return [self._serialize_row(row, 0.0) for _, row in pool.head(limit).iterrows()]

    def recommend(
        self,
        mode: str,
        item_id: str,
        profile: Any,
        limit: int = 8,
        article_type: str | None = None,
        subcategory: str | None = None,
        limit_per_category: int = 4,
        include_categories: list[str] | None = None,
    ) -> dict:
        if mode == "similar":
            return {
                "mode": "similar",
                "item_id": item_id,
                "results": self.similar_by_id(
                    item_id=item_id,
                    profile=profile,
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
            return {"mode": "outfit", **outfit}

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
            article_type=article_type or seed.get("articleType"),
            subcategory=subcategory or seed.get("subCategory"),
        )

        pool = pool[pool.index != seed_idx]

        return self._rank_by_visual_similarity(
            self.features[seed_idx],
            pool,
            limit,
        )

    def similar_by_image(
        self,
        image: Image.Image,
        profile: Any,
        limit: int = 8,
        article_type: str | None = None,
        subcategory: str | None = None,
    ) -> list[dict]:
        query_features = self.extractor.extract(image)
        pool = self._filtered_pool(
            profile,
            article_type=article_type,
            subcategory=subcategory,
        )
        return self._rank_by_visual_similarity(query_features, pool, limit)

    def outfit_by_image(
        self,
        image: Image.Image,
        profile: Any,
        seed_subcategory: str = "Topwear",
        limit_per_category: int = 4,
    ) -> dict:
        query_features = self.extractor.extract(image)
        categories = upload_outfit_categories_for(seed_subcategory)
        outfit = {
            "mode": "upload_outfit",
            "seed": {
                "item_id": "uploaded-image",
                "product_name": "Uploaded item",
                "subcategory": seed_subcategory,
                "score": 1.0,
            },
            "matches": {},
        }

        for category in categories:
            pool = self._filtered_pool(profile, subcategory=category)
            outfit["matches"][category] = self._rank_by_visual_similarity(
                query_features,
                pool,
                limit_per_category,
            )

        return outfit

    def outfit_by_id(
        self,
        item_id: str,
        profile: Any,
        limit_per_category: int = 4,
        include_categories: list[str] | None = None,
    ) -> dict:
        seed_idx = self._index_for_item(item_id)
        seed = self.metadata.iloc[seed_idx]
        seed_features = self.features[seed_idx]

        categories = include_categories or outfit_categories_for(
            seed.get("subCategory"),
            seed.get("articleType"),
        )

        outfit = {"seed": self._serialize_row(seed, 1.0), "matches": {}}

        for category in categories:
            pool = self._filtered_pool(profile, subcategory=category)
            pool = pool[pool.index != seed_idx]
            ranked = self._rank_outfit_pool(seed, seed_features, pool, limit_per_category)
            outfit["matches"][category] = ranked

        return outfit

    def image_path_for(self, item_id: str) -> Path:
        idx = self._index_for_item(item_id)
        return Path(str(self.metadata.iloc[idx]["image_path"]))

    def _index_for_item(self, item_id: str) -> int:
        matches = self.metadata.index[
            self.metadata["item_id"].astype(str) == str(item_id)
        ].tolist()

        if not matches:
            raise KeyError(f"Unknown item_id: {item_id}")

        return matches[0]

    def _filtered_pool(
        self,
        profile: Any,
        article_type: str | None = None,
        subcategory: str | None = None,
        color: str | None = None,
        usage: str | None = None,
    ) -> pd.DataFrame:
        pool = self.metadata.copy()

        profile_gender = getattr(profile, "gender", None)
        profile_style = getattr(profile, "clothing_style", None)

        if profile_gender and "gender" in pool:
            gender = profile_gender.lower()
            pool = pool[
                pool["gender"].astype(str).str.lower().isin([gender, "unisex"])
            ]

        requested_usage = usage or profile_style

        if requested_usage and "usage" in pool:
            pool = pool[
                pool["usage"].astype(str).str.lower() == requested_usage.lower()
            ]

        if article_type and "articleType" in pool:
            pool = pool[
                pool["articleType"].astype(str).str.lower() == article_type.lower()
            ]

        if subcategory and "subCategory" in pool:
            pool = pool[
                pool["subCategory"].astype(str).str.lower() == subcategory.lower()
            ]

        if color:
            normalized = normalize_color(color)
            pool = pool[
                pool["fashion_color"].astype(str).str.lower() == normalized.lower()
            ]

        return pool

    def _rank_by_visual_similarity(
        self,
        query_features: np.ndarray,
        pool: pd.DataFrame,
        limit: int,
    ) -> list[dict]:
        if pool.empty:
            return []

        indices = pool.index.to_numpy()
        scores = cosine_similarity([query_features], self.features[indices])[0]
        order = np.argsort(scores)[::-1][:limit]

        return [
            self._serialize_row(self.metadata.iloc[int(indices[i])], float(scores[i]))
            for i in order
        ]

    def _rank_outfit_pool(
        self,
        seed: pd.Series,
        seed_features: np.ndarray,
        pool: pd.DataFrame,
        limit: int,
    ) -> list[dict]:
        if pool.empty:
            return []

        indices = pool.index.to_numpy()

        style_scores = cosine_similarity(
            [seed_features],
            self.features[indices],
        )[0]

        color_scores = pool["fashion_color"].apply(
            lambda color: color_score(seed.get("fashion_color"), color)
        ).to_numpy()

        final_scores = (color_scores * 0.60) + (style_scores * 0.40)
        order = np.argsort(final_scores)[::-1][:limit]

        return [
            self._serialize_row(self.metadata.iloc[int(indices[i])], float(final_scores[i]))
            for i in order
        ]

    def _serialize_row(self, row: pd.Series, score: float) -> dict:
        item_id = str(row.get("item_id"))

        return {
            "item_id": item_id,
            "image_url": f"/images/{item_id}",
            "image_path": str(row.get("image_path", "")),
            "score": clean_float(score),
            "product_name": clean_value(row.get("productDisplayName")),
            "gender": clean_value(row.get("gender")),
            "master_category": clean_value(row.get("masterCategory")),
            "subcategory": clean_value(row.get("subCategory")),
            "article_type": clean_value(row.get("articleType")),
            "color": clean_value(row.get("fashion_color")),
            "usage": clean_value(row.get("usage")),
        }


def normalize_color(color: object) -> str:
    if color is None or (isinstance(color, float) and math.isnan(color)):
        return "Pattern"

    label = str(color).strip()
    return COLOR_ALIASES.get(
        label,
        label if label in DEFAULT_COLOR_RANKS else label.split()[0].title(),
    )


def color_score(seed_color: object, candidate_color: object) -> float:
    seed = normalize_color(seed_color)
    candidate = normalize_color(candidate_color)

    ranks = COLOR_COMPATIBILITY.get(seed, DEFAULT_COLOR_RANKS)

    try:
        return 1 - (ranks.index(candidate) / max(len(ranks) - 1, 1))
    except ValueError:
        return 0.45


def outfit_categories_for(subcategory: object, article_type: object) -> list[str]:
    sub = str(subcategory or "").lower()
    article = str(article_type or "").lower()

    topwear = {
        "shirts",
        "formal shirts",
        "tshirts",
        "tops",
        "sweatshirts",
        "jackets",
        "blazers",
    }
    bottomwear = {
        "jeans",
        "trousers",
        "track pants",
        "shorts",
        "formal pants",
        "formal pant",
    }

    if sub == "topwear" or article in topwear:
        return ["Bottomwear"]

    if sub == "bottomwear" or article in bottomwear:
        return ["Topwear"]

    if sub == "shoes":
        return ["Topwear", "Bottomwear"]

    return ["Topwear", "Bottomwear"]


def upload_outfit_categories_for(seed_subcategory: object) -> list[str]:
    seed = str(seed_subcategory or "Topwear").lower()

    if seed == "topwear":
        return ["Bottomwear", "Shoes", "Accessories"]

    if seed == "bottomwear":
        return ["Topwear", "Shoes", "Accessories"]

    if seed == "shoes":
        return ["Topwear", "Bottomwear", "Accessories"]

    return ["Bottomwear", "Shoes", "Accessories"]


def clean_value(value: object) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None

    return str(value)


def clean_float(value: float) -> float:
    return round(float(value), 4)


@lru_cache(maxsize=1)
def get_service() -> FashionRecommendationService:
    return FashionRecommendationService()