from typing import Literal, Optional
from pydantic import BaseModel, Field

class UserProfile(BaseModel):
    gender : Optional[str] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    skin_color: Optional[str] = None
    clothing_style: Optional[str] = Field(
        default=None,
        description = "Casual, Formal, Sport, Party, etc. Mapped to dataset usage when possible."

    )
    size: Optional[str] = None


class SimilarRequest(BaseModel):
    item_id: str
    profile: UserProfile = Field(default_factory=UserProfile)
    article_type: Optional[str] = None
    subcategory: Optional[str] = None
    limit: int =  Field(default = 8, ge=1, le=50)


class OutfitRequest(BaseModel):
    item_id: str
    profile: UserProfile = Field(default_factory = UserProfile)
    limit_per_category: int = Field(default = 4, ge=1,le=20)
    include_categories : Optional[list[str]] = None

class RecommendationRequest(BaseModel):
    mode: Literal["similar", "outfit"] = Field(
        description = "Use similar for similar clothes, or 'outfits' to create a matching outfit" 
    )

    item_id : str
    profile: UserProfile = Field(default_factory=UserProfile)
    article_type: Optional[str] = None 
    subcategory: Optional[str] = None
    limit: int = Field(default=8, ge=1, le=50)
    limit_per_category: int = Field(default=4, ge=1, le=20)
    include_categories: Optional[list[str]] = None


class RecommendationItem(BaseModel):
    item_id : str
    image_url : str
    image_path : str
    score : float
    product_name : Optional[str] = None
    gender : Optional[str] = None
    master_category : Optional[str] = None
    subcategory : Optional[str] = None
    article_type: Optional[str] = None
    color: Optional[str] = None
    usage: Optional[str] = None

    







