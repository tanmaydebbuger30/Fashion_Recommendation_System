from io import BytesIO


from fastapi import Depends, FastAPI, File, Form, HTPPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from .recommender import FashionRecommendationService, get_service
# from .schemas import OutfitRequest, RecommendationRequest, SimilarRequest, UserProfile



# Creating fastAPI application
app = FastAPI(
    title="Fashion Recommendation API",
    version = "0.1.0",
    description = "Backend API for similar-clothing search and color-aware outfit creations."
)

# Adding CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)



@app.get("/", tags = ["system"])
def root()->dict:
    return{
        "messsage": "Welcome to the Fashion Recommendation System API!",
        "docs" : "/docs",
        "health" : "/health",
        "main_recommendation_endpoint" : "/recommend",
    }
    # Health check endpoint
    # Dependency Injection. Service is injected into the function.
@app.get("/health", tags = ["system"])
def health(service: FashionRecommendationService = Depends(get_service))->dict:
    return {"status": "ok", "items": len(service.metadata)}

# Main recommendation endpoint
# Dependency Injection. Service is injected into the function.


@app.get("/items", tags=["catalog"])
def items(
    gender: str | None = None,
    clothing_style: str | None = Query(default = None, description = "Casual, Formal, Sports, etc."),
    article_type: str | None = None,
    subcategory: str | None = None,
    limit: int = Query(default = 40, ge=1, le=200),
    services : FashionRecommendationService = Depends(get_service),

) -> list[dict]:
 profile = UserProfile(gender = gender, clothing_style= clothing_style)
 return services.list_item(profile, limit = limit, article_type = article_type, subcategory = subcategory)








   