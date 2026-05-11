from io import BytesIO


from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

from .recommender import FashionRecommendationService, get_service
from .schemas import OutfitRequest, RecommendationRequest, SimilarRequest, UserProfile



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
    return services.list_items(profile, limit = limit, article_type = article_type, subcategory = subcategory)

@app.post("/recommend", tags = ["recommendations"])
def recommend(
    request: RecommendationRequest,
    service: FashionRecommendationService = Depends(get_service),

) -> dict:
    try:
        return service.recommend(
            mode = request.mode,
            item_id = request.item_id,
            profile = request.profile,
            limit = request.limit,
            article_type = request.article_type,
            subcategory = request.subcategory,
            limit_per_category = request.limit_per_category,
            include_categories = request.include_categories,

        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from exc


@app.post("/recommend/similar", tags = ["recommendations"])
def recommend_similar(
    request: SimilarRequest,
    service : FashionRecommendationService = Depends(get_service),

) -> list[dict]:
    try:
        return service.similar_by_id(
            item_id = request.item_id,
            profile = request.profile,
            limit = request.limit,
            article_type = request.article_type,
            subcategory = request.subcategory,
        )
    except KeyError as exc:
        raise HTTPException(status_code = 404, detail = str(exc)) from exc


@app.post("/recommend/outfit", tags = ["recommendations"])
def recommed_outfit(
    request: OutfitRequest,
    service : FashionRecommendationService = Depends(get_service),
) -> list[dict]:
    try:
        return service.outfit_by_id(
            item_id = request.item_id,
            profile = request.profile,
            limit_per_category = request.limit_per_category,
            include_categories =  request.include_categories,
        )
    except KeyError as exc:
        raise HTTPException(status_code = 404, detail = str(exc)) from exc


@app.post("/recommend/similar/upload", tags=["recommendations"])
async def recommend_from_upload(
    file: UploadFile = File(...),
    gender : str | None = Form(default=None),
    clothing_style : str | None = Form(default=None),
    article_type: str | None = Form(default= None),
    subcategory: str | None = Form(default = None), 
    limit : int = Form(default = 8),
    service: FashionRecommendationService = Depends(get_service),

) -> list[dict]:
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code = 400, detail = "Could not read upload image") from exc

    profile = UserProfile(gender = gender, clothing_style = clothing_style)
    return service.similar_by_image(
        image = image,
        profile = profile,
        limit = limit,
        article_type  = article_type,
        subcategory = subcategory
    )


@app.post("/recommend/outfit/upload", tags=["recommendations"])
async def recommend_outfit_from_upload(
    file: UploadFile = File(...),
    gender: str | None = Form(default=None),
    clothing_style: str | None = Form(default=None),
    seed_subcategory: str = Form(default="Topwear"),
    limit_per_category: int = Form(default=4),
    service: FashionRecommendationService = Depends(get_service),
) -> dict:
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not read upload image") from exc

    profile = UserProfile(gender=gender, clothing_style=clothing_style)
    return service.outfit_by_image(
        image=image,
        profile=profile,
        seed_subcategory=seed_subcategory,
        limit_per_category=limit_per_category,
    )


@app.get("/images/{item_id}", tags=["images"])
def image(item_id: str, service: FashionRecommendationService = Depends(get_service))-> FileResponse:
    try:
        path = service.image_path_for(item_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail =str(exc)) from exc
    if not path.exists():
        raise HTTPException(status_code = 404, detail= f"Image file not found for item_id = {item_id}")
    return FileResponse(path)









   