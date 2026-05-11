# Fashion Recommendation System - Run Guide

This guide explains how to run the project from a fresh clone or a clean local setup.

## 1. Project Overview

This project has two main parts:

- **Backend:** FastAPI + Python recommender system
- **Frontend:** HTML/CSS/JavaScript UI served locally

The recommender supports:

- Catalog browsing
- Similar product recommendations
- Outfit recommendations
- Image upload recommendations through DripDrop
- Add to cart UI preview
- Mock login/signup UI

Current frontend auth is a preview using browser `localStorage`. MongoDB and real Google login are planned backend integrations, not required to run the current app.

## 2. Folder Structure

Expected project structure:

```text
Trail/
  backend/
    app/
      __init__.py
      main.py
      recommender.py
      schemas.py
      config.py
      build_recommender_model.py

  Frontend/
    index.html

  img/
    updated_db/
      Casual/
        Jeans/
          Men/
          Women/
        Tshirts/
          Men/
          Women/
      Formals/
        Blazers/
          Men/
          Women/
        Formal_pant/
          Men/
          Women/
        Formal_shirt/
          Men/
          Women/

  notebooks/
    models/
      fashion_features.npy
      fashion_metadata.csv
      fashion_recommender.pkl

  requirements.txt
```

## 3. Requirements

Recommended Python version:

```text
Python 3.12
```

Install dependencies from:

```text
requirements.txt
```

Main packages used:

```text
fastapi
uvicorn
python-multipart
numpy
pandas
scikit-learn
Pillow
torch
torchvision
joblib
opencv-python
```

## 4. First-Time Setup

From the project root:

```bash
cd /Users/tanmay/Desktop/Capstone/code/Trail
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If FastAPI packages are missing, install them directly:

```bash
pip install fastapi uvicorn python-multipart pydantic
```

## 5. Build or Rebuild the Recommender Model

Run this whenever images are added, removed, or moved inside `img/updated_db`.

```bash
cd /Users/tanmay/Desktop/Capstone/code/Trail
source .venv/bin/activate
python backend/app/build_recommender_model.py
```

Expected success output looks like:

```text
Model rebuilt successfully
Images processed: 5798
Features shape: (5798, 512)
Saved to: /Users/tanmay/Desktop/Capstone/code/Trail/notebooks/models/fashion_recommender.pkl
```

This creates or updates:

```text
notebooks/models/fashion_features.npy
notebooks/models/fashion_metadata.csv
notebooks/models/fashion_recommender.pkl
```

Important: rebuilding overwrites the previous model files.

## 6. Image Database Rules

The model builder reads images from this folder:

```text
img/updated_db
```

Folder meaning:

```text
updated_db / usage / article_type_folder / gender / image_file
```

Example:

```text
img/updated_db/Casual/Jeans/Men/example.jpg
```

becomes:

```json
{
  "usage": "Casual",
  "gender": "Men",
  "articleType": "Jeans",
  "subCategory": "Bottomwear"
}
```

Supported image extensions:

```text
.jpg
.jpeg
.png
.webp
```

Current category mappings:

```text
Jeans         -> Bottomwear / Jeans
Tshirts       -> Topwear / Tshirts
Blazers       -> Topwear / Blazers
Formal_pant   -> Bottomwear / Formal Pants
Formal_shirt  -> Topwear / Formal Shirts
```

If a product appears under the wrong gender, move the image into the correct `Men` or `Women` folder, then rebuild the model.

## 7. Start the Backend

Open a terminal and run:

```bash
cd /Users/tanmay/Desktop/Capstone/code/Trail/backend
source ../.venv/bin/activate
python -m uvicorn app.main:app --reload --port 8000
```

Backend URL:

```text
http://127.0.0.1:8000
```

API docs:

```text
http://127.0.0.1:8000/docs
```

Quick backend checks:

```text
GET /health
GET /items
POST /recommend
POST /recommend/similar/upload
POST /recommend/outfit/upload
```

A healthy backend log includes lines like:

```text
Application startup complete.
GET /items ... 200 OK
POST /recommend ... 200 OK
GET /images/... 200 OK
```

## 8. Start the Frontend

Open a second terminal and run:

```bash
cd /Users/tanmay/Desktop/Capstone/code/Trail/Frontend
python3 -m http.server 5173
```

Frontend URL:

```text
http://127.0.0.1:5173
```

Do not close the backend terminal while using the frontend.

## 9. How to Use the App

1. Open the frontend URL.
2. Sign up or log in using the preview form.
3. Go to `Shop`.
4. Choose gender and style.
5. Click `Load Products`.
6. Click a product card.
7. Use:

```text
Recommended Outfits
Similar Outfits
Add to cart
```

For upload flow:

1. Go to `DripDrop`.
2. Choose style target.
3. Upload an image.
4. Click:

```text
Recommend similar kind of it
```

or:

```text
Create for me
```

For uploaded topwear, `Create for me` recommends complementary bottomwear, and shoes/accessories if those categories exist in the database.

## 10. Common Issues

### Backend says `ModuleNotFoundError: No module named app`

You are probably running Uvicorn from the wrong folder.

Run it from `backend`:

```bash
cd /Users/tanmay/Desktop/Capstone/code/Trail/backend
python -m uvicorn app.main:app --reload --port 8000
```

### Frontend loads but images do not show

Make sure backend is running on:

```text
http://127.0.0.1:8000
```

Image requests should show in the backend terminal:

```text
GET /images/<item_id> 200 OK
```

### Recommendations return wrong gender

Gender comes from folder structure. Check that images are in the correct folder:

```text
Men
Women
```

Then rebuild the model.

### Search does not find products

Use terms that map to article types:

```text
Tshirt
Jeans
Blazers
Formal Shirts
Formal Pants
pants
```

Also make sure the selected style matches your folder:

```text
Casual
Formals
```

### Upload outfit does not show shoes or accessories

That means shoes/accessories are not currently in the image database. Add folders and images, update the category mapping in `build_recommender_model.py`, then rebuild the model.

## 11. API Examples

Similar recommendation:

```json
{
  "mode": "similar",
  "item_id": "YOUR_ITEM_ID",
  "profile": {
    "gender": "Men",
    "clothing_style": "Casual"
  },
  "limit": 8
}
```

Outfit recommendation:

```json
{
  "mode": "outfit",
  "item_id": "YOUR_ITEM_ID",
  "profile": {
    "gender": "Women",
    "clothing_style": "Formals"
  },
  "limit_per_category": 4
}
```

## 12. Development Notes

Current app status:

```text
Backend: FastAPI working
Frontend: full preview UI working
Model: ResNet feature extraction + similarity search
Outfit logic: category matching + visual similarity
Auth: localStorage preview only
Google login: mocked only
MongoDB: not connected yet
```

Future production work:

```text
Add MongoDB users and profiles
Add password hashing
Add real Google OAuth
Persist cart and saved outfits
Add product sizes and inventory
Add shoes/accessories folders and category mappings
```
