# DripDrop

# Fashion Recommendation System

A full-stack fashion recommendation platform combining a FastAPI backend, a computer-vision recommender engine, a Claude-powered AI stylist, and a lightweight HTML/CSS/JS frontend. Browse a fashion catalog, get similar-item and outfit recommendations, upload your own photo for style matches, or just ask "what should I wear?" in plain English.

## ✨ Features

- 🛍️ **Catalog Browsing** — Explore products by gender and style
- 🔁 **Similar Product Recommendations** — Find visually similar items
- 👔 **Outfit Recommendations** — Get complementary pieces (top → bottom, etc.)
- 📸 **DripDrop Image Upload** — Upload a photo and get similar or complementary recommendations
- 🤖 **AI Stylist Chat** — Ask natural language questions like *"I have a date night, what should I wear?"* and get grounded recommendations powered by Claude
- 🛒 **Add to Cart (Preview UI)**
- 🔐 **Mock Login/Signup** — Local preview auth (not production-ready)

## 🧱 Tech Stack

| Layer | Tech |
|---|---|
| Backend | FastAPI, Python |
| AI Layer | Claude API |
| ML / Recommender | ResNet feature extraction, scikit-learn similarity search, PyTorch, OpenCV, Pillow |
| Frontend | HTML, CSS, JavaScript |
| Data | Local image dataset + generated feature/metadata files |

## 📁 Project Structure

```
Trail/
├── backend/
│   └── app/
│       ├── main.py
│       ├── recommender.py
│       ├── schemas.py
│       ├── config.py
│       └── build_recommender_model.py
├── Frontend/
│   └── index.html
├── img/
│   └── updated_db/
│       ├── Casual/
│       │   ├── Jeans/{Men,Women}
│       │   └── Tshirts/{Men,Women}
│       └── Formals/
│           ├── Blazers/{Men,Women}
│           ├── Formal_pant/{Men,Women}
│           └── Formal_shirt/{Men,Women}
├── notebooks/
│   └── models/
│       ├── fashion_features.npy
│       ├── fashion_metadata.csv
│       └── fashion_recommender.pkl
└── requirements.txt
```

## 🚀 Getting Started

### Prerequisites

- Python 3.12
- A Claude API key (for the AI Stylist feature)

### 1. Clone & set up environment

```bash
git clone <your-repo-url>
cd Trail

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

If FastAPI-related packages are missing:

```bash
pip install fastapi uvicorn python-multipart pydantic
```

### 2. Configure your API key

Create a `.env` file in the project root (or `backend/`) with your Claude API key:

```bash
ANTHROPIC_API_KEY=your_api_key_here
```

This is required for the `/chat` AI Stylist endpoint to work.

### 3. Build the recommender model

Run this once initially, and again whenever images are added/removed/moved in `img/updated_db`:

```bash
python backend/app/build_recommender_model.py
```

Expected output:

```
Model rebuilt successfully
Images processed: 5798
Features shape: (5798, 512)
Saved to: .../notebooks/models/fashion_recommender.pkl
```

### 4. Start the backend

```bash
cd backend
source ../.venv/bin/activate
python -m uvicorn app.main:app --reload --port 8000
```

- App: `http://127.0.0.1:8000`
- Docs: `http://127.0.0.1:8000/docs`

### 5. Start the frontend

In a **second terminal**:

```bash
cd Frontend
python3 -m http.server 5173
```

- App: `http://127.0.0.1:5173`

> ⚠️ Keep the backend terminal running while using the frontend.

## 🖼️ Image Database Structure

Images are organized as:

```
img/updated_db/<usage>/<articleType>/<gender>/<image_file>
```

Example: `img/updated_db/Casual/Jeans/Men/example.jpg` maps to:

```json
{
  "usage": "Casual",
  "gender": "Men",
  "articleType": "Jeans",
  "subCategory": "Bottomwear"
}
```

**Category mappings:**

| Folder | Maps to |
|---|---|
| Jeans | Bottomwear / Jeans |
| Tshirts | Topwear / Tshirts |
| Blazers | Topwear / Blazers |
| Formal_pant | Bottomwear / Formal Pants |
| Formal_shirt | Topwear / Formal Shirts |

Supported extensions: `.jpg`, `.jpeg`, `.png`, `.webp`

To fix a misclassified item, move the image to the correct folder and rebuild the model.

## 🤖 AI Stylist (Claude-Powered)

On top of the recommender engine, the app includes a conversational styling assistant powered by the **Claude API**. Instead of clicking through filters, users can just describe what they need — e.g. *"I have a date night, what should I wear?"* — and get a grounded outfit suggestion.

**How it works:**

1. User sends a natural language message to the `/chat` endpoint.
2. The backend calls the **Claude API**, passing along relevant catalog/item metadata (style, gender, category, etc.) as context.
3. Claude reasons over the request (occasion, vibe, weather, etc.) and the available catalog metadata to decide what kind of items fit.
4. The backend then calls the **existing `/recommend` endpoints** using Claude's interpreted intent (e.g. gender, clothing_style, category) to fetch actual matching products from the recommender.
5. The final response combines Claude's natural-language styling advice with real catalog items and images — not just generic text.

**Example flow:**

```
User: "I have a date night, what should I wear?"

→ Claude interprets: occasion = date night, style = smart casual/formal
→ Backend calls /recommend with inferred profile (gender, clothing_style)
→ Response: styling advice + actual recommended products from the catalog
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/items` | List catalog items |
| POST | `/recommend` | Get recommendations |
| POST | `/recommend/similar/upload` | Similar items from uploaded image |
| POST | `/recommend/outfit/upload` | Outfit match from uploaded image |
| POST | `/chat` | Natural language styling assistant powered by Claude, grounded in catalog data |

**Similar recommendation example:**

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

**Outfit recommendation example:**

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

**AI Stylist chat example:**

Request:

```json
{
  "message": "I have a date night, what should I wear?",
  "profile": {
    "gender": "Men"
  }
}
```

Response:

```json
{
  "reply": "For a date night, I'd go smart casual — think a fitted shirt with dark denim or tailored pants. Here are a few pieces that would work:",
  "recommendations": [
    { "item_id": "...", "articleType": "Formal Shirt", "gender": "Men" },
    { "item_id": "...", "articleType": "Jeans", "gender": "Men" }
  ]
}
```

## 🧭 Usage Walkthrough

1. Open the frontend and sign up/log in (preview auth)
2. Go to **Shop** → choose gender and style → **Load Products**
3. Click a product to view **Similar Outfits** / **Recommended Outfits**, or **Add to Cart**
4. Or go to **DripDrop** → choose a style target → upload an image → **Recommend similar kind of it** or **Create for me**
5. Or just ask the **AI Stylist** a question like *"I have a date night, what should I wear?"*

## 🩹 Troubleshooting

<details>
<summary><code>ModuleNotFoundError: No module named 'app'</code></summary>

Run uvicorn from the `backend` directory, not the project root:

```bash
cd backend
python -m uvicorn app.main:app --reload --port 8000
```
</details>

<details>
<summary>Frontend loads but images don't show</summary>

Confirm the backend is running at `http://127.0.0.1:8000` and check for `GET /images/<item_id> 200 OK` in the backend logs.
</details>

<details>
<summary>Recommendations return the wrong gender</summary>

Gender is derived from folder structure (`Men` / `Women`). Move the image to the correct folder and rebuild the model.
</details>

<details>
<summary>Search doesn't find products</summary>

Use terms matching article types: `Tshirt`, `Jeans`, `Blazers`, `Formal Shirts`, `Formal Pants`, `pants`. Also confirm the selected style (`Casual` / `Formals`) matches the folder.
</details>

<details>
<summary>Uploaded outfit doesn't show shoes or accessories</summary>

Those categories aren't in the image database yet. Add the folders, update the mapping in `build_recommender_model.py`, and rebuild.
</details>

<details>
<summary>AI Stylist (<code>/chat</code>) returns an error</summary>

Make sure `ANTHROPIC_API_KEY` is set correctly in your `.env` file and that the backend has restarted after adding it.
</details>

## 🗺️ Roadmap

- [ ] MongoDB-backed users and profiles
- [ ] Password hashing
- [ ] Real Google OAuth
- [ ] Persistent cart and saved outfits
- [ ] Product sizes and inventory
- [ ] Shoes and accessories categories
- [ ] Expand AI Stylist context (weather, wardrobe history, personal preferences)

## 📌 Current Status

| Component | Status |
|---|---|
| Backend (FastAPI) | ✅ Working |
| Frontend UI | ✅ Full preview working |
| Recommender model | ✅ ResNet features + similarity search |
| Outfit logic | ✅ Category matching + visual similarity |
| AI Stylist (Claude) | ✅ Working |
| Auth | ⚠️ localStorage preview only |
| Google login | ⚠️ Mocked |
| MongoDB | ❌ Not connected |
