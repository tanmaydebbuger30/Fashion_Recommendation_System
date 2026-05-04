import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Fashion Recommendation System", layout="wide")
st.set_option("client.showErrorDetails", True)

st.markdown("""
<style>
    .stButton>button {
        background: linear-gradient(90deg, #FF69B4, #FF1493);
        color: white;
        font-weight: bold;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# PATH CONFIG
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # notebooks/
PROJECT_ROOT = os.path.dirname(BASE_DIR)                # Trail/

EXTRACTED_FOLDER = os.path.join(PROJECT_ROOT, "img", "archive")
IMAGES_FOLDER = os.path.join(EXTRACTED_FOLDER, "images")

MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PKL_PATH = os.path.join(MODELS_DIR, "fashion_recommender.pkl")


# ----------------------------
# DATASET SETUP
# ----------------------------
@st.cache_resource
def setup_dataset():
    if not os.path.exists(EXTRACTED_FOLDER):
        st.error(f"Dataset folder not found: {EXTRACTED_FOLDER}")
        st.stop()

    if not os.path.exists(IMAGES_FOLDER):
        st.error(f"Images folder not found: {IMAGES_FOLDER}")
        st.stop()

    image_count = len([
        f for f in os.listdir(IMAGES_FOLDER)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    st.sidebar.success(f"✅ Dataset ready! {image_count:,} images found")
    return EXTRACTED_FOLDER


# ----------------------------
# FEATURE EXTRACTOR
# ----------------------------
class FeatureExtractor:
    def __init__(self, model_name="resnet18"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_name == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        self.model = nn.Sequential(*list(backbone.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract_features(self, image):
        if isinstance(image, str):
            image = Image.open(image)

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if image.mode != "RGB":
            image = image.convert("RGB")

        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(image_tensor)

        return features.squeeze().cpu().numpy()


@st.cache_resource
def load_extractor():
    return FeatureExtractor("resnet18")


# ----------------------------
# RECOMMENDER
# ----------------------------
class FashionRecommender:
    def __init__(self, features, metadata_df):
        if len(features) == 0:
            raise ValueError("No features available")

        self.features = features
        self.metadata = metadata_df.reset_index(drop=True)

        self.knn_model = NearestNeighbors(
            n_neighbors=min(20, len(features)),
            metric="cosine",
            algorithm="brute"
        )
        self.knn_model.fit(features)

    def get_recommendations(self, item_index, n_recommendations=6):
        query_features = self.features[item_index].reshape(1, -1)

        distances, indices = self.knn_model.kneighbors(
            query_features,
            n_neighbors=min(n_recommendations + 1, len(self.features))
        )

        indices = indices[0][1:]
        distances = distances[0][1:]

        recommendations = self.metadata.iloc[indices].copy()
        recommendations["similarity_score"] = 1 - distances
        return recommendations

    def find_similar_to_uploaded(self, uploaded_features, n_recommendations=6):
        uploaded_features = uploaded_features.reshape(1, -1)

        distances, indices = self.knn_model.kneighbors(
            uploaded_features,
            n_neighbors=min(n_recommendations, len(self.features))
        )

        indices = indices[0]
        distances = distances[0]

        recommendations = self.metadata.iloc[indices].copy()
        recommendations["similarity_score"] = 1 - distances
        return recommendations


# ----------------------------
# HELPERS
# ----------------------------
def safe_open_image(path):
    try:
        if os.path.exists(path):
            return Image.open(path).convert("RGB")
        return None
    except Exception:
        return None


# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PKL_PATH):
        st.error(f"Model not found: {MODEL_PKL_PATH}")
        st.stop()

    try:
        with open(MODEL_PKL_PATH, "rb") as f:
            model_data = joblib.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    features = model_data["features"]
    metadata = model_data["metadata"]

    if "image_path" in metadata.columns:
        metadata["image_path"] = metadata["image_path"].astype(str).apply(
            lambda x: os.path.join(IMAGES_FOLDER, os.path.basename(x))
        )
    elif "filename" in metadata.columns:
        metadata["image_path"] = metadata["filename"].astype(str).apply(
            lambda x: os.path.join(IMAGES_FOLDER, x)
        )
    else:
        st.error("Metadata must contain either 'image_path' or 'filename'.")
        st.stop()

    st.sidebar.success("✅ Model Loaded!")
    st.sidebar.info(f"Items: {len(metadata):,}")

    recommender = FashionRecommender(features, metadata)
    extractor = load_extractor()

    return recommender, extractor, metadata


# ----------------------------
# UI: BROWSE MODE
# ----------------------------
def browse_catalog_mode(recommender, metadata):
    st.header("Browse Fashion Catalog")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.write(f"**Total Items:** {len(metadata):,}")
    with col2:
        n_recommendations = st.slider("Recommendations", 3, 9, 6)

    st.subheader("Select an Item")

    if "sample_items" not in st.session_state or st.button("Shuffle"):
        st.session_state.sample_items = metadata.sample(min(10, len(metadata)))

    sample_items = st.session_state.sample_items
    cols = st.columns(5)
    selected_idx = None

    for idx, (_, item) in enumerate(sample_items.iterrows()):
        with cols[idx % 5]:
            img = safe_open_image(item["image_path"])
            if img is not None:
                st.image(img, use_container_width=True)
            else:
                st.write("Image not found")

            if "filename" in item:
                st.caption(str(item["filename"])[:25])

            if st.button("Select", key=f"btn_{idx}"):
                selected_idx = item.name

    if selected_idx is not None:
        st.markdown("---")
        st.subheader("Recommendations")

        query_item = metadata.iloc[selected_idx]

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### Query Item")
            query_img = safe_open_image(query_item["image_path"])
            if query_img is not None:
                st.image(query_img, use_container_width=True)
                if "filename" in query_item:
                    st.caption(str(query_item["filename"])[:30])
            else:
                st.error("Error loading query image")

        with col2:
            recommendations = recommender.get_recommendations(selected_idx, n_recommendations)

            if len(recommendations) > 0:
                cols = st.columns(3)
                for idx, (_, rec) in enumerate(recommendations.iterrows()):
                    with cols[idx % 3]:
                        rec_img = safe_open_image(rec["image_path"])
                        if rec_img is not None:
                            st.image(rec_img, use_container_width=True)
                        else:
                            st.write("Image not found")

                        st.write(f"**{rec['similarity_score']:.1%}**")
                        if "filename" in rec:
                            st.caption(str(rec["filename"])[:20])


# ----------------------------
# UI: UPLOAD MODE
# ----------------------------
def upload_image_mode(recommender, extractor, metadata):
    st.header("Upload Your Image")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Could not open uploaded image: {e}")
            return

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Your Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Similar Items")
            n_recommendations = st.slider("Results", 3, 12, 6)

            with st.spinner("Analyzing..."):
                features = extractor.extract_features(image)

            recommendations = recommender.find_similar_to_uploaded(features, n_recommendations)

            cols = st.columns(3)
            for idx, (_, rec) in enumerate(recommendations.iterrows()):
                with cols[idx % 3]:
                    rec_img = safe_open_image(rec["image_path"])
                    if rec_img is not None:
                        st.image(rec_img, use_container_width=True)
                    else:
                        st.write("Image not found")

                    st.write(f"**{rec['similarity_score']:.1%}**")
                    if "filename" in rec:
                        st.caption(str(rec["filename"])[:25])


# ----------------------------
# UI: ANALYTICS
# ----------------------------
def analytics_dashboard(metadata):
    st.header("Dataset Analytics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Items", f"{len(metadata):,}")
    with col2:
        if "image_id" in metadata.columns:
            st.metric("Unique Images", f"{metadata['image_id'].nunique():,}")
        elif "filename" in metadata.columns:
            st.metric("Unique Images", f"{metadata['filename'].nunique():,}")
        else:
            st.metric("Unique Images", f"{len(metadata):,}")
    with col3:
        st.metric("Features", "512D")

    st.markdown("---")
    st.subheader("Sample Images")

    cols = st.columns(6)
    sample_items = metadata.sample(min(12, len(metadata)))

    for idx, (_, item) in enumerate(sample_items.iterrows()):
        with cols[idx % 6]:
            img = safe_open_image(item["image_path"])
            if img is not None:
                st.image(img, use_container_width=True)

    st.markdown("---")
    st.subheader("Dataset Info")
    st.dataframe(metadata.head(20), use_container_width=True)


# ----------------------------
# MAIN
# ----------------------------
def main():
    st.markdown(
        '<h1 style="text-align: center;">Fashion Recommendation System</h1>',
        unsafe_allow_html=True
    )

    setup_dataset()
    recommender, extractor, metadata = load_model()

    st.sidebar.title("Navigation")
    mode = st.sidebar.radio(
        "Select Mode",
        ["Browse Catalog", "Upload Image", "Analytics"],
        label_visibility="collapsed"
    )

    if mode == "Browse Catalog":
        browse_catalog_mode(recommender, metadata)
    elif mode == "Upload Image":
        upload_image_mode(recommender, extractor, metadata)
    else:
        analytics_dashboard(metadata)


if __name__ == "__main__":
    main()