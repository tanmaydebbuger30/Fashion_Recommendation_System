import os
from pathlib import Path


# CHANGE THSE SETTEING according to your PC  /  Laptop

PROJECT_ROOT = Path(
    os.getenv(
        "FASHION_PROJECT_ROOT",
        "/Users/tanmay/Desktop/Capstone/code/Trail"
    )
)


MODEL_PATH = Path(
    os.getenv(
        "FASHION_MODEL_PATH",
        str(PROJECT_ROOT/"notebooks" / "models"/ "fashion_recommender.pkl"),

    )
)


STYLES_CSV_PATH = Path(
    os.getenv(
        "FASHION_STYLES_CSV",
        str(PROJECT_ROOT / "img" / "archive" / "styles.csv")
    )
)

IMAGE_ROOT = Path(
    os.getenv(
        "FASHION_IMAGE_ROOT",
        str(PROJECT_ROOT / "img" / "archive" / "myntradataset" / "images" / "subset"),
    )
)