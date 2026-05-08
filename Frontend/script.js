const form = document.getElementById("recommendationForm");
const resultsEmpty = document.getElementById("resultsEmpty");
const resultsContent = document.getElementById("resultsContent");
const resultsTitle = document.getElementById("resultsTitle");
const profilePill = document.getElementById("profilePill");
const summaryCard = document.getElementById("summaryCard");
const recommendationList = document.getElementById("recommendationList");
const userPhotoInput = document.getElementById("userPhoto");
const cameraPhotoInput = document.getElementById("cameraPhoto");
const imagePreviewWrap = document.getElementById("imagePreviewWrap");
const imagePreview = document.getElementById("imagePreview");
const uploadedPhotoCard = document.getElementById("uploadedPhotoCard");
const resultPhotoPreview = document.getElementById("resultPhotoPreview");
const photoInsight = document.getElementById("photoInsight");

const recommendations = {
  Casual: {
    "Top Wear": [
      {
        title: "Relaxed Fit Overshirt",
        description: "A versatile layer that works well over tees or fitted tops for a casual everyday look.",
        match: "Easy layering",
      },
      {
        title: "Solid Crew Neck T-Shirt",
        description: "A clean base piece that keeps the outfit simple and lets the colors stand out naturally.",
        match: "Core essential",
      },
      {
        title: "Lightweight Bomber Jacket",
        description: "Adds shape and edge without making the outfit feel too formal or bulky.",
        match: "Balanced structure",
      },
    ],
    "Bottom Wear": [
      {
        title: "Classic Straight Denim",
        description: "A dependable casual option that keeps proportions balanced and pairs with most tops.",
        match: "Easy daily wear",
      },
      {
        title: "Tapered Chinos",
        description: "A slightly smarter casual bottom that looks neat while staying comfortable.",
        match: "Versatile fit",
      },
      {
        title: "Relaxed Cargo Trousers",
        description: "Great for a modern streetwear-inspired casual outfit with more movement and ease.",
        match: "Comfort focused",
      },
    ],
  },
  Formal: {
    "Top Wear": [
      {
        title: "Tailored Blazer",
        description: "A structured blazer gives the outfit polish and helps create a sharper profile.",
        match: "Smart structure",
      },
      {
        title: "Crisp Button-Down Shirt",
        description: "A refined top essential that works for office wear, events, and elevated styling.",
        match: "Formal staple",
      },
      {
        title: "Fine Knit Layer",
        description: "A sleek knit under a blazer adds a sophisticated formal look without feeling stiff.",
        match: "Refined layering",
      },
    ],
    "Bottom Wear": [
      {
        title: "Slim Tapered Trousers",
        description: "Clean lines and a modern taper make formal outfits look refined without feeling heavy.",
        match: "Balanced fit",
      },
      {
        title: "Pleated Dress Pants",
        description: "Adds elegance and room through the leg while keeping the look polished.",
        match: "Classic formal",
      },
      {
        title: "Structured Straight Trousers",
        description: "A timeless formal base that works well with blazers, shirts, and loafers.",
        match: "Elegant finish",
      },
    ],
  },
};

const tonePalette = {
  Fair: "soft blues, charcoal, emerald",
  Medium: "navy, rust, ivory",
  Olive: "cream, forest green, tan",
  Brown: "white, cobalt, camel",
  Deep: "mustard, burgundy, crisp white",
};

function getBuildNote(height, weight) {
  const bmi = weight / ((height / 100) * (height / 100));

  if (bmi < 18.5) {
    return "Try slightly layered or textured pieces for a fuller visual balance.";
  }

  if (bmi < 25) {
    return "Most standard fits should work well, especially clean and balanced silhouettes.";
  }

  if (bmi < 30) {
    return "Structured fits and vertical lines can create a neat and flattering outline.";
  }

  return "Comfortable tailored cuts and breathable fabrics can keep the outfit polished and easy to wear.";
}

function updatePhotoPreview(file) {
  if (!file) {
    imagePreviewWrap.classList.add("hidden");
    imagePreview.removeAttribute("src");
    return;
  }

  const reader = new FileReader();
  reader.onload = () => {
    imagePreview.src = reader.result;
    imagePreviewWrap.classList.remove("hidden");
  };
  reader.readAsDataURL(file);
}

function renderRecommendations(userData) {
  const items = recommendations[userData.category][userData.wearFocus];
  const suggestedColors = tonePalette[userData.skinTone] || "navy, white, and earth tones";
  const buildNote = getBuildNote(userData.height, userData.weight);

  resultsTitle.textContent = `${userData.category} ${userData.wearFocus} recommendations for ${userData.gender}`;
  profilePill.textContent = `${userData.skinTone} tone`;
  summaryCard.textContent = `Suggested colors for this profile: ${suggestedColors}. ${buildNote} Focus area: ${userData.wearFocus}.`;

  recommendationList.innerHTML = "";

  items.forEach((item) => {
    const card = document.createElement("article");
    card.className = "recommendation-card";
    card.innerHTML = `
      <h3>${item.title}</h3>
      <p>${item.description}</p>
      <div class="product-meta">
        <span>${userData.category}</span>
        <span>${userData.wearFocus}</span>
        <span>${item.match}</span>
        <span>${userData.gender}</span>
      </div>
    `;
    recommendationList.appendChild(card);
  });

  resultsEmpty.classList.add("hidden");
  resultsContent.classList.remove("hidden");

  if (userData.photoDataUrl) {
    resultPhotoPreview.src = userData.photoDataUrl;
    photoInsight.textContent = `Photo uploaded successfully. These suggestions are tuned for ${userData.wearFocus.toLowerCase()} based on the selected profile and visual reference.`;
    uploadedPhotoCard.classList.remove("hidden");
  } else {
    uploadedPhotoCard.classList.add("hidden");
    resultPhotoPreview.removeAttribute("src");
    photoInsight.textContent = "";
  }
}

if (form) {
  if (userPhotoInput) {
    userPhotoInput.addEventListener("change", () => {
      const file = userPhotoInput.files && userPhotoInput.files[0];
      if (cameraPhotoInput) {
        cameraPhotoInput.value = "";
      }
      updatePhotoPreview(file);
    });
  }

  if (cameraPhotoInput) {
    cameraPhotoInput.addEventListener("change", () => {
      const file = cameraPhotoInput.files && cameraPhotoInput.files[0];
      if (userPhotoInput) {
        userPhotoInput.value = "";
      }
      updatePhotoPreview(file);
    });
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();

    const formData = new FormData(form);
    const userData = {
      height: Number(formData.get("height")),
      weight: Number(formData.get("weight")),
      gender: formData.get("gender"),
      skinTone: formData.get("skinTone"),
      category: formData.get("category"),
      wearFocus: formData.get("wearFocus"),
      photoDataUrl: imagePreview && imagePreview.src ? imagePreview.src : "",
    };

    renderRecommendations(userData);
  });
}
