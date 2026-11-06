
import os
import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
import rasterio
import matplotlib.pyplot as plt

from config import DEVICE, CHECKPOINT_DIR
from model import FloodTransformer
from data_utils import normalize_image


# --------------------------------------------------
# 1. PAGE SETUP
# --------------------------------------------------

st.set_page_config(
    page_title="Flood Detection System",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Flood Detection Using SAR and Optical Images")
st.write(
    "Attention-Guided Multimodal Transformer "
    "for Flood Detection"
)

st.info(
    "Upload a matching SAR image (2 bands) and optical "
    "image (13 bands) in GeoTIFF format."
)


# --------------------------------------------------
# 2. LOAD TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    model_path = os.path.join(
        CHECKPOINT_DIR,
        "flood_transformer.pth"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained checkpoint not found: {model_path}. "
            "Please run train.py first."
        )

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE
    )

    # Support the checkpoint format used by train.py.
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise ValueError("The checkpoint format is invalid.")

    # Remove DataParallel prefixes if present.
    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }

    model = FloodTransformer().to(DEVICE)

    # Strict loading prevents silently ignoring model weights.
    model.load_state_dict(state_dict, strict=True)

    model.eval()
    return model


# --------------------------------------------------
# 3. READ AND PREPARE GEOTIFF
# --------------------------------------------------

def prepare_image(uploaded_file, expected_channels):
    with rasterio.open(uploaded_file) as src:
        image = src.read().astype(np.float32)
        valid_mask = src.dataset_mask() > 0

    if image.shape[0] != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} bands, "
            f"but received {image.shape[0]} bands."
        )

    # Resize image to 128 x 128.
    image_tensor = torch.from_numpy(image).unsqueeze(0)

    image_tensor = F.interpolate(
        image_tensor,
        size=(128, 128),
        mode="bilinear",
        align_corners=False
    )

    image = image_tensor.squeeze(0).numpy()

    # Resize the validity mask.
    mask_tensor = torch.from_numpy(
        valid_mask.astype(np.float32)
    )[None, None]

    mask_tensor = F.interpolate(
        mask_tensor,
        size=(128, 128),
        mode="nearest"
    )

    valid_mask = mask_tensor[0, 0].numpy() > 0.5

    # Use the same normalization as the training dataset.
    image = normalize_image(image)

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return image, valid_mask


# --------------------------------------------------
# 4. UPLOAD IMAGES
# --------------------------------------------------

st.subheader("Upload Satellite Images")

col1, col2 = st.columns(2)

with col1:
    sar_file = st.file_uploader(
        "SAR GeoTIFF (2 bands)",
        type=["tif", "tiff"],
        key="sar"
    )

with col2:
    optical_file = st.file_uploader(
        "Optical GeoTIFF (13 bands)",
        type=["tif", "tiff"],
        key="optical"
    )


# --------------------------------------------------
# 5. PREDICTION
# --------------------------------------------------

if sar_file is not None and optical_file is not None:

    if st.button("Detect Flood", type="primary"):

        try:
            with st.spinner("Loading model and analysing images..."):

                model = load_model()

                sar, sar_mask = prepare_image(sar_file, 2)
                optical, optical_mask = prepare_image(
                    optical_file, 13
                )

                valid_mask = sar_mask & optical_mask
                valid_pixels = int(valid_mask.sum())

                if valid_pixels == 0:
                    st.error(
                        "No valid pixels were found in both images."
                    )
                    st.stop()

                sar_tensor = torch.from_numpy(
                    sar
                ).unsqueeze(0).to(DEVICE)

                optical_tensor = torch.from_numpy(
                    optical
                ).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    logits = model(sar_tensor, optical_tensor)

                    probabilities = torch.softmax(logits, dim=1)

                    predictions = torch.argmax(
                        probabilities, dim=1
                    )[0].cpu().numpy()

                    flood_scores = probabilities[
                        0, 1
                    ].cpu().numpy()

            # --------------------------------------------------
            # 6. COUNT PREDICTED CLASSES
            # --------------------------------------------------

            class0_count = int(
                np.sum((predictions == 0) & valid_mask)
            )

            class1_count = int(
                np.sum((predictions == 1) & valid_mask)
            )

            class1_percent = (
                100.0 * class1_count / valid_pixels
            )

            st.subheader("Prediction Results")

            metric1, metric2, metric3 = st.columns(3)

            metric1.metric(
                "Valid Pixels",
                f"{valid_pixels:,}"
            )

            metric2.metric(
                "Predicted Class 1 Pixels",
                f"{class1_count:,}"
            )

            metric3.metric(
                "Class 1 Share",
                f"{class1_percent:.2f}%"
            )

            st.write("Class 0 pixels:", class0_count)
            st.write("Class 1 pixels:", class1_count)

            if class1_count == 0:
                st.warning(
                    "No Class 1 pixels were predicted. "
                    "This does not prove that the area is flood-free."
                )

            # --------------------------------------------------
            # 7. DISPLAY IMAGES AND FLOOD MAP
            # --------------------------------------------------

            st.subheader("Satellite Image and Prediction Map")

            # Optical band preview (bands 4, 3, 2 in 1-based indexing).
            rgb = optical[[3, 2, 1], :, :]
            rgb = np.moveaxis(rgb, 0, -1)
            rgb = np.clip(rgb, 0, 1)

            prediction_display = predictions.astype(np.float32)
            prediction_display[~valid_mask] = np.nan

            score_display = flood_scores.copy()
            score_display[~valid_mask] = np.nan

            fig, axes = plt.subplots(
                1, 3, figsize=(16, 5)
            )

            axes[0].imshow(rgb)
            axes[0].set_title("Optical Image Preview")
            axes[0].axis("off")

            axes[1].imshow(
                prediction_display,
                cmap="gray",
                vmin=0,
                vmax=1,
                interpolation="nearest"
            )
            axes[1].set_title(
                "Predicted Classes\n0 = Non-flood, 1 = Flood"
            )
            axes[1].axis("off")

            axes[2].imshow(
                score_display,
                cmap="viridis",
                vmin=0,
                vmax=1
            )
            axes[2].set_title("Class 1 Model Score")
            axes[2].axis("off")

            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "Class 1 share is the percentage of valid pixels "
                "predicted as Class 1. The model score is not a "
                "verified probability of actual flooding."
            )

        except Exception as error:
            st.error(
                f"{type(error).__name__}: {error}"
            )

            st.info(
                "If model loading fails, check whether the checkpoint "
                "and current model architecture match."
            )

else:
    st.info(
        "Upload both the SAR and optical GeoTIFF images "
        "to begin flood detection."
    )
