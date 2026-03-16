import io
import json
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import models, transforms


# ---------------------------
# Configuration / constants
# ---------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_MODEL_PATH = "best_model.pth"  # expected to be next to this script
DEFAULT_TOPK = 5


# ---------------------------
# Utilities: model & mapping
# ---------------------------
@st.cache_resource
def load_checkpoint(path: str = DEFAULT_MODEL_PATH, use_gpu: bool = True):
    """Load checkpoint and try to extract useful metadata (state_dict, class mapping).
    Returns (state_dict, metadata dict).
    """
    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    try:
        ckpt = torch.load(path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint at {path}: {e}")

    # Some people save {'state_dict': ..., 'class_to_idx': ..., 'epoch': ...}
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and any(k.startswith("module.") or k.startswith("fc.") for k in ckpt.keys()):
        # sometimes the raw state_dict is saved directly as the checkpoint dict
        state_dict = ckpt
    else:
        # could be just a state_dict or something else
        state_dict = ckpt

    metadata = {}
    # extract class mapping if present
    if isinstance(ckpt, dict):
        if "class_to_idx" in ckpt:
            metadata["class_to_idx"] = ckpt["class_to_idx"]
        if "idx_to_class" in ckpt:
            metadata["idx_to_class"] = ckpt["idx_to_class"]
        if "num_classes" in ckpt:
            metadata["num_classes"] = ckpt["num_classes"]

    return state_dict, metadata, device


def build_model_from_state(state_dict: dict, device: str = "cpu") -> torch.nn.Module:
    """Construct a torchvision.resnet34 and load the provided state_dict.
    We try to infer the final fc size from the state_dict.
    """
    # Handle DataParallel 'module.' prefix if present
    sd = state_dict.copy()
    new_sd = {}
    for k, v in sd.items():
        new_key = k
        if k.startswith("module."):
            new_key = k[len("module."):]
        new_sd[new_key] = v

    # try to detect number of classes from fc weight
    num_classes = None
    if "fc.weight" in new_sd:
        num_classes = new_sd["fc.weight"].shape[0]
    elif "classifier.weight" in new_sd:
        num_classes = new_sd["classifier.weight"].shape[0]

    # build base model
    model = models.resnet34(pretrained=False)

    if num_classes is not None:
        # replace final fc
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)

    model.load_state_dict(new_sd, strict=False)
    model.to(device)
    model.eval()

    return model


def load_class_mapping_from_metadata(metadata: dict) -> Dict[int, str]:
    """Return idx->class mapping if available in metadata. If only class_to_idx present,
    invert it.
    """
    if "idx_to_class" in metadata:
        return {int(k): v for k, v in metadata["idx_to_class"].items()}
    if "class_to_idx" in metadata:
        # invert
        inv = {int(v): k for k, v in metadata["class_to_idx"].items()}
        return inv
    return {}


# ---------------------------
# Image preprocessing & predict
# ---------------------------

def get_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def predict_image(img: Image.Image, model: torch.nn.Module, idx2class: Dict[int, str], device: str, topk: int = DEFAULT_TOPK) -> List[Tuple[str, float]]:
    t = get_transform()
    x = t(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        top_probs, top_idxs = probs.topk(topk, dim=1)

    top_probs = top_probs.cpu().numpy()[0]
    top_idxs = top_idxs.cpu().numpy()[0]

    results = []
    for p, idx in zip(top_probs, top_idxs):
        name = idx2class.get(int(idx), f"class_{int(idx)}")
        results.append((name, float(p)))
    return results


# ---------------------------
# Streamlit app
# ---------------------------

st.set_page_config(page_title="Celebrity Classifier", layout="centered")
st.title("Celebrity classifier — ResNet34")
st.caption("Upload a photo and the app will predict which celebrity (from the model's training set) the image most likely belongs to.")

# Sidebar: model & settings
st.sidebar.header("Model & settings")
model_file = st.sidebar.text_input("best_model.pth", value=DEFAULT_MODEL_PATH)
use_gpu = st.sidebar.checkbox("Use GPU if available", value=True)
top_k = st.sidebar.slider("Top K", min_value=1, max_value=10, value=DEFAULT_TOPK)

st.sidebar.markdown("---")
st.sidebar.markdown("If your checkpoint includes a `class_to_idx` or `idx_to_class`, the app will try to use it.\nYou can also upload a `classes.json` file mapping indices to labels (e.g. {\"0\": \"BTS Jungkook\", \"1\": \"IU\"}).")

# Allow user to upload an explicit classes.json
classes_file = st.sidebar.file_uploader("(Optional) Upload classes.json (idx->label)", type=["json"])

# Load model (cached)
load_status = st.sidebar.empty()
try:
    state_dict, metadata, device = load_checkpoint(model_file, use_gpu)
    model = build_model_from_state(state_dict, device)
    idx2class = load_class_mapping_from_metadata(metadata)
    load_status.success(f"Model loaded from '{model_file}' — device: {device}")
except Exception as e:
    load_status.error(f"Failed to load model: {e}")
    st.stop()

# If user provided classes.json, override
if classes_file is not None:
    try:
        classes_json = json.load(classes_file)
        # ensure keys are ints
        idx2class = {int(k): v for k, v in classes_json.items()}
        st.sidebar.success("Loaded classes.json — will use provided mapping.")
    except Exception as e:
        st.sidebar.error(f"Failed to parse classes.json: {e}")

if not idx2class:
    st.sidebar.warning("No class mapping found in checkpoint or uploaded file. Results will show numeric class indices.")

# Main UI: upload
st.header("Try it — upload an image")
uploaded_file = st.file_uploader("Upload a face photo (jpg, png)", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns([1, 1])

if uploaded_file is not None:
    try:
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        st.error(f"Cannot open image: {e}")
        st.stop()

    with col1:
        st.image(image, caption="Input image", use_column_width=True)

    with col2:
        st.markdown("**Predictions**")
        try:
            results = predict_image(image, model, idx2class, device, topk=top_k)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        # Create DataFrame for easy display
        df = pd.DataFrame(results, columns=["label", "probability"])  # probability in [0,1]
        df["probability"] = df["probability"].map(lambda x: round(x, 4))
        st.table(df)

        # bar chart
        chart_df = df.set_index("label")
        st.bar_chart(chart_df)

        st.markdown("---")
        st.caption("Notes: \n- This app uses the model file you specified. \n- Make sure the model was trained on aligned face crops (or use aligned inputs) for best results.")

else:
    st.info("Upload an image to get a prediction.")

# # Footer: instructions to run
# st.markdown("---")
# st.markdown("**Run locally:**\n```
# # install requirements
# pip install torch torchvision streamlit pillow pandas

# # then run
# streamlit run celebrity_classifier_app.py
# ```")

#st.markdown("If your `best_model.pth` was trained with a different input size/normalization, edit `IMAGENET_MEAN/STD` and transform in the source." )


# End of file
