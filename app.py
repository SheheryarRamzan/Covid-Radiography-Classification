# app.py
import io
import os
from typing import Tuple

import streamlit as st
import torch
import torch.nn.functional as F
import timm
import numpy as np
import cv2
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.cm as cm

# ================== CONFIG ==================
CLASS_NAMES = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PTH = "best_chest_xray_model.pth"  # ensure this path is correct

# ================== MODEL DEFINITION ==================
class ChestXRayClassifier(torch.nn.Module):
    def __init__(self, num_classes=4, model_name='efficientnet_b0', dropout_rate=0.3):
        super().__init__()
        # create backbone with no final classifier (num_classes=0)
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        # figure out feature dim dynamically by running a dummy through backbone
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feat = self.backbone(dummy)
            feature_dim = feat.shape[1]
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(feature_dim, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate * 0.7),
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate * 0.5),
            torch.nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

# ================ HELPERS FOR LAST CONV LAYER ================
def find_last_conv_module(model: torch.nn.Module):
    """
    Returns the last module in the model that produces a 4D activation (C,H,W).
    This is robust across timm model variants.
    """
    last_conv = None

    def _search(module):
        nonlocal last_conv
        for name, child in module.named_children():
            # check child's output shape by running a small forward if it's not trivial
            if len(list(child.children())) > 0:
                _search(child)
            else:
                # Heuristic: conv layers usually have 'Conv' in class name or are 2D convs
                cls_name = child.__class__.__name__.lower()
                if "conv" in cls_name or "conv2d" in cls_name:
                    last_conv = child

    _search(model)
    if last_conv is None:
        # fallback - return backbone itself
        return model
    return last_conv

# ================== PREPROCESS / POSTPROCESS ==================
val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

def load_dicom_bytes(file_bytes: io.BytesIO) -> np.ndarray:
    dicom = pydicom.dcmread(file_bytes, force=True)
    arr = dicom.pixel_array
    arr = apply_voi_lut(arr, dicom)
    # normalize to 0-255
    if arr.max() > 255:
        arr = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    else:
        if arr.shape[2] == 1:
            arr = np.concatenate([arr] * 3, axis=2)
    return arr

def load_image_from_uploaded(uploaded_file) -> Tuple[np.ndarray, str]:
    """Return RGB numpy array and original filename for either DICOM or image files"""
    name = uploaded_file.name
    try:
        content = uploaded_file.read()
        buf = io.BytesIO(content)
        if name.lower().endswith((".dcm", ".dicom")):
            arr = load_dicom_bytes(buf)
        else:
            img = Image.open(buf).convert("RGB")
            arr = np.array(img)
        return arr, name
    except Exception as e:
        raise ValueError(f"Failed to load file `{name}`: {e}")

def preprocess_for_model(image_np: np.ndarray):
    augmented = val_transform(image=image_np)
    tensor = augmented["image"].unsqueeze(0).to(DEVICE)
    return tensor

# ================== MODEL LOADING (cached) ==================
@st.cache_resource
def load_model():
    model = ChestXRayClassifier(num_classes=len(CLASS_NAMES))
    if not os.path.exists(MODEL_PTH):
        st.error(f"Model file `{MODEL_PTH}` not found. Place your .pth file next to app or update MODEL_PTH.")
    else:
        try:
            state = torch.load(MODEL_PTH, map_location=DEVICE)
            # if the saved dict contains 'model_state_dict'
            if isinstance(state, dict) and 'model_state_dict' in state:
                state = state['model_state_dict']
            model.load_state_dict(state)
        except Exception as e:
            st.error(f"Error loading model weights: {e}")
    model.to(DEVICE).eval()
    return model

model = load_model()

# ================== PREDICTION ==================
def predict(image_tensor: torch.Tensor):
    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    return probs

# ================== GRAD-CAM CORE ==================
class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        # register hooks
        self._fwd = self.target_layer.register_forward_hook(self._forward_hook)
        self._bwd = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        # output shape: [N, C, H, W]
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        # grad_output is a tuple; take first element
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor: torch.Tensor, class_idx: int = None):
        self.model.zero_grad()
        input_tensor = input_tensor.to(next(self.model.parameters()).device)
        input_tensor.requires_grad_(True)
        out = self.model(input_tensor)  # shape [1, num_classes]
        if class_idx is None:
            class_idx = int(torch.argmax(out, dim=1).item())
        score = out[0, class_idx]
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("GradCAM couldn't capture gradients or activations. Target layer might be incorrect.")

        # global-average-pool gradients over spatial dims
        pooled_grads = torch.mean(self.gradients[0], dim=(1, 2))  # shape [C]
        activations = self.activations[0]  # [C, H, W]
        # weight channels
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_grads[i]
        heatmap = torch.sum(activations, dim=0)
        heatmap = torch.relu(heatmap)
        if torch.max(heatmap) != 0:
            heatmap = heatmap / (torch.max(heatmap) + 1e-8)
        heatmap_np = heatmap.cpu().numpy()
        return heatmap_np

    def close(self):
        if self._fwd:
            self._fwd.remove()
        if self._bwd:
            self._bwd.remove()

def make_overlay(image_rgb: np.ndarray, heatmap: np.ndarray, cmap_name="jet", alpha=0.4, intensity=1.0):
    """
    image_rgb: HxWx3 uint8
    heatmap: h'xw' normalized 0..1 (may be smaller than image, we'll resize)
    intensity: multiplies heatmap before colormap mapping
    """
    h, w = image_rgb.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_resized = np.clip(heatmap_resized * intensity, 0, 1)
    # colormap
    cmap = cm.get_cmap(cmap_name)
    colored = cmap(heatmap_resized)[:, :, :3]  # RGB float 0..1
    colored = (colored * 255).astype(np.uint8)
    overlay = (colored * alpha + image_rgb * (1 - alpha)).astype(np.uint8)
    # also produce heatmap alone for visualization
    heatmap_vis = (colored * 0.8 + 255 * np.stack([1 - heatmap_resized]*3, axis=2) * 0.2).astype(np.uint8)
    return overlay, heatmap_vis

# ================== STREAMLIT UI ==================
st.set_page_config(layout="wide", page_title="Chest X-ray Classifier + Grad-CAM")

st.title("🩻 Chest X-Ray Classifier + Grad-CAM")
st.write("Upload a chest X-ray (PNG, JPG, or DICOM). The app will show prediction, confidence, and Grad-CAM localization.")

uploaded_file = st.file_uploader("Upload an image file", type=["png", "jpg", "jpeg", "dcm", "dicom"])

# visualization controls
st.sidebar.header("Grad-CAM Visualization Controls")
cmap_name = st.sidebar.selectbox("Colormap", options=["jet", "inferno", "magma", "plasma"], index=0)
intensity = st.sidebar.slider("Heatmap intensity", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
alpha = st.sidebar.slider("Overlay opacity", min_value=0.05, max_value=1.0, value=0.4, step=0.05)
show_heatmap_only = st.sidebar.checkbox("Show heatmap only (no overlay)", value=False)

if uploaded_file is None:
    st.info("Please upload an image (PNG/JPG/DICOM) to run prediction and Grad-CAM.")
else:
    try:
        # load image bytes into numpy RGB
        image_np, fname = load_image_from_uploaded(uploaded_file)
        # keep a copy for display (uint8)
        display_image = image_np.copy()
        st.write(f"**File:** {fname} — shape: {display_image.shape}")

        # preprocess for model
        img_tensor = preprocess_for_model(display_image)

        # predict
        probs = predict(img_tensor)
        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])

        # Display predictions & confidences
        col_left, col_right = st.columns([1, 2])
        with col_left:
            st.subheader("Prediction")
            st.metric("Predicted class", pred_class, delta=None)
            st.write("**Confidence scores:**")
            for i, cls in enumerate(CLASS_NAMES):
                st.write(f"- {cls}: {probs[i]:.4f}")

        # Generate Grad-CAM
        # find a sensible target layer
        target_layer = find_last_conv_module(model.backbone)
        gradcam = GradCAM(model, target_layer)

        try:
            heatmap = gradcam(img_tensor, class_idx=pred_idx)
        except Exception as e:
            # fallback: try with no specified class
            heatmap = gradcam(img_tensor, class_idx=None)

        gradcam.close()

        # overlay & heatmap images
        overlay_img, heatmap_vis = make_overlay(display_image, heatmap, cmap_name=cmap_name, alpha=alpha, intensity=intensity)

        # zoomable / full size toggle
        col1, col2, col3 = st.columns([1,2,1])  # middle column wider
        with col2:
          st.image(overlay_img, caption="Overlay (full)")

        # side-by-side display
        st.markdown(
            "<h3 style='text-align: center;'>Original vs Grad-CAM</h3>",
            unsafe_allow_html=True
        )

        left_col, right_col = st.columns(2)
        with left_col:
            st.image(display_image, caption="Original X-ray")
        with right_col:
            if show_heatmap_only:
                st.image(heatmap_vis, caption=f"Grad-CAM heatmap ({cmap_name})")
            else:
                st.image(overlay_img, caption=f"Overlay: {pred_class} (conf {confidence:.3f})")


    except Exception as e:
        st.error(f"Processing error: {e}")
        st.exception(e)
