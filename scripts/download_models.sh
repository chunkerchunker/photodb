#!/bin/bash
# Download required model files for detection and age/gender estimation

set -e

MODELS_DIR="${1:-models}"
mkdir -p "$MODELS_DIR"

echo "Downloading models to $MODELS_DIR..."

# Ensure omegaconf is installed (required by YOLO model)
if ! python -c "import omegaconf" 2>/dev/null; then
    echo "Installing omegaconf (required by YOLO model)..."
    if command -v uv &> /dev/null; then
        uv pip install omegaconf
    else
        pip install omegaconf
    fi
fi

# YOLOv8x person_face model (from HuggingFace)
if [ ! -f "$MODELS_DIR/yolov8x_person_face.pt" ]; then
    echo "Downloading yolov8x_person_face.pt (137 MB)..."
    wget -q --show-progress -O "$MODELS_DIR/yolov8x_person_face.pt" \
        "https://huggingface.co/iitolstykh/YOLO-Face-Person-Detector/resolve/main/yolov8x_person_face.pt"
else
    echo "yolov8x_person_face.pt already exists, skipping..."
fi

# MiVOLO model (mivolo_d1 face+body from Google Drive)
# This is the original MiVOLO checkpoint in .pth.tar format required by the mivolo library
# File ID: 11i8pKctxz3wVkDBlWKvhYIh7kpVFXSZ4
if [ ! -f "$MODELS_DIR/mivolo_d1.pth.tar" ]; then
    echo "Downloading MiVOLO checkpoint mivolo_d1.pth.tar (~330 MB)..."

    # Ensure gdown is installed for Google Drive downloads
    if ! python -c "import gdown" 2>/dev/null; then
        echo "Installing gdown for Google Drive download..."
        if command -v uv &> /dev/null; then
            uv pip install gdown
        else
            pip install gdown
        fi
    fi

    # Download using gdown (use uv run if available for consistent environment)
    if command -v uv &> /dev/null; then
        uv run gdown 11i8pKctxz3wVkDBlWKvhYIh7kpVFXSZ4 -O "$MODELS_DIR/mivolo_d1.pth.tar"
    else
        gdown 11i8pKctxz3wVkDBlWKvhYIh7kpVFXSZ4 -O "$MODELS_DIR/mivolo_d1.pth.tar"
    fi
else
    echo "mivolo_d1.pth.tar already exists, skipping..."
fi

# On macOS, export CoreML model for faster inference via Neural Engine
# NOTE: dynamic=True export causes SIGSEGV with coremltools 9.0 + PyTorch 2.8 on Apple Silicon.
# We export with nms=False (required for Ultralytics 8.3+) but without dynamic=True.
# The default config uses PyTorch MPS batch inference (DETECTION_PREFER_COREML=False).
# This CoreML export is retained as a fallback (enable with DETECTION_PREFER_COREML=true).
if [[ "$OSTYPE" == "darwin"* ]]; then
    COREML_DIR="$MODELS_DIR/yolov8x_person_face.mlpackage"

    if [ ! -d "$COREML_DIR" ]; then
        echo ""
        echo "Exporting CoreML model for macOS (5x faster inference via Neural Engine)..."

        # Ensure coremltools is installed
        if ! python -c "import coremltools" 2>/dev/null; then
            echo "Installing coremltools..."
            if command -v uv &> /dev/null; then
                uv pip install "coremltools>=7.0"
            else
                pip install "coremltools>=7.0"
            fi
        fi

        # Export to CoreML (nms=False required for Ultralytics 8.3+)
        if command -v uv &> /dev/null; then
            uv run python -c "
import torch
# Patch torch.load for PyTorch 2.6+
original = torch.load
torch.load = lambda *a, **k: original(*a, **{**k, 'weights_only': False})
from ultralytics import YOLO
model = YOLO('$MODELS_DIR/yolov8x_person_face.pt')
model.export(format='coreml', nms=False, imgsz=640)
"
        else
            python -c "
import torch
original = torch.load
torch.load = lambda *a, **k: original(*a, **{**k, 'weights_only': False})
from ultralytics import YOLO
model = YOLO('$MODELS_DIR/yolov8x_person_face.pt')
model.export(format='coreml', nms=False, imgsz=640)
"
        fi
    else
        echo "CoreML model already exists, skipping export..."
    fi
fi

# ArcFace recognition model (w600k_r50 from buffalo_l pack)
ARCFACE_MODEL="$HOME/.insightface/models/buffalo_l/w600k_r50.onnx"
if [ ! -f "$ARCFACE_MODEL" ]; then
    echo ""
    echo "Downloading ArcFace recognition model (buffalo_l pack)..."
    mkdir -p "$(dirname "$ARCFACE_MODEL")"
    curl -L "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip" -o /tmp/buffalo_l.zip
    unzip -o /tmp/buffalo_l.zip -d "$HOME/.insightface/models/buffalo_l/"
    rm /tmp/buffalo_l.zip
    echo "ArcFace model extracted to $(dirname "$ARCFACE_MODEL")"
else
    echo "ArcFace w600k_r50.onnx already exists, skipping..."
fi

echo ""
echo "Done! Models downloaded to $MODELS_DIR"
ls -lh "$MODELS_DIR"
