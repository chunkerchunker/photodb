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
if [[ "$OSTYPE" == "darwin"* ]]; then
    if [ ! -d "$MODELS_DIR/yolov8x_person_face.mlpackage" ]; then
        echo ""
        echo "Exporting CoreML model for macOS (5x faster inference)..."

        # Ensure coremltools is installed
        if ! python -c "import coremltools" 2>/dev/null; then
            echo "Installing coremltools..."
            if command -v uv &> /dev/null; then
                uv pip install "coremltools>=7.0"
            else
                pip install "coremltools>=7.0"
            fi
        fi

        # Export to CoreML
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

echo ""
echo "=== InsightFace Models ==="
echo "InsightFace buffalo_l models will auto-download on first use."
echo "Models are cached in ~/.insightface/models/buffalo_l/"
echo ""
echo "To pre-download, run:"
echo "  python -c \"from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l').prepare(ctx_id=0)\""
echo ""

# Optional: Add explicit pre-download
if [ "${PREDOWNLOAD_INSIGHTFACE:-false}" = "true" ]; then
    echo "Pre-downloading InsightFace buffalo_l model..."
    python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l').prepare(ctx_id=0)"
fi

echo ""
echo "Done! Models downloaded to $MODELS_DIR"
ls -lh "$MODELS_DIR"
