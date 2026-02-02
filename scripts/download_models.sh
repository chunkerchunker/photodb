#!/bin/bash
# Download required model files for detection and age/gender estimation

set -e

MODELS_DIR="${1:-models}"
mkdir -p "$MODELS_DIR"

echo "Downloading models to $MODELS_DIR..."

# YOLOv8x person_face model (from HuggingFace)
if [ ! -f "$MODELS_DIR/yolov8x_person_face.pt" ]; then
    echo "Downloading yolov8x_person_face.pt (137 MB)..."
    wget -q --show-progress -O "$MODELS_DIR/yolov8x_person_face.pt" \
        "https://huggingface.co/iitolstykh/YOLO-Face-Person-Detector/resolve/main/yolov8x_person_face.pt"
else
    echo "yolov8x_person_face.pt already exists, skipping..."
fi

# MiVOLO v2 model (from HuggingFace) - now in safetensors format
if [ ! -f "$MODELS_DIR/mivolo_v2.safetensors" ]; then
    echo "Downloading mivolo_v2.safetensors (115 MB)..."
    wget -q --show-progress -O "$MODELS_DIR/mivolo_v2.safetensors" \
        "https://huggingface.co/iitolstykh/mivolo_v2/resolve/main/model.safetensors"
else
    echo "mivolo_v2.safetensors already exists, skipping..."
fi

# Also download MiVOLO config file
if [ ! -f "$MODELS_DIR/mivolo_v2_config.json" ]; then
    echo "Downloading mivolo_v2_config.json..."
    wget -q -O "$MODELS_DIR/mivolo_v2_config.json" \
        "https://huggingface.co/iitolstykh/mivolo_v2/resolve/main/config.json"
else
    echo "mivolo_v2_config.json already exists, skipping..."
fi

echo ""
echo "Done! Models downloaded to $MODELS_DIR"
ls -lh "$MODELS_DIR"
