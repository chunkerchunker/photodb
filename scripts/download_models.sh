#!/bin/bash
# Download required model files for detection and age/gender estimation

set -e

MODELS_DIR="${1:-models}"
mkdir -p "$MODELS_DIR"

echo "Downloading models to $MODELS_DIR..."

# YOLOv8x person_face model (from MiVOLO repo)
if [ ! -f "$MODELS_DIR/yolov8x_person_face.pt" ]; then
    echo "Downloading yolov8x_person_face.pt..."
    wget -O "$MODELS_DIR/yolov8x_person_face.pt" \
        "https://github.com/WildChlamydia/MiVOLO/releases/download/v1.0/yolov8x_person_face.pt"
else
    echo "yolov8x_person_face.pt already exists, skipping..."
fi

# MiVOLO model
if [ ! -f "$MODELS_DIR/mivolo_imdb.pth.tar" ]; then
    echo "Downloading mivolo_imdb.pth.tar..."
    wget -O "$MODELS_DIR/mivolo_imdb.pth.tar" \
        "https://github.com/WildChlamydia/MiVOLO/releases/download/v1.0/mivolo_imdb.pth.tar"
else
    echo "mivolo_imdb.pth.tar already exists, skipping..."
fi

echo ""
echo "Done! Models downloaded to $MODELS_DIR"
ls -lh "$MODELS_DIR"
