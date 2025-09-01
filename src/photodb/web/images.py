from flask import send_file, abort
from pathlib import Path
import os
from typing import Any


def serve_image(normalized_path: str) -> Any:
    if not normalized_path:
        abort(404, "No image path provided")

    img_base = os.environ.get("IMG_PATH", "./photos/processed")

    if not normalized_path.startswith("/"):
        image_path = Path(img_base) / normalized_path
    else:
        image_path = Path(normalized_path)

    if not image_path.exists():
        abort(404, f"Image not found: {normalized_path}")

    if not image_path.is_file():
        abort(400, "Path is not a file")

    mimetype = "image/jpeg"
    if image_path.suffix.lower() in [".png"]:
        mimetype = "image/png"
    elif image_path.suffix.lower() in [".gif"]:
        mimetype = "image/gif"
    elif image_path.suffix.lower() in [".webp"]:
        mimetype = "image/webp"

    return send_file(
        image_path,
        mimetype=mimetype,
        max_age=86400,  # Cache for 1 day
        conditional=True,
    )
