"""Apple Vision scene classifier using VNClassifyImageRequest.

Uses Apple's Vision framework to classify images into 1303 scene taxonomy labels.
Only available on macOS.
"""

import logging
import sys
import time
from typing import Any, Dict

if sys.platform != "darwin":
    raise ImportError("Apple Vision only available on macOS")

import objc
import Quartz
import Vision
from Foundation import NSURL  # type: ignore[attr-defined]

from .. import config as defaults

logger = logging.getLogger(__name__)


class AppleVisionClassifier:
    """Classify scene content using Apple Vision Framework (1303 labels).

    Uses VNClassifyImageRequest to analyze images and return scene classification
    labels with confidence scores. This is a lightweight classifier that runs
    efficiently on Apple Silicon via the Neural Engine.
    """

    def __init__(self):
        """Initialize the classifier."""
        logger.info("AppleVisionClassifier initialized")

    def classify(
        self,
        image_path: str,
        top_k: int = defaults.APPLE_VISION_TOP_K,
        min_confidence: float = defaults.APPLE_VISION_MIN_CONFIDENCE,
    ) -> Dict[str, Any]:
        """Classify scene content in an image.

        Args:
            image_path: Path to the image file to classify.
            top_k: Maximum number of classifications to return (default: 10).
            min_confidence: Minimum confidence threshold for classifications (default: 0.01).

        Returns:
            Dict containing:
                - status: "success" or "error"
                - classifications: List of dicts with "identifier" and "confidence"
                - processing_time_ms: Time taken in milliseconds
                - error: Error message if status is "error"
        """
        start_time = time.time()

        try:
            # Wrap in autorelease pool to drain ObjC objects (CIImage, VNImageRequestHandler,
            # IOSurface textures) after each call. Without this, autoreleased objects accumulate
            # in long-lived ThreadPoolExecutor threads and are never freed.
            with objc.autorelease_pool():
                # Load image using CoreImage
                image_url = NSURL.fileURLWithPath_(image_path)
                ci_image = Quartz.CIImage.imageWithContentsOfURL_(image_url)  # type: ignore[attr-defined]

                if ci_image is None:
                    return {
                        "status": "error",
                        "classifications": [],
                        "error": f"Failed to load image: {image_path}",
                        "processing_time_ms": int((time.time() - start_time) * 1000),
                    }

                # Create Vision request handler and classification request
                handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(ci_image, None)  # type: ignore[attr-defined]
                request = Vision.VNClassifyImageRequest.alloc().init()  # type: ignore[attr-defined]

                # Perform the classification
                success, error = handler.performRequests_error_([request], None)

                if not success:
                    return {
                        "status": "error",
                        "classifications": [],
                        "error": str(error) if error else "Classification failed",
                        "processing_time_ms": int((time.time() - start_time) * 1000),
                    }

                # Process results — extract Python primitives before pool drains
                results = request.results() or []
                classifications = []

                for observation in results:
                    conf = float(observation.confidence())
                    if conf >= min_confidence:
                        classifications.append(
                            {
                                "identifier": str(observation.identifier()),
                                "confidence": conf,
                            }
                        )

            # Sort by confidence descending and limit to top_k
            classifications.sort(key=lambda x: x["confidence"], reverse=True)
            classifications = classifications[:top_k]

            return {
                "status": "success",
                "classifications": classifications,
                "processing_time_ms": int((time.time() - start_time) * 1000),
            }

        except Exception as e:
            logger.error(f"Classification failed for {image_path}: {e}")
            return {
                "status": "error",
                "classifications": [],
                "error": str(e),
                "processing_time_ms": int((time.time() - start_time) * 1000),
            }
