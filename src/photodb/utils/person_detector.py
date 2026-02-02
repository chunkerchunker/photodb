"""
Person detection using YOLO for face+body detection and FaceNet for embeddings.
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from ultralytics import YOLO


class PersonDetector:
    """Detect faces and bodies in images using YOLO, extract face embeddings."""

    # Class IDs from yolov8x_person_face model
    FACE_CLASS_ID = 1
    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        force_cpu: bool = False,
    ):
        """
        Initialize person detection and embedding models.

        Args:
            model_path: Path to YOLO model. If None, uses DETECTION_MODEL_PATH env var
                       or defaults to 'yolov8n.pt' (will auto-download).
            device: Device to use ('mps', 'cuda', 'cpu'). Auto-detects if not specified.
            force_cpu: Force CPU-only mode.
        """
        # Determine device
        if force_cpu:
            self.device = "cpu"
        elif device is not None:
            self.device = device
        else:
            # Auto-detect best available device
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

        # Get model path from env or parameter
        if model_path is None:
            model_path = os.environ.get("DETECTION_MODEL_PATH", "yolov8n.pt")

        # Get minimum confidence from env
        self.min_confidence = float(os.environ.get("DETECTION_CONFIDENCE", "0.5"))

        # Load YOLO model
        self.model = YOLO(model_path)

        # Load FaceNet embedding model
        self.facenet = InceptionResnetV1(pretrained="vggface2").eval()
        if self.device in ["cuda", "mps"]:
            self.facenet = self.facenet.to(self.device)

    def detect(self, image_path: str) -> Dict[str, Any]:
        """
        Detect faces and bodies in an image.

        Args:
            image_path: Path to the image file.

        Returns:
            Dictionary with:
                - status: 'success', 'no_detections', or 'error'
                - detections: List of person detections with face/body info
                - image_dimensions: Dict with width and height
                - error: Error message (only if status is 'error')
        """
        try:
            # Load image
            img = Image.open(image_path)

            # Convert RGBA to RGB if necessary
            if img.mode == "RGBA":
                img = img.convert("RGB")
            elif img.mode != "RGB":
                img = img.convert("RGB")

            img_width, img_height = img.size

            # Run YOLO detection
            results = self.model(
                img,
                conf=self.min_confidence,
                device=self.device if self.device != "mps" else "cpu",  # YOLO may have MPS issues
                verbose=False,
            )

            # Parse detections
            faces: List[Dict[str, Any]] = []
            bodies: List[Dict[str, Any]] = []

            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    for i, box in enumerate(result.boxes):
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        xyxy = box.xyxy[0].cpu().numpy()

                        detection = {
                            "bbox": {
                                "x1": float(xyxy[0]),
                                "y1": float(xyxy[1]),
                                "x2": float(xyxy[2]),
                                "y2": float(xyxy[3]),
                                "width": float(xyxy[2] - xyxy[0]),
                                "height": float(xyxy[3] - xyxy[1]),
                            },
                            "confidence": conf,
                            "class_id": cls_id,
                        }

                        if cls_id == self.FACE_CLASS_ID:
                            faces.append(detection)
                        elif cls_id == self.PERSON_CLASS_ID:
                            bodies.append(detection)

            # Match faces to bodies
            if not faces and not bodies:
                return {
                    "status": "no_detections",
                    "detections": [],
                    "image_dimensions": {"width": img_width, "height": img_height},
                }

            matched_detections = self._match_faces_to_bodies(faces, bodies)

            # Extract embeddings for faces
            for detection in matched_detections:
                if detection["face"] is not None:
                    try:
                        embedding = self.extract_embedding(img, detection["face"]["bbox"])
                        detection["face"]["embedding"] = embedding
                        detection["face"]["embedding_norm"] = float(np.linalg.norm(embedding))
                    except Exception:
                        # If embedding extraction fails, continue without embedding
                        detection["face"]["embedding"] = None
                        detection["face"]["embedding_norm"] = None

            return {
                "status": "success",
                "detections": matched_detections,
                "image_dimensions": {"width": img_width, "height": img_height},
            }

        except FileNotFoundError:
            return {
                "status": "error",
                "detections": [],
                "image_dimensions": {"width": 0, "height": 0},
                "error": f"File not found: {image_path}",
            }
        except Exception as e:
            return {
                "status": "error",
                "detections": [],
                "image_dimensions": {"width": 0, "height": 0},
                "error": str(e),
            }

    def _match_faces_to_bodies(
        self, faces: List[Dict[str, Any]], bodies: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Match each face to its containing body based on spatial containment.

        Args:
            faces: List of face detections with bbox.
            bodies: List of body detections with bbox.

        Returns:
            List of matched detections with {'face': face_data|None, 'body': body_data|None}
        """
        matched: List[Dict[str, Any]] = []
        used_bodies: set = set()

        # Match each face to best containing body
        for face in faces:
            best_body = None
            best_containment = 0.0

            for i, body in enumerate(bodies):
                if i in used_bodies:
                    continue

                containment = self._compute_containment(face["bbox"], body["bbox"])
                if containment > best_containment:
                    best_containment = containment
                    best_body = (i, body)

            if best_body is not None and best_containment > 0.3:  # Minimum 30% overlap
                used_bodies.add(best_body[0])
                matched.append({"face": face, "body": best_body[1]})
            else:
                matched.append({"face": face, "body": None})

        # Add unmatched bodies
        for i, body in enumerate(bodies):
            if i not in used_bodies:
                matched.append({"face": None, "body": body})

        return matched

    def _compute_containment(self, face_bbox: Dict[str, Any], body_bbox: Dict[str, Any]) -> float:
        """
        Compute how much of the face bbox is contained within the body bbox.

        Args:
            face_bbox: Face bounding box with x1, y1, x2, y2.
            body_bbox: Body bounding box with x1, y1, x2, y2.

        Returns:
            Containment ratio (0.0 to 1.0), where 1.0 means fully contained.
        """
        # Calculate intersection
        x1 = max(face_bbox["x1"], body_bbox["x1"])
        y1 = max(face_bbox["y1"], body_bbox["y1"])
        x2 = min(face_bbox["x2"], body_bbox["x2"])
        y2 = min(face_bbox["y2"], body_bbox["y2"])

        # No intersection
        if x1 >= x2 or y1 >= y2:
            return 0.0

        intersection_area = (x2 - x1) * (y2 - y1)
        face_area = (face_bbox["x2"] - face_bbox["x1"]) * (face_bbox["y2"] - face_bbox["y1"])

        if face_area == 0:
            return 0.0

        return intersection_area / face_area

    def extract_embedding(self, image: Image.Image, bbox: Dict[str, Any]) -> List[float]:
        """
        Extract face embedding from a cropped face region.

        Args:
            image: PIL Image to extract face from.
            bbox: Bounding box with x1, y1, x2, y2.

        Returns:
            512-dimensional face embedding as list of floats.
        """
        # Crop face from image
        x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])

        # Ensure bounds are within image
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.width, x2)
        y2 = min(image.height, y2)

        face_crop = image.crop((x1, y1, x2, y2))

        # Resize to 160x160 (FaceNet input size)
        face_crop = face_crop.resize((160, 160), Image.BILINEAR)

        # Convert to tensor
        face_tensor = torch.tensor(np.array(face_crop)).permute(2, 0, 1).float()

        # Normalize to [-1, 1] range (as expected by FaceNet)
        face_tensor = (face_tensor - 127.5) / 128.0

        # Add batch dimension
        face_tensor = face_tensor.unsqueeze(0)

        # Move to device
        if self.device in ["cuda", "mps"]:
            face_tensor = face_tensor.to(self.device)

        # Extract embedding
        with torch.no_grad():
            try:
                embedding = self.facenet(face_tensor)
            except RuntimeError as e:
                # Handle MPS issues by falling back to CPU
                if "MPS" in str(e):
                    face_tensor = face_tensor.cpu()
                    facenet_cpu = InceptionResnetV1(pretrained="vggface2").eval()
                    embedding = facenet_cpu(face_tensor)
                else:
                    raise

        return embedding.cpu().numpy().flatten().tolist()
