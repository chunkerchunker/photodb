"""
Face detection and embedding extraction using MTCNN and FaceNet.
"""

from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from typing import Dict, List, Tuple


class FaceExtractor:
    """Extract face bounding boxes and embeddings for database storage."""

    def __init__(self, device: str | None = None, force_cpu_fallback: bool = False):
        """
        Initialize face detection and embedding models.

        Args:
            device: 'mps', 'cuda' or 'cpu'. Auto-detects if not specified.
            force_cpu_fallback: Force CPU-only mode to avoid MPS issues.
        """
        if force_cpu_fallback:
            device = "cpu"
            # print("Forcing CPU mode to avoid MPS compatibility issues")
        elif device is None:
            # Auto-detect best available device
            if torch.backends.mps.is_available():
                device = "mps"  # Apple Silicon GPU
            elif torch.cuda.is_available():
                device = "cuda"  # NVIDIA GPU
            else:
                device = "cpu"
        self.device = device
        # print(f"Using device: {device}")

        # Face detection with MTCNN
        self.mtcnn = MTCNN(
            image_size=160,
            margin=20,  # Add margin around face for better embeddings
            keep_all=True,  # Return all detected faces
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],  # Detection thresholds
            device=device,
        )

        # Face embeddings with FaceNet
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval()
        if device in ["cuda", "mps"]:
            self.resnet = self.resnet.to(device)

    def extract_from_image(self, image_path: str) -> Dict:
        """
        Extract faces and embeddings from a single image.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with face data ready for database storage
        """
        img = Image.open(image_path)

        # Convert RGBA to RGB if necessary
        if img.mode == "RGBA":
            img = img.convert("RGB")

        # Detect faces with MPS fallback handling
        try:
            # MTCNN returns aligned faces and probabilities when called with return_prob=True
            aligned_faces, probs = self.mtcnn(img, return_prob=True)

            # Get bounding boxes separately
            if aligned_faces is not None:
                boxes, _ = self.mtcnn.detect(img)
            else:
                boxes = None

        except RuntimeError as e:
            if "MPS" in str(e) and "Adaptive pool" in str(e):
                # MPS has issues with certain operations, fallback to CPU
                print(f"MPS error detected, falling back to CPU for face detection: {e}")
                return self._extract_with_cpu_fallback(img, image_path)
            else:
                raise e

        if aligned_faces is None or boxes is None:
            return {"status": "no_faces_detected", "faces": [], "image_dimensions": img.size}

        # Handle single vs multiple faces
        if aligned_faces.dim() == 3:
            aligned_faces = aligned_faces.unsqueeze(0)
            probs = [probs]

        # Convert boxes to list format if needed
        if not isinstance(boxes, list):
            boxes = boxes.tolist() if hasattr(boxes, "tolist") else [boxes]

        # Generate embeddings with MPS fallback handling
        try:
            if self.device in ["cuda", "mps"]:
                aligned_faces = aligned_faces.to(self.device)

            with torch.no_grad():
                embeddings = self.resnet(aligned_faces)
        except RuntimeError as e:
            if "MPS" in str(e):
                # MPS error during embedding generation, fallback to CPU
                print(f"MPS error during embedding generation, falling back to CPU: {e}")
                aligned_faces = aligned_faces.cpu()
                resnet_cpu = InceptionResnetV1(pretrained="vggface2").eval()
                with torch.no_grad():
                    embeddings = resnet_cpu(aligned_faces)
            else:
                raise e

        # Prepare for database storage
        faces = []
        for i, (box, prob, embedding) in enumerate(zip(boxes, probs, embeddings)):
            embedding_np = embedding.cpu().numpy()

            face = {
                "face_id": i,
                "bbox": {
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3]),
                    "width": float(box[2] - box[0]),
                    "height": float(box[3] - box[1]),
                },
                "confidence": float(prob),
                "embedding": embedding_np.tolist(),  # 512-dim vector
                "embedding_norm": float(np.linalg.norm(embedding_np)),
            }

            # Calculate face position relative to image
            face["position"] = {
                "center_x": (face["bbox"]["x1"] + face["bbox"]["x2"]) / 2 / img.width,
                "center_y": (face["bbox"]["y1"] + face["bbox"]["y2"]) / 2 / img.height,
                "relative_size": (face["bbox"]["width"] * face["bbox"]["height"])
                / (img.width * img.height),
            }

            faces.append(face)

        return {
            "status": "success",
            "faces": faces,
            "face_count": len(faces),
            "image_dimensions": {"width": img.width, "height": img.height},
        }

    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two face embeddings.

        Args:
            embedding1: First face embedding (512-dim)
            embedding2: Second face embedding (512-dim)

        Returns:
            Cosine similarity score (higher = more similar)
        """
        e1 = np.array(embedding1)
        e2 = np.array(embedding2)

        # Cosine similarity
        similarity = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
        return float(similarity)

    def find_similar_faces(
        self,
        query_embedding: List[float],
        face_embeddings: List[Tuple[int, List[float]]],
        threshold: float = 0.6,
    ) -> List[Tuple[int, float]]:
        """
        Find similar faces from a collection.

        Args:
            query_embedding: Query face embedding
            face_embeddings: List of (id, embedding) tuples
            threshold: Similarity threshold (0.6 is typical for same person)

        Returns:
            List of (id, similarity_score) for matching faces
        """
        matches = []
        for face_id, embedding in face_embeddings:
            similarity = self.compute_similarity(query_embedding, embedding)
            if similarity >= threshold:
                matches.append((face_id, similarity))

        # Sort by similarity descending
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def _extract_with_cpu_fallback(self, img: Image.Image, image_path: str = "") -> Dict:
        """
        Extract faces using CPU-only models as fallback for MPS issues.
        """
        try:
            # Create CPU-only MTCNN and FaceNet models
            mtcnn_cpu = MTCNN(
                image_size=160,
                margin=20,
                keep_all=True,
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7],
                device="cpu",
            )

            resnet_cpu = InceptionResnetV1(pretrained="vggface2").eval()
            # Keep on CPU (no .to() call needed)

            # Detect faces on CPU - need to call twice to get both aligned faces and boxes
            # First get aligned faces and probabilities
            aligned_faces, probs = mtcnn_cpu(img, return_prob=True)

            if aligned_faces is None:
                return {"status": "no_faces_detected", "faces": [], "image_dimensions": img.size}

            # Second call to get bounding boxes (this is how MTCNN works)
            boxes, _ = mtcnn_cpu.detect(img)

            if boxes is None:
                return {"status": "no_faces_detected", "faces": [], "image_dimensions": img.size}

            # Handle single vs multiple faces
            if aligned_faces.dim() == 3:
                aligned_faces = aligned_faces.unsqueeze(0)
                probs = [probs]

            # Convert boxes to list format if needed
            if not isinstance(boxes, list):
                boxes = boxes.tolist() if hasattr(boxes, "tolist") else [boxes]

            # Generate embeddings on CPU
            with torch.no_grad():
                embeddings = resnet_cpu(aligned_faces)

            # Prepare for database storage
            faces = []
            for i, (box, prob, embedding) in enumerate(zip(boxes, probs, embeddings)):
                embedding_np = embedding.cpu().numpy()

                face = {
                    "face_id": i,
                    "bbox": {
                        "x1": float(box[0]),
                        "y1": float(box[1]),
                        "x2": float(box[2]),
                        "y2": float(box[3]),
                        "width": float(box[2] - box[0]),
                        "height": float(box[3] - box[1]),
                    },
                    "confidence": float(prob),
                    "embedding": embedding_np.tolist(),
                    "embedding_norm": float(np.linalg.norm(embedding_np)),
                }

                # Calculate face position relative to image
                face["position"] = {
                    "center_x": (face["bbox"]["x1"] + face["bbox"]["x2"]) / 2 / img.width,
                    "center_y": (face["bbox"]["y1"] + face["bbox"]["y2"]) / 2 / img.height,
                    "relative_size": (face["bbox"]["width"] * face["bbox"]["height"])
                    / (img.width * img.height),
                }

                faces.append(face)

            return {
                "status": "success",
                "faces": faces,
                "face_count": len(faces),
                "image_dimensions": {"width": img.width, "height": img.height},
            }

        except Exception as e:
            print(f"Error during CPU fallback extraction for {image_path}: {e}")
            return {"status": "error", "faces": [], "image_dimensions": img.size, "error": str(e)}
