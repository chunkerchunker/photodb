"""
Scene analysis stage: Taxonomy and prompt-based tagging.

Uses Apple Vision for scene taxonomy (macOS) and MobileCLIP with
configurable prompts for scene and face tagging.
"""

import logging
import sys
from pathlib import Path
from typing import List

from .base import BaseStage
from .. import config as defaults
from ..database.models import (
    AnalysisOutput,
    DetectionTag,
    Photo,
    PhotoTag,
    SceneAnalysis,
)
from ..utils.mobileclip_analyzer import MobileCLIPAnalyzer
from ..utils.prompt_cache import PromptCache

logger = logging.getLogger(__name__)

_apple_vision_available = sys.platform == "darwin"
AppleVisionClassifier = None  # Type hint for conditional import
if _apple_vision_available:
    try:
        from ..utils.apple_vision_classifier import AppleVisionClassifier
    except ImportError:
        _apple_vision_available = False
        logger.warning("Apple Vision import failed - not available")


class SceneAnalysisStage(BaseStage):
    """Stage for scene taxonomy and prompt-based tagging."""

    def __init__(self, repository, config: dict):
        super().__init__(repository, config)
        self.stage_name = "scene_analysis"

        self.analyzer = MobileCLIPAnalyzer()
        self.analyzer.warmup()
        self.prompt_cache = PromptCache(repository, device=self.analyzer.device)

        if _apple_vision_available and AppleVisionClassifier is not None:
            self.apple_classifier = AppleVisionClassifier()
            self.apple_classifier.warmup()
        else:
            self.apple_classifier = None
            logger.warning("Apple Vision not available (not macOS or import failed)")

        # Load category configs
        self.scene_categories = repository.get_prompt_categories(target="scene")
        self.face_categories = repository.get_prompt_categories(target="face")

        logger.info(
            f"SceneAnalysisStage initialized: "
            f"{len(self.scene_categories)} scene categories, "
            f"{len(self.face_categories)} face categories"
        )

    def process_photo(self, photo: Photo, file_path: Path) -> bool:
        """Process scene taxonomy and tagging for a photo."""
        if photo.id is None:
            logger.error(f"Photo {file_path} has no ID")
            return False

        photo_id = photo.id  # Capture for type narrowing

        try:
            if not photo.med_path:
                logger.warning(f"No medium path for photo {photo_id}")
                return False

            normalized_path = Path(self.config["IMG_PATH"]) / photo.med_path
            if not normalized_path.exists():
                logger.error(f"Normalized image not found: {normalized_path}")
                return False

            # 1. Apple Vision taxonomy (macOS only)
            taxonomy_output_id = None
            taxonomy_labels = None
            taxonomy_confidences = None

            if self.apple_classifier:
                taxonomy_result = self.apple_classifier.classify(
                    str(normalized_path), top_k=defaults.APPLE_VISION_TOP_K
                )

                taxonomy_output = AnalysisOutput.create(
                    photo_id=photo_id,
                    model_type="classifier",
                    model_name="apple_vision_classify",
                    output=taxonomy_result,
                    processing_time_ms=taxonomy_result.get("processing_time_ms"),
                    device="ane",
                )
                self.repository.create_analysis_output(taxonomy_output)
                taxonomy_output_id = taxonomy_output.id

                if taxonomy_result["status"] == "success":
                    taxonomy_labels = [c["identifier"] for c in taxonomy_result["classifications"]]
                    taxonomy_confidences = [
                        c["confidence"] for c in taxonomy_result["classifications"]
                    ]

            # 2. Encode image with MobileCLIP
            image_embedding = self.analyzer.encode_image(str(normalized_path))

            # 3. Classify against all scene categories
            all_photo_tags: List[PhotoTag] = []
            scene_results = {}

            for category in self.scene_categories:
                try:
                    if category.selection_mode == "single":
                        scores = self.prompt_cache.classify(image_embedding, category.name)
                        # Get top result
                        top_label = max(scores, key=lambda k: scores[k])
                        top_score = scores[top_label]

                        if top_score >= category.min_confidence:
                            prompt_ids = self.prompt_cache.get_prompt_ids(category.name)
                            labels, _, _ = self.prompt_cache.get_category(category.name)
                            idx = labels.index(top_label)

                            all_photo_tags.append(
                                PhotoTag.create(
                                    photo_id=photo_id,
                                    prompt_id=prompt_ids[idx],
                                    confidence=top_score,
                                    rank_in_category=1,
                                )
                            )
                        scene_results[category.name] = scores
                    else:
                        # Multi-select
                        results = self.prompt_cache.classify_multi(image_embedding, category.name)
                        for rank, (label, conf, prompt_id) in enumerate(results, 1):
                            all_photo_tags.append(
                                PhotoTag.create(
                                    photo_id=photo_id,
                                    prompt_id=prompt_id,
                                    confidence=conf,
                                    rank_in_category=rank,
                                )
                            )
                        scene_results[category.name] = {r[0]: r[1] for r in results}
                except Exception as e:
                    logger.warning(f"Failed to classify category {category.name}: {e}")

            # 4. Store scene analysis output
            mobileclip_output = AnalysisOutput.create(
                photo_id=photo_id,
                model_type="tagger",
                model_name="mobileclip",
                output={"scene": scene_results},
                device=self.analyzer.device,
            )
            self.repository.create_analysis_output(mobileclip_output)

            # 5. Save photo tags
            if all_photo_tags:
                self.repository.bulk_upsert_photo_tags(all_photo_tags)

            # 6. Save scene analysis record
            scene_analysis = SceneAnalysis.create(
                photo_id=photo_id,
                taxonomy_labels=taxonomy_labels,
                taxonomy_confidences=taxonomy_confidences,
                taxonomy_output_id=taxonomy_output_id,
                mobileclip_output_id=mobileclip_output.id,
            )
            self.repository.upsert_scene_analysis(scene_analysis)

            # 7. Process face tags if detections exist
            detections = self.repository.get_detections_for_photo(photo_id)
            face_detections = [d for d in detections if d.has_face()]

            if face_detections and self.face_categories:
                self._process_face_tags(photo_id, normalized_path, face_detections)

            logger.info(
                f"Scene analysis complete for {file_path}: "
                f"{len(all_photo_tags)} scene tags, "
                f"{len(face_detections)} faces"
            )
            return True

        except Exception as e:
            logger.error(f"Scene analysis failed for {file_path}: {e}")
            return False

    def _process_face_tags(self, photo_id: int, image_path: Path, detections) -> None:
        """Tag each face detection with face categories."""
        # Build bboxes
        bboxes = []
        for det in detections:
            bboxes.append(
                {
                    "x1": det.face_bbox_x,
                    "y1": det.face_bbox_y,
                    "x2": det.face_bbox_x + det.face_bbox_width,
                    "y2": det.face_bbox_y + det.face_bbox_height,
                }
            )

        # Batch encode faces
        face_embeddings = self.analyzer.encode_faces_batch(str(image_path), bboxes)

        if face_embeddings.shape[0] == 0:
            return

        # Store face analysis output
        face_results = {}
        all_detection_tags: List[DetectionTag] = []

        for det, face_emb in zip(detections, face_embeddings):
            face_emb = face_emb.unsqueeze(0)
            face_results[det.id] = {}

            for category in self.face_categories:
                try:
                    if category.selection_mode == "single":
                        scores = self.prompt_cache.classify(face_emb, category.name)
                        top_label = max(scores, key=lambda k: scores[k])
                        top_score = scores[top_label]

                        if top_score >= category.min_confidence:
                            prompt_ids = self.prompt_cache.get_prompt_ids(category.name)
                            labels, _, _ = self.prompt_cache.get_category(category.name)
                            idx = labels.index(top_label)

                            all_detection_tags.append(
                                DetectionTag.create(
                                    detection_id=det.id,
                                    prompt_id=prompt_ids[idx],
                                    confidence=top_score,
                                    rank_in_category=1,
                                )
                            )
                        face_results[det.id][category.name] = scores
                    else:
                        results = self.prompt_cache.classify_multi(face_emb, category.name)
                        for rank, (label, conf, prompt_id) in enumerate(results, 1):
                            all_detection_tags.append(
                                DetectionTag.create(
                                    detection_id=det.id,
                                    prompt_id=prompt_id,
                                    confidence=conf,
                                    rank_in_category=rank,
                                )
                            )
                        face_results[det.id][category.name] = {r[0]: r[1] for r in results}
                except Exception as e:
                    logger.warning(f"Failed face category {category.name}: {e}")

        # Save tags
        if all_detection_tags:
            self.repository.bulk_upsert_detection_tags(all_detection_tags)

        # Store output
        face_output = AnalysisOutput.create(
            photo_id=photo_id,
            model_type="tagger",
            model_name="mobileclip",
            output={"faces": face_results},
            device=self.analyzer.device,
        )
        self.repository.create_analysis_output(face_output)
