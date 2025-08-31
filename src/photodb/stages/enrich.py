import os
import base64
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

import instructor
from instructor.batch import BatchProcessor, filter_successful, filter_errors
from anthropic import Anthropic
from anthropic.types import MessageParam

from .base import BaseStage
from ..database.models import Photo, LLMAnalysis
from ..database.repository import PhotoRepository
from ..models.photo_analysis import PhotoAnalysisResponse

logger = logging.getLogger(__name__)


class EnrichStage(BaseStage):
    """Stage 3: LLM-based photo analysis for enriched metadata."""

    def __init__(self, repository: PhotoRepository, config: dict):
        super().__init__(repository, config)
        self.stage_name = "enrich"

        # LLM configuration
        self.model_name = config.get("LLM_MODEL", "claude-sonnet-4-20250514")
        self.provider_name = config.get("LLM_PROVIDER", "anthropic")
        self.api_key = config.get("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        self.batch_size = int(config.get("BATCH_SIZE", 100))

        # Initialize Instructor client and batch processor
        self.api_available = self.api_key is not None
        if self.api_available:
            self.client = instructor.from_anthropic(Anthropic(api_key=self.api_key))
            self.batch_processor = BatchProcessor(
                f"anthropic/{self.model_name}", PhotoAnalysisResponse
            )
            self.system_prompt = self._load_system_prompt()
        else:
            logger.warning("No LLM API key found - will skip LLM analysis")
            self.client = None
            self.batch_processor = None
            self.system_prompt = None

    def _load_system_prompt(self) -> str:
        """Load the system prompt from file."""
        # Navigate to project root from src/photodb/stages/enrich.py
        system_prompt_file = (
            Path(__file__).parent.parent.parent.parent / "prompts" / "system_prompt.md"
        )
        try:
            with open(system_prompt_file, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            logger.warning(f"Could not load system prompt: {e}. Using fallback.")
            return "You are an expert photo archivist and computer-vision analyst. Extract factual, verifiable metadata from images with precision and transparency about uncertainty."

    def should_process(self, file_path: Path, force: bool = False) -> bool:
        """Check if a file should be processed for LLM analysis."""
        if force:
            return True

        photo = self.repository.get_photo_by_filename(str(file_path))
        if not photo:
            return False

        # Check if photo has normalized image and no existing analysis
        return photo.normalized_path != "" and not self.repository.has_llm_analysis(photo.id)

    def process_photo(self, photo: Photo, file_path: Path) -> bool:
        """Process individual photo for LLM analysis (synchronous fallback).

        Note: file_path parameter not used - we use the normalized image path instead.
        """
        try:
            start_time = time.time()

            # Check if API is available
            if not self.api_available:
                logger.debug(f"Skipping LLM analysis for {photo.id} - no API key")
                # Create placeholder analysis record to mark as processed
                placeholder_analysis = LLMAnalysis.create(
                    photo_id=photo.id,
                    model_name="placeholder",
                    analysis={"description": "API not available", "confidence": 0.0},
                    processing_duration_ms=0,
                )
                self.repository.create_llm_analysis(placeholder_analysis)
                return True

            # Get normalized image path
            img_path = self.config.get("IMG_PATH", "./photos/processed")
            normalized_path = Path(img_path) / f"{photo.id}.png"

            if not normalized_path.exists():
                logger.error(f"Normalized image not found: {normalized_path}")
                return False

            # Get existing metadata for context
            metadata = self.repository.get_metadata(photo.id)
            exif_context = self._build_exif_context(metadata) if metadata else ""

            # Analyze photo with LLM
            analysis_result = self._analyze_single_photo(normalized_path, exif_context)

            if not analysis_result:
                logger.error(f"LLM analysis failed for photo: {photo.id}")
                return False

            # Extract key fields for indexing
            extracted_fields = self._extract_key_fields(analysis_result)

            # Create and save analysis record
            processing_time = int((time.time() - start_time) * 1000)
            llm_analysis = LLMAnalysis.create(
                photo_id=photo.id,
                model_name=self.model_name,
                analysis=analysis_result,
                model_version=self.model_name.split("-")[-1] if "-" in self.model_name else None,
                processing_duration_ms=processing_time,
                **extracted_fields,
            )

            self.repository.create_llm_analysis(llm_analysis)
            logger.info(f"Successfully analyzed photo {photo.id} in {processing_time}ms")
            return True

        except Exception as e:
            logger.error(f"Error analyzing photo {photo.id}: {e}")
            # Save error record
            error_analysis = LLMAnalysis.create(
                photo_id=photo.id, model_name=self.model_name, analysis={}, error_message=str(e)
            )
            self.repository.create_llm_analysis(error_analysis)
            return False

    def process_batch(self, photos: List[Photo]) -> Optional[str]:
        """Submit a batch of photos for LLM analysis using Instructor's batch processing."""
        if not self.api_available or not self.batch_processor:
            logger.warning("Cannot process batch - no API key or batch processor available")
            return None

        logger.info(f"Using Instructor batch API for {len(photos)} photos")

        # Prepare message conversations for batch processing
        messages_list = []
        photo_ids = []

        for photo in photos:
            img_path = self.config.get("IMG_PATH", "./photos/processed")
            normalized_path = Path(img_path) / f"{photo.id}.png"

            if not normalized_path.exists():
                logger.warning(f"Skipping {photo.id} - normalized image not found")
                continue

            metadata = self.repository.get_metadata(photo.id)
            exif_context = self._build_exif_context(metadata) if metadata else ""

            # Create message content using shared method
            content = self._create_message_content(normalized_path, exif_context)

            # Build message conversation for this photo
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": content})

            messages_list.append(messages)
            photo_ids.append(photo.id)

        if not messages_list:
            return None

        try:
            # Create batch file using Instructor's batch processor
            batch_requests_dir = os.getenv("BATCH_REQUESTS_PATH", "./batch_requests")
            Path(batch_requests_dir).mkdir(parents=True, exist_ok=True)
            batch_file_path = Path(batch_requests_dir) / f"batch_requests_{int(time.time())}.jsonl"
            batch_file = self.batch_processor.create_batch_from_messages(
                messages_list=messages_list,
                file_path=str(batch_file_path),
                max_tokens=4000,
                temperature=0.1,
            )

            # Submit batch job
            batch_id = self.batch_processor.submit_batch(batch_file)

            # Create batch job record
            from ..database.models import BatchJob

            batch_job = BatchJob.create(provider_batch_id=batch_id, photo_count=len(photo_ids))
            batch_job.status = "submitted"
            self.repository.create_batch_job(batch_job)

            logger.info(f"Instructor batch submitted: {batch_id}")
            return batch_id

        except Exception as e:
            logger.error(f"Error submitting Instructor batch: {e}")
            return None

    # Note: _create_batch_request method removed - now using Instructor's native batch processing

    def monitor_batch(self, batch_id: str) -> Optional[dict]:
        """Monitor the status of a batch job using Instructor's batch processor."""
        if not self.api_available or not self.batch_processor:
            return None

        try:
            # Get batch status using Instructor's batch processor
            status = self.batch_processor.get_batch_status(batch_id)

            # Update our database record
            batch_job = self.repository.get_batch_job_by_provider_id(batch_id)
            if batch_job:
                # Map status to our internal status
                if status in ["in_progress", "validating", "finalizing"]:
                    batch_job.status = "processing"
                elif status == "completed":
                    batch_job.status = "completed"
                elif status in ["failed", "expired", "cancelled"]:
                    batch_job.status = "failed"

                if status in ["completed", "failed", "expired", "cancelled"]:
                    from datetime import datetime

                    batch_job.completed_at = datetime.now()

                    # Process results if batch is completed
                    if status == "completed":
                        processed_count = self._process_instructor_batch_results(batch_id)
                        batch_job.processed_count = processed_count
                        batch_job.failed_count = batch_job.photo_count - processed_count

                self.repository.update_batch_job(batch_job)

            return {
                "batch_id": batch_id,
                "status": status,
                "photo_count": batch_job.photo_count if batch_job else 0,
                "processed_count": batch_job.processed_count if batch_job else 0,
                "failed_count": batch_job.failed_count if batch_job else 0,
                "submitted_at": batch_job.submitted_at.isoformat() if batch_job else None,
                "completed_at": (
                    batch_job.completed_at.isoformat()
                    if batch_job and batch_job.completed_at
                    else None
                ),
            }

        except Exception as e:
            logger.error(f"Error monitoring Instructor batch {batch_id}: {e}")
            return None

    def _process_instructor_batch_results(self, batch_id: str) -> int:
        """Process completed Instructor batch results and save to database."""
        try:
            if not self.batch_processor:
                return 0

            # Retrieve results using Instructor's batch processor
            all_results = self.batch_processor.retrieve_results(batch_id)

            # Filter successful and error results
            successful_results = filter_successful(all_results)
            error_results = filter_errors(all_results)

            processed_count = 0

            # Process successful results
            for result in successful_results:
                custom_id = "unknown"
                try:
                    custom_id = result.custom_id or "unknown"

                    # Extract photo ID from custom_id (format: "photo_{photo_id}")
                    photo_id = None
                    if custom_id.startswith("photo_"):
                        photo_id = custom_id[6:]  # Remove "photo_" prefix

                    # Get structured response from Instructor result
                    analysis_data = result.result.model_dump() if result.result else {}

                    # If no photo_id from custom_id, try to get it from structured response
                    if not photo_id and result.result:
                        photo_id = result.result.image.id

                    if not photo_id:
                        logger.error(f"Instructor result not matched to photo for batch {batch_id}")
                        continue

                    # Extract key fields from structured data
                    extracted_fields = self._extract_key_fields(analysis_data)

                    # Create and save LLM analysis
                    llm_analysis = LLMAnalysis.create(
                        photo_id=photo_id,
                        model_name=self.model_name,
                        analysis=analysis_data,
                        batch_id=batch_id,
                        model_version=(
                            self.model_name.split("-")[-1] if "-" in self.model_name else None
                        ),
                        **extracted_fields,
                    )

                    self.repository.create_llm_analysis(llm_analysis)
                    processed_count += 1
                    logger.debug(f"Processed Instructor batch result for photo {photo_id}")

                except Exception as e:
                    logger.error(f"Error processing successful batch result for {custom_id}: {e}")
                    continue

            # Process error results
            for result in error_results:
                custom_id = "unknown"
                try:
                    custom_id = result.custom_id or "unknown"

                    # Extract photo ID from custom_id (format: "photo_{photo_id}")
                    if custom_id.startswith("photo_"):
                        photo_id = custom_id[6:]  # Remove "photo_" prefix

                        # Handle error response
                        error_msg = f"Batch processing failed: {getattr(result, 'error_message', 'Unknown error')}"

                        error_analysis = LLMAnalysis.create(
                            photo_id=photo_id,
                            model_name=self.model_name,
                            analysis={},
                            batch_id=batch_id,
                            error_message=error_msg,
                        )
                        self.repository.create_llm_analysis(error_analysis)

                except Exception as e:
                    logger.error(f"Error processing error batch result for {custom_id}: {e}")
                    continue

            logger.info(
                f"Processed {processed_count} successful results and {len(error_results)} errors from Instructor batch {batch_id}"
            )
            return processed_count

        except Exception as e:
            logger.error(f"Error processing Instructor batch results: {e}")
            return 0


    def _create_message_content(self, image_path: Path, exif_context: str) -> List[Dict[str, Any]]:
        """Create message content for photo analysis.

        Shared by both single photo analysis and batch processing.
        """
        # Read and encode image
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode()

        # Build prompt
        prompt = self._build_analysis_prompt(exif_context)

        return [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_data,
                },
            },
            {"type": "text", "text": prompt},
        ]

    def _analyze_single_photo(
        self, image_path: Path, exif_context: str
    ) -> Optional[Dict[str, Any]]:
        """Analyze a single photo using Instructor for structured output."""
        try:
            if not self.client:
                return None

            # Create message content using shared method
            content = self._create_message_content(image_path, exif_context)

            # Create user message - content is properly typed as list of content blocks
            user_message: MessageParam = {
                "role": "user",
                "content": content,  # type: ignore # Mixed content types from _create_message_content
            }

            # Use Instructor to get structured response
            # Note: type ignore due to Instructor's mixed OpenAI/Anthropic type hints
            response = self.client.messages.create(  # type: ignore
                model=self.model_name,
                max_tokens=4000,
                response_model=PhotoAnalysisResponse,
                system=self.system_prompt,  # Use system parameter instead of message
                messages=[user_message],  # type: ignore
            )

            # Convert Pydantic model to dict for database storage
            return response.model_dump() if response else None

        except Exception as e:
            logger.error(f"Structured LLM analysis failed: {e}")
            return None

    def _build_analysis_prompt(self, exif_context: str) -> str:
        """Build the analysis prompt using separate user prompt template."""

        # Load user prompt template
        # Navigate to project root from src/photodb/stages/enrich.py
        user_template_file = (
            Path(__file__).parent.parent.parent.parent / "prompts" / "user_prompt_template.md"
        )
        try:
            with open(user_template_file, "r", encoding="utf-8") as f:
                user_template = f.read().strip()

                # Replace EXIF context placeholder
                if exif_context:
                    exif_section = f"EXIF data available:\n{exif_context}"
                else:
                    exif_section = "No EXIF data available."

                user_prompt = user_template.replace("{EXIF_CONTEXT}", exif_section)
                return user_prompt

        except Exception as e:
            logger.warning(f"Could not load prompt template: {e}. Using fallback.")

            # Fallback prompt if template loading fails
            fallback_prompt = """Extract metadata from this image focusing on: people, activities, events, location cues, time/season cues, objects, text, and scene details.

Return structured data following the PhotoAnalysisResponse schema."""

            if exif_context:
                fallback_prompt += f"\n\nEXIF context:\n{exif_context}"

            return fallback_prompt

    def _build_exif_context(self, metadata) -> str:
        """Build context string from EXIF metadata."""
        context_parts = []

        if metadata.captured_at:
            context_parts.append(f"Captured: {metadata.captured_at}")

        if metadata.latitude and metadata.longitude:
            context_parts.append(f"Location: {metadata.latitude}, {metadata.longitude}")

        if metadata.extra:
            # Include key camera settings
            camera_info = []
            if "make" in metadata.extra:
                camera_info.append(f"Camera: {metadata.extra['make']}")
            if "model" in metadata.extra:
                camera_info.append(f"Model: {metadata.extra['model']}")
            if "f_number" in metadata.extra:
                camera_info.append(f"Aperture: f/{metadata.extra['f_number']}")
            if "exposure_time" in metadata.extra:
                camera_info.append(f"Shutter: {metadata.extra['exposure_time']}s")
            if "iso" in metadata.extra:
                camera_info.append(f"ISO: {metadata.extra['iso']}")

            if camera_info:
                context_parts.extend(camera_info)

        return "\n".join(context_parts)


    def _extract_key_fields(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key fields from structured analysis for database indexing."""
        extracted = {}

        # Extract description from scene info
        if "scene" in analysis:
            scene = analysis["scene"]
            if "short_caption" in scene:
                extracted["description"] = scene["short_caption"]
            elif "long_description" in scene:
                extracted["description"] = scene["long_description"]

        # Extract objects list
        if "objects" in analysis and "items" in analysis["objects"]:
            extracted["objects"] = [item["label"] for item in analysis["objects"]["items"]]

        # Extract people count
        if "people" in analysis and "count" in analysis["people"]:
            extracted["people_count"] = analysis["people"]["count"]

        # Extract location information
        if "location" in analysis:
            location = analysis["location"]
            if "environment" in location:
                extracted["location_description"] = location["environment"]
            # Add hypotheses if available
            if "hypotheses" in location and location["hypotheses"]:
                best_location = max(location["hypotheses"], key=lambda x: x.get("confidence", 0))
                if best_location["confidence"] > 0.5:
                    extracted["location_description"] = (
                        f"{location['environment']} - {best_location['value']}"
                    )

        # Extract activities/emotional context
        if "activities" in analysis:
            activities = analysis["activities"]
            if "event_hypotheses" in activities and activities["event_hypotheses"]:
                best_event = max(
                    activities["event_hypotheses"], key=lambda x: x.get("confidence", 0)
                )
                if best_event["confidence"] > 0.5:
                    extracted["emotional_tone"] = best_event["type"]

        # Extract overall confidence - use aesthetic score or a derived confidence
        confidence_score = 0.5  # default
        if "image" in analysis and "technical" in analysis["image"]:
            tech = analysis["image"]["technical"]
            if "quality" in tech and "aesthetic_score" in tech["quality"]:
                confidence_score = tech["quality"]["aesthetic_score"]

        extracted["confidence_score"] = confidence_score

        return extracted
