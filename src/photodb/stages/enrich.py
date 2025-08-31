import os
import base64
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

import instructor
from anthropic import Anthropic
from anthropic.types import MessageParam

from .base import BaseStage
from ..database.models import Photo, LLMAnalysis
from ..database.pg_repository import PostgresPhotoRepository
from ..models.photo_analysis import PhotoAnalysisResponse

logger = logging.getLogger(__name__)


class EnrichStage(BaseStage):
    """Stage 3: LLM-based photo analysis for enriched metadata."""

    def __init__(self, repository: PostgresPhotoRepository, config: dict):
        super().__init__(repository, config)
        self.stage_name = "enrich"

        # LLM configuration
        self.model_name = config.get("LLM_MODEL", "claude-sonnet-4-20250514")
        self.provider_name = config.get("LLM_PROVIDER", "anthropic")
        self.api_key = config.get("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        self.batch_size = int(config.get("BATCH_SIZE", 100))

        # Initialize Instructor client and load system prompt
        self.api_available = self.api_key is not None
        if self.api_available:
            self.client = instructor.from_anthropic(Anthropic(api_key=self.api_key))
            self.system_prompt = self._load_system_prompt()
        else:
            logger.warning("No LLM API key found - will skip LLM analysis")
            self.client = None
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
        """Submit a batch of photos for LLM analysis."""
        if not self.api_available:
            logger.warning("Cannot process batch - no API key available")
            return None

        try:
            logger.info(f"Processing batch of {len(photos)} photos")

            # Try to use Anthropic batch API with structured response support
            try:
                return self._process_batch_with_api(photos)
            except (AttributeError, ImportError):
                logger.info(
                    "Anthropic batch API not available, using grouped individual processing"
                )
                return self._process_batch_grouped(photos)

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return None

    def _process_batch_with_api(self, photos: List[Photo]) -> Optional[str]:
        """Process batch using Anthropic's batch API with structured response support.

        This method now supports structured responses by:
        1. Including system prompt and structured output instructions
        2. Processing batch results with Pydantic model validation
        3. Providing the same PhotoAnalysisResponse objects as individual processing

        The Anthropic batch API supports all Messages API parameters including system prompts.
        """
        from anthropic import Anthropic

        client = Anthropic(api_key=self.api_key)

        # Check if batch API is available
        if not hasattr(client, "messages") or not hasattr(client.messages, "batches"):
            raise AttributeError("Message Batches API not available in this SDK version")

        logger.info(f"Using Anthropic batch API for {len(photos)} photos")

        # Prepare batch requests
        requests = []
        for photo in photos:
            img_path = self.config.get("IMG_PATH", "./photos/processed")
            normalized_path = Path(img_path) / f"{photo.id}.png"

            if not normalized_path.exists():
                logger.warning(f"Skipping {photo.id} - normalized image not found")
                continue

            metadata = self.repository.get_metadata(photo.id)
            exif_context = self._build_exif_context(metadata) if metadata else ""

            custom_id = f"photo_{photo.id}"
            request = self._create_batch_request(custom_id, normalized_path, exif_context)
            requests.append(request)

        if not requests:
            return None

        # Create and submit batch using Message Batches API
        # Note: File upload is handled within _create_batch_input_file but not needed for Message Batches API

        # Convert our batch format to Message Batches API format
        batch_requests = []
        for request in requests:
            batch_requests.append({"custom_id": request["custom_id"], "params": request["body"]})

        batch = client.messages.batches.create(requests=batch_requests)

        # Create batch job record
        from ..database.models import BatchJob

        batch_job = BatchJob.create(provider_batch_id=batch.id, photo_count=len(requests))

        # Map Anthropic processing_status to our status
        if batch.processing_status == "in_progress":
            batch_job.status = "processing"
        elif batch.processing_status == "ended":
            batch_job.status = "completed"
        elif batch.processing_status == "canceling":
            batch_job.status = "failed"
        else:
            batch_job.status = "submitted"

        self.repository.create_batch_job(batch_job)

        logger.info(f"Anthropic batch submitted: {batch.id}")
        return batch.id

    def _process_batch_grouped(self, photos: List[Photo]) -> Optional[str]:
        """Process photos as a grouped batch using individual structured API calls.

        This method processes photos individually but tracks them as a batch.
        Each photo gets structured analysis using Instructor + Pydantic models.
        """
        logger.info(f"Using grouped individual processing for {len(photos)} photos")

        batch_id = f"grouped_batch_{int(time.time())}_{len(photos)}"

        # Create batch job record
        from ..database.models import BatchJob

        batch_job = BatchJob.create(provider_batch_id=batch_id, photo_count=len(photos))
        self.repository.create_batch_job(batch_job)

        # Process photos individually but track as a batch
        processed_count = 0
        failed_count = 0

        for photo in photos:
            try:
                # Get normalized image path
                img_path = self.config.get("IMG_PATH", "./photos/processed")
                normalized_path = Path(img_path) / f"{photo.id}.png"

                if not normalized_path.exists():
                    logger.warning(f"Skipping {photo.id} - normalized image not found")
                    failed_count += 1
                    continue

                # Get metadata context
                metadata = self.repository.get_metadata(photo.id)
                exif_context = self._build_exif_context(metadata) if metadata else ""

                # Analyze photo using structured format
                analysis_result = self._analyze_single_photo(normalized_path, exif_context)

                if analysis_result:
                    # Extract key fields from structured analysis
                    extracted_fields = self._extract_key_fields(analysis_result)

                    llm_analysis = LLMAnalysis.create(
                        photo_id=photo.id,
                        model_name=self.model_name,
                        analysis=analysis_result,
                        batch_id=batch_id,
                        model_version=(
                            self.model_name.split("-")[-1] if "-" in self.model_name else None
                        ),
                        **extracted_fields,
                    )

                    self.repository.create_llm_analysis(llm_analysis)
                    processed_count += 1
                else:
                    failed_count += 1

            except Exception as e:
                logger.error(f"Error processing photo {photo.id} in batch: {e}")
                failed_count += 1

        # Update batch job status
        from datetime import datetime

        batch_job.status = "completed"
        batch_job.completed_at = datetime.now()
        batch_job.processed_count = processed_count
        batch_job.failed_count = failed_count

        self.repository.update_batch_job(batch_job)

        logger.info(
            f"Grouped batch {batch_id} completed: {processed_count} processed, {failed_count} failed"
        )
        return batch_id

    def _create_batch_input_file(self, client, requests: List[Dict[str, Any]]) -> str:
        """Create input file for batch processing."""
        import tempfile

        # Create temporary file with batch requests
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for request in requests:
                f.write(json.dumps(request) + "\n")
            temp_file_path = f.name

        # Upload file to Anthropic using beta files API
        from pathlib import Path

        file_response = client.beta.files.upload(file=Path(temp_file_path))

        # Clean up temp file
        import os

        os.unlink(temp_file_path)

        return file_response.id

    def _create_batch_request(
        self, custom_id: str, image_path: Path, exif_context: str
    ) -> Dict[str, Any]:
        """Create a batch request for a single photo."""
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode()

        prompt = self._build_analysis_prompt(exif_context)

        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/messages",
            "body": {
                "model": self.model_name,
                "max_tokens": 4000,  # Increased for structured responses
                "system": self.system_prompt or "You are an expert photo analyst.",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_data,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            },
        }

    def _store_photo_mapping(self, batch_id: str, photo_mapping: Dict[str, Photo]):
        """Store mapping of custom_id to photo for batch result processing."""
        # Store in a temporary table or file for later retrieval
        # For now, we can reconstruct this from the photo IDs in custom_id
        logger.debug(f"Storing photo mapping for batch {batch_id}: {len(photo_mapping)} photos")
        pass

    def monitor_batch(self, batch_id: str) -> Optional[dict]:
        """Monitor the status of a batch job."""
        if not self.api_available:
            return None

        try:
            # Check if this is a real Anthropic batch or grouped batch
            if batch_id.startswith("grouped_batch_"):
                return self._monitor_grouped_batch(batch_id)
            else:
                return self._monitor_anthropic_batch(batch_id)

        except Exception as e:
            logger.error(f"Error monitoring batch {batch_id}: {e}")
            return None

    def _monitor_anthropic_batch(self, batch_id: str) -> Optional[dict]:
        """Monitor real Anthropic batch API job."""
        try:
            from anthropic import Anthropic

            client = Anthropic(api_key=self.api_key)

            # Check if batch API is available
            if not hasattr(client, "messages") or not hasattr(client.messages, "batches"):
                logger.warning("Message Batches API not available, cannot monitor Anthropic batch")
                return None

            # Get batch status from Anthropic
            batch = client.messages.batches.retrieve(batch_id)

            # Update our database record
            batch_job = self.repository.get_batch_job_by_provider_id(batch_id)
            if batch_job:
                # Map Anthropic processing_status to our status
                if batch.processing_status == "in_progress":
                    batch_job.status = "processing"
                elif batch.processing_status == "ended":
                    batch_job.status = "completed"
                elif batch.processing_status == "canceling":
                    batch_job.status = "failed"

                if batch.processing_status in ["ended", "canceling"]:
                    from datetime import datetime

                    batch_job.completed_at = datetime.now()

                    # Process results if batch is completed
                    if batch.processing_status == "ended":
                        processed_count = self._process_batch_results(batch)
                        batch_job.processed_count = processed_count
                        batch_job.failed_count = batch_job.photo_count - processed_count

                self.repository.update_batch_job(batch_job)

            return {
                "batch_id": batch_id,
                "status": batch.processing_status,
                "photo_count": batch_job.photo_count if batch_job else 0,
                "processed_count": batch_job.processed_count if batch_job else 0,
                "failed_count": batch_job.failed_count if batch_job else 0,
                "submitted_at": batch_job.submitted_at.isoformat() if batch_job else None,
                "completed_at": (
                    batch_job.completed_at.isoformat()
                    if batch_job and batch_job.completed_at
                    else None
                ),
                "anthropic_status": {
                    "request_counts": getattr(batch, "request_counts", {}),
                    "ended_at": getattr(batch, "ended_at", None),
                    "created_at": getattr(batch, "created_at", None),
                },
            }

        except Exception as e:
            logger.error(f"Error monitoring Anthropic batch {batch_id}: {e}")
            return None

    def _monitor_grouped_batch(self, batch_id: str) -> Optional[dict]:
        """Monitor grouped batch (already completed in process_batch)."""
        batch_job = self.repository.get_batch_job_by_provider_id(batch_id)
        if not batch_job:
            return None

        return {
            "batch_id": batch_id,
            "status": batch_job.status,
            "photo_count": batch_job.photo_count,
            "processed_count": batch_job.processed_count,
            "failed_count": batch_job.failed_count,
            "submitted_at": batch_job.submitted_at.isoformat(),
            "completed_at": batch_job.completed_at.isoformat() if batch_job.completed_at else None,
        }

    def _process_batch_results(self, batch) -> int:
        """Process completed batch results and save to database."""
        try:
            from anthropic import Anthropic
            import json

            client = Anthropic(api_key=self.api_key)

            processed_count = 0

            # Get results using Message Batches API results() method
            result_stream = client.messages.batches.results(batch.id)

            # Process each result
            for entry in result_stream:
                custom_id = "unknown"
                try:
                    custom_id = entry.custom_id

                    # Extract photo ID from custom_id (format: "photo_{photo_id}")
                    if custom_id.startswith("photo_"):
                        photo_id = custom_id[6:]  # Remove "photo_" prefix

                        # Check if the request was successful
                        if entry.result.type == "succeeded":
                            # Get the message response
                            message_response = entry.result.message
                            analysis_text = None

                            if message_response.content:
                                # Get text content from the message
                                for content_block in message_response.content:
                                    # Check if this is a text content block
                                    if (
                                        hasattr(content_block, "type")
                                        and content_block.type == "text"
                                    ):
                                        analysis_text = content_block.text
                                        break

                            if analysis_text:
                                # Parse analysis result with structured validation
                                try:
                                    # Parse JSON, handling potential markdown wrapper
                                    analysis_json = self._parse_json_with_fallback(analysis_text)

                                    # Validate with Pydantic model for structured output
                                    structured_response = PhotoAnalysisResponse.model_validate(
                                        analysis_json
                                    )
                                    analysis_data = structured_response.model_dump()

                                    # Extract key fields from structured data
                                    extracted_fields = self._extract_key_fields(analysis_data)

                                    # Create and save LLM analysis
                                    llm_analysis = LLMAnalysis.create(
                                        photo_id=photo_id,
                                        model_name=self.model_name,
                                        analysis=analysis_data,
                                        batch_id=batch.id,
                                        model_version=(
                                            self.model_name.split("-")[-1]
                                            if "-" in self.model_name
                                            else None
                                        ),
                                        **extracted_fields,
                                    )

                                    self.repository.create_llm_analysis(llm_analysis)
                                    processed_count += 1
                                    logger.debug(
                                        f"Processed structured batch result for photo {photo_id}"
                                    )

                                except (json.JSONDecodeError, ValueError) as e:
                                    logger.warning(
                                        f"Failed to parse structured response for {photo_id}: {e}"
                                    )
                                    # Save as fallback analysis if structured parsing fails
                                    llm_analysis = LLMAnalysis.create(
                                        photo_id=photo_id,
                                        model_name=self.model_name,
                                        analysis={
                                            "description": analysis_text,
                                            "confidence": 0.8,
                                            "parsing_error": str(e),
                                        },
                                        batch_id=batch.id,
                                        error_message=f"Structured parsing failed: {e}",
                                    )
                                    self.repository.create_llm_analysis(llm_analysis)
                                    processed_count += 1
                        else:
                            # Handle error response based on result type
                            error_msg = f"Batch processing failed: {entry.result.type}"
                            if entry.result.type == "errored" and hasattr(entry.result, "error"):
                                error_msg = f"API error: {entry.result.error}"
                            elif entry.result.type == "canceled":
                                error_msg = "Request was canceled"
                            elif entry.result.type == "expired":
                                error_msg = "Request expired"

                            error_analysis = LLMAnalysis.create(
                                photo_id=photo_id,
                                model_name=self.model_name,
                                analysis={},
                                batch_id=batch.id,
                                error_message=error_msg,
                            )
                            self.repository.create_llm_analysis(error_analysis)

                except Exception as e:
                    logger.error(f"Error processing batch result for {custom_id}: {e}")
                    continue

            logger.info(f"Processed {processed_count} results from batch {batch.id}")
            return processed_count

        except Exception as e:
            logger.error(f"Error processing batch results: {e}")
            return 0

    def _analyze_single_photo(
        self, image_path: Path, exif_context: str
    ) -> Optional[Dict[str, Any]]:
        """Analyze a single photo using Instructor for structured output."""
        try:
            if not self.client:
                return None

            # Read and encode image
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode()

            # Build prompt with structured output request
            prompt = self._build_analysis_prompt(exif_context)

            # Create user message
            user_message: MessageParam = {  # type: ignore
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": image_data},
                    },
                    {"type": "text", "text": prompt},
                ],
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

    def _parse_json_with_fallback(self, text: str) -> Dict[str, Any]:
        """Parse JSON text, handling potential markdown code block wrapper.

        Args:
            text: Raw text that may contain JSON, possibly wrapped in ```json ... ```

        Returns:
            Parsed JSON as dictionary

        Raises:
            json.JSONDecodeError: If parsing fails even after unwrapping attempt
        """
        try:
            # Try direct JSON parsing first
            return json.loads(text)
        except json.JSONDecodeError:
            # Check if wrapped in ```json ... ``` markers
            if "```json" in text and "```" in text:
                import re

                json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
                if json_match:
                    cleaned_text = json_match.group(1).strip()
                    return json.loads(cleaned_text)
            # Re-raise the original error if no markdown wrapper found
            raise

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
