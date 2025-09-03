import json
import os
import base64
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

import instructor
from instructor.batch import BatchProcessor, filter_successful, filter_errors

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
        self.provider_name = config.get("LLM_PROVIDER", "anthropic").lower()
        self.batch_size = int(config.get("BATCH_SIZE", 100))

        # Initialize Instructor client based on provider
        self.client = None
        self.batch_processor = None
        self.api_available = False
        self.model_name = None

        try:
            if self.provider_name == "bedrock":
                # Bedrock configuration
                import boto3
                from botocore.config import Config

                self.model_name = config.get(
                    "BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0"
                )
                aws_region = config.get("AWS_REGION", os.getenv("AWS_REGION", "us-east-1"))
                aws_profile = config.get("AWS_PROFILE", os.getenv("AWS_PROFILE"))

                session_kwargs = {}
                if aws_profile:
                    session_kwargs["profile_name"] = aws_profile

                session = boto3.Session(**session_kwargs)
                bedrock_config = Config(
                    region_name=aws_region, retries={"max_attempts": 3, "mode": "adaptive"}
                )

                # Store the bedrock client directly (instructor doesn't support images yet)
                self.bedrock_client = session.client(
                    service_name="bedrock-runtime", config=bedrock_config
                )
                # Also create bedrock client for batch operations
                self.bedrock_control_client = session.client(
                    service_name="bedrock", config=bedrock_config
                )

                # Don't use instructor for Bedrock images - use native API
                self.client = None  # Will use native bedrock API

                # Validate Bedrock setup
                if self._validate_bedrock_setup():
                    self.api_available = True
                    logger.info(f"Initialized Bedrock client with model {self.model_name}")
                else:
                    self.api_available = False
                    logger.error("Bedrock setup validation failed")

            else:  # anthropic provider
                from anthropic import Anthropic

                self.model_name = config.get("LLM_MODEL", "claude-3-5-sonnet-20241022")
                api_key = config.get("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY")

                if api_key:
                    anthropic_client = Anthropic(api_key=api_key)
                    self.client = instructor.from_anthropic(anthropic_client)
                    self.batch_processor = BatchProcessor(
                        f"anthropic/{self.model_name}", PhotoAnalysisResponse
                    )
                    self.api_available = True
                    logger.info(f"Initialized Anthropic client with model {self.model_name}")
                else:
                    logger.warning("No Anthropic API key found - will skip LLM analysis")

        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}", exc_info=True)
            self.api_available = False

        # Load system prompt if API is available
        if self.api_available:
            self.system_prompt = self._load_system_prompt()
        else:
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
            normalized_path = Path(photo.normalized_path)

            if not normalized_path.exists():
                logger.error(f"Normalized image not found: {normalized_path}")
                return False

            # Get existing metadata for context
            metadata = self.repository.get_metadata(photo.id)
            exif_context = self._build_exif_context(metadata) if metadata else ""

            # Analyze photo with LLM
            try:
                analysis_result = self._analyze_single_photo(normalized_path, exif_context)

                if not analysis_result:
                    logger.error(
                        f"LLM analysis returned None for photo: {photo.id} - check previous error logs"
                    )
                    return False

            except Exception as e:
                logger.error(
                    f"Exception during LLM analysis for photo {photo.id}: {e}", exc_info=True
                )
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
            logger.error(f"Error analyzing photo {photo.id}: {e}", exc_info=True)
            # Save error record
            error_analysis = LLMAnalysis.create(
                photo_id=photo.id, model_name=self.model_name, analysis={}, error_message=str(e)
            )
            self.repository.create_llm_analysis(error_analysis)
            return False

    def process_batch(self, photos: List[Photo]) -> Optional[str]:
        """Submit a batch of photos for LLM analysis using Instructor's batch processing."""
        if not self.api_available:
            logger.warning("Cannot process batch - API not available")
            return None

        # Handle batch processing based on provider
        if self.provider_name == "anthropic":
            if not self.batch_processor:
                logger.warning("Anthropic batch processor not available")
                return None
            return self._process_anthropic_batch(photos)
        elif self.provider_name == "bedrock":
            return self._process_bedrock_batch(photos)
        else:
            logger.info(f"Batch processing not supported for {self.provider_name} provider")
            return None

    def _process_anthropic_batch(self, photos: List[Photo]) -> Optional[str]:
        """Process batch using Anthropic's batch API via instructor."""
        logger.info(f"Using Instructor batch API for {len(photos)} photos")

        # Prepare message conversations for batch processing
        messages_list = []
        photo_ids = []

        for photo in photos:
            normalized_path = Path(photo.normalized_path)

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

            batch_job = BatchJob.create(
                provider_batch_id=batch_id, photo_ids=photo_ids, model_name=self.model_name
            )
            batch_job.status = "submitted"
            self.repository.create_batch_job(batch_job)

            logger.info(f"Instructor batch submitted: {batch_id}")
            return batch_id

        except Exception as e:
            logger.error(f"Error submitting Instructor batch: {e}")
            return None

    def _process_bedrock_batch(self, photos: List[Photo]) -> Optional[str]:
        """Process batch using Bedrock's native batch inference API."""
        logger.info(f"Using Bedrock batch inference API for {len(photos)} photos")

        try:
            # Create input data file for Bedrock batch inference
            batch_requests_dir = os.getenv("BATCH_REQUESTS_PATH", "./batch_requests")
            Path(batch_requests_dir).mkdir(parents=True, exist_ok=True)

            # Generate unique identifier for this batch (Bedrock allows: [a-zA-Z0-9-+.])
            timestamp = int(time.time())
            batch_name = f"photodb-batch-{timestamp}"
            input_file_path = Path(batch_requests_dir) / f"{batch_name}-input.jsonl"

            # Prepare batch input data
            photo_ids = []
            with open(input_file_path, "w") as f:
                for i, photo in enumerate(photos):
                    normalized_path = Path(photo.normalized_path)

                    if not normalized_path.exists():
                        logger.warning(f"Skipping {photo.id} - normalized image not found")
                        continue

                    metadata = self.repository.get_metadata(photo.id)
                    exif_context = self._build_exif_context(metadata) if metadata else ""

                    # Create message content
                    content = self._create_message_content(normalized_path, exif_context)

                    # Bedrock batch request format
                    request = {
                        "recordId": f"photo_{i}",
                        "modelInput": {
                            "anthropic_version": "bedrock-2023-05-31",
                            "max_tokens": 4000,
                            "temperature": 0.1,
                            "system": self.system_prompt,
                            "messages": [{"role": "user", "content": content}],
                        },
                    }

                    f.write(json.dumps(request) + "\n")
                    photo_ids.append(photo.id)

            if not photo_ids:
                logger.warning("No valid photos to process in batch")
                return None

            # Check required configuration
            s3_bucket = self.config.get("BEDROCK_BATCH_S3_BUCKET")
            role_arn = self.config.get("BEDROCK_BATCH_ROLE_ARN")

            if not s3_bucket:
                logger.error("BEDROCK_BATCH_S3_BUCKET not configured")
                return None

            if not role_arn:
                logger.error("BEDROCK_BATCH_ROLE_ARN not configured")
                return None

            import boto3

            s3_client = boto3.client("s3")
            input_s3_key = f"batch-input/{batch_name}-input.jsonl"
            output_s3_prefix = f"batch-output/{batch_name}/"

            # Upload input file
            s3_client.upload_file(str(input_file_path), s3_bucket, input_s3_key)

            # Use the stored bedrock control client for batch operations
            bedrock_client = self.bedrock_control_client

            job_response = bedrock_client.create_model_invocation_job(
                jobName=batch_name,
                modelId=self.model_name,
                roleArn=role_arn,
                inputDataConfig={
                    "s3InputDataConfig": {"s3Uri": f"s3://{s3_bucket}/{input_s3_key}"}
                },
                outputDataConfig={
                    "s3OutputDataConfig": {"s3Uri": f"s3://{s3_bucket}/{output_s3_prefix}"}
                },
            )

            job_arn = job_response["jobArn"]

            # Create batch job record
            from ..database.models import BatchJob

            batch_job = BatchJob.create(
                provider_batch_id=job_arn, photo_ids=photo_ids, model_name=self.model_name
            )
            batch_job.status = "submitted"
            self.repository.create_batch_job(batch_job)

            logger.info(f"Bedrock batch job submitted: {job_arn}")
            return job_arn

        except Exception as e:
            logger.error(f"Error submitting Bedrock batch job: {e}")
            return None

    # Note: _create_batch_request method removed - now using Instructor's native batch processing

    def monitor_batch(self, batch_id: str) -> Optional[dict]:
        """Monitor the status of a batch job based on provider."""
        if not self.api_available:
            return None

        # Determine provider based on batch_id format
        if batch_id.startswith("arn:aws:bedrock"):
            return self._monitor_bedrock_batch(batch_id)
        else:
            return self._monitor_anthropic_batch(batch_id)

    def _monitor_anthropic_batch(self, batch_id: str) -> Optional[dict]:
        """Monitor Anthropic batch job using Instructor's batch processor."""
        if not self.batch_processor:
            return None

        try:
            # Get batch status using Instructor's batch processor
            status = self.batch_processor.get_batch_status(batch_id)["status"]

            # Update our database record
            batch_job = self.repository.get_batch_job_by_provider_id(batch_id)
            if batch_job:
                # Map status to our internal status
                if status in ["in_progress", "validating", "finalizing"]:
                    batch_job.status = "processing"
                elif status in ["completed", "ended", "JOB_STATE_SUCCEEDED"]:
                    batch_job.status = "completed"
                elif status in ["failed", "expired", "cancelled"]:
                    batch_job.status = "failed"

                if batch_job.status in ["completed", "failed"]:
                    from datetime import datetime

                    batch_job.completed_at = datetime.now()

                    # Process results if batch is completed
                    if batch_job.status == "completed":
                        processed_count = self._process_instructor_batch_results(
                            batch_id, batch_job.photo_ids
                        )
                        batch_job.processed_count = processed_count
                        batch_job.failed_count = batch_job.photo_count - processed_count

                        # Extract and save token usage and cost information
                        usage_data = self._extract_usage_from_raw_results(batch_id)
                        total_usage = usage_data.get("total_usage", {})

                        if total_usage:
                            batch_job.total_input_tokens = total_usage.get("input_tokens", 0)
                            batch_job.total_output_tokens = total_usage.get("output_tokens", 0)
                            batch_job.total_cache_creation_tokens = total_usage.get(
                                "cache_creation_tokens", 0
                            )
                            batch_job.total_cache_read_tokens = total_usage.get(
                                "cache_read_tokens", 0
                            )

                            # Calculate cost
                            batch_job.actual_cost_cents = self._calculate_batch_cost(
                                total_usage, self.model_name, batch_discount=True
                            )

                            logger.info(
                                f"Batch {batch_id} completed: "
                                f"Input tokens: {batch_job.total_input_tokens:,}, "
                                f"Output tokens: {batch_job.total_output_tokens:,}, "
                                f"Cost: ${batch_job.actual_cost_cents / 100:.2f}"
                            )

                self.repository.update_batch_job(batch_job)

            return {
                "batch_id": batch_id,
                "status": batch_job.status if batch_job else "processing",
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

    def _monitor_bedrock_batch(self, job_arn: str) -> Optional[dict]:
        """Monitor Bedrock batch job using native API."""
        try:
            # Use the stored bedrock control client
            bedrock_client = self.bedrock_control_client

            # Get job status
            response = bedrock_client.get_model_invocation_job(jobIdentifier=job_arn)
            job_status = response["status"]

            # Update our database record
            batch_job = self.repository.get_batch_job_by_provider_id(job_arn)
            if batch_job:
                # Map Bedrock status to our internal status
                if job_status in ["InProgress", "Validating"]:
                    batch_job.status = "processing"
                elif job_status in ["Completed", "PartiallyCompleted"]:
                    batch_job.status = "completed"
                elif job_status in ["Failed", "Stopped"]:
                    batch_job.status = "failed"
                elif job_status == "Submitted":
                    batch_job.status = "submitted"

                if batch_job.status in ["completed", "failed"]:
                    from datetime import datetime

                    batch_job.completed_at = datetime.now()

                    # Process results if completed
                    if batch_job.status == "completed":
                        processed_count = self._process_bedrock_batch_results(
                            job_arn, batch_job.photo_ids, response
                        )
                        batch_job.processed_count = processed_count
                        batch_job.failed_count = batch_job.photo_count - processed_count

                        logger.info(
                            f"Bedrock batch {job_arn} completed: Processed {processed_count} photos"
                        )

                self.repository.update_batch_job(batch_job)

            return {
                "batch_id": job_arn,
                "status": batch_job.status if batch_job else job_status.lower(),
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
            logger.error(f"Error monitoring Bedrock batch {job_arn}: {e}")
            return None

    def _process_bedrock_batch_results(
        self, job_arn: str, photo_ids: List[str], job_response: dict
    ) -> int:
        """Process completed Bedrock batch results."""
        try:
            import boto3

            # Download results from S3
            s3_bucket = self.config.get("BEDROCK_BATCH_S3_BUCKET")
            if not s3_bucket:
                logger.error("BEDROCK_BATCH_S3_BUCKET not configured")
                return 0

            s3_client = boto3.client("s3")

            # Extract job name from ARN for S3 path
            # The ARN format should be: arn:aws:bedrock:region:account:model-invocation-job/job-name
            logger.debug(f"Batch job ARN: {job_arn}")
            job_name = job_arn.split("/")[-1]
            logger.debug(f"Extracted job name: {job_name}")

            # The output path should match what we used in job creation
            # We used: f"batch-output/{batch_name}/" where batch_name was f"photodb-batch-{timestamp}"
            # But the job name in the ARN might be different, so let's try both patterns
            output_prefix = f"batch-output/{job_name}/"
            logger.debug(f"S3 output prefix: {output_prefix}")

            # List output files - try the expected path first
            logger.debug(f"Looking for output files with prefix: {output_prefix}")
            response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=output_prefix)

            processed_count = 0

            if "Contents" in response:
                logger.info(
                    f"Found {len(response['Contents'])} output files with prefix {output_prefix}"
                )
                for obj_info in response["Contents"]:
                    logger.debug(f"Output file: {obj_info['Key']} (size: {obj_info['Size']})")
            else:
                # If not found, try to see what's actually in the bucket
                logger.warning(f"No output files found with prefix {output_prefix}")
                logger.debug("Listing all objects in batch-output/ to see what's there...")

                try:
                    all_output_response = s3_client.list_objects_v2(
                        Bucket=s3_bucket, Prefix="batch-output/"
                    )

                    if "Contents" in all_output_response:
                        logger.info(
                            f"Found {len(all_output_response['Contents'])} files in batch-output/:"
                        )
                        for obj_info in all_output_response["Contents"]:
                            logger.info(f"  - {obj_info['Key']} (size: {obj_info['Size']})")

                        # Try to find files that match our job name pattern
                        # Bedrock output files end with .out, not .jsonl
                        matching_files = []
                        for obj in all_output_response["Contents"]:
                            # Look for files containing the job name and ending with .jsonl.out
                            if (
                                job_name in obj["Key"]
                                and (
                                    obj["Key"].endswith(".jsonl.out") or obj["Key"].endswith(".out")
                                )
                                and obj["Size"] > 0
                            ):  # Skip empty files/directories
                                logger.info(f"Found matching output file: {obj['Key']}")
                                matching_files.append(obj)

                        if matching_files:
                            response = {"Contents": matching_files}
                            logger.info(f"Will process {len(matching_files)} output files")
                        else:
                            logger.error(f"No matching .out files found for job {job_name}")
                    else:
                        logger.error("No files found in batch-output/ at all")
                        # Try listing the entire bucket to see what's there
                        logger.debug("Listing entire bucket to see available files...")
                        try:
                            all_bucket_response = s3_client.list_objects_v2(Bucket=s3_bucket)
                            if "Contents" in all_bucket_response:
                                logger.info(
                                    f"Found {len(all_bucket_response['Contents'])} total files in bucket:"
                                )
                                for obj_info in all_bucket_response["Contents"][
                                    :20
                                ]:  # Show first 20 files
                                    logger.info(f"  - {obj_info['Key']} (size: {obj_info['Size']})")
                                if len(all_bucket_response["Contents"]) > 20:
                                    logger.info(
                                        f"  ... and {len(all_bucket_response['Contents']) - 20} more files"
                                    )
                            else:
                                logger.error("Bucket is completely empty")
                        except Exception as e:
                            logger.error(f"Error listing entire bucket: {e}")
                        return 0

                except Exception as e:
                    logger.error(f"Error listing bucket contents: {e}")
                    return 0

                if "Contents" not in response:
                    logger.error(f"Still no output files found for job {job_name}")
                    return 0

            if "Contents" in response:
                for obj in response["Contents"]:
                    # Process both .jsonl and .out files (Bedrock batch outputs use .out extension)
                    if obj["Key"].endswith(".jsonl") or obj["Key"].endswith(".out"):
                        logger.debug(f"Processing result file: {obj['Key']}")
                        # Download and process each result file
                        local_path = f"/tmp/{os.path.basename(obj['Key'])}"
                        s3_client.download_file(s3_bucket, obj["Key"], local_path)

                        # Process results
                        logger.debug(f"Downloaded to {local_path}, reading file")
                        with open(local_path, "r") as f:
                            line_count = 0
                            for line in f:
                                line_count += 1
                                if not line.strip():
                                    continue

                                try:
                                    logger.debug(f"Processing line {line_count}: {line[:100]}...")
                                    result = json.loads(line)
                                    record_id = result.get("recordId")
                                    logger.debug(f"Record ID: {record_id}")

                                    # Extract photo index from record_id
                                    if record_id and record_id.startswith("photo_"):
                                        photo_idx = int(record_id.split("_")[1])
                                        if photo_idx < len(photo_ids):
                                            photo_id = photo_ids[photo_idx]

                                            # Extract analysis from model output
                                            model_output = result.get("modelOutput", {})
                                            logger.debug(
                                                f"Model output keys: {list(model_output.keys())}"
                                            )
                                            logger.debug(
                                                f"Full model output: {json.dumps(model_output, indent=2)[:500]}..."
                                            )

                                            if "content" in model_output:
                                                content = model_output["content"]
                                                if content and len(content) > 0:
                                                    text_response = content[0].get("text", "")

                                                    # Parse JSON response
                                                    try:
                                                        analysis_data = json.loads(text_response)
                                                    except json.JSONDecodeError:
                                                        analysis_data = {
                                                            "description": text_response,
                                                            "confidence": 0.8,
                                                        }

                                                    # Extract key fields
                                                    extracted_fields = self._extract_key_fields(
                                                        analysis_data
                                                    )

                                                    # Create LLM analysis record
                                                    llm_analysis = LLMAnalysis.create(
                                                        photo_id=photo_id,
                                                        model_name=self.model_name,
                                                        analysis=analysis_data,
                                                        batch_id=job_arn,
                                                        **extracted_fields,
                                                    )

                                                    self.repository.create_llm_analysis(
                                                        llm_analysis
                                                    )
                                                    processed_count += 1

                                except Exception as e:
                                    logger.error(f"Error processing Bedrock result: {e}")
                                    logger.debug(f"Problematic line: {line}")
                                    continue

                        # Clean up temp file
                        os.unlink(local_path)

            logger.info(f"Processed {processed_count} Bedrock batch results")
            return processed_count

        except Exception as e:
            logger.error(f"Error processing Bedrock batch results: {e}")
            return 0

    def _extract_usage_from_raw_results(self, results: str) -> Dict[str, Any]:
        """Extract token usage information from raw batch results."""
        try:
            total_usage = {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_read_tokens": 0,
            }

            per_photo_usage = []  # List of (custom_id, usage_dict) tuples

            lines = results.strip().split("\n")
            for line in lines:
                if not line.strip():
                    continue

                data = json.loads(line)
                custom_id = data.get("custom_id", "unknown")
                result = data.get("result", {})
                message = result.get("message", {})
                usage = message.get("usage", {})

                if usage:
                    photo_usage = {
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": usage.get("output_tokens", 0),
                        "cache_creation_tokens": usage.get("cache_creation_input_tokens", 0),
                        "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
                    }

                    # Accumulate totals
                    for key in total_usage:
                        total_usage[key] += photo_usage.get(key, 0)

                    per_photo_usage.append((custom_id, photo_usage))

            return {
                "total_usage": total_usage,
                "per_photo_usage": per_photo_usage,
            }

        except Exception as e:
            logger.error(f"Error extracting usage from batch results: {e}")
            return {"total_usage": {}, "per_photo_usage": []}

    def _calculate_batch_cost(
        self, usage: Dict[str, int], model_name: str, batch_discount: bool = True
    ) -> int:
        """Calculate cost in cents based on token usage.

        Pricing as of 2024:
        - Claude 3.5 Sonnet: $3.00 per million input tokens, $15.00 per million output tokens
        - Batch API: 50% discount
        - Cache read: 90% discount on input tokens
        - Cache creation: 25% extra on input tokens
        """
        # Base prices in cents per million tokens
        price_map = {
            "claude-3-5-sonnet": {"input": 300, "output": 1500},
            "claude-3-sonnet": {"input": 300, "output": 1500},
            "claude-3-opus": {"input": 1500, "output": 7500},
            "claude-3-haiku": {"input": 25, "output": 125},
        }

        # Get base prices for the model
        base_prices = None
        for model_key in price_map:
            if model_key in model_name.lower():
                base_prices = price_map[model_key]
                break

        if not base_prices:
            # Default to Sonnet pricing if model not recognized
            base_prices = price_map["claude-3-sonnet"]

        # Calculate costs in cents directly (prices are in cents per million tokens)
        input_cost_cents = (usage.get("input_tokens", 0) * base_prices["input"]) / 1_000_000
        output_cost_cents = (usage.get("output_tokens", 0) * base_prices["output"]) / 1_000_000

        # Cache creation costs 25% extra
        cache_creation_cost_cents = (
            usage.get("cache_creation_tokens", 0) * base_prices["input"] * 1.25
        ) / 1_000_000

        # Cache read costs 90% less
        cache_read_cost_cents = (
            usage.get("cache_read_tokens", 0) * base_prices["input"] * 0.1
        ) / 1_000_000

        total_cost_cents = (
            input_cost_cents + output_cost_cents + cache_creation_cost_cents + cache_read_cost_cents
        )

        # Apply batch discount if applicable
        if batch_discount:
            total_cost_cents *= 0.5

        # Return cost in cents (rounded)
        return round(total_cost_cents)

    def _process_instructor_batch_results(self, batch_id: str, photo_ids: List[str]) -> int:
        """Process completed Instructor batch results and save to database."""
        try:
            if not self.batch_processor:
                return 0

            # Retrieve raw results using Instructor's batch processor
            raw_results = self.batch_processor.provider.retrieve_results(batch_id)
            all_results = self.batch_processor.parse_results(raw_results)

            # First, extract token usage from raw results
            usage_data = self._extract_usage_from_raw_results(raw_results)
            per_photo_usage_map = {
                custom_id: usage for custom_id, usage in usage_data.get("per_photo_usage", [])
            }

            # Retrieve results using Instructor's batch processor
            all_results = self.batch_processor.retrieve_results(batch_id)

            # Filter successful and error results
            successful_results = filter_successful(all_results)
            error_results = filter_errors(all_results)

            processed_count = 0

            # Process successful results - use array indexing to match photo_ids
            for i, result in enumerate(successful_results):
                custom_id = "unknown"
                try:
                    custom_id = result.custom_id

                    photo_id = None
                    if custom_id:
                        if custom_id.startswith("request-"):
                            custom_id = custom_id[len("request-") :]
                            photo_id = photo_ids[int(custom_id)]
                        elif custom_id.startswith("photoid-"):
                            custom_id = custom_id[len("photoid-") :]
                            photo_id = custom_id

                    # TODO: it looks like Instructor doesn't allow setting custom_id yet, so the photoid- prefix isn't currently used.

                    if not photo_id:
                        logger.error(
                            f"Failed to extract photo id for batch {batch_id} result {i}: {custom_id}"
                        )
                        continue

                    # Get structured response from Instructor result
                    analysis_data = result.result.model_dump() if result.result else {}

                    # Extract key fields from structured data
                    extracted_fields = self._extract_key_fields(analysis_data)

                    # Get token usage for this photo if available
                    photo_usage = per_photo_usage_map.get(result.custom_id, {})

                    # Create and save LLM analysis
                    llm_analysis = LLMAnalysis.create(
                        photo_id=photo_id,
                        model_name=self.model_name,
                        analysis=analysis_data,
                        batch_id=batch_id,
                        model_version=(
                            self.model_name.split("-")[-1] if "-" in self.model_name else None
                        ),
                        input_tokens=photo_usage.get("input_tokens"),
                        output_tokens=photo_usage.get("output_tokens"),
                        cache_creation_tokens=photo_usage.get("cache_creation_tokens"),
                        cache_read_tokens=photo_usage.get("cache_read_tokens"),
                        **extracted_fields,
                    )

                    self.repository.create_llm_analysis(llm_analysis)
                    processed_count += 1
                    logger.debug(f"Processed Instructor batch result for photo {photo_id}")

                except Exception as e:
                    logger.error(f"Error processing successful batch result for {custom_id}: {e}")
                    continue

            # Process error results - use array indexing to match photo_ids
            for i, result in enumerate(error_results):
                custom_id = "unknown"
                try:
                    custom_id = result.custom_id or "unknown"

                    # Get photo_id from the array using the result index
                    if i >= len(photo_ids):
                        logger.error(
                            f"Instructor error result index {i} exceeds photo_ids length {len(photo_ids)} for batch {batch_id}"
                        )
                        continue

                    photo_id = photo_ids[i]

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
            if not self.client and self.provider_name != "bedrock":
                logger.error(f"No {self.provider_name} client available for analysis")
                return None

            if self.provider_name == "bedrock" and not hasattr(self, "bedrock_client"):
                logger.error("Bedrock client not properly initialized")
                return None

            # Create message content using shared method
            content = self._create_message_content(image_path, exif_context)

            # Create messages for instructor
            messages = [
                {
                    "role": "user",
                    "content": content,
                }
            ]

            # Handle different providers
            if self.provider_name == "bedrock":
                # Use native Bedrock API for image support
                response = self._call_bedrock_native(content)
            else:
                # Use Instructor for Anthropic
                create_params = {
                    "model": self.model_name,
                    "max_tokens": 4000,
                    "response_model": PhotoAnalysisResponse,
                    "messages": messages,
                }

                if self.system_prompt:
                    create_params["system"] = self.system_prompt

                response = self.client.messages.create(**create_params)

            # Convert response to dict for database storage
            if self.provider_name == "bedrock":
                return response  # Already a dict from native API
            else:
                return response.model_dump() if response else None  # Pydantic model from instructor

        except Exception as e:
            logger.error(f"Structured LLM analysis failed: {e}", exc_info=True)
            return None

    def _call_bedrock_native(self, content: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Call Bedrock native API for image analysis."""
        try:
            if not hasattr(self, "bedrock_client") or not self.bedrock_client:
                logger.error("Bedrock client not available")
                return None

            logger.debug(f"Making Bedrock API call with model {self.model_name}")

            # Note: EXIF context is already included in the content parameter

            # Prepare the request for Bedrock native API
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4000,
                "temperature": 0.1,
                "system": self.system_prompt,
                "messages": [{"role": "user", "content": content}],
            }

            logger.debug(f"Request body prepared with {len(content)} content blocks")

            # Invoke the model
            response = self.bedrock_client.invoke_model(
                modelId=self.model_name,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body),
            )

            logger.debug("Bedrock API call successful")

            # Parse response
            response_body = json.loads(response["body"].read())

            # Extract the text content from Claude's response
            content_blocks = response_body.get("content", [])
            if content_blocks and len(content_blocks) > 0:
                text_response = content_blocks[0].get("text", "")

                # Try to parse as JSON (structured response)
                try:
                    analysis_data = json.loads(text_response)
                except json.JSONDecodeError:
                    # If not JSON, create a simple structure
                    analysis_data = {"description": text_response, "confidence": 0.8}

                # Add usage metrics if available
                if "usage" in response_body:
                    analysis_data["_usage"] = response_body["usage"]

                return analysis_data

            return None

        except Exception as e:
            logger.error(f"Bedrock native API call failed: {e}", exc_info=True)
            return None

    def _validate_bedrock_setup(self) -> bool:
        """Validate Bedrock setup and permissions."""
        try:
            # Check if we can list foundation models (basic permission test)
            response = self.bedrock_control_client.list_foundation_models()

            # Check if our specific model is available
            available_models = [model["modelId"] for model in response.get("modelSummaries", [])]

            if self.model_name not in available_models:
                logger.error(f"Model {self.model_name} not found in available models")
                logger.info(
                    f"Available models: {', '.join(available_models[:10])}"
                )  # Show first 10
                return False

            logger.debug(f"Model {self.model_name} is available")
            return True

        except Exception as e:
            logger.error(f"Failed to validate Bedrock setup: {e}", exc_info=True)

            # Common error checks
            error_str = str(e).lower()
            if "accessdenied" in error_str or "unauthorizedoperation" in error_str:
                logger.error("Permission denied - check your AWS credentials and IAM permissions")
            elif "regionnotsupported" in error_str or "endpointconnectionerror" in error_str:
                logger.error("Region not supported or connection error - check AWS_REGION setting")
            elif "modelnotfound" in error_str:
                logger.error(
                    "Model not found - check BEDROCK_MODEL_ID and ensure model access is enabled"
                )

            return False

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

        return "\n".join(context_parts)

    def _extract_key_fields(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key fields from structured analysis for database indexing."""
        extracted = {}

        try:
            # Extract description and confidence
            if "description" in analysis:
                extracted["description"] = analysis["description"]

            # Extract objects list
            if (
                "objects" in analysis
                and analysis["objects"]
                and "items" in analysis["objects"]
                and analysis["objects"]["items"]
            ):
                extracted["objects"] = [
                    item.get("label") for item in analysis["objects"]["items"] if item.get("label")
                ]

            # Extract people count
            if "people" in analysis and analysis["people"] and "count" in analysis["people"]:
                extracted["people_count"] = analysis["people"]["count"]

            # Extract location information
            if "location" in analysis and analysis["location"]:
                location = analysis["location"]
                if "environment" in location and location["environment"]:
                    extracted["location_description"] = location["environment"]
                # Add hypotheses if available
                if "hypotheses" in location and location["hypotheses"]:
                    best_location = max(
                        location["hypotheses"], key=lambda x: x.get("confidence", 0)
                    )
                    if best_location.get("confidence", 0) > 0.5 and best_location.get("value"):
                        extracted["location_description"] = (
                            f"{location.get('environment', '')} - {best_location['value']}"
                        )

            # Extract activities/emotional context
            if "activities" in analysis and analysis["activities"]:
                activities = analysis["activities"]
                if "event_hypotheses" in activities and activities["event_hypotheses"]:
                    best_event = max(
                        activities["event_hypotheses"], key=lambda x: x.get("confidence", 0)
                    )
                    if best_event.get("confidence", 0) > 0.5 and best_event.get("type"):
                        extracted["emotional_tone"] = best_event["type"]
        except Exception as e:
            logger.error(f"Error extracting key fields from analysis: {e} - analysis: {analysis}")

        return extracted
