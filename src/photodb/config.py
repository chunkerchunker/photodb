"""
Centralized configuration with environment variable overrides.

Every UPPERCASE constant defined here can be overridden by setting an
environment variable of the same name.  Type coercion is automatic:
bool, int, float, and str are inferred from the default value.

    # in any module
    from .. import config as defaults
    url = defaults.DATABASE_URL   # already env-resolved
"""

import os as _os
import sys as _sys

# --- General ---

# PostgreSQL connection string
DATABASE_URL = "postgresql://localhost/photodb"
# Directory to scan for new photos
INGEST_PATH = "./photos/raw"
# Root directory for processed output (med/, full/, faces/ subdirs)
IMG_PATH = "./photos/processed"
# Default collection partition for multi-tenant isolation
COLLECTION_ID = 1
# Root logger level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL = "INFO"
# Path to the rotating log file
LOG_FILE = "./logs/photodb.log"
# Multiplier applied to target image dimensions during normalization
RESIZE_SCALE = 1.0

# --- Image Processing ---

# WebP lossy quality for normalized images and face crops (0-100)
WEBP_QUALITY = 95
# Max pixel count before refusing to open an image (memory guard)
MAX_IMAGE_PIXELS = 178_956_970
# WebP encoder effort level (0=fast, 6=best compression)
WEBP_METHOD = 4
# PNG compression level for lossless saves (0-9)
PNG_COMPRESS_LEVEL = 9

# --- Detection ---

# Path to YOLOv8x person_face model weights
DETECTION_MODEL_PATH = "models/yolov8x_person_face.pt"
# Force CPU-only YOLO inference (disables CoreML/CUDA/MPS)
DETECTION_FORCE_CPU = False
# Minimum YOLO detection confidence to keep a face/body box
DETECTION_MIN_CONFIDENCE = 0.5
# Prefer CoreML (.mlpackage) on macOS for Neural Engine acceleration.
# Disabled by default: CoreML + MPS coexistence causes SIGSEGV with
# ultralytics 8.3+ / coremltools 9.0 / PyTorch 2.8. PyTorch MPS with
# batch inference is faster overall (MobileCLIP also gets MPS).
DETECTION_PREFER_COREML = False

# --- Age/Gender ---

# Path to MiVOLO checkpoint for age/gender estimation
MIVOLO_MODEL_PATH = "models/mivolo_d1.pth.tar"
# Force CPU-only MiVOLO inference
MIVOLO_FORCE_CPU = False
# Minimum IoU to match a MiVOLO prediction to a YOLO detection
AGE_GENDER_MIN_IOU = 0.3

# --- Embeddings ---

# InsightFace model pack name (buffalo_l = ArcFace ResNet-100)
EMBEDDING_MODEL_NAME = "buffalo_l"
# Root directory for InsightFace model downloads
EMBEDDING_MODEL_ROOT = _os.path.expanduser("~/.insightface/models")
# Padding ratio added around face bbox before ArcFace cropping
FACE_CROP_PADDING = 0.2

# --- Clustering ---

# Minimum points to form an HDBSCAN cluster
HDBSCAN_MIN_CLUSTER_SIZE = 3
# Core-point neighborhood size for HDBSCAN density estimation
HDBSCAN_MIN_SAMPLES = 2
# Minimum HDBSCAN core probability to mark a point as a core member
CORE_PROBABILITY_THRESHOLD = 0.8
# Cosine distance threshold for incremental epsilon-ball assignment
CLUSTERING_THRESHOLD = 0.45
# Floor clamp when converting lambda_birth to epsilon
MIN_EPSILON = 0.1
# Faces smaller than this (px, either dimension) are excluded from clustering
MIN_FACE_SIZE_PX = 50
# Faces below this detection confidence are excluded from clustering
MIN_FACE_CONFIDENCE = 0.9
# Pool threshold = CLUSTERING_THRESHOLD * this factor (stricter for pool)
POOL_CLUSTERING_THRESHOLD_FACTOR = 0.7
# Min unassigned faces in a neighborhood before attempting a pool sub-cluster
UNASSIGNED_CLUSTER_THRESHOLD = 5
# k for k-NN search during incremental cluster assignment
CLUSTERING_K_NEIGHBORS = 5
# Multiplier applied to epsilon for verified clusters (tighter acceptance)
VERIFIED_THRESHOLD_MULTIPLIER = 0.8
# Fraction of members that must change before medoid is recomputed
MEDOID_RECOMPUTE_THRESHOLD = 0.25
# k for Metal/MPS GPU sparse-graph k-NN during HDBSCAN bootstrap
KNN_NEIGHBORS = 60
# Batch size for GPU k-NN computation (balances VRAM vs. throughput)
KNN_BATCH_SIZE = 2000
# Embedding growth ratio that triggers a staleness warning
HDBSCAN_STALENESS_THRESHOLD = 1.25

# --- Maintenance ---

# Days in the unassigned pool before a face gets a singleton cluster
UNASSIGNED_MAX_AGE_DAYS = 30
# Cluster confidence assigned to maintenance-created singleton clusters
SINGLETON_CLUSTER_CONFIDENCE = 0.5
# Minimum faces in a cluster before epsilon can be calculated
EPSILON_MIN_FACES = 3
# Percentile of centroid distances used to derive cluster epsilon
EPSILON_PERCENTILE = 90.0
# Cosine distance threshold for auto-associating clusters to the same person
# pgvector <=> returns cosine distance (0=identical, 2=opposite)
# Looser than per-cluster epsilon since this is cross-context matching
PERSON_ASSOCIATION_THRESHOLD = 0.8

# --- Capture Import ---

# PostgreSQL connection string for the Capture system database
CAPTURE_DATABASE_URL = "postgresql://localhost/capture"
# Root directory for Capture photo files on disk
CAPTURE_BASE_PATH = "/Volumes/media/Pictures/capture"

# --- Scene Analysis ---

# MobileCLIP model variant for prompt-based tagging
CLIP_MODEL_NAME = "MobileCLIP-S2"
# MobileCLIP pretrained weights source
CLIP_PRETRAINED = "datacompdr"
# Number of top Apple Vision taxonomy labels to keep per photo
APPLE_VISION_TOP_K = 15
# Minimum Apple Vision classification confidence to return
APPLE_VISION_MIN_CONFIDENCE = 0.01

# --- LLM / Enrichment ---

# LLM backend: "anthropic" (direct API) or "bedrock" (AWS)
LLM_PROVIDER = "anthropic"
# Anthropic model ID for direct API calls
LLM_MODEL = "claude-sonnet-4-20250514"
# API key for Anthropic direct API (falls back to ANTHROPIC_API_KEY)
LLM_API_KEY = None
# AWS Bedrock model ID for batch inference
BEDROCK_MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"
# AWS region for Bedrock API calls
AWS_REGION = "us-east-1"
# AWS CLI profile name (optional)
AWS_PROFILE = None
# S3 bucket for Bedrock batch processing (required for batch mode)
BEDROCK_BATCH_S3_BUCKET = None
# IAM role ARN for Bedrock batch processing (required for batch mode)
BEDROCK_BATCH_ROLE_ARN = None
# Number of photos per LLM batch request
BATCH_SIZE = 100
# Batches smaller than this are skipped (not worth the overhead)
MIN_BATCH_SIZE = 10
# Seconds between batch-completion polling cycles
BATCH_CHECK_INTERVAL = 300
# Directory for batch request JSONL files
BATCH_REQUESTS_PATH = "./batch_requests"

# --- Batch Coordinator ---

# Maximum number of items per batched inference call
BATCH_COORDINATOR_MAX_SIZE = 32
# Maximum milliseconds to wait for a batch to fill before processing
BATCH_COORDINATOR_MAX_WAIT_MS = 50
# Enable/disable batch coordinator for ML inference (set False to use per-item inference)
BATCH_COORDINATOR_ENABLED = True
# Enable YOLO batch inference via BatchCoordinator.
# Works with PyTorch MPS backend (DETECTION_PREFER_COREML=False).
# CoreML dynamic batch export causes SIGSEGV (coremltools 9.0 + PyTorch 2.8).
YOLO_BATCH_ENABLED = True
# Enable experimental InsightFace batch embedding (disabled by default — ONNX CoreML EP may crash)
INSIGHTFACE_BATCH_ENABLED = False

# --- Connection Pool ---

# Minimum idle connections kept in the PostgreSQL pool
DEFAULT_MIN_CONN = 2
# Maximum connections the pool will open
DEFAULT_MAX_CONN = 100

# ---------------------------------------------------------------------------
# Environment override: every UPPERCASE constant above can be set via an
# environment variable of the same name.  Type is coerced to match the default.
# ---------------------------------------------------------------------------


def _load_env() -> None:
    module = _sys.modules[__name__]
    for name in list(vars(module)):
        if name.startswith("_") or not name.isupper():
            continue
        env_val = _os.environ.get(name)
        if env_val is None:
            continue
        current = getattr(module, name)
        if current is None:
            setattr(module, name, env_val)
        elif isinstance(current, bool):
            setattr(module, name, env_val.lower() in ("true", "1", "yes"))
        elif isinstance(current, int):
            setattr(module, name, int(env_val))
        elif isinstance(current, float):
            setattr(module, name, float(env_val))
        else:
            setattr(module, name, env_val)


_load_env()

# LLM_API_KEY falls back to ANTHROPIC_API_KEY
if LLM_API_KEY is None:
    LLM_API_KEY = _os.environ.get("ANTHROPIC_API_KEY")
