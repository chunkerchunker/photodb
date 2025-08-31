# Stage 3: LLM-Based Photo Analysis

## Overview

Stage 3 processes normalized photos through LLM analysis to extract advanced metadata including scene descriptions, object identification, emotional context, and other semantic information that cannot be captured through traditional EXIF data.

## Input/Output

**Input**: Photo UUID from the `photos` table
**Output**: Structured analysis data stored in `llm_analysis` table

## Processing Logic

### Prerequisites

- Photo must exist in `photos` table (Stage 1 complete)
- Normalized image file must exist at `IMG_PATH/{photo_id}.png`
- EXIF metadata should be available (Stage 2 preferred but not required)

### Skip Conditions

A photo is skipped if:

- Already has a row in `llm_analysis` table (unless `--force` flag is used)
- Normalized image file is missing
- Photo record doesn't exist in database

### LLM Processing Workflow

1. **Batch Preparation**
   - Collect all eligible photos for processing
   - Prepare batch requests with image data and EXIF context
   - Submit to LLM provider's batch API

2. **Batch Monitoring**
   - Track batch job status
   - Handle provider rate limits and retries
   - Monitor for completion or failures

3. **Result Processing**
   - Parse LLM responses into structured format
   - Validate response schema
   - Store results in `llm_analysis` table
   - Update processing status

## Database Schema

### llm_analysis Table

```sql
CREATE TABLE llm_analysis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    photo_id UUID NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    
    -- LLM processing metadata
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    processed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    batch_id VARCHAR(255), -- Provider batch job ID
    
    -- Analysis results (JSON structure matching analyze_photo.md prompt)
    analysis JSONB NOT NULL,
    
    -- Extracted key fields for indexing/querying
    description TEXT, -- Main scene description
    objects TEXT[], -- Array of identified objects
    people_count INTEGER, -- Number of people detected
    location_description TEXT, -- Described location if not in EXIF
    emotional_tone VARCHAR(50), -- Happy, sad, neutral, etc.
    
    -- Processing metadata
    confidence_score DECIMAL(3,2), -- Overall confidence 0.00-1.00
    processing_duration_ms INTEGER,
    error_message TEXT, -- If processing failed
    
    UNIQUE(photo_id),
    INDEX(objects),
    INDEX(emotional_tone),
    INDEX(people_count),
    INDEX(processed_at)
);
```

## Batch Processing Integration

### CLI Batch Processing Flow

1. **Discovery Phase**
   - CLI identifies all photos needing Stage 3 processing
   - Creates batch request payload

2. **Batch Submission**
   - Submit batch to LLM provider
   - Store batch job ID and photo mapping
   - Return immediately (non-blocking)

3. **Status Tracking**
   - Separate CLI command: `process-photos --check-batches`
   - Polls provider for batch completion
   - Downloads and processes results when ready

4. **Recovery & Retry**
   - Failed individual requests are retried separately
   - Partial batch results are processed incrementally

### CLI Commands

```bash
# Submit photos for LLM analysis (returns immediately)
uv run process-photos /path --stage enrich

# Check status of running batches
uv run process-photos --check-batches

# Force reprocessing of specific photos
uv run process-photos /path --stage enrich --force

# Process only photos with failed LLM analysis
uv run process-photos /path --stage enrich --retry-failed
```

### Batch Job Tracking Table

```sql
CREATE TABLE batch_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    provider_batch_id VARCHAR(255) UNIQUE NOT NULL,
    status VARCHAR(20) NOT NULL, -- submitted, processing, completed, failed
    submitted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    photo_count INTEGER NOT NULL,
    processed_count INTEGER DEFAULT 0,
    failed_count INTEGER DEFAULT 0,
    error_message TEXT
);
```

## Configuration

Environment variables:

- `LLM_PROVIDER`: Provider name (default: "anthropic")
- `LLM_MODEL`: Model identifier (default: "claude-sonnet-4-20250514")
- `LLM_API_KEY`: Provider API key
- `BATCH_SIZE`: Photos per batch request (default: 100)
- `BATCH_CHECK_INTERVAL`: Polling interval in seconds (default: 300)

## Error Handling

- Individual photo failures are logged but don't block batch processing
- Network timeouts trigger automatic retries
- Invalid LLM responses are logged with error details
- Corrupted image files are marked as permanently failed

## Performance Considerations

- Batch processing reduces API costs and improves throughput
- JSONB storage enables efficient querying of analysis results
- Indexed fields support fast filtering and search
- Asynchronous processing doesn't block CLI completion
