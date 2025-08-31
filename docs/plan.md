# Personal photo indexing pipeline

This is an overview plan for a personal photo indexing system.  All processing code should be implemented in python using uv as the packager and using existing common libraries when available.

The end goal of the system is to have a database of photos that is searchable and navigable via a number of attributes: faces, places, dates, free-form descriptions, and any combinations thereof.

The following plan phases are intended to incrementally build up the database.

A separate process, outside of this pipeline, ingests the raw photos into a directory structure that we don't manage.

Environment variables are described in [.env](/.env).

A `process_photos` CLI is the overall entry point to the pipeline.  The program takes as input either a directory path or a single image path.  Images may be of any standard format, including jpeg, png, and heic.  If the input is a directory path, then the program scans the directory structure for any new images (as described below) and processes them individually.  Ingest paths are relative to the `INGEST_PATH` env var.

The database is stored as sqlite (at `DB_PATH` env var) using [schema.sql](/schema.sql).

The following stages should be implemented as separate modules that can be called indepedently (allowing for reprocessing as needed).

Generally, each stage has a way to detect if it has been run for the input photo.  Stages will skip reprocessing unless a `force` parameter is specified.

## Stage 1: Normalize photo

Input to this pipeline stage is a single ingested photo path.

A photo has been processed by this stage if it has a corresponding row in the `photos` table, based on `filename` matching the ingest relpath.

Normalization:

* Generate a UUID for the photo.
* If the photo exceeds the dimensions below, resize it smaller, preserving aspect ratio.
* Convert to png if required.
* Store normalized photo in `IMG_PATH`/{id}.png
* create corresponding row in `photos` (or update created_at if reprocessing).

Maximum dimensions (use closest match aspect ratio for determining max dimensions):

| Aspect Ratio | Max Dimensions |
|--------------|----------------|
| 1:1          | 1092x1092 px   |
| 3:4          | 951x1268 px    |
| 2:3          | 896x1344 px    |
| 9:16         | 819x1456 px    |
| 1:2          | 784x1568 px    |

## Stage 2: Extract basic metadata

Input to this pipeline stage is a photo id.

A photo has been processed by this stage if it has a corresponding row in the `metadata` table.

Extract all available EXIF/TIFF/IFD metadata from the ingest photo and create a corresponding `metadata` row.  All metadata is stored in the `extra` column.  Captured-at timestamp and location are also stored in their own columns.

## Stage 3: Extract LLM-based metadata

This stage sends the normalized photo and complete exif data to an LLM for advanced metadata extraction using the prompt info in [analyze_photo.md](../prompts/analyze_photo.md).

Take advantage of LLM provider batch processing.

The LLM provider and model should be configured in the .env file, defaulted to Anthropic Claude Sonnet 4 (claude-sonnet-4-20250514).

## Stage N

More stages to be defined..
