# PhotoDB Web Server

## Quick Start

```bash
# Start the web server (default port 5000)
uv run photodb-web

# Start with custom port
uv run photodb-web --port 8080

# Start in debug mode
uv run photodb-web --debug

# Bind to all interfaces (for network access)
uv run photodb-web --host 0.0.0.0
```

## Features

The web server provides a date-based browsing interface for your photo collection:

1. **Year Index** (`/`): Shows all years that have photos
2. **Month Grid** (`/year/<year>`): Shows months in a year with photo previews
3. **Photo Grid** (`/year/<year>/month/<month>`): Displays photo thumbnails with descriptions
4. **Photo Detail** (`/photo/<photo_id>`): Full photo view with all metadata and AI analysis

## Navigation

- Click on a year to see months
- Click on a month to see photos
- Click on a photo to see full details
- Breadcrumb navigation at the top of each page

## Photo Details Include

- Full-size image from normalized_path
- Basic information (filename, paths, dates)
- GPS location (with Google Maps link if available)
- AI Analysis summary (description, objects, emotional tone)
- Full AI Analysis JSON (expandable)
- EXIF metadata (expandable)

## Requirements

- PostgreSQL database with photos already processed
- Photos must have been normalized (normalized_path populated)
- Optional: LLM analysis for descriptions

## Performance Notes

- Images are served directly from normalized_path (no resizing)
- Browser caching enabled (1 day cache)
- Pagination for months with >48 photos
- CSS-based grid layout for responsive design