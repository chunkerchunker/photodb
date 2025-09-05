# Photo Browse Website Task Plan

## Overview
Create a simple Flask-based web application for browsing the PhotoDB collection organized by date, with drill-down navigation from years to months to individual photos.

## Architecture

### Technology Stack
- **Backend**: Flask 3.x with Jinja2 templates
- **Database**: Existing PostgreSQL with pg_repository
- **Frontend**: Bootstrap 5 + minimal vanilla JavaScript
- **Image Serving**: Serve directly from `normalized_path` locations (no resizing)

### URL Structure
```
/                           # Year index page
/year/<year>               # Month grid for specific year
/year/<year>/month/<month> # Photo thumbnails for specific month
/photo/<photo_id>          # Individual photo detail view
/api/image/<photo_id>      # Serve image from normalized_path
```

## Implementation Plan

### Phase 1: Core Setup
1. **Create Flask Application Structure**
   - `src/photodb/web/__init__.py` - Flask app factory
   - `src/photodb/web/app.py` - Main application
   - `src/photodb/web/routes.py` - Route definitions
   - `src/photodb/web/templates/` - Jinja2 templates
   - `src/photodb/web/static/` - CSS, JS, images

2. **Database Integration**
   - Reuse existing `pg_repository.py` for data access
   - Add new query methods:
     - `get_years_with_photos()` - Distinct years from metadata.captured_at
     - `get_months_in_year(year)` - Distinct months with photo count
     - `get_photos_by_month(year, month)` - Photos with metadata and analysis
     - `get_photo_details(photo_id)` - Full photo data including JSONB fields

### Phase 2: Templates & Views

#### Base Template (`base.html`)
```html
- Bootstrap 5 CDN
- Responsive navbar with home link
- Breadcrumb navigation
- Container for content
```

#### Year Index (`index.html`)
```html
- Grid of year cards
- Each card shows:
  - Year (large)
  - Photo count for that year
  - Sample thumbnail (first photo of year)
- Click navigates to /year/<year>
```

#### Month Grid (`year.html`)
```html
- Grid of month cards (3x4 layout)
- Each card shows:
  - Month name
  - Photo count
  - 2x2 thumbnail preview grid
- Click navigates to /year/<year>/month/<month>
```

#### Photo Grid (`month.html`)
```html
- Responsive image grid (4-6 columns)
- Each image shows:
  - Image from normalized_path (CSS constrained size)
  - Filename (without path)
  - Photo ID (small, muted)
  - Description (first 50 chars from llm_analysis if available)
- Click opens photo detail
- Use CSS object-fit: cover for uniform grid
```

#### Photo Detail (`photo.html`)
```html
- Two-column layout:
  Left: Full-size image (max 800px wide)
  Right: Metadata panels
    - Basic Info (filename, ID, dates)
    - Location (if available)
    - LLM Analysis (formatted JSON)
    - Technical Details (EXIF data)
```

### Phase 3: Features

#### Image Serving
```python
# src/photodb/web/images.py
- Read image path from photo.normalized_path
- Serve the normalized image directly (no resizing)
- Use Flask's send_file() for efficient serving
- Set appropriate cache headers
- Handle missing files gracefully
```

#### LLM Analysis Display
```python
# Pretty-format the analysis JSONB field
- Parse JSON structure
- Display sections:
  - Description
  - Objects detected
  - People count
  - Location description
  - Emotional tone
  - Technical quality
- Use collapsible Bootstrap cards
```

#### Navigation Breadcrumbs
```
Home > 2024 > March > Photo Detail
```

### Phase 4: Enhancements

#### Search/Filter (Optional)
- Filter by objects detected
- Search descriptions
- Filter by location

#### Performance
- Implement pagination for large months (>100 photos)
- Add Redis caching for year/month queries
- Use lazy loading for images in grid view
- Set browser cache headers for images

## Database Queries

### Required SQL Queries

```sql
-- Years with photos
SELECT DISTINCT EXTRACT(YEAR FROM m.captured_at) as year,
       COUNT(*) as photo_count
FROM metadata m
WHERE m.captured_at IS NOT NULL
GROUP BY year
ORDER BY year DESC;

-- Months in a year
SELECT EXTRACT(MONTH FROM m.captured_at) as month,
       COUNT(*) as photo_count
FROM metadata m
WHERE EXTRACT(YEAR FROM m.captured_at) = :year
  AND m.captured_at IS NOT NULL
GROUP BY month
ORDER BY month;

-- Photos by month
SELECT p.*, m.*, la.description, la.analysis
FROM photo p
JOIN metadata m ON p.id = m.photo_id
LEFT JOIN llm_analysis la ON p.id = la.photo_id
WHERE EXTRACT(YEAR FROM m.captured_at) = :year
  AND EXTRACT(MONTH FROM m.captured_at) = :month
ORDER BY m.captured_at;

-- Photo details
SELECT p.*, m.*, la.*, ps.*
FROM photo p
JOIN metadata m ON p.id = m.photo_id
LEFT JOIN llm_analysis la ON p.id = la.photo_id
LEFT JOIN processing_status ps ON p.id = ps.photo_id
WHERE p.id = :photo_id;
```

## Dependencies

### Python Packages to Add
```toml
flask = "^3.0"
python-dotenv = "^1.0"  # For config
```

### Frontend Dependencies (via CDN)
- Bootstrap 5.3
- Bootstrap Icons
- Optional: Lightbox2 for photo viewing

## File Structure
```
src/photodb/web/
├── __init__.py
├── app.py              # Flask application
├── routes.py           # Route handlers
├── queries.py          # Database queries
├── images.py           # Image serving from normalized_path
├── templates/
│   ├── base.html
│   ├── index.html      # Year list
│   ├── year.html       # Month grid
│   ├── month.html      # Photo grid
│   └── photo.html      # Photo detail
└── static/
    ├── css/
    │   └── style.css
    └── js/
        └── app.js
```

## CLI Integration

Add new command to `src/photodb/cli.py`:
```python
@cli.command()
@click.option('--port', default=5000)
@click.option('--debug/--no-debug', default=False)
def web(port, debug):
    """Start the photo browsing web server."""
    from photodb.web import create_app
    app = create_app()
    app.run(port=port, debug=debug)
```

## Development Steps

1. **Setup Flask skeleton** with basic routing
2. **Create database query layer** extending pg_repository
3. **Build templates** starting with base.html
4. **Implement year index** with real data
5. **Add month view** with navigation
6. **Create photo grid** with images from normalized_path
7. **Build photo detail page** with all metadata
8. **Add image serving** from normalized_path
9. **Format LLM analysis** display
10. **Test with sample data** and optimize queries

## Success Criteria

- [ ] Navigate from years → months → photos intuitively
- [ ] Display all available metadata and LLM analysis
- [ ] Serve images directly from normalized_path
- [ ] Responsive design works on mobile/tablet/desktop
- [ ] Page load times < 500ms for index pages
- [ ] Gracefully handle missing data (no captured_at, no analysis)

## Future Enhancements

- Add photo upload capability
- Implement batch reprocessing UI
- Add map view for geotagged photos
- Create slideshow mode
- Export collections as ZIP
- Add user authentication for private collections