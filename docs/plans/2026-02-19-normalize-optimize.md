# Normalize Stage Optimization Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Cut normalize time from ~2.3s/photo to ~0.8-1.2s/photo by eliminating redundant file decodes and using a faster WebP encoder setting.

**Architecture:** Add an `open_and_orient()` method to ImageHandler that decodes + applies EXIF rotation in one step. Simplify normalize.py to call it once and reuse the oriented image for both full and medium saves. Change WebP encode method from 6 (slowest) to 4 (~2x faster, ~5% larger files). All Pillow-only — no new dependencies.

**Tech Stack:** Pillow (existing), pillow_heif (existing)

---

### Task 1: Change WEBP_METHOD default from 6 to 4

**Files:**
- Modify: `src/photodb/config.py:40`

This is the simplest optimization: WebP method 6 is the slowest encoder preset (best compression). Method 4 is ~2x faster for ~5% larger files. Since we encode WebP twice per photo (full + medium), this saves significant time.

**Step 1: Change the default**

In `src/photodb/config.py`, change line 40 from:
```python
WEBP_METHOD = 6
```
to:
```python
WEBP_METHOD = 4
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_image.py -v`
Expected: All 14 tests pass (no test directly asserts method value).

**Step 3: Commit**

```bash
git add src/photodb/config.py
git commit -m "change WEBP_METHOD default from 6 to 4 for ~2x faster encoding"
```

---

### Task 2: Add `open_and_orient()` to ImageHandler

**Files:**
- Modify: `src/photodb/utils/image.py`
- Modify: `tests/test_image.py`

Add a new method that combines `open_image()` + EXIF rotation into one step. This is the key building block: callers get a correctly oriented image from a single decode, eliminating the need for `_apply_exif_orientation` to reopen the file.

**Step 1: Write the test**

Add to `tests/test_image.py`:

```python
def test_open_and_orient(self, sample_image):
    """Test open_and_orient returns correctly oriented image."""
    img = ImageHandler.open_and_orient(sample_image)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.size == (100, 100)

def test_open_and_orient_unsupported(self, temp_dir):
    """Test open_and_orient rejects unsupported formats."""
    txt_file = temp_dir / "test.txt"
    txt_file.write_text("not an image")
    with pytest.raises(ValueError, match="Unsupported format"):
        ImageHandler.open_and_orient(txt_file)

def test_open_and_orient_exif_rotation(self, temp_dir):
    """Test that EXIF orientation is applied and tag stripped."""
    import piexif
    # Create a 100x200 image with EXIF orientation 6 (90° CCW rotation)
    img_path = temp_dir / "rotated.jpg"
    img = Image.new("RGB", (100, 200), color="red")
    exif_dict = {"0th": {piexif.ImageIFD.Orientation: 6}}
    exif_bytes = piexif.dump(exif_dict)
    img.save(img_path, "JPEG", exif=exif_bytes)

    result = ImageHandler.open_and_orient(img_path)
    # After 90° CCW rotation, 100x200 becomes 200x100
    assert result.size == (200, 100)
    # EXIF orientation tag should be stripped
    exif = result.getexif()
    assert exif.get(0x0112) in (None, 1)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_image.py::TestImageHandler::test_open_and_orient -v`
Expected: FAIL — `AttributeError: type object 'ImageHandler' has no attribute 'open_and_orient'`

**Step 3: Implement `open_and_orient`**

Add to `src/photodb/utils/image.py` in the `ImageHandler` class, after `open_image`:

```python
@classmethod
def open_and_orient(cls, file_path: Path) -> Image.Image:
    """
    Open an image and apply EXIF orientation correction.

    Combines open_image() + exif_transpose() in one step. The returned
    image has rotation baked in and the EXIF orientation tag stripped.
    Dimensions reflect the final oriented size.

    Args:
        file_path: Path to the image file

    Returns:
        PIL Image object with EXIF orientation applied (caller must close it!)
    """
    image = cls.open_image(file_path)
    try:
        oriented = ImageOps.exif_transpose(image)
        if oriented is not image:
            image.close()
            image = oriented
    except Exception as e:
        logger.debug(f"Could not apply EXIF orientation: {e}")
    return image
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_image.py -v`
Expected: All tests pass (14 existing + 3 new = 17).

**Step 5: Commit**

```bash
git add src/photodb/utils/image.py tests/test_image.py
git commit -m "add ImageHandler.open_and_orient() for single-decode EXIF rotation"
```

---

### Task 3: Simplify normalize.py to single-decode path

**Files:**
- Modify: `src/photodb/stages/normalize.py`

This is the main payoff. Replace 5 file opens with 1 by using `open_and_orient()` and passing pre-oriented images to `save_as_webp` without `original_path`.

**Step 1: Rewrite `process_photo`**

Replace the entire `process_photo` method in `src/photodb/stages/normalize.py`:

```python
def process_photo(self, photo: Photo, file_path: Path) -> bool:
    """Normalize a photo to WebP format (both medium and full-size versions).

    Returns:
        bool: True if processing was successful, False otherwise
    """
    image = None
    try:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.full_output_dir.mkdir(parents=True, exist_ok=True)

        output_filename = f"{self._generate_photo_id(Path(photo.orig_path))}.webp"
        med_output_path = self.output_dir / output_filename
        full_output_path = self.full_output_dir / output_filename

        # Single decode + EXIF rotation (replaces 5 separate file opens)
        image = ImageHandler.open_and_orient(file_path)

        # Store original dimensions (post-rotation = final orientation)
        photo.width = image.width
        photo.height = image.height
        logger.debug(f"Original size (post-rotation): {image.width}x{image.height}")

        # Save full-size WebP (no original_path needed — rotation already applied)
        ImageHandler.save_as_webp(image, full_output_path, quality=self.WEBP_QUALITY)
        logger.debug(f"Full-size photo saved to {full_output_path}")
        photo.full_path = str(full_output_path)

        # Calculate and apply resize for medium version
        original_size = (image.width, image.height)
        new_size = ImageHandler.calculate_resize_dimensions(original_size, {})

        if new_size and new_size != original_size:
            logger.debug(f"Resizing to: {new_size[0]}x{new_size[1]}")
            med_image = ImageHandler.resize_image(image, new_size)
            photo.med_width = new_size[0]
            photo.med_height = new_size[1]
        else:
            med_image = image
            photo.med_width = image.width
            photo.med_height = image.height

        # Save medium WebP (no original_path needed — rotation already applied)
        ImageHandler.save_as_webp(med_image, med_output_path, quality=self.WEBP_QUALITY)
        logger.debug(f"Medium photo saved to {med_output_path}")
        logger.debug(f"Final dimensions: {photo.med_width}x{photo.med_height}")

        photo.med_path = str(med_output_path)
        self.repository.update_photo(photo)
        return True

    except Exception as e:
        logger.error(f"Failed to normalize photo {file_path}: {e}")
        return False
    finally:
        if image:
            image.close()
```

Also clean up the imports at the top of the file. Remove:
```python
import pillow_heif
from PIL import Image
from PIL.ExifTags import Base as ExifBase
```
and:
```python
pillow_heif.register_heif_opener()

# EXIF orientations that swap width/height (involve 90° or 270° rotation)
ORIENTATION_SWAPS_DIMENSIONS = {5, 6, 7, 8}
```

The only imports needed are:
```python
from pathlib import Path
import logging

from .base import BaseStage
from .. import config as defaults
from ..database.models import Photo
from ..utils.image import ImageHandler
```

**Step 2: Run full test suite**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All tests pass.

**Step 3: Commit**

```bash
git add src/photodb/stages/normalize.py
git commit -m "simplify normalize to single decode via open_and_orient

Reduces per-photo file opens from 5 to 1:
- Was: open_image x2 + Image.open for EXIF + _apply_exif_orientation x2
- Now: open_and_orient x1, reuse oriented image for both saves

Also removes ORIENTATION_SWAPS_DIMENSIONS logic — dimensions after
exif_transpose are already correct."
```

---

### Task 4: Verify with benchmarks

Run single-thread normalize on the 83-photo test set:

```bash
just local "/Volumes/media/Pictures/capture/Order\#04236/Album\#23070/" --skip-directory-scan --force --collection-id 3 --show-progress --stage normalize
```

Record the avg/photo time. Expected: significant improvement from baseline ~2.3s.

Run 4-thread full pipeline:

```bash
just local "/Volumes/media/Pictures/capture/Order\#04236/Album\#23070/" --skip-directory-scan --force --collection-id 3 --show-progress --parallel 4
```

Compare overall wall time. Spot-check a few output files for correct orientation.

---

## Files Changed Summary

| File | Action | What changes |
|---|---|---|
| `src/photodb/config.py` | Edit line 40 | `WEBP_METHOD = 6` → `WEBP_METHOD = 4` |
| `src/photodb/utils/image.py` | Add method | New `open_and_orient()` classmethod |
| `src/photodb/stages/normalize.py` | Simplify | Single decode, remove EXIF handling, remove PIL imports |
| `tests/test_image.py` | Add 3 tests | Tests for `open_and_orient` including EXIF rotation |

## What's NOT Changing

| Optimization | Why skipped |
|---|---|
| `Image.thumbnail()` shrink-on-load | Only helps for JPEG during decode from file; contradicts single-decode approach since image is already loaded |
| Skip full-size for WebP/JPEG sources | Changes assumption that all `full/` files are WebP; downstream stages depend on this |
| `_apply_exif_orientation` removal | Keep it — still used by `save_as_png`. Just no longer called from normalize path |
