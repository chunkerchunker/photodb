#!/usr/bin/env python3
"""Test script to extract and print date information from an image using ExifExtractor."""

import sys
from pathlib import Path
from src.photodb.utils.exif import ExifExtractor


def test_exif_date(image_path: str):
    """Load an image and extract date information using ExifExtractor."""
    file_path = Path(image_path)

    if not file_path.exists():
        print(f"Error: File {image_path} does not exist")
        return

    if not file_path.is_file():
        print(f"Error: {image_path} is not a file")
        return

    print(f"Loading image: {file_path}")

    # Create ExifExtractor instance
    extractor = ExifExtractor(file_path)

    # Check for errors during loading
    if extractor.error:
        print(f"Error loading metadata: {extractor.error}")
        return

    # Extract datetime
    datetime_obj = extractor.extract_datetime()

    print("\nDate Information:")
    print(f"  Extracted datetime: {datetime_obj}")

    if datetime_obj:
        print(f"  Formatted date: {datetime_obj.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Year: {datetime_obj.year}")
        print(f"  Month: {datetime_obj.month}")
        print(f"  Day: {datetime_obj.day}")
    else:
        print("  No datetime found in EXIF data")

    # Show some basic image info
    print("\nBasic Image Info:")
    print(f"  Format: {extractor.img_info.get('format', 'Unknown')}")
    print(f"  Size: {extractor.img_info.get('size', {})}")
    print(f"  Mode: {extractor.img_info.get('mode', 'Unknown')}")

    # Show if EXIF data exists
    print("\nEXIF Data:")
    print(f"  Has EXIF: {'Yes' if extractor.exif_dict else 'No'}")

    if extractor.exif_dict:
        # Show available EXIF sections
        sections = list(extractor.exif_dict.keys())
        print(f"  EXIF sections: {sections}")

        # Show datetime-related tags if they exist
        datetime_tags = []
        if "Exif" in extractor.exif_dict:
            exif_section = extractor.exif_dict["Exif"]
            for tag, value in exif_section.items():
                if "date" in str(tag).lower() or tag in [
                    36867,
                    36868,
                ]:  # DateTimeOriginal, DateTimeDigitized
                    datetime_tags.append((tag, value))

        if "0th" in extractor.exif_dict:
            zeroth_section = extractor.exif_dict["0th"]
            for tag, value in zeroth_section.items():
                if "date" in str(tag).lower() or tag == 306:  # DateTime
                    datetime_tags.append((tag, value))

        if datetime_tags:
            print("  DateTime tags found:")
            for tag, value in datetime_tags:
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="ignore")
                print(f"    Tag {tag}: {value}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python test_exif_date.py <image_path>")
        print("Example: python test_exif_date.py /path/to/photo.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    test_exif_date(image_path)


if __name__ == "__main__":
    main()
