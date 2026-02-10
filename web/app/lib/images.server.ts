import fs from "node:fs";
import path from "node:path";
import dotenv from "dotenv";
import { getPhotoById } from "./db.server";

// Load environment variables from .env file
dotenv.config({ path: path.join(process.cwd(), "..", ".env") });

export async function getImagePath(collectionId: number, photoId: number, useOriginal = false): Promise<string | null> {
  const photo = await getPhotoById(collectionId, photoId);
  if (!photo) {
    return null;
  }

  // When requesting full-size: prefer full_path (WebP), fallback to orig_path, then med_path
  // When requesting medium: use med_path
  let imagePath: string | null = null;

  if (useOriginal) {
    // Try full_path first (browser-compatible WebP at full resolution)
    if (photo.full_path && fs.existsSync(photo.full_path)) {
      imagePath = photo.full_path;
    }
    // Fallback to orig_path (may be HEIC/etc that some browsers can't display)
    else if (photo.orig_path && fs.existsSync(photo.orig_path)) {
      imagePath = photo.orig_path;
    }
    // Final fallback to med_path
    else if (photo.med_path && fs.existsSync(photo.med_path)) {
      imagePath = photo.med_path;
    }
  } else {
    imagePath = photo.med_path;
  }

  if (!imagePath) {
    return null;
  }

  // Check if file exists (for med_path case)
  if (!fs.existsSync(imagePath)) {
    console.error(`Image not found: ${imagePath}`);
    return null;
  }

  return imagePath;
}

export async function getImageBuffer(
  collectionId: number,
  photoId: number,
  useOriginal = false,
): Promise<Buffer | null> {
  const imagePath = await getImagePath(collectionId, photoId, useOriginal);
  if (!imagePath) {
    return null;
  }

  try {
    return fs.readFileSync(imagePath);
  } catch (error) {
    console.error(`Failed to read image: ${imagePath}`, error);
    return null;
  }
}

export function getMimeType(filePath: string): string {
  const ext = path.extname(filePath).toLowerCase();
  switch (ext) {
    case ".jpg":
    case ".jpeg":
      return "image/jpeg";
    case ".png":
      return "image/png";
    case ".gif":
      return "image/gif";
    case ".webp":
      return "image/webp";
    case ".bmp":
      return "image/bmp";
    case ".svg":
      return "image/svg+xml";
    case ".heic":
    case ".heif":
      return "image/heic";
    case ".tiff":
    case ".tif":
      return "image/tiff";
    default:
      return "application/octet-stream";
  }
}
