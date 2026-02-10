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

  // Use original (filename) if requested and available, otherwise fall back to normalized
  let imagePath = useOriginal && photo.filename ? photo.filename : photo.normalized_path;

  if (!imagePath) {
    return null;
  }

  // Check if file exists
  if (!fs.existsSync(imagePath)) {
    // If original doesn't exist, try normalized as fallback
    if (useOriginal && photo.normalized_path && fs.existsSync(photo.normalized_path)) {
      imagePath = photo.normalized_path;
    } else {
      console.error(`Image not found: ${imagePath}`);
      return null;
    }
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
    default:
      return "application/octet-stream";
  }
}
