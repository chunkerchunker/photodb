import fs from "node:fs";
import path from "node:path";
import dotenv from "dotenv";
import { getPhotoById } from "./db.server";

// Load environment variables from .env file
dotenv.config({ path: path.join(process.cwd(), "..", ".env") });

export async function getImagePath(photoId: number): Promise<string | null> {
  const photo = await getPhotoById(photoId);
  if (!photo || !photo.normalized_path) {
    return null;
  }

  const imagePath = photo.normalized_path;

  // Check if file exists
  if (!fs.existsSync(imagePath)) {
    console.error(`Image not found: ${imagePath}`);
    return null;
  }

  return imagePath;
}

export async function getImageBuffer(photoId: number): Promise<Buffer | null> {
  const imagePath = await getImagePath(photoId);
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
