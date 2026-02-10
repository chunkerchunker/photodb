import fs from "node:fs";
import { requireCollectionId } from "~/lib/auth.server";
import { getImagePath, getMimeType } from "~/lib/images.server";
import type { Route } from "./+types/api.image.$id";

export async function loader({ request, params }: Route.LoaderArgs) {
  const { collectionId } = await requireCollectionId(request);
  const photoId = parseInt(params.id, 10);
  const url = new URL(request.url);
  const useOriginal = url.searchParams.get("full") === "true";

  const imagePath = await getImagePath(collectionId, photoId, useOriginal);

  if (!imagePath) {
    return new Response("Image not found", { status: 404 });
  }

  try {
    const imageBuffer = fs.readFileSync(imagePath);
    const mimeType = getMimeType(imagePath);

    return new Response(imageBuffer, {
      status: 200,
      headers: {
        "Content-Type": mimeType,
        "Cache-Control": "public, max-age=31536000, immutable",
      },
    });
  } catch (error) {
    console.error(`Failed to serve image ${photoId}:`, error);
    return new Response("Failed to load image", { status: 500 });
  }
}
