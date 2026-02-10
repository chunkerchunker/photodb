import fs from "node:fs";
import { requireCollectionId } from "~/lib/auth.server";
import { getFacePath, getMimeType } from "~/lib/images.server";
import type { Route } from "./+types/api.face.$id";

export async function loader({ request, params }: Route.LoaderArgs) {
  const { collectionId } = await requireCollectionId(request);
  const detectionId = parseInt(params.id, 10);

  const facePath = await getFacePath(collectionId, detectionId);

  if (!facePath) {
    return new Response("Face image not found", { status: 404 });
  }

  try {
    const imageBuffer = fs.readFileSync(facePath);
    const mimeType = getMimeType(facePath);

    return new Response(imageBuffer, {
      status: 200,
      headers: {
        "Content-Type": mimeType,
        "Cache-Control": "public, max-age=31536000, immutable",
      },
    });
  } catch (error) {
    console.error(`Failed to serve face ${detectionId}:`, error);
    return new Response("Failed to load face image", { status: 500 });
  }
}
