import fs from "fs";
import { getImagePath, getMimeType } from "~/lib/images.server";
import type { Route } from "./+types/api.image.$id";

export async function loader({ params }: Route.LoaderArgs) {
	const photoId = parseInt(params.id);

	const imagePath = await getImagePath(photoId);

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
