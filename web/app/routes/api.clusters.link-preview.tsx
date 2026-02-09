import { requireCollectionId } from "~/lib/auth.server";
import { previewClusterLink } from "~/lib/db.server";
import type { Route } from "./+types/api.clusters.link-preview";

export async function loader({ request }: Route.LoaderArgs) {
  const { collectionId } = await requireCollectionId(request);
  const url = new URL(request.url);
  const sourceClusterId = url.searchParams.get("source");
  const targetClusterId = url.searchParams.get("target");

  if (!sourceClusterId || !targetClusterId) {
    return Response.json({ found: false, error: "Missing cluster IDs" }, { status: 400 });
  }

  const preview = await previewClusterLink(collectionId, sourceClusterId, targetClusterId);
  return Response.json(preview);
}
