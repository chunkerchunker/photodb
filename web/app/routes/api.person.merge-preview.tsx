import { requireCollectionId } from "~/lib/auth.server";
import { previewPersonMerge } from "~/lib/db.server";
import type { Route } from "./+types/api.person.merge-preview";

export async function loader({ request }: Route.LoaderArgs) {
  const { collectionId } = await requireCollectionId(request);
  const url = new URL(request.url);
  const sourcePersonId = url.searchParams.get("source");
  const targetPersonId = url.searchParams.get("target");

  if (!sourcePersonId || !targetPersonId) {
    return Response.json({ found: false, error: "Missing person IDs" }, { status: 400 });
  }

  const preview = await previewPersonMerge(collectionId, sourcePersonId, targetPersonId);
  return Response.json(preview);
}
