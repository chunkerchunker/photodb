import { requireCollectionId } from "~/lib/auth.server";
import { assignClusterToPerson, linkClustersToSamePerson } from "~/lib/db.server";
import type { Route } from "./+types/api.clusters.merge";

/**
 * API endpoint to link two clusters as the same person.
 * This does NOT merge clusters - it assigns them to the same person_id.
 * The endpoint path is kept as /api/clusters/merge for backwards compatibility.
 *
 * Accepts either:
 * - sourceClusterId + targetClusterId (link via cluster)
 * - sourceClusterId + targetPersonId (link directly to a person)
 */
export async function action({ request }: Route.ActionArgs) {
  if (request.method !== "POST") {
    return Response.json({ success: false, message: "Method not allowed" }, { status: 405 });
  }

  const { collectionId } = await requireCollectionId(request);
  const formData = await request.formData();
  const sourceClusterId = formData.get("sourceClusterId") as string;
  const targetClusterId = formData.get("targetClusterId") as string;
  const targetPersonId = formData.get("targetPersonId") as string;

  if (!sourceClusterId) {
    return Response.json({ success: false, message: "Missing source cluster ID" }, { status: 400 });
  }

  if (targetPersonId) {
    const result = await assignClusterToPerson(collectionId, sourceClusterId, targetPersonId);
    return Response.json(result);
  }

  if (!targetClusterId) {
    return Response.json({ success: false, message: "Missing target cluster or person ID" }, { status: 400 });
  }

  const result = await linkClustersToSamePerson(collectionId, sourceClusterId, targetClusterId);
  return Response.json(result);
}
