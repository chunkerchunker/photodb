import { linkClustersToSamePerson } from "~/lib/db.server";
import type { Route } from "./+types/api.clusters.merge";

/**
 * API endpoint to link two clusters as the same person.
 * This does NOT merge clusters - it assigns them to the same person_id.
 * The endpoint path is kept as /api/clusters/merge for backwards compatibility.
 */
export async function action({ request }: Route.ActionArgs) {
  if (request.method !== "POST") {
    return Response.json({ success: false, message: "Method not allowed" }, { status: 405 });
  }

  const formData = await request.formData();
  const sourceClusterId = formData.get("sourceClusterId") as string;
  const targetClusterId = formData.get("targetClusterId") as string;

  if (!sourceClusterId || !targetClusterId) {
    return Response.json({ success: false, message: "Missing cluster IDs" }, { status: 400 });
  }

  const result = await linkClustersToSamePerson(sourceClusterId, targetClusterId);
  return Response.json(result);
}
