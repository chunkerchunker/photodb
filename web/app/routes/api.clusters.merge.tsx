import { mergeClusters } from "~/lib/db.server";
import type { Route } from "./+types/api.clusters.merge";

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

  const result = await mergeClusters(sourceClusterId, targetClusterId);
  return Response.json(result);
}
