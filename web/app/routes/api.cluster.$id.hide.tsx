import { setClusterHidden } from "~/lib/db.server";
import type { Route } from "./+types/api.cluster.$id.hide";

export async function action({ request, params }: Route.ActionArgs) {
  if (request.method !== "POST") {
    return Response.json({ success: false, message: "Method not allowed" }, { status: 405 });
  }

  const clusterId = params.id;
  if (!clusterId) {
    return Response.json({ success: false, message: "Missing cluster ID" }, { status: 400 });
  }

  const formData = await request.formData();
  const hidden = formData.get("hidden") === "true";

  const result = await setClusterHidden(clusterId, hidden);
  return Response.json(result);
}
