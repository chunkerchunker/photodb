import { requireCollectionId } from "~/lib/auth.server";
import { setClusterPersonName } from "~/lib/db.server";
import type { Route } from "./+types/api.cluster.$id.rename";

export async function action({ request, params }: Route.ActionArgs) {
  if (request.method !== "POST") {
    return Response.json({ success: false, message: "Method not allowed" }, { status: 405 });
  }

  const { collectionId } = await requireCollectionId(request);
  const clusterId = params.id;
  if (!clusterId) {
    return Response.json({ success: false, message: "Missing cluster ID" }, { status: 400 });
  }

  const formData = await request.formData();
  const firstName = (formData.get("firstName") as string)?.trim();
  const lastName = (formData.get("lastName") as string)?.trim();

  if (!firstName) {
    return Response.json({ success: false, message: "First name is required" }, { status: 400 });
  }

  const result = await setClusterPersonName(collectionId, clusterId, firstName, lastName || undefined);
  return Response.json(result);
}
