import { requireCollectionId } from "~/lib/auth.server";
import { deletePersonAggregation } from "~/lib/db.server";
import type { Route } from "./+types/api.person.$id.delete";

export async function action({ request, params }: Route.ActionArgs) {
  if (request.method !== "POST") {
    return Response.json({ success: false, message: "Method not allowed" }, { status: 405 });
  }

  const { collectionId } = await requireCollectionId(request);
  const personId = params.id;
  if (!personId) {
    return Response.json({ success: false, message: "Person ID required" }, { status: 400 });
  }

  const result = await deletePersonAggregation(collectionId, personId);
  return Response.json(result);
}
