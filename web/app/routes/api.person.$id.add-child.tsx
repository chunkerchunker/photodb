import { requireCollectionId } from "~/lib/auth.server";
import { addPersonChild } from "~/lib/db.server";
import type { Route } from "./+types/api.person.$id.add-child";

export async function action({ request, params }: Route.ActionArgs) {
  if (request.method !== "POST") {
    return Response.json({ success: false, message: "Method not allowed" }, { status: 405 });
  }

  const { collectionId } = await requireCollectionId(request);
  const parentId = params.id;
  if (!parentId) {
    return Response.json({ success: false, message: "Person ID required" }, { status: 400 });
  }

  const formData = await request.formData();
  const relatedPersonId = formData.get("relatedPersonId") as string;
  const role = (formData.get("role") as string) || "parent";

  if (!relatedPersonId) {
    return Response.json({ success: false, message: "Related person ID required" }, { status: 400 });
  }

  const result = await addPersonChild(collectionId, parentId, relatedPersonId, role);
  return Response.json(result);
}
