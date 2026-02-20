import { requireCollectionId } from "~/lib/auth.server";
import { removePersonSibling } from "~/lib/db.server";
import type { Route } from "./+types/api.person.$id.remove-sibling";

export async function action({ request, params }: Route.ActionArgs) {
  if (request.method !== "POST") {
    return Response.json({ success: false, message: "Method not allowed" }, { status: 405 });
  }
  await requireCollectionId(request);
  const personId = params.id;
  if (!personId) {
    return Response.json({ success: false, message: "Person ID required" }, { status: 400 });
  }
  const formData = await request.formData();
  const siblingId = formData.get("siblingId") as string;
  if (!siblingId) {
    return Response.json({ success: false, message: "Sibling ID required" }, { status: 400 });
  }
  const result = await removePersonSibling(personId, siblingId);
  return Response.json(result);
}
