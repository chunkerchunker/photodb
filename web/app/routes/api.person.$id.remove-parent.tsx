import { requireCollectionId } from "~/lib/auth.server";
import { removePersonParent } from "~/lib/db.server";
import type { Route } from "./+types/api.person.$id.remove-parent";

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
  const parentId = formData.get("parentId") as string;

  if (!parentId) {
    return Response.json({ success: false, message: "Parent ID required" }, { status: 400 });
  }

  const result = await removePersonParent(personId, parentId);
  return Response.json(result);
}
