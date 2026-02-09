import { requireCollectionId } from "~/lib/auth.server";
import { setPersonName } from "~/lib/db.server";
import type { Route } from "./+types/api.person.$id.rename";

export async function action({ request, params }: Route.ActionArgs) {
  if (request.method !== "POST") {
    return Response.json({ success: false, message: "Method not allowed" }, { status: 405 });
  }

  const { collectionId } = await requireCollectionId(request);
  const personId = params.id;
  if (!personId) {
    return Response.json({ success: false, message: "Person ID required" }, { status: 400 });
  }

  const formData = await request.formData();
  const firstName = formData.get("firstName") as string;
  const lastName = formData.get("lastName") as string;

  if (!firstName?.trim()) {
    return Response.json({ success: false, message: "First name is required" }, { status: 400 });
  }

  const result = await setPersonName(collectionId, personId, firstName.trim(), lastName?.trim());
  return Response.json(result);
}
