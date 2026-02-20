import { requireCollectionId } from "~/lib/auth.server";
import { removePersonPartnership } from "~/lib/db.server";
import type { Route } from "./+types/api.person.$id.remove-partner";

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
  const partnerId = formData.get("partnerId") as string;

  if (!partnerId) {
    return Response.json({ success: false, message: "Partner ID required" }, { status: 400 });
  }

  const result = await removePersonPartnership(personId, partnerId);
  return Response.json(result);
}
