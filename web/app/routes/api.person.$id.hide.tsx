import { setPersonHidden } from "~/lib/db.server";
import type { Route } from "./+types/api.person.$id.hide";

export async function action({ request, params }: Route.ActionArgs) {
  if (request.method !== "POST") {
    return Response.json({ success: false, message: "Method not allowed" }, { status: 405 });
  }

  const personId = params.id;
  if (!personId) {
    return Response.json({ success: false, message: "Person ID required" }, { status: 400 });
  }

  const formData = await request.formData();
  const hidden = formData.get("hidden") === "true";

  const result = await setPersonHidden(personId, hidden);
  return Response.json(result);
}
