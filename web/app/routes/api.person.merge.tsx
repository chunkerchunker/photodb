import { requireCollectionId } from "~/lib/auth.server";
import { mergePersons } from "~/lib/db.server";
import type { Route } from "./+types/api.person.merge";

export async function action({ request }: Route.ActionArgs) {
  if (request.method !== "POST") {
    return Response.json({ success: false, message: "Method not allowed" }, { status: 405 });
  }

  const { collectionId } = await requireCollectionId(request);
  const formData = await request.formData();
  const sourcePersonId = formData.get("sourcePersonId") as string;
  const targetPersonId = formData.get("targetPersonId") as string;

  if (!sourcePersonId || !targetPersonId) {
    return Response.json({ success: false, message: "Missing person IDs" }, { status: 400 });
  }

  const result = await mergePersons(collectionId, sourcePersonId, targetPersonId);
  return Response.json(result);
}
