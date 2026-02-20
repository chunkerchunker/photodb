import { requireCollectionId } from "~/lib/auth.server";
import { createPlaceholderPerson } from "~/lib/db.server";
import type { Route } from "./+types/api.person.create-placeholder";

export async function action({ request }: Route.ActionArgs) {
  if (request.method !== "POST") {
    return Response.json({ success: false, message: "Method not allowed" }, { status: 405 });
  }

  const { collectionId } = await requireCollectionId(request);
  const formData = await request.formData();
  const name = (formData.get("name") as string) || null;
  const gender = (formData.get("gender") as string) || "U";
  const description = (formData.get("description") as string) || null;

  const result = await createPlaceholderPerson(collectionId, name, gender, description);
  return Response.json({ success: true, ...result });
}
