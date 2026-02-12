import { requireUser } from "~/lib/auth.server";
import { updateUserDefaultCollection } from "~/lib/db.server";
import type { Route } from "./+types/api.user.set-default-collection";

export async function action({ request }: Route.ActionArgs) {
  const user = await requireUser(request);
  const formData = await request.formData();

  const collectionId = parseInt(formData.get("collectionId") as string, 10);

  if (Number.isNaN(collectionId)) {
    return Response.json({ success: false, message: "Invalid collection ID" }, { status: 400 });
  }

  try {
    await updateUserDefaultCollection(user.id, collectionId);
    return Response.json({ success: true, message: "Default collection updated" });
  } catch (error) {
    console.error("Failed to update default collection:", error);
    return Response.json({ success: false, message: "Failed to update default collection" }, { status: 500 });
  }
}
