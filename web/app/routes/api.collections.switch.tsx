import { redirect } from "react-router";
import { requireUser } from "~/lib/auth.server";
import { updateUserDefaultCollection } from "~/lib/db.server";
import type { Route } from "./+types/api.collections.switch";

export async function action({ request }: Route.ActionArgs) {
  const user = await requireUser(request);
  const formData = await request.formData();
  const collectionId = parseInt(formData.get("collectionId") as string, 10);

  if (Number.isNaN(collectionId)) {
    return { success: false, error: "Invalid collection ID" };
  }

  try {
    await updateUserDefaultCollection(user.id, collectionId);
    // Redirect to home to show the new collection
    return redirect("/");
  } catch (error) {
    console.error("Failed to switch collection:", error);
    return { success: false, error: "Failed to switch collection" };
  }
}
