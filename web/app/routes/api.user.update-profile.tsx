import { requireUser } from "~/lib/auth.server";
import { updateUserProfile } from "~/lib/db.server";
import type { Route } from "./+types/api.user.update-profile";

export async function action({ request }: Route.ActionArgs) {
  const user = await requireUser(request);
  const formData = await request.formData();

  const firstName = (formData.get("firstName") as string)?.trim();
  const lastName = (formData.get("lastName") as string)?.trim() || null;

  if (!firstName) {
    return Response.json({ success: false, message: "First name is required" }, { status: 400 });
  }

  try {
    await updateUserProfile(user.id, firstName, lastName);
    return Response.json({ success: true, message: "Profile updated" });
  } catch (error) {
    console.error("Failed to update profile:", error);
    return Response.json({ success: false, message: "Failed to update profile" }, { status: 500 });
  }
}
