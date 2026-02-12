import { requireUser } from "~/lib/auth.server";
import { setCollectionMemberPerson } from "~/lib/db.server";
import type { Route } from "./+types/api.collection.$id.set-member-person";

export async function action({ request, params }: Route.ActionArgs) {
  const user = await requireUser(request);
  const collectionId = parseInt(params.id, 10);

  if (Number.isNaN(collectionId)) {
    return Response.json({ success: false, message: "Invalid collection ID" }, { status: 400 });
  }

  const formData = await request.formData();
  const personIdStr = formData.get("personId") as string;
  const personId = personIdStr ? parseInt(personIdStr, 10) : null;

  if (personIdStr && Number.isNaN(personId)) {
    return Response.json({ success: false, message: "Invalid person ID" }, { status: 400 });
  }

  try {
    await setCollectionMemberPerson(user.id, collectionId, personId);
    return Response.json({ success: true, message: personId ? "Person linked" : "Person unlinked" });
  } catch (error) {
    console.error("Failed to set member person:", error);
    const message = error instanceof Error ? error.message : "Failed to set member person";
    return Response.json({ success: false, message }, { status: 500 });
  }
}
