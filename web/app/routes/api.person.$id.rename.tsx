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
  const middleName = formData.get("middleName") as string;
  const maidenName = formData.get("maidenName") as string;
  const preferredName = formData.get("preferredName") as string;
  const suffix = formData.get("suffix") as string;
  const alternateNamesRaw = formData.get("alternateNames") as string;
  const alternateNames = alternateNamesRaw ? JSON.parse(alternateNamesRaw) : [];

  if (!firstName?.trim()) {
    return Response.json({ success: false, message: "First name is required" }, { status: 400 });
  }

  const result = await setPersonName(
    collectionId,
    personId,
    firstName.trim(),
    lastName?.trim(),
    middleName?.trim(),
    maidenName?.trim(),
    preferredName?.trim(),
    suffix?.trim(),
    alternateNames,
  );
  return Response.json(result);
}
