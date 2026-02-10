import { redirect } from "react-router";
import { startImpersonation } from "~/lib/auth.server";
import type { Route } from "./+types/api.admin.impersonate";

export async function action({ request }: Route.ActionArgs) {
  const formData = await request.formData();
  const userId = parseInt(formData.get("userId") as string, 10);

  if (Number.isNaN(userId)) {
    return { success: false, error: "Invalid user ID" };
  }

  const cookie = await startImpersonation(request, userId);

  throw redirect("/", {
    headers: {
      "Set-Cookie": cookie,
    },
  });
}
