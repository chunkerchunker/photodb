import { redirect } from "react-router";
import { stopImpersonation } from "~/lib/auth.server";
import type { Route } from "./+types/api.admin.stop-impersonate";

export async function action({ request }: Route.ActionArgs) {
  const cookie = await stopImpersonation(request);

  throw redirect("/", {
    headers: {
      "Set-Cookie": cookie,
    },
  });
}
