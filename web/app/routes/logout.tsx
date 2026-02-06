import type { Route } from "./+types/logout";
import { handleLogout } from "~/lib/auth.server";

export async function action({ request }: Route.ActionArgs) {
  return await handleLogout(request);
}

export async function loader() {
  return null;
}
