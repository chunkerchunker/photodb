import { handleLogout } from "~/lib/auth.server";
import type { Route } from "./+types/logout";

export async function action({ request }: Route.ActionArgs) {
  return await handleLogout(request);
}

export async function loader() {
  return null;
}
