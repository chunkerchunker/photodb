import { redirect } from "react-router";
import { getViewModeCookie } from "~/lib/cookies.server";
import type { Route } from "./+types/clusters.hidden.redirect";

export async function loader({ request }: Route.LoaderArgs) {
  const viewMode = await getViewModeCookie(request);
  if (viewMode === "grid") {
    return redirect("/clusters/hidden/grid");
  }
  return redirect("/clusters/hidden/wall");
}
