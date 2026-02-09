import { redirect } from "react-router";
import { getViewModeCookie } from "~/lib/cookies.server";
import type { Route } from "./+types/person.$id.redirect";

export async function loader({ request, params }: Route.LoaderArgs) {
  const viewMode = await getViewModeCookie(request);
  if (viewMode === "grid") {
    return redirect(`/person/${params.id}/grid`);
  }
  return redirect(`/person/${params.id}/wall`);
}
