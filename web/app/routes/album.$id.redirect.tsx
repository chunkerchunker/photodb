import { redirect } from "react-router";
import { getViewModeCookie } from "~/lib/cookies.server";
import type { Route } from "./+types/album.$id.redirect";

export async function loader({ request, params }: Route.LoaderArgs) {
  const albumId = params.id;
  const viewMode = await getViewModeCookie(request);
  if (viewMode === "grid") {
    return redirect(`/album/${albumId}/grid`);
  }
  return redirect(`/album/${albumId}/wall`);
}
