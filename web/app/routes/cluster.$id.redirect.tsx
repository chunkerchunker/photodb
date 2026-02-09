import { redirect } from "react-router";
import { getViewModeCookie } from "~/lib/cookies.server";
import type { Route } from "./+types/cluster.$id.redirect";

export async function loader({ request, params }: Route.LoaderArgs) {
  const viewMode = await getViewModeCookie(request);
  if (viewMode === "grid") {
    return redirect(`/cluster/${params.id}/grid`);
  }
  return redirect(`/cluster/${params.id}/wall`);
}
