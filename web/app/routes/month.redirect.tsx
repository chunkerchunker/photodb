import { redirect } from "react-router";
import { getViewModeCookie } from "~/lib/cookies.server";
import type { Route } from "./+types/month.redirect";

export async function loader({ request, params }: Route.LoaderArgs) {
  const viewMode = await getViewModeCookie(request);
  const { year, month } = params;
  if (viewMode === "grid") {
    return redirect(`/year/${year}/month/${month}/grid`);
  }
  return redirect(`/year/${year}/month/${month}/wall`);
}
