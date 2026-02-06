
import { redirect } from "react-router";
import type { Route } from "./+types/month.redirect";

export function loader({ request, params }: Route.LoaderArgs) {
  const cookieHeader = request.headers.get("Cookie");
  const viewMode = (cookieHeader?.match(/viewMode=(wall|grid)/)?.[1] as "wall" | "grid") || "grid";
  const { year, month } = params;

  if (viewMode === "wall") {
    return redirect(`/year/${year}/month/${month}/wall`);
  } else {
    return redirect(`/year/${year}/month/${month}/grid`);
  }
}
