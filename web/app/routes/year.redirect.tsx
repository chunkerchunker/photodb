import { redirect } from "react-router";
import type { Route } from "./+types/year.redirect";

export function loader({ request, params }: Route.LoaderArgs) {
  const cookieHeader = request.headers.get("Cookie");
  const viewMode = (cookieHeader?.match(/viewMode=(wall|grid)/)?.[1] as "wall" | "grid") || "grid";
  const { year } = params;

  if (viewMode === "wall") {
    return redirect(`/year/${year}/wall`);
  } else {
    return redirect(`/year/${year}/grid`);
  }
}
