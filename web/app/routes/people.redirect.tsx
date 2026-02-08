import { redirect } from "react-router";
import type { Route } from "./+types/people.redirect";

export function loader({ request }: Route.LoaderArgs) {
  const cookieHeader = request.headers.get("Cookie");
  const viewMode = (cookieHeader?.match(/viewMode=(wall|grid)/)?.[1] as "wall" | "grid") || "grid";

  if (viewMode === "wall") {
    return redirect("/people/wall");
  } else {
    return redirect("/people/grid");
  }
}
