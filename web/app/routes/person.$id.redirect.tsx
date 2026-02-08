import { redirect } from "react-router";
import type { Route } from "./+types/person.$id.redirect";

export function loader({ request, params }: Route.LoaderArgs) {
  const cookieHeader = request.headers.get("Cookie");
  const viewMode = (cookieHeader?.match(/viewMode=(wall|grid)/)?.[1] as "wall" | "grid") || "grid";

  if (viewMode === "wall") {
    return redirect(`/person/${params.id}/wall`);
  } else {
    return redirect(`/person/${params.id}/grid`);
  }
}
