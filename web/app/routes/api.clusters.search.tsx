import { searchClusters } from "~/lib/db.server";
import type { Route } from "./+types/api.clusters.search";

export async function loader({ request }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const query = url.searchParams.get("q") || "";
  const exclude = url.searchParams.get("exclude") || undefined;

  const clusters = await searchClusters(query, exclude, 20);

  return Response.json({ clusters });
}
