import { index, type RouteConfig, route } from "@react-router/dev/routes";

export default [
  index("routes/home.tsx"),
  route("wall", "routes/home.wall.tsx"),
  route("clusters", "routes/clusters.tsx"),
  route("clusters/hidden", "routes/clusters.hidden.tsx"),
  route("cluster/:id", "routes/cluster.$id.tsx"),
  route("year/:year", "routes/year.tsx"),
  route("year/:year/wall", "routes/year.wall.tsx"),
  route("year/:year/month/:month", "routes/month.tsx"),
  route("year/:year/month/:month/wall", "routes/month.wall.tsx"),
  route("photo/:id", "routes/photo.$id.tsx"),
  route("face/:id/similar", "routes/face.$id.similar.tsx"),
  route("api/image/:id", "routes/api.image.$id.tsx"),
  route("api/clusters/search", "routes/api.clusters.search.tsx"),
  route("api/clusters/merge", "routes/api.clusters.merge.tsx"),
] satisfies RouteConfig;
