import { index, type RouteConfig, route } from "@react-router/dev/routes";

export default [
  index("routes/home.tsx"),
  route("clusters", "routes/clusters.tsx"),
  route("cluster/:id", "routes/cluster.$id.tsx"),
  route("year/:year", "routes/year.tsx"),
  route("year/:year/month/:month", "routes/month.tsx"),
  route("photo/:id", "routes/photo.$id.tsx"),
  route("api/image/:id", "routes/api.image.$id.tsx"),
] satisfies RouteConfig;
