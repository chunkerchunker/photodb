import { index, type RouteConfig, route } from "@react-router/dev/routes";

export default [
  route("login", "routes/login.tsx"),
  route("logout", "routes/logout.tsx"),

  // Redirectors (Root paths)
  index("routes/home.redirect.tsx"),
  route("year/:year", "routes/year.redirect.tsx"),
  route("year/:year/month/:month", "routes/month.redirect.tsx"),

  // Grid Views
  route("grid", "routes/home.grid.tsx"),
  route("year/:year/grid", "routes/year.grid.tsx"),
  route("year/:year/month/:month/grid", "routes/month.grid.tsx"),

  // Wall Views
  route("wall", "routes/home.wall.tsx"),
  route("year/:year/wall", "routes/year.wall.tsx"),
  route("year/:year/month/:month/wall", "routes/month.wall.tsx"),

  // Other routes
  route("clusters", "routes/clusters.tsx"),
  route("clusters/hidden", "routes/clusters.hidden.tsx"),
  route("people", "routes/people.tsx"),
  route("cluster/:id", "routes/cluster.$id.tsx"),
  route("photo/:id", "routes/photo.$id.tsx"),
  route("face/:id/similar", "routes/face.$id.similar.tsx"),
  route("api/image/:id", "routes/api.image.$id.tsx"),
  route("api/clusters/search", "routes/api.clusters.search.tsx"),
  route("api/clusters/merge", "routes/api.clusters.merge.tsx"),
  route("api/cluster/:id/hide", "routes/api.cluster.$id.hide.tsx"),
  route("api/cluster/:id/rename", "routes/api.cluster.$id.rename.tsx"),
] satisfies RouteConfig;
