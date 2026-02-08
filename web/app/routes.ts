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

  // Clusters - Redirectors
  route("clusters", "routes/clusters.redirect.tsx"),
  route("cluster/:id", "routes/cluster.$id.redirect.tsx"),

  // Clusters - Grid Views
  route("clusters/grid", "routes/clusters.grid.tsx"),
  route("cluster/:id/grid", "routes/cluster.$id.grid.tsx"),

  // Clusters - Wall Views
  route("clusters/wall", "routes/clusters.wall.tsx"),
  route("cluster/:id/wall", "routes/cluster.$id.wall.tsx"),

  // Clusters - Hidden
  route("clusters/hidden", "routes/clusters.hidden.tsx"),

  // People - Redirectors
  route("people", "routes/people.redirect.tsx"),
  route("person/:id", "routes/person.$id.redirect.tsx"),

  // People - Grid Views
  route("people/grid", "routes/people.grid.tsx"),
  route("person/:id/grid", "routes/person.$id.grid.tsx"),

  // People - Wall Views
  route("people/wall", "routes/people.wall.tsx"),
  route("person/:id/wall", "routes/person.$id.wall.tsx"),
  route("photo/:id", "routes/photo.$id.tsx"),
  route("face/:id/similar", "routes/face.$id.similar.tsx"),
  route("api/image/:id", "routes/api.image.$id.tsx"),
  route("api/clusters/search", "routes/api.clusters.search.tsx"),
  route("api/clusters/merge", "routes/api.clusters.merge.tsx"),
  route("api/clusters/link-preview", "routes/api.clusters.link-preview.tsx"),
  route("api/cluster/:id/hide", "routes/api.cluster.$id.hide.tsx"),
  route("api/cluster/:id/rename", "routes/api.cluster.$id.rename.tsx"),
  route("api/person/:id/rename", "routes/api.person.$id.rename.tsx"),
  route("api/person/:id/hide", "routes/api.person.$id.hide.tsx"),
  route("api/person/:id/representative", "routes/api.person.$id.representative.tsx"),
] satisfies RouteConfig;
