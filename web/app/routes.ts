import { index, type RouteConfig, route } from "@react-router/dev/routes";

export default [
  route("login", "routes/login.tsx"),
  route("logout", "routes/logout.tsx"),
  route("profile", "routes/profile.tsx"),
  route("collections", "routes/collections.tsx"),
  route("api/collections/switch", "routes/api.collections.switch.tsx"),
  route("api/user/update-profile", "routes/api.user.update-profile.tsx"),
  route("api/user/set-default-collection", "routes/api.user.set-default-collection.tsx"),
  route("api/collection/:id/set-member-person", "routes/api.collection.$id.set-member-person.tsx"),

  // Admin routes
  route("admin/users", "routes/admin.users.tsx"),
  route("api/admin/impersonate", "routes/api.admin.impersonate.tsx"),
  route("api/admin/stop-impersonate", "routes/api.admin.stop-impersonate.tsx"),

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
  route("clusters/hidden", "routes/clusters.hidden.redirect.tsx"),
  route("clusters/hidden/grid", "routes/clusters.hidden.grid.tsx"),
  route("clusters/hidden/wall", "routes/clusters.hidden.wall.tsx"),

  // People - Redirectors
  route("people", "routes/people.redirect.tsx"),
  route("person/:id", "routes/person.$id.redirect.tsx"),

  // People - Grid Views
  route("people/grid", "routes/people.grid.tsx"),
  route("person/:id/grid", "routes/person.$id.grid.tsx"),

  // People - Wall Views
  route("people/wall", "routes/people.wall.tsx"),
  route("person/:id/wall", "routes/person.$id.wall.tsx"),
  route("person/:id/family-tree", "routes/person.$id.family-tree.tsx"),

  // People - Hidden
  route("people/hidden", "routes/people.hidden.tsx"),

  // Albums - Redirectors
  route("albums", "routes/albums.redirect.tsx"),
  route("album/:id", "routes/album.$id.redirect.tsx"),

  // Albums - Grid Views
  route("albums/grid", "routes/albums.grid.tsx"),
  route("album/:id/grid", "routes/album.$id.grid.tsx"),

  // Albums - Wall Views
  route("albums/wall", "routes/albums.wall.tsx"),
  route("album/:id/wall", "routes/album.$id.wall.tsx"),

  route("photo/:id", "routes/photo.$id.tsx"),
  route("face/:id/similar", "routes/face.$id.similar.tsx"),
  route("api/image/:id", "routes/api.image.$id.tsx"),
  route("api/face/:id", "routes/api.face.$id.tsx"),
  route("api/clusters/search", "routes/api.clusters.search.tsx"),
  route("api/clusters/merge", "routes/api.clusters.merge.tsx"),
  route("api/clusters/link-preview", "routes/api.clusters.link-preview.tsx"),
  route("api/cluster/:id/hide", "routes/api.cluster.$id.hide.tsx"),
  route("api/cluster/:id/rename", "routes/api.cluster.$id.rename.tsx"),
  route("api/person/:id/rename", "routes/api.person.$id.rename.tsx"),
  route("api/person/:id/hide", "routes/api.person.$id.hide.tsx"),
  route("api/person/:id/representative", "routes/api.person.$id.representative.tsx"),
  route("api/person/:id/delete", "routes/api.person.$id.delete.tsx"),
  route("api/person/create-placeholder", "routes/api.person.create-placeholder.tsx"),
  route("api/person/merge-preview", "routes/api.person.merge-preview.tsx"),
  route("api/person/merge", "routes/api.person.merge.tsx"),
  route("api/person/:id/add-parent", "routes/api.person.$id.add-parent.tsx"),
  route("api/person/:id/add-child", "routes/api.person.$id.add-child.tsx"),
  route("api/person/:id/add-partner", "routes/api.person.$id.add-partner.tsx"),
  route("api/person/:id/remove-parent", "routes/api.person.$id.remove-parent.tsx"),
  route("api/person/:id/remove-partner", "routes/api.person.$id.remove-partner.tsx"),
] satisfies RouteConfig;
