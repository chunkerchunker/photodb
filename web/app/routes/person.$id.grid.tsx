import { EyeOff, Grid, Loader2, Pencil, Star, Unlink, User, Users } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useFetcher, useRevalidator } from "react-router";
import { Breadcrumb } from "~/components/breadcrumb";
import { CoverflowIcon } from "~/components/coverflow-icon";
import { Layout } from "~/components/layout";
import { Badge } from "~/components/ui/badge";
import { Button } from "~/components/ui/button";
import { Card, CardContent } from "~/components/ui/card";
import { ContextMenu, ContextMenuContent, ContextMenuItem, ContextMenuTrigger } from "~/components/ui/context-menu";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "~/components/ui/dialog";
import { Input } from "~/components/ui/input";
import { dataWithViewMode } from "~/lib/cookies.server";
import { getClustersByPerson, getPersonById, unlinkClusterFromPerson } from "~/lib/db.server";
import type { Route } from "./+types/person.$id.grid";

export function meta({ data }: Route.MetaArgs) {
  const personName = data?.person?.person_name || "Person";
  return [
    { title: `PhotoDB - ${personName}` },
    { name: "description", content: `View ${personName}'s photo clusters` },
  ];
}

export async function loader({ params }: Route.LoaderArgs) {
  const personId = params.id;
  if (!personId) {
    throw new Response("Person ID required", { status: 400 });
  }

  const person = await getPersonById(personId);
  if (!person) {
    throw new Response("Person not found", { status: 404 });
  }

  const clusters = await getClustersByPerson(personId);

  return dataWithViewMode({ person, clusters }, "grid");
}

export async function action({ request, params }: Route.ActionArgs) {
  const formData = await request.formData();
  const intent = formData.get("intent");
  const personId = params.id;

  if (intent === "unlink-cluster") {
    const clusterId = formData.get("clusterId") as string;
    if (!clusterId) {
      return { success: false, message: "Cluster ID required" };
    }
    return await unlinkClusterFromPerson(clusterId);
  }

  return { success: false, message: "Unknown action" };
}

function getFaceCropStyle(
  bbox: {
    bbox_x: number;
    bbox_y: number;
    bbox_width: number;
    bbox_height: number;
  },
  imageWidth: number,
  imageHeight: number,
  containerSize = 128,
) {
  const scaleX = containerSize / bbox.bbox_width;
  const scaleY = containerSize / bbox.bbox_height;

  const left = -bbox.bbox_x * scaleX;
  const top = -bbox.bbox_y * scaleY;
  const width = imageWidth * scaleX;
  const height = imageHeight * scaleY;

  return {
    transform: `translate(${left}px, ${top}px)`,
    transformOrigin: "0 0",
    width: `${width}px`,
    height: `${height}px`,
  };
}

type Cluster = Awaited<ReturnType<typeof getClustersByPerson>>[number];

export default function PersonDetailView({ loaderData }: Route.ComponentProps) {
  const { person, clusters: initialClusters } = loaderData;
  const [clusters, setClusters] = useState<Cluster[]>(initialClusters);

  // Dialog state
  const [renameDialogOpen, setRenameDialogOpen] = useState(false);
  const [editFirstName, setEditFirstName] = useState(person.first_name || "");
  const [editLastName, setEditLastName] = useState(person.last_name || "");
  const [pendingUnlinkClusterId, setPendingUnlinkClusterId] = useState<string | null>(null);

  const renameFetcher = useFetcher();
  const hideFetcher = useFetcher();
  const unlinkFetcher = useFetcher();
  const representativeFetcher = useFetcher();
  const revalidator = useRevalidator();

  const isSubmitting =
    renameFetcher.state !== "idle" ||
    hideFetcher.state !== "idle" ||
    unlinkFetcher.state !== "idle" ||
    representativeFetcher.state !== "idle";

  // Reset clusters when loader data changes
  useEffect(() => {
    setClusters(initialClusters);
  }, [initialClusters]);

  // Revalidate after rename completes
  useEffect(() => {
    if (renameFetcher.data?.success) {
      setRenameDialogOpen(false);
      revalidator.revalidate();
    }
  }, [renameFetcher.data, revalidator]);

  // Revalidate after hide completes
  useEffect(() => {
    if (hideFetcher.data?.success) {
      revalidator.revalidate();
    }
  }, [hideFetcher.data, revalidator]);

  // Handle unlink completion
  useEffect(() => {
    if (unlinkFetcher.data?.success && pendingUnlinkClusterId) {
      // Remove cluster from local state
      setClusters((prev) => prev.filter((c) => c.id.toString() !== pendingUnlinkClusterId));
      setPendingUnlinkClusterId(null);
    }
  }, [unlinkFetcher.data, pendingUnlinkClusterId]);

  // Revalidate after representative change
  useEffect(() => {
    if (representativeFetcher.data?.success) {
      revalidator.revalidate();
    }
  }, [representativeFetcher.data, revalidator]);

  const handleSaveRename = () => {
    if (editFirstName.trim()) {
      renameFetcher.submit(
        { firstName: editFirstName.trim(), lastName: editLastName.trim() },
        { method: "post", action: `/api/person/${person.id}/rename` },
      );
    }
  };

  const handleHideAll = () => {
    hideFetcher.submit({ hidden: "true" }, { method: "post", action: `/api/person/${person.id}/hide` });
  };

  const handleUnhideAll = () => {
    hideFetcher.submit({ hidden: "false" }, { method: "post", action: `/api/person/${person.id}/hide` });
  };

  const handleUnlinkCluster = (clusterId: string) => {
    setPendingUnlinkClusterId(clusterId);
    unlinkFetcher.submit({ intent: "unlink-cluster", clusterId }, { method: "post" });
  };

  const handleSetRepresentative = (clusterId: string) => {
    representativeFetcher.submit({ clusterId }, { method: "post", action: `/api/person/${person.id}/representative` });
  };

  const visibleClusters = clusters.filter((c) => !c.hidden);
  const hiddenClusters = clusters.filter((c) => c.hidden);
  const totalFaces = clusters.reduce((sum, c) => sum + (c.face_count || 0), 0);

  return (
    <Layout>
      <div className="space-y-6">
        <Breadcrumb
          items={[{ label: "People", href: "/people" }, { label: person.person_name || `Person ${person.id}` }]}
        />

        {/* Person Header */}
        <div className="flex items-start justify-between">
          <div className="flex items-center space-x-4">
            {person.photo_id && person.bbox_x !== null && person.normalized_width && person.normalized_height ? (
              <div className="relative w-24 h-24 bg-gray-100 rounded-lg border overflow-hidden">
                <img
                  src={`/api/image/${person.photo_id}`}
                  alt={person.person_name || `Person ${person.id}`}
                  className="absolute max-w-none max-h-none"
                  style={getFaceCropStyle(
                    {
                      bbox_x: person.bbox_x,
                      bbox_y: person.bbox_y,
                      bbox_width: person.bbox_width,
                      bbox_height: person.bbox_height,
                    },
                    person.normalized_width,
                    person.normalized_height,
                    96,
                  )}
                />
              </div>
            ) : (
              <div className="w-24 h-24 bg-gray-200 rounded-lg flex items-center justify-center">
                <User className="h-10 w-10 text-gray-400" />
              </div>
            )}
            <div>
              <h1 className="text-3xl font-bold text-gray-900">{person.person_name || `Person ${person.id}`}</h1>
              <div className="text-gray-600 mt-1">
                {totalFaces} photo{totalFaces !== 1 ? "s" : ""} across {clusters.length} cluster
                {clusters.length !== 1 ? "s" : ""}
              </div>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <div className="flex items-center rounded-lg border bg-gray-50 p-1" title="View mode">
              <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium bg-white text-gray-900 shadow-sm">
                <Grid className="h-4 w-4" />
              </div>
              <Link
                to={`/person/${person.id}/wall`}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium text-gray-500 hover:text-gray-700 transition-colors"
              >
                <CoverflowIcon className="size-4" />
              </Link>
            </div>
            <Button variant="outline" size="sm" onClick={() => setRenameDialogOpen(true)} disabled={isSubmitting}>
              <Pencil className="h-4 w-4 mr-1" />
              Rename
            </Button>
            {visibleClusters.length > 0 && (
              <Button variant="outline" size="sm" onClick={handleHideAll} disabled={isSubmitting}>
                <EyeOff className="h-4 w-4 mr-1" />
                Hide All
              </Button>
            )}
            {hiddenClusters.length > 0 && (
              <Button variant="outline" size="sm" onClick={handleUnhideAll} disabled={isSubmitting}>
                Unhide All ({hiddenClusters.length})
              </Button>
            )}
          </div>
        </div>

        {/* Clusters Grid */}
        {clusters.length > 0 ? (
          <div className="space-y-6">
            {/* Visible Clusters */}
            {visibleClusters.length > 0 && (
              <div>
                <h2 className="text-lg font-semibold text-gray-800 mb-4">Clusters</h2>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-4">
                  {visibleClusters.map((cluster) => (
                    <ContextMenu key={cluster.id}>
                      <ContextMenuTrigger asChild>
                        <div>
                          <Link to={`/cluster/${cluster.id}`}>
                            <Card className="hover:shadow-lg transition-all h-full">
                              <CardContent className="p-4">
                                <div className="text-center space-y-3">
                                  {cluster.photo_id &&
                                  cluster.bbox_x !== null &&
                                  cluster.normalized_width &&
                                  cluster.normalized_height ? (
                                    <div className="relative w-32 h-32 mx-auto bg-gray-100 rounded-lg border overflow-hidden">
                                      <img
                                        src={`/api/image/${cluster.photo_id}`}
                                        alt={`Cluster ${cluster.id}`}
                                        className="absolute max-w-none max-h-none"
                                        style={getFaceCropStyle(
                                          {
                                            bbox_x: cluster.bbox_x,
                                            bbox_y: cluster.bbox_y,
                                            bbox_width: cluster.bbox_width,
                                            bbox_height: cluster.bbox_height,
                                          },
                                          cluster.normalized_width,
                                          cluster.normalized_height,
                                        )}
                                        loading="lazy"
                                      />
                                    </div>
                                  ) : (
                                    <div className="w-full h-32 bg-gray-200 rounded-lg flex items-center justify-center">
                                      <Users className="h-8 w-8 text-gray-400" />
                                    </div>
                                  )}

                                  <div className="space-y-1">
                                    <div className="text-sm text-gray-600">
                                      {cluster.face_count} photo{cluster.face_count !== 1 ? "s" : ""}
                                    </div>
                                    {cluster.verified && <Badge variant="secondary">Verified</Badge>}
                                  </div>
                                </div>
                              </CardContent>
                            </Card>
                          </Link>
                        </div>
                      </ContextMenuTrigger>
                      <ContextMenuContent>
                        <ContextMenuItem onClick={() => handleSetRepresentative(cluster.id.toString())}>
                          <Star className="h-4 w-4 mr-2" />
                          Use as Person Photo
                        </ContextMenuItem>
                        <ContextMenuItem onClick={() => handleUnlinkCluster(cluster.id.toString())}>
                          <Unlink className="h-4 w-4 mr-2" />
                          Unlink from Person
                        </ContextMenuItem>
                      </ContextMenuContent>
                    </ContextMenu>
                  ))}
                </div>
              </div>
            )}

            {/* Hidden Clusters */}
            {hiddenClusters.length > 0 && (
              <div>
                <h2 className="text-lg font-semibold text-gray-500 mb-4">Hidden Clusters ({hiddenClusters.length})</h2>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-4 opacity-50">
                  {hiddenClusters.map((cluster) => (
                    <Link key={cluster.id} to={`/cluster/${cluster.id}`}>
                      <Card className="hover:shadow-lg transition-all h-full">
                        <CardContent className="p-4">
                          <div className="text-center space-y-3">
                            {cluster.photo_id &&
                            cluster.bbox_x !== null &&
                            cluster.normalized_width &&
                            cluster.normalized_height ? (
                              <div className="relative w-32 h-32 mx-auto bg-gray-100 rounded-lg border overflow-hidden">
                                <img
                                  src={`/api/image/${cluster.photo_id}`}
                                  alt={`Cluster ${cluster.id}`}
                                  className="absolute max-w-none max-h-none grayscale"
                                  style={getFaceCropStyle(
                                    {
                                      bbox_x: cluster.bbox_x,
                                      bbox_y: cluster.bbox_y,
                                      bbox_width: cluster.bbox_width,
                                      bbox_height: cluster.bbox_height,
                                    },
                                    cluster.normalized_width,
                                    cluster.normalized_height,
                                  )}
                                  loading="lazy"
                                />
                              </div>
                            ) : (
                              <div className="w-full h-32 bg-gray-200 rounded-lg flex items-center justify-center">
                                <Users className="h-8 w-8 text-gray-400" />
                              </div>
                            )}

                            <div className="text-sm text-gray-500">
                              {cluster.face_count} photo{cluster.face_count !== 1 ? "s" : ""}
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    </Link>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="text-center py-12">
            <Users className="h-16 w-16 text-gray-400 mx-auto mb-4" />
            <div className="text-gray-500 text-lg">No clusters found for this person.</div>
          </div>
        )}

        {/* Rename Dialog */}
        <Dialog open={renameDialogOpen} onOpenChange={setRenameDialogOpen}>
          <DialogContent className="max-w-sm">
            <DialogHeader>
              <DialogTitle>Rename Person</DialogTitle>
              <DialogDescription>Enter the person's name.</DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <label htmlFor="firstName" className="text-sm font-medium">
                  First Name
                </label>
                <Input
                  id="firstName"
                  value={editFirstName}
                  onChange={(e) => setEditFirstName(e.target.value)}
                  placeholder="First name"
                  autoFocus
                />
              </div>
              <div className="space-y-2">
                <label htmlFor="lastName" className="text-sm font-medium">
                  Last Name
                </label>
                <Input
                  id="lastName"
                  value={editLastName}
                  onChange={(e) => setEditLastName(e.target.value)}
                  placeholder="Last name (optional)"
                />
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setRenameDialogOpen(false)}>
                Cancel
              </Button>
              <Button onClick={handleSaveRename} disabled={!editFirstName.trim() || isSubmitting}>
                {renameFetcher.state !== "idle" ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                    Saving...
                  </>
                ) : (
                  "Save"
                )}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    </Layout>
  );
}
