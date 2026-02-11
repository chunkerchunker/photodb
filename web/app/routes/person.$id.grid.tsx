import { EyeOff, Pencil, Star, Unlink, User, Users } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useFetcher, useRevalidator } from "react-router";
import { Breadcrumb } from "~/components/breadcrumb";
import { Layout } from "~/components/layout";
import { RenamePersonDialog } from "~/components/rename-person-dialog";
import { SecondaryControls } from "~/components/secondary-controls";
import { Badge } from "~/components/ui/badge";
import { Button } from "~/components/ui/button";
import { Card, CardContent } from "~/components/ui/card";
import { ContextMenu, ContextMenuContent, ContextMenuItem, ContextMenuTrigger } from "~/components/ui/context-menu";
import { requireCollectionId } from "~/lib/auth.server";
import { dataWithViewMode } from "~/lib/cookies.server";
import { getClustersByPerson, getPersonById, unlinkClusterFromPerson } from "~/lib/db.server";
import type { Route } from "./+types/person.$id.grid";

export function meta({ data }: Route.MetaArgs) {
  const personName = data?.person?.person_name || "Person";
  return [
    { title: `Storyteller - ${personName}` },
    { name: "description", content: `View ${personName}'s photo clusters` },
  ];
}

export async function loader({ request, params }: Route.LoaderArgs) {
  const { collectionId } = await requireCollectionId(request);
  const personId = params.id;
  if (!personId) {
    throw new Response("Person ID required", { status: 400 });
  }

  const person = await getPersonById(collectionId, personId);
  if (!person) {
    throw new Response("Person not found", { status: 404 });
  }

  const clusters = await getClustersByPerson(collectionId, personId);

  return dataWithViewMode({ person, clusters }, "grid");
}

export async function action({ request, params }: Route.ActionArgs) {
  const { collectionId } = await requireCollectionId(request);
  const formData = await request.formData();
  const intent = formData.get("intent");
  const personId = params.id;

  if (intent === "unlink-cluster") {
    const clusterId = formData.get("clusterId") as string;
    if (!clusterId) {
      return { success: false, message: "Cluster ID required" };
    }
    return await unlinkClusterFromPerson(collectionId, clusterId);
  }

  return { success: false, message: "Unknown action" };
}

type Cluster = Awaited<ReturnType<typeof getClustersByPerson>>[number];

export default function PersonDetailView({ loaderData }: Route.ComponentProps) {
  const { person, clusters: initialClusters } = loaderData;
  const [clusters, setClusters] = useState<Cluster[]>(initialClusters);

  // Dialog state
  const [renameDialogOpen, setRenameDialogOpen] = useState(false);
  const [pendingUnlinkClusterId, setPendingUnlinkClusterId] = useState<string | null>(null);

  const hideFetcher = useFetcher();
  const unlinkFetcher = useFetcher();
  const representativeFetcher = useFetcher();
  const revalidator = useRevalidator();

  const isSubmitting =
    hideFetcher.state !== "idle" || unlinkFetcher.state !== "idle" || representativeFetcher.state !== "idle";

  // Reset clusters when loader data changes
  useEffect(() => {
    setClusters(initialClusters);
  }, [initialClusters]);

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

  const handleRenameSuccess = () => {
    revalidator.revalidate();
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
            {person.detection_id ? (
              <div className="w-24 h-24 bg-gray-100 rounded-lg border overflow-hidden">
                <img
                  src={`/api/face/${person.detection_id}`}
                  alt={person.person_name || `Person ${person.id}`}
                  className="w-full h-full object-cover"
                />
              </div>
            ) : (
              <div className="w-24 h-24 bg-gray-200 rounded-lg flex items-center justify-center">
                <User className="h-10 w-10 text-gray-400" />
              </div>
            )}
            <div>
              <div className="flex items-center gap-3">
                <h1 className="text-3xl font-bold text-gray-900">{person.person_name || `Person ${person.id}`}</h1>
                <button
                  onClick={() => setRenameDialogOpen(true)}
                  disabled={isSubmitting}
                  className="text-gray-400 hover:text-gray-600 disabled:opacity-50"
                >
                  <Pencil className="h-4 w-4" />
                </button>
              </div>
              <div className="text-gray-600 mt-1">
                {totalFaces} photo{totalFaces !== 1 ? "s" : ""} across {clusters.length} cluster
                {clusters.length !== 1 ? "s" : ""}
              </div>
            </div>
          </div>

          <SecondaryControls variant="grid">
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
          </SecondaryControls>
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
                                  {cluster.detection_id ? (
                                    <div className="w-32 h-32 mx-auto bg-gray-100 rounded-lg border overflow-hidden">
                                      <img
                                        src={`/api/face/${cluster.detection_id}`}
                                        alt={`Cluster ${cluster.id}`}
                                        className="w-full h-full object-cover"
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
                            {cluster.detection_id ? (
                              <div className="w-32 h-32 mx-auto bg-gray-100 rounded-lg border overflow-hidden">
                                <img
                                  src={`/api/face/${cluster.detection_id}`}
                                  alt={`Cluster ${cluster.id}`}
                                  className="w-full h-full object-cover grayscale"
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
        <RenamePersonDialog
          open={renameDialogOpen}
          onOpenChange={setRenameDialogOpen}
          personId={person.id.toString()}
          currentFirstName={person.first_name || ""}
          currentLastName={person.last_name || ""}
          onSuccess={handleRenameSuccess}
        />
      </div>
    </Layout>
  );
}
