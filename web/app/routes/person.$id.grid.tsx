import { EyeOff, Link2, Loader2, Pencil, Search, Sparkles, Star, Trash2, Unlink, User, Users } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { Link, useFetcher, useLocation, useNavigate, useRevalidator } from "react-router";
import { Breadcrumb } from "~/components/breadcrumb";
import { ClusterLinkDialog } from "~/components/cluster-merge-dialog";
import { DeletePersonDialog } from "~/components/delete-person-dialog";
import { Layout } from "~/components/layout";
import { RenamePersonDialog } from "~/components/rename-person-dialog";
import { SearchBox } from "~/components/search-box";
import { SecondaryControls } from "~/components/secondary-controls";
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
import { requireCollectionId } from "~/lib/auth.server";
import { dataWithViewMode } from "~/lib/cookies.server";
import { deletePersonRow, getClustersByPerson, getPersonById, unlinkClusterFromPerson } from "~/lib/db.server";
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

  if (intent === "delete-person") {
    if (!personId) {
      return { success: false, message: "Person ID required" };
    }
    return await deletePersonRow(collectionId, personId);
  }

  return { success: false, message: "Unknown action" };
}

type Cluster = Awaited<ReturnType<typeof getClustersByPerson>>[number];

export default function PersonDetailView({ loaderData }: Route.ComponentProps) {
  const { person, clusters: initialClusters } = loaderData;
  const [clusters, setClusters] = useState<Cluster[]>(initialClusters);

  // Dialog state
  const [renameDialogOpen, setRenameDialogOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [deletePersonDialogOpen, setDeletePersonDialogOpen] = useState(false);
  const [linkDialogOpen, setLinkDialogOpen] = useState(false);
  const [pendingUnlinkClusterId, setPendingUnlinkClusterId] = useState<string | null>(null);
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const navigate = useNavigate();
  const location = useLocation();

  // Navigate away from deleted person: go back if there's history, else to people index
  const navigateAway = useCallback(() => {
    if (location.key !== "default") {
      navigate(-1);
    } else {
      navigate("/people", { replace: true });
    }
  }, [location.key, navigate]);

  const hideFetcher = useFetcher();
  const searchFetcher = useFetcher();
  const unlinkFetcher = useFetcher();
  const representativeFetcher = useFetcher();
  const deletePersonFetcher = useFetcher();
  const revalidator = useRevalidator();

  const isSubmitting =
    hideFetcher.state !== "idle" ||
    unlinkFetcher.state !== "idle" ||
    representativeFetcher.state !== "idle" ||
    deletePersonFetcher.state !== "idle";

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
      if (unlinkFetcher.data.deletedPerson) {
        // Person was auto-created and now empty — navigate away
        navigateAway();
        return;
      }
      // Remove cluster from local state
      setClusters((prev) => prev.filter((c) => c.id.toString() !== pendingUnlinkClusterId));
      setPendingUnlinkClusterId(null);
    }
  }, [unlinkFetcher.data, pendingUnlinkClusterId, navigateAway]);

  // Revalidate after representative change
  useEffect(() => {
    if (representativeFetcher.data?.success) {
      revalidator.revalidate();
    }
  }, [representativeFetcher.data, revalidator]);

  // Navigate away after person deletion
  useEffect(() => {
    if (deletePersonFetcher.data?.success) {
      navigateAway();
    }
  }, [deletePersonFetcher.data, navigateAway]);

  // Debounced server-side search
  // biome-ignore lint/correctness/useExhaustiveDependencies: intentionally omit searchFetcher.load to avoid re-trigger loop in debounced search
  useEffect(() => {
    const trimmed = searchQuery.trim();
    if (!trimmed) return;
    const timer = setTimeout(() => {
      searchFetcher.load(`/clusters/grid?search=${encodeURIComponent(trimmed)}`);
    }, 300);
    return () => clearTimeout(timer);
  }, [searchQuery]);

  const isSearching = searchQuery.trim().length > 0;
  const searchResults = isSearching ? searchFetcher.data?.items || [] : [];
  const searchResultCount =
    isSearching && searchFetcher.state === "idle" && searchFetcher.data ? searchFetcher.data.totalItems : undefined;

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
          items={[
            { label: "People", href: "/people" },
            { label: person.auto_created ? "Auto-grouped" : person.person_name || `Person ${person.id}` },
          ]}
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
                {person.auto_created ? (
                  <span title="Auto-grouped">
                    <Sparkles className="h-7 w-7 text-gray-400" />
                  </span>
                ) : (
                  <h1 className="text-3xl font-bold text-gray-900">{person.person_name || `Person ${person.id}`}</h1>
                )}
                <button
                  type="button"
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
                Hide
              </Button>
            )}
            {hiddenClusters.length > 0 && (
              <Button variant="outline" size="sm" onClick={handleUnhideAll} disabled={isSubmitting}>
                Unhide ({hiddenClusters.length})
              </Button>
            )}
            {clusters.length > 0 && (
              <Button variant="outline" size="sm" onClick={() => setLinkDialogOpen(true)} disabled={isSubmitting}>
                <Link2 className="h-4 w-4 mr-1" />
                Same Person...
              </Button>
            )}
            {clusters.length > 0 ? (
              <Button variant="outline" size="sm" onClick={() => setDeleteDialogOpen(true)} disabled={isSubmitting}>
                <Trash2 className="h-4 w-4 mr-1" />
                Remove All Clusters
              </Button>
            ) : (
              <Button
                variant="outline"
                size="sm"
                onClick={() => setDeletePersonDialogOpen(true)}
                disabled={isSubmitting}
              >
                <Trash2 className="h-4 w-4 mr-1" />
                Delete Person
              </Button>
            )}
          </SecondaryControls>
        </div>

        <SearchBox
          open={searchOpen}
          onOpenChange={setSearchOpen}
          query={searchQuery}
          onQueryChange={setSearchQuery}
          placeholder="Search all people..."
          resultCount={searchResultCount}
        />

        {/* Search Results */}
        {isSearching ? (
          searchResults.length > 0 ? (
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-4">
              {searchResults.map(
                (item: {
                  item_type: string;
                  id: number;
                  person_name: string | null;
                  auto_created: boolean;
                  face_count: number;
                  cluster_count: number;
                  detection_id: number | null;
                }) => {
                  const isPerson = item.item_type === "person";
                  const linkTo = isPerson ? `/person/${item.id}` : `/cluster/${item.id}`;
                  return (
                    <Link key={`${item.item_type}:${item.id}`} to={linkTo}>
                      <Card className="hover:shadow-lg transition-all h-full">
                        <CardContent className="p-4">
                          <div className="text-center space-y-3">
                            {item.detection_id ? (
                              <div className="w-32 h-32 mx-auto bg-gray-100 rounded-lg border overflow-hidden">
                                <img
                                  src={`/api/face/${item.detection_id}`}
                                  alt={isPerson ? item.person_name || `Person ${item.id}` : `Cluster ${item.id}`}
                                  className="w-full h-full object-cover"
                                  loading="lazy"
                                />
                              </div>
                            ) : (
                              <div className="w-full h-32 bg-gray-200 rounded-lg flex items-center justify-center">
                                {isPerson ? (
                                  <User className="h-8 w-8 text-gray-400" />
                                ) : (
                                  <Users className="h-8 w-8 text-gray-400" />
                                )}
                              </div>
                            )}
                            <div className="space-y-1">
                              <div
                                className={`font-semibold truncate ${
                                  isPerson && item.auto_created
                                    ? "text-gray-400"
                                    : item.person_name
                                      ? "text-gray-900"
                                      : "text-blue-600"
                                }`}
                              >
                                {isPerson && item.auto_created ? (
                                  <Sparkles className="h-4 w-4 mx-auto text-gray-400" />
                                ) : (
                                  item.person_name || `Cluster #${item.id}`
                                )}
                              </div>
                              <div className="text-sm text-gray-600">
                                {item.face_count} photo{item.face_count !== 1 ? "s" : ""}
                                {isPerson && item.cluster_count > 1 && (
                                  <span className="text-gray-400"> · {item.cluster_count} clusters</span>
                                )}
                              </div>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    </Link>
                  );
                },
              )}
            </div>
          ) : searchFetcher.state !== "idle" ? (
            <div className="flex justify-center py-12">
              <div className="flex items-center space-x-2 text-gray-500">
                <Loader2 className="h-5 w-5 animate-spin" />
                <span>Searching...</span>
              </div>
            </div>
          ) : (
            <div className="text-center py-12">
              <Search className="h-12 w-12 text-gray-300 mx-auto mb-4" />
              <div className="text-gray-500">No results match &ldquo;{searchQuery}&rdquo;</div>
              <button
                type="button"
                onClick={() => setSearchQuery("")}
                className="mt-2 text-sm text-blue-600 hover:underline"
              >
                Clear search
              </button>
            </div>
          )
        ) : /* Clusters Grid */
        clusters.length > 0 ? (
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
          currentMiddleName={person.middle_name || ""}
          currentMaidenName={person.maiden_name || ""}
          currentPreferredName={person.preferred_name || ""}
          currentSuffix={person.suffix || ""}
          currentAlternateNames={person.alternate_names || []}
          onSuccess={handleRenameSuccess}
        />

        {/* Link Person Dialog */}
        {clusters.length > 0 && (
          <ClusterLinkDialog
            open={linkDialogOpen}
            onOpenChange={setLinkDialogOpen}
            sourceClusterId={clusters[0].id.toString()}
            sourceClusterName={person.person_name || `Person ${person.id}`}
            onLinkComplete={(personId) => {
              if (personId && personId !== person.id) {
                navigate(`/person/${personId}`, { replace: true });
              } else {
                revalidator.revalidate();
              }
            }}
          />
        )}

        {/* Remove All Clusters Dialog */}
        <DeletePersonDialog
          open={deleteDialogOpen}
          onOpenChange={setDeleteDialogOpen}
          personId={person.id.toString()}
          personName={person.person_name || `Person ${person.id}`}
          clusterCount={clusters.length}
          onSuccess={navigateAway}
        />

        {/* Delete Person Dialog (no clusters) */}
        <Dialog open={deletePersonDialogOpen} onOpenChange={setDeletePersonDialogOpen}>
          <DialogContent className="max-w-sm">
            <DialogHeader>
              <DialogTitle>Delete Person</DialogTitle>
              <DialogDescription>
                Permanently delete{" "}
                <span className="font-medium text-gray-700">{person.person_name || `Person ${person.id}`}</span>? This
                cannot be undone.
              </DialogDescription>
            </DialogHeader>
            <DialogFooter>
              <Button
                variant="outline"
                onClick={() => setDeletePersonDialogOpen(false)}
                disabled={deletePersonFetcher.state !== "idle"}
              >
                Cancel
              </Button>
              <Button
                variant="destructive"
                disabled={deletePersonFetcher.state !== "idle"}
                onClick={() => deletePersonFetcher.submit({ intent: "delete-person" }, { method: "post" })}
              >
                {deletePersonFetcher.state !== "idle" ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                    Deleting...
                  </>
                ) : (
                  <>
                    <Trash2 className="h-4 w-4 mr-1" />
                    Delete
                  </>
                )}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    </Layout>
  );
}
