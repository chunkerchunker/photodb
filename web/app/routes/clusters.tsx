import { EyeOff, GitMerge, Loader2, Pencil, Search, Users, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useFetcher, useRevalidator } from "react-router";
import { ClusterMergeDialog } from "~/components/cluster-merge-dialog";
import { Layout } from "~/components/layout";
import { Button } from "~/components/ui/button";
import { Card, CardContent } from "~/components/ui/card";
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuTrigger,
} from "~/components/ui/context-menu";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "~/components/ui/dialog";
import { Input } from "~/components/ui/input";
import { getClusters, getClustersCount, getHiddenClustersCount } from "~/lib/db.server";
import type { Route } from "./+types/clusters";

export function meta() {
  return [
    { title: "PhotoDB - Face Clusters" },
    {
      name: "description",
      content: "Browse face clusters sorted by face count",
    },
  ];
}

const LIMIT = 24; // 4x6 grid

export async function loader({ request }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const page = parseInt(url.searchParams.get("page") || "1", 10);
  const offset = (page - 1) * LIMIT;

  try {
    const clusters = await getClusters(LIMIT, offset);
    const totalClusters = await getClustersCount();
    const hiddenCount = await getHiddenClustersCount();
    const hasMore = offset + clusters.length < totalClusters;

    return {
      clusters,
      totalClusters,
      hiddenCount,
      hasMore,
      page,
    };
  } catch (error) {
    console.error("Failed to load clusters:", error);
    return {
      clusters: [],
      totalClusters: 0,
      hiddenCount: 0,
      hasMore: false,
      page,
    };
  }
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
) {
  // Calculate scale to fit the face in the container (128px)
  const containerSize = 128;
  const scaleX = containerSize / bbox.bbox_width;
  const scaleY = containerSize / bbox.bbox_height;

  // Convert normalized coordinates to percentages for CSS positioning
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

type Cluster = Route.ComponentProps["loaderData"]["clusters"][number];

export default function ClustersView({ loaderData }: Route.ComponentProps) {
  const {
    clusters: initialClusters,
    totalClusters,
    hiddenCount,
    hasMore: initialHasMore,
    page: initialPage,
  } = loaderData;

  const [clusters, setClusters] = useState<Cluster[]>(initialClusters);
  const [page, setPage] = useState(initialPage);
  const [hasMore, setHasMore] = useState(initialHasMore);

  // Search state
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const searchInputRef = useRef<HTMLInputElement>(null);

  // Selection state (shift-click)
  const [selectedClusterIds, setSelectedClusterIds] = useState<Set<number>>(new Set());

  // Context menu state
  const [contextCluster, setContextCluster] = useState<Cluster | null>(null);
  const [renameDialogOpen, setRenameDialogOpen] = useState(false);
  const [mergeDialogOpen, setMergeDialogOpen] = useState(false);
  const [editFirstName, setEditFirstName] = useState("");
  const [editLastName, setEditLastName] = useState("");
  const [pendingHideClusterId, setPendingHideClusterId] = useState<number | null>(null);
  const renameFetcher = useFetcher();
  const hideFetcher = useFetcher();
  const revalidator = useRevalidator();

  // Drag and drop state
  const [draggingClusterId, setDraggingClusterId] = useState<number | null>(null);
  const [dropTargetClusterId, setDropTargetClusterId] = useState<number | null>(null);
  const [pendingMerge, setPendingMerge] = useState<{ sourceId: number; targetId: number } | null>(null);

  const fetcher = useFetcher<typeof loader>();
  const mergeFetcher = useFetcher();
  const loadMoreRef = useRef<HTMLDivElement>(null);

  // Reset state when initial data changes (e.g., navigation)
  useEffect(() => {
    setClusters(initialClusters);
    setPage(initialPage);
    setHasMore(initialHasMore);
  }, [initialClusters, initialPage, initialHasMore]);

  // Keyboard shortcut for search (Cmd+F / Ctrl+F)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "f") {
        e.preventDefault();
        setSearchOpen(true);
      }
      if (e.key === "Escape" && searchOpen) {
        setSearchOpen(false);
        setSearchQuery("");
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [searchOpen]);

  // Focus search input when opened
  useEffect(() => {
    if (searchOpen && searchInputRef.current) {
      searchInputRef.current.focus();
    }
  }, [searchOpen]);

  // Append new clusters when fetcher returns data
  useEffect(() => {
    if (fetcher.data && fetcher.data.clusters.length > 0) {
      setClusters((prev) => {
        const existingIds = new Set(prev.map((c) => c.id));
        const newClusters = fetcher.data!.clusters.filter((c) => !existingIds.has(c.id));
        return [...prev, ...newClusters];
      });
      setPage(fetcher.data.page);
      setHasMore(fetcher.data.hasMore);
    }
  }, [fetcher.data]);

  const loadMore = useCallback(() => {
    if (fetcher.state === "idle" && hasMore) {
      fetcher.load(`/clusters?page=${page + 1}`);
    }
  }, [fetcher, hasMore, page]);

  // Intersection Observer for infinite scroll
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          loadMore();
        }
      },
      { rootMargin: "200px" },
    );

    const currentRef = loadMoreRef.current;
    if (currentRef) {
      observer.observe(currentRef);
    }

    return () => {
      if (currentRef) {
        observer.unobserve(currentRef);
      }
    };
  }, [loadMore]);

  const isLoading = fetcher.state === "loading";
  const isMerging = mergeFetcher.state !== "idle";

  // Revalidate after drag-and-drop merge completes
  useEffect(() => {
    if (mergeFetcher.data?.success && pendingMerge) {
      setPendingMerge(null);
      revalidator.revalidate();
    }
  }, [mergeFetcher.data, pendingMerge, revalidator]);

  // Drag handlers
  const handleDragStart = (e: React.DragEvent, clusterId: number) => {
    setDraggingClusterId(clusterId);
    e.dataTransfer.effectAllowed = "move";
    e.dataTransfer.setData("text/plain", clusterId.toString());
  };

  const handleDragEnd = () => {
    setDraggingClusterId(null);
    setDropTargetClusterId(null);
  };

  const handleDragOver = (e: React.DragEvent, clusterId: number) => {
    e.preventDefault();
    if (draggingClusterId && draggingClusterId !== clusterId) {
      e.dataTransfer.dropEffect = "move";
      setDropTargetClusterId(clusterId);
    }
  };

  const handleDragLeave = () => {
    setDropTargetClusterId(null);
  };

  const handleDrop = (e: React.DragEvent, targetClusterId: number) => {
    e.preventDefault();
    // Try dataTransfer first, fall back to draggingClusterId state
    const dataTransferId = parseInt(e.dataTransfer.getData("text/plain"), 10);
    const sourceClusterId = !Number.isNaN(dataTransferId) ? dataTransferId : draggingClusterId;

    if (sourceClusterId && sourceClusterId !== targetClusterId) {
      setPendingMerge({ sourceId: sourceClusterId, targetId: targetClusterId });
    }
    setDraggingClusterId(null);
    setDropTargetClusterId(null);
  };

  const confirmMerge = () => {
    if (pendingMerge) {
      mergeFetcher.submit(
        {
          sourceClusterId: pendingMerge.sourceId.toString(),
          targetClusterId: pendingMerge.targetId.toString(),
        },
        { method: "post", action: "/api/clusters/merge" },
      );
    }
  };

  const cancelMerge = () => {
    setPendingMerge(null);
  };

  // Helper to get cluster display name
  const getClusterName = (clusterId: number) => {
    const cluster = clusters.find((c) => c.id === clusterId);
    return cluster?.person_name || `Cluster #${clusterId}`;
  };

  // Context menu handlers
  const handleRename = (cluster: Cluster) => {
    setContextCluster(cluster);
    // Parse existing name into first/last
    const parts = cluster.person_name?.split(" ") || [];
    setEditFirstName(parts[0] || "");
    setEditLastName(parts.slice(1).join(" ") || "");
    setRenameDialogOpen(true);
  };

  const handleHide = (cluster: Cluster) => {
    setPendingHideClusterId(cluster.id);
    hideFetcher.submit({ hidden: "true" }, { method: "post", action: `/api/cluster/${cluster.id}/hide` });
  };

  // Revalidate after hide completes and remove from local state
  useEffect(() => {
    if (hideFetcher.data?.success && pendingHideClusterId !== null) {
      // Remove the hidden cluster from local state immediately
      // This ensures the UI updates even if the cluster was loaded via infinite scroll
      setClusters((prev) => prev.filter((c) => c.id !== pendingHideClusterId));
      setPendingHideClusterId(null);
      // Reset pagination state since total count changed
      setPage(1);
      setHasMore(true);
      revalidator.revalidate();
    }
  }, [hideFetcher.data, pendingHideClusterId, revalidator]);

  const handleMerge = (cluster: Cluster) => {
    setContextCluster(cluster);
    setMergeDialogOpen(true);
  };

  const handleSaveRename = () => {
    if (contextCluster && editFirstName.trim()) {
      renameFetcher.submit(
        { firstName: editFirstName.trim(), lastName: editLastName.trim() },
        { method: "post", action: `/api/cluster/${contextCluster.id}/rename` },
      );
    }
  };

  // Update local state when rename completes
  useEffect(() => {
    if (renameFetcher.data?.success && contextCluster) {
      const newName = editLastName.trim() ? `${editFirstName.trim()} ${editLastName.trim()}` : editFirstName.trim();
      setClusters((prev) => prev.map((c) => (c.id === contextCluster.id ? { ...c, person_name: newName } : c)));
      setRenameDialogOpen(false);
      setContextCluster(null);
    }
  }, [renameFetcher.data, contextCluster, editFirstName, editLastName]);

  const handleMergeComplete = () => {
    setContextCluster(null);
    revalidator.revalidate();
  };

  // Toggle cluster selection (shift-click)
  const toggleClusterSelection = (clusterId: number, e: React.MouseEvent) => {
    if (e.shiftKey) {
      e.preventDefault();
      setSelectedClusterIds((prev) => {
        const next = new Set(prev);
        if (next.has(clusterId)) {
          next.delete(clusterId);
        } else {
          next.add(clusterId);
        }
        return next;
      });
    }
  };

  // Filter clusters based on search query, always including selected clusters
  const filteredClusters = searchQuery.trim()
    ? clusters.filter((cluster) => {
        // Always include selected clusters
        if (selectedClusterIds.has(cluster.id)) return true;
        const query = searchQuery.toLowerCase().trim();
        // Match cluster ID
        if (cluster.id.toString().includes(query)) return true;
        // Match person name (first or last)
        if (cluster.person_name?.toLowerCase().includes(query)) return true;
        return false;
      })
    : clusters;

  return (
    <Layout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Users className="h-8 w-8 text-gray-700" />
            <h1 className="text-3xl font-bold text-gray-900">Face Clusters</h1>
          </div>
          <div className="flex items-center space-x-4">
            {hiddenCount > 0 && (
              <Link to="/clusters/hidden">
                <Button variant="outline" size="sm">
                  <EyeOff className="h-4 w-4 mr-1" />
                  Hidden ({hiddenCount})
                </Button>
              </Link>
            )}
            <span className="text-gray-600">
              {totalClusters} cluster{totalClusters !== 1 ? "s" : ""}
            </span>
          </div>
        </div>

        {/* Search Box */}
        {searchOpen && (
          <div className="relative w-full max-w-lg mx-auto mb-4">
            <div className="bg-white rounded-lg shadow-lg border">
              <div className="flex items-center px-4 py-3">
                <Search className="h-5 w-5 text-gray-400 mr-3" />
                <input
                  ref={searchInputRef}
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search by name or cluster ID..."
                  className="flex-1 outline-none text-lg placeholder:text-gray-400"
                  autoComplete="off"
                />
                {searchQuery && (
                  <button
                    type="button"
                    onClick={() => setSearchQuery("")}
                    className="p-1 hover:bg-gray-100 rounded mr-2"
                  >
                    <X className="h-4 w-4 text-gray-400" />
                  </button>
                )}
                <button
                  type="button"
                  onClick={() => {
                    setSearchOpen(false);
                    setSearchQuery("");
                  }}
                  className="text-xs text-gray-400 hover:text-gray-600"
                >
                  <kbd className="px-1.5 py-0.5 bg-gray-100 rounded">Esc</kbd>
                </button>
              </div>
              {searchQuery && (
                <div className="px-4 py-2 text-xs text-gray-500 border-t">
                  {filteredClusters.length} result{filteredClusters.length !== 1 ? "s" : ""}
                </div>
              )}
            </div>
          </div>
        )}

        {clusters.length > 0 ? (
          <>
            {searchQuery && filteredClusters.length === 0 ? (
              <div className="text-center py-12">
                <Search className="h-12 w-12 text-gray-300 mx-auto mb-4" />
                <div className="text-gray-500">No clusters match "{searchQuery}"</div>
                <button
                  type="button"
                  onClick={() => setSearchQuery("")}
                  className="mt-2 text-sm text-blue-600 hover:underline"
                >
                  Clear search
                </button>
              </div>
            ) : (
              <>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-4">
                  {filteredClusters.map((cluster) => {
                    const isDragging = draggingClusterId === cluster.id;
                    const isDropTarget = dropTargetClusterId === cluster.id;
                    const isSelected = selectedClusterIds.has(cluster.id);

                    return (
                      <ContextMenu key={cluster.id}>
                        <ContextMenuTrigger asChild>
                          <div
                            role="button"
                            tabIndex={0}
                            draggable
                            onClick={(e) => toggleClusterSelection(cluster.id, e)}
                            onDragStart={(e) => handleDragStart(e, cluster.id)}
                            onDragEnd={handleDragEnd}
                            onDragOver={(e) => handleDragOver(e, cluster.id)}
                            onDragLeave={handleDragLeave}
                            onDrop={(e) => handleDrop(e, cluster.id)}
                            className={`transition-all ${isDragging ? "opacity-50" : ""}`}
                          >
                            <Link
                              to={`/cluster/${cluster.id}`}
                              draggable={false}
                              onClick={(e) => {
                                if (e.shiftKey) e.preventDefault();
                              }}
                            >
                              <Card
                                className={`hover:shadow-lg transition-all h-full ${
                                  isSelected
                                    ? "ring-2 ring-amber-500 ring-offset-2 bg-amber-50"
                                    : isDropTarget
                                      ? "ring-2 ring-blue-500 ring-offset-2 bg-blue-50"
                                      : ""
                                }`}
                              >
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
                                      <div
                                        className={`font-semibold truncate ${
                                          cluster.person_name ? "text-gray-900" : "text-blue-600"
                                        }`}
                                        title={cluster.person_name || `Cluster #${cluster.id}`}
                                      >
                                        {cluster.person_name || `Cluster #${cluster.id}`}
                                      </div>
                                      <div className="text-sm text-gray-600">
                                        {cluster.face_count} face{cluster.face_count !== 1 ? "s" : ""}
                                        {cluster.person_name && <span className="text-gray-400"> · #{cluster.id}</span>}
                                      </div>
                                    </div>
                                  </div>
                                </CardContent>
                              </Card>
                            </Link>
                          </div>
                        </ContextMenuTrigger>
                        <ContextMenuContent>
                          <ContextMenuItem onClick={() => handleRename(cluster)}>
                            <Pencil className="h-4 w-4 mr-2" />
                            Rename
                          </ContextMenuItem>
                          <ContextMenuItem onClick={() => handleHide(cluster)}>
                            <EyeOff className="h-4 w-4 mr-2" />
                            Hide
                          </ContextMenuItem>
                          <ContextMenuSeparator />
                          <ContextMenuItem onClick={() => handleMerge(cluster)}>
                            <GitMerge className="h-4 w-4 mr-2" />
                            Merge into...
                          </ContextMenuItem>
                        </ContextMenuContent>
                      </ContextMenu>
                    );
                  })}
                </div>

                {/* Infinite scroll trigger */}
                <div ref={loadMoreRef} className="flex justify-center py-8">
                  {isLoading && (
                    <div className="flex items-center space-x-2 text-gray-500">
                      <Loader2 className="h-5 w-5 animate-spin" />
                      <span>Loading more clusters...</span>
                    </div>
                  )}
                  {!hasMore && clusters.length > 0 && !searchQuery && (
                    <span className="text-gray-400 text-sm">All clusters loaded</span>
                  )}
                </div>
              </>
            )}
          </>
        ) : (
          <div className="text-center py-12">
            <Users className="h-16 w-16 text-gray-400 mx-auto mb-4" />
            <div className="text-gray-500 text-lg">No face clusters found in the database.</div>
            <div className="text-gray-400 text-sm mt-2">
              Clusters are created when faces are detected and grouped together.
            </div>
          </div>
        )}

        {/* Drag-and-drop Merge Confirmation Dialog */}
        <Dialog open={pendingMerge !== null} onOpenChange={(open) => !open && cancelMerge()}>
          <DialogContent showCloseButton={false} className="max-w-sm">
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2">
                <GitMerge className="h-5 w-5" />
                Merge Clusters
              </DialogTitle>
              <DialogDescription>
                {pendingMerge && (
                  <>
                    Merge <strong>{getClusterName(pendingMerge.sourceId)}</strong> into{" "}
                    <strong>{getClusterName(pendingMerge.targetId)}</strong>?
                    <br />
                    <br />
                    All faces will be moved to the target cluster.
                  </>
                )}
              </DialogDescription>
            </DialogHeader>
            <DialogFooter>
              <Button variant="outline" onClick={cancelMerge} disabled={isMerging}>
                Cancel
              </Button>
              <Button onClick={confirmMerge} disabled={isMerging}>
                {isMerging ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                    Merging...
                  </>
                ) : (
                  "Merge"
                )}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Rename Dialog */}
        <Dialog open={renameDialogOpen} onOpenChange={setRenameDialogOpen}>
          <DialogContent className="max-w-sm">
            <DialogHeader>
              <DialogTitle>{contextCluster?.person_name ? "Rename Cluster" : "Set Name"}</DialogTitle>
              <DialogDescription>Enter a name for this person.</DialogDescription>
            </DialogHeader>
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <label htmlFor="renameFirstName" className="text-sm font-medium text-gray-700">
                    First Name
                  </label>
                  <Input
                    id="renameFirstName"
                    placeholder="First name"
                    value={editFirstName}
                    onChange={(e) => setEditFirstName(e.target.value)}
                    autoComplete="off"
                    data-form-type="other"
                    data-1p-ignore
                    data-lpignore="true"
                    autoFocus
                  />
                </div>
                <div className="space-y-1">
                  <label htmlFor="renameLastName" className="text-sm font-medium text-gray-700">
                    Last Name
                  </label>
                  <Input
                    id="renameLastName"
                    placeholder="Last name"
                    value={editLastName}
                    onChange={(e) => setEditLastName(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        handleSaveRename();
                      }
                    }}
                    autoComplete="off"
                    data-form-type="other"
                    data-1p-ignore
                    data-lpignore="true"
                  />
                </div>
              </div>
              <DialogFooter>
                <Button variant="outline" onClick={() => setRenameDialogOpen(false)}>
                  Cancel
                </Button>
                <Button onClick={handleSaveRename} disabled={!editFirstName.trim() || renameFetcher.state !== "idle"}>
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
            </div>
          </DialogContent>
        </Dialog>

        {/* Context Menu Merge Dialog */}
        {contextCluster && (
          <ClusterMergeDialog
            open={mergeDialogOpen}
            onOpenChange={setMergeDialogOpen}
            sourceClusterId={contextCluster.id.toString()}
            sourceClusterName={contextCluster.person_name || `Cluster #${contextCluster.id}`}
            onMergeComplete={handleMergeComplete}
          />
        )}
      </div>
    </Layout>
  );
}
