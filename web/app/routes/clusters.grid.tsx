import { AlertTriangle, EyeOff, Grid, Link2, Loader2, Pencil, Search, User, Users, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useFetcher, useRevalidator } from "react-router";
import { ClusterLinkDialog } from "~/components/cluster-merge-dialog";
import { CoverflowIcon } from "~/components/coverflow-icon";
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
import { dataWithViewMode } from "~/lib/cookies.server";
import { getClustersGroupedByPerson, getClustersGroupedCount, getHiddenClustersCount } from "~/lib/db.server";
import type { Route } from "./+types/clusters.grid";

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
    const items = await getClustersGroupedByPerson(LIMIT, offset);
    const totalItems = await getClustersGroupedCount();
    const hiddenCount = await getHiddenClustersCount();
    const hasMore = offset + items.length < totalItems;

    return dataWithViewMode(
      {
        items,
        totalItems,
        hiddenCount,
        hasMore,
        page,
      },
      "grid",
    );
  } catch (error) {
    console.error("Failed to load clusters:", error);
    return dataWithViewMode(
      {
        items: [],
        totalItems: 0,
        hiddenCount: 0,
        hasMore: false,
        page,
      },
      "grid",
    );
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

type Item = Route.ComponentProps["loaderData"]["items"][number];

export default function ClustersView({ loaderData }: Route.ComponentProps) {
  const { items: initialItems, totalItems, hiddenCount, hasMore: initialHasMore, page: initialPage } = loaderData;

  const [items, setItems] = useState<Item[]>(initialItems);
  const [page, setPage] = useState(initialPage);
  const [hasMore, setHasMore] = useState(initialHasMore);

  // Search state
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const searchInputRef = useRef<HTMLInputElement>(null);

  // Selection state (shift-click) - stores "type:id" keys for uniqueness
  const [selectedItemKeys, setSelectedItemKeys] = useState<Set<string>>(new Set());

  // Context menu state
  const [contextItem, setContextItem] = useState<Item | null>(null);
  const [renameDialogOpen, setRenameDialogOpen] = useState(false);
  const [linkDialogOpen, setLinkDialogOpen] = useState(false);
  const [editFirstName, setEditFirstName] = useState("");
  const [editLastName, setEditLastName] = useState("");
  const [pendingHideClusterId, setPendingHideClusterId] = useState<number | null>(null);
  const [pendingHidePersonId, setPendingHidePersonId] = useState<number | null>(null);
  const renameFetcher = useFetcher();
  const hideFetcher = useFetcher();
  const revalidator = useRevalidator();

  // Drag and drop state - uses item key ("person:1" or "cluster:2") for visual state
  // but cluster IDs for actual linking operations
  const [draggingItemKey, setDraggingItemKey] = useState<string | null>(null);
  const [dropTargetItemKey, setDropTargetItemKey] = useState<string | null>(null);
  // pendingLink uses cluster IDs (primary_cluster_id for persons)
  const [pendingLink, setPendingLink] = useState<{ sourceId: number; targetId: number } | null>(null);
  const [linkPreview, setLinkPreview] = useState<{
    willMergePersons: boolean;
    source?: { personName: string | null; personClusterCount: number };
    target?: { personName: string | null; personClusterCount: number };
  } | null>(null);
  const [isLoadingPreview, setIsLoadingPreview] = useState(false);

  const fetcher = useFetcher<typeof loader>();
  const linkFetcher = useFetcher();
  const loadMoreRef = useRef<HTMLDivElement>(null);

  // Reset state when initial data changes (e.g., navigation)
  useEffect(() => {
    setItems(initialItems);
    setPage(initialPage);
    setHasMore(initialHasMore);
  }, [initialItems, initialPage, initialHasMore]);

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

  // Append new items when fetcher returns data
  useEffect(() => {
    if (fetcher.data && fetcher.data.items.length > 0) {
      setItems((prev) => {
        const existingKeys = new Set(prev.map((item) => `${item.item_type}:${item.id}`));
        const newItems = fetcher.data!.items.filter((item) => !existingKeys.has(`${item.item_type}:${item.id}`));
        return [...prev, ...newItems];
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
  const isLinking = linkFetcher.state !== "idle";

  // Revalidate after drag-and-drop link completes
  useEffect(() => {
    if (linkFetcher.data?.success && pendingLink) {
      setPendingLink(null);
      revalidator.revalidate();
    }
  }, [linkFetcher.data, pendingLink, revalidator]);

  // Helper to get item key
  const getItemKey = (item: Item) => `${item.item_type}:${item.id}`;

  // Drag handlers
  const handleDragStart = (e: React.DragEvent, item: Item) => {
    const itemKey = getItemKey(item);
    setDraggingItemKey(itemKey);
    e.dataTransfer.effectAllowed = "move";
    // Store both item key and cluster ID for the drop handler
    e.dataTransfer.setData("text/plain", `${itemKey}|${item.primary_cluster_id}`);
  };

  const handleDragEnd = () => {
    setDraggingItemKey(null);
    setDropTargetItemKey(null);
  };

  const handleDragOver = (e: React.DragEvent, item: Item) => {
    e.preventDefault();
    const itemKey = getItemKey(item);
    if (draggingItemKey && draggingItemKey !== itemKey) {
      e.dataTransfer.dropEffect = "move";
      setDropTargetItemKey(itemKey);
    }
  };

  const handleDragLeave = () => {
    setDropTargetItemKey(null);
  };

  const handleDrop = (e: React.DragEvent, targetItem: Item) => {
    e.preventDefault();
    const targetItemKey = getItemKey(targetItem);

    // Parse dataTransfer: "type:id|clusterId"
    const data = e.dataTransfer.getData("text/plain");
    const [sourceItemKey, sourceClusterIdStr] = data.split("|");
    const sourceClusterId = parseInt(sourceClusterIdStr, 10);
    const targetClusterId = targetItem.primary_cluster_id;

    if (sourceItemKey && sourceItemKey !== targetItemKey && sourceClusterId && targetClusterId) {
      setPendingLink({ sourceId: sourceClusterId, targetId: targetClusterId });
      // Fetch preview to check if persons will be merged
      setIsLoadingPreview(true);
      setLinkPreview(null);
      fetch(`/api/clusters/link-preview?source=${sourceClusterId}&target=${targetClusterId}`)
        .then((res) => res.json())
        .then((preview) => {
          setLinkPreview(preview);
        })
        .catch((err) => {
          console.error("Failed to fetch link preview:", err);
        })
        .finally(() => {
          setIsLoadingPreview(false);
        });
    }
    setDraggingItemKey(null);
    setDropTargetItemKey(null);
  };

  const confirmLink = () => {
    if (pendingLink) {
      linkFetcher.submit(
        {
          sourceClusterId: pendingLink.sourceId.toString(),
          targetClusterId: pendingLink.targetId.toString(),
        },
        { method: "post", action: "/api/clusters/merge" },
      );
    }
  };

  const cancelLink = () => {
    setPendingLink(null);
    setLinkPreview(null);
  };

  // Helper to get item display name from cluster ID
  const getItemNameByClusterId = (clusterId: number) => {
    // Find the item that has this cluster ID as its primary_cluster_id
    const item = items.find((i) => i.primary_cluster_id === clusterId);
    if (!item) return `Cluster #${clusterId}`;
    return item.person_name || `Cluster #${clusterId}`;
  };

  // Context menu handlers
  const handleRename = (item: Item) => {
    setContextItem(item);
    // Parse existing name into first/last
    const parts = item.person_name?.split(" ") || [];
    setEditFirstName(parts[0] || "");
    setEditLastName(parts.slice(1).join(" ") || "");
    setRenameDialogOpen(true);
  };

  const handleHide = (item: Item) => {
    if (item.item_type === "person") {
      // Hide all clusters for this person
      setPendingHidePersonId(item.id);
      hideFetcher.submit({ hidden: "true" }, { method: "post", action: `/api/person/${item.id}/hide` });
    } else {
      // Hide single cluster
      setPendingHideClusterId(item.id);
      hideFetcher.submit({ hidden: "true" }, { method: "post", action: `/api/cluster/${item.id}/hide` });
    }
  };

  // Revalidate after hide completes and remove from local state
  useEffect(() => {
    if (hideFetcher.data?.success) {
      if (pendingHideClusterId !== null) {
        // Remove the hidden cluster from local state immediately
        setItems((prev) => prev.filter((item) => !(item.item_type === "cluster" && item.id === pendingHideClusterId)));
        setPendingHideClusterId(null);
      }
      if (pendingHidePersonId !== null) {
        // Remove the hidden person from local state immediately
        setItems((prev) => prev.filter((item) => !(item.item_type === "person" && item.id === pendingHidePersonId)));
        setPendingHidePersonId(null);
      }
      // Reset pagination state since total count changed
      setPage(1);
      setHasMore(true);
      revalidator.revalidate();
    }
  }, [hideFetcher.data, pendingHideClusterId, pendingHidePersonId, revalidator]);

  const handleLink = (item: Item) => {
    setContextItem(item);
    setLinkDialogOpen(true);
  };

  const handleSaveRename = () => {
    if (contextItem && editFirstName.trim()) {
      const action =
        contextItem.item_type === "person"
          ? `/api/person/${contextItem.id}/rename`
          : `/api/cluster/${contextItem.id}/rename`;
      renameFetcher.submit(
        { firstName: editFirstName.trim(), lastName: editLastName.trim() },
        { method: "post", action },
      );
    }
  };

  // Update local state when rename completes
  useEffect(() => {
    if (renameFetcher.data?.success && contextItem) {
      const newName = editLastName.trim() ? `${editFirstName.trim()} ${editLastName.trim()}` : editFirstName.trim();
      setItems((prev) =>
        prev.map((item) =>
          item.item_type === contextItem.item_type && item.id === contextItem.id
            ? { ...item, person_name: newName }
            : item,
        ),
      );
      setRenameDialogOpen(false);
      setContextItem(null);
    }
  }, [renameFetcher.data, contextItem, editFirstName, editLastName]);

  const handleLinkComplete = () => {
    setContextItem(null);
    revalidator.revalidate();
  };

  // Toggle item selection (shift-click)
  const toggleItemSelection = (item: Item, e: React.MouseEvent) => {
    if (e.shiftKey) {
      e.preventDefault();
      const itemKey = getItemKey(item);
      setSelectedItemKeys((prev) => {
        const next = new Set(prev);
        if (next.has(itemKey)) {
          next.delete(itemKey);
        } else {
          next.add(itemKey);
        }
        return next;
      });
    }
  };

  // Filter items based on search query, always including selected items
  const filteredItems = searchQuery.trim()
    ? items.filter((item) => {
        const itemKey = getItemKey(item);
        // Always include selected items
        if (selectedItemKeys.has(itemKey)) return true;
        const query = searchQuery.toLowerCase().trim();
        // Match ID
        if (item.id.toString().includes(query)) return true;
        // Match person name
        if (item.person_name?.toLowerCase().includes(query)) return true;
        return false;
      })
    : items;

  return (
    <Layout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Users className="h-8 w-8 text-gray-700" />
            <h1 className="text-3xl font-bold text-gray-900">Face Clusters</h1>
          </div>
          <div className="flex items-center space-x-4">
            <div className="flex items-center rounded-lg border bg-gray-50 p-1">
              <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium bg-white text-gray-900 shadow-sm">
                <Grid className="h-4 w-4" />
              </div>
              <Link
                to="/clusters/wall"
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium text-gray-500 hover:text-gray-700 transition-colors"
              >
                <CoverflowIcon className="size-4" />
              </Link>
            </div>
            {hiddenCount > 0 && (
              <Link to="/clusters/hidden">
                <Button variant="outline" size="sm">
                  <EyeOff className="h-4 w-4 mr-1" />
                  Hidden ({hiddenCount})
                </Button>
              </Link>
            )}
            <span className="text-gray-600">
              {totalItems} item{totalItems !== 1 ? "s" : ""}
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
                  {filteredItems.length} result{filteredItems.length !== 1 ? "s" : ""}
                </div>
              )}
            </div>
          </div>
        )}

        {items.length > 0 ? (
          <>
            {searchQuery && filteredItems.length === 0 ? (
              <div className="text-center py-12">
                <Search className="h-12 w-12 text-gray-300 mx-auto mb-4" />
                <div className="text-gray-500">No results match "{searchQuery}"</div>
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
                  {filteredItems.map((item) => {
                    const itemKey = getItemKey(item);
                    const isDragging = draggingItemKey === itemKey;
                    const isDropTarget = dropTargetItemKey === itemKey;
                    const isSelected = selectedItemKeys.has(itemKey);
                    const isPerson = item.item_type === "person";
                    const linkTo = isPerson ? `/person/${item.id}` : `/cluster/${item.id}`;

                    return (
                      <ContextMenu key={itemKey}>
                        <ContextMenuTrigger asChild>
                          <div
                            role="button"
                            tabIndex={0}
                            draggable
                            onClick={(e) => toggleItemSelection(item, e)}
                            onDragStart={(e) => handleDragStart(e, item)}
                            onDragEnd={handleDragEnd}
                            onDragOver={(e) => handleDragOver(e, item)}
                            onDragLeave={handleDragLeave}
                            onDrop={(e) => handleDrop(e, item)}
                            className={`transition-all ${isDragging ? "opacity-50" : ""}`}
                          >
                            <Link
                              to={linkTo}
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
                                    {item.photo_id &&
                                    item.bbox_x !== null &&
                                    item.normalized_width &&
                                    item.normalized_height ? (
                                      <div className="relative w-32 h-32 mx-auto bg-gray-100 rounded-lg border overflow-hidden">
                                        <img
                                          src={`/api/image/${item.photo_id}`}
                                          alt={
                                            isPerson ? item.person_name || `Person ${item.id}` : `Cluster ${item.id}`
                                          }
                                          className="absolute max-w-none max-h-none"
                                          style={getFaceCropStyle(
                                            {
                                              bbox_x: item.bbox_x,
                                              bbox_y: item.bbox_y,
                                              bbox_width: item.bbox_width,
                                              bbox_height: item.bbox_height,
                                            },
                                            item.normalized_width,
                                            item.normalized_height,
                                          )}
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
                                          item.person_name ? "text-gray-900" : "text-blue-600"
                                        }`}
                                        title={item.person_name || `Cluster #${item.id}`}
                                      >
                                        {item.person_name || `Cluster #${item.id}`}
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
                          </div>
                        </ContextMenuTrigger>
                        <ContextMenuContent>
                          <ContextMenuItem onClick={() => handleRename(item)}>
                            <Pencil className="h-4 w-4 mr-2" />
                            Rename
                          </ContextMenuItem>
                          <ContextMenuItem onClick={() => handleHide(item)}>
                            <EyeOff className="h-4 w-4 mr-2" />
                            Hide{isPerson ? " All" : ""}
                          </ContextMenuItem>
                          {!isPerson && (
                            <>
                              <ContextMenuSeparator />
                              <ContextMenuItem onClick={() => handleLink(item)}>
                                <Link2 className="h-4 w-4 mr-2" />
                                Same Person...
                              </ContextMenuItem>
                            </>
                          )}
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
                      <span>Loading more...</span>
                    </div>
                  )}
                  {!hasMore && items.length > 0 && !searchQuery && (
                    <span className="text-gray-400 text-sm">All items loaded</span>
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

        {/* Drag-and-drop Link Confirmation Dialog */}
        <Dialog open={pendingLink !== null} onOpenChange={(open) => !open && cancelLink()}>
          <DialogContent showCloseButton={false} className="max-w-md">
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2">
                <Link2 className="h-5 w-5" />
                Link as Same Person
              </DialogTitle>
              <DialogDescription>
                {pendingLink && (
                  <>
                    Link <strong>{getItemNameByClusterId(pendingLink.sourceId)}</strong> and{" "}
                    <strong>{getItemNameByClusterId(pendingLink.targetId)}</strong> as the same person?
                  </>
                )}
              </DialogDescription>
            </DialogHeader>

            {isLoadingPreview ? (
              <div className="flex items-center justify-center py-4 text-gray-500">
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Loading...
              </div>
            ) : linkPreview?.willMergePersons ? (
              <div className="flex items-start gap-2 p-3 bg-amber-50 border border-amber-200 rounded-lg text-amber-800 text-sm">
                <AlertTriangle className="h-5 w-5 mt-0.5 flex-shrink-0" />
                <div>
                  <strong>Person records will be merged.</strong>
                  <br />"{linkPreview.source?.personName}" will be merged into "{linkPreview.target?.personName}
                  ".
                  {(linkPreview.source?.personClusterCount ?? 0) > 1 && (
                    <>
                      <br />
                      All {linkPreview.source?.personClusterCount} clusters of "{linkPreview.source?.personName}" will
                      be reassigned.
                    </>
                  )}
                </div>
              </div>
            ) : (
              <p className="text-sm text-gray-600">
                Both clusters will be preserved and assigned to the same identity.
              </p>
            )}

            <DialogFooter>
              <Button variant="outline" onClick={cancelLink} disabled={isLinking}>
                Cancel
              </Button>
              <Button onClick={confirmLink} disabled={isLinking || isLoadingPreview}>
                {isLinking ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                    {linkPreview?.willMergePersons ? "Merging..." : "Linking..."}
                  </>
                ) : linkPreview?.willMergePersons ? (
                  "Merge Persons"
                ) : (
                  "Link"
                )}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Rename Dialog */}
        <Dialog open={renameDialogOpen} onOpenChange={setRenameDialogOpen}>
          <DialogContent className="max-w-sm">
            <DialogHeader>
              <DialogTitle>
                {contextItem?.person_name
                  ? `Rename ${contextItem.item_type === "person" ? "Person" : "Cluster"}`
                  : "Set Name"}
              </DialogTitle>
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

        {/* Context Menu Link Dialog - only for clusters, not persons */}
        {contextItem && contextItem.item_type === "cluster" && (
          <ClusterLinkDialog
            open={linkDialogOpen}
            onOpenChange={setLinkDialogOpen}
            sourceClusterId={contextItem.id.toString()}
            sourceClusterName={contextItem.person_name || `Cluster #${contextItem.id}`}
            onLinkComplete={handleLinkComplete}
          />
        )}
      </div>
    </Layout>
  );
}
