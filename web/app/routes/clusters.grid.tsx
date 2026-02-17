import { EyeOff, Link2, Loader2, Pencil, Search, Sparkles, User, Users } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { Link, useFetcher, useRevalidator } from "react-router";
import { ClusterLinkDialog } from "~/components/cluster-merge-dialog";
import { Layout } from "~/components/layout";
import { MergeConfirmationDialog } from "~/components/merge-confirmation-dialog";
import { RenamePersonDialog } from "~/components/rename-person-dialog";
import { SearchBox } from "~/components/search-box";
import { ControlsCount, SecondaryControls } from "~/components/secondary-controls";
import { Button } from "~/components/ui/button";
import { Card, CardContent } from "~/components/ui/card";
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuTrigger,
} from "~/components/ui/context-menu";
import { useDragToMerge } from "~/hooks/use-drag-to-merge";
import { useInfiniteScroll } from "~/hooks/use-infinite-scroll";
import { requireCollectionId } from "~/lib/auth.server";
import { dataWithViewMode } from "~/lib/cookies.server";
import { getClustersGroupedByPerson, getClustersGroupedCount, getHiddenClustersCount } from "~/lib/db.server";
import type { Route } from "./+types/clusters.grid";

export function meta() {
  return [
    { title: "Storyteller - Face Clusters" },
    {
      name: "description",
      content: "Browse face clusters sorted by face count",
    },
  ];
}

const LIMIT = 24; // 4x6 grid

export async function loader({ request }: Route.LoaderArgs) {
  const { collectionId } = await requireCollectionId(request);
  const url = new URL(request.url);
  const page = parseInt(url.searchParams.get("page") || "1", 10);
  const offset = (page - 1) * LIMIT;

  try {
    const items = await getClustersGroupedByPerson(collectionId, LIMIT, offset);
    const totalItems = await getClustersGroupedCount(collectionId);
    const hiddenCount = await getHiddenClustersCount(collectionId);
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

type Item = Route.ComponentProps["loaderData"]["items"][number];

export default function ClustersView({ loaderData }: Route.ComponentProps) {
  const { items: initialItems, totalItems, hiddenCount, hasMore: initialHasMore, page: initialPage } = loaderData;

  const [items, setItems] = useState<Item[]>(initialItems);
  const [page, setPage] = useState(initialPage);
  const [hasMore, setHasMore] = useState(initialHasMore);

  // Search state
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");

  // Selection state (shift-click) - stores "type:id" keys for uniqueness
  const [selectedItemKeys, setSelectedItemKeys] = useState<Set<string>>(new Set());

  // Context menu state
  const [contextItem, setContextItem] = useState<Item | null>(null);
  const [renameDialogOpen, setRenameDialogOpen] = useState(false);
  const [linkDialogOpen, setLinkDialogOpen] = useState(false);
  const [pendingHideClusterId, setPendingHideClusterId] = useState<number | null>(null);
  const [pendingHidePersonId, setPendingHidePersonId] = useState<number | null>(null);
  const hideFetcher = useFetcher();
  const revalidator = useRevalidator();

  // Helper to get item key
  const getItemKey = (item: Item) => `${item.item_type}:${item.id}`;

  // Drag-and-drop merge
  const drag = useDragToMerge({
    items,
    getItemKey,
    getMergeId: (item) => item.primary_cluster_id,
    getItemName: (item) => item.person_name || `Cluster #${item.id}`,
    previewUrl: (src, tgt) => `/api/clusters/link-preview?source=${src}&target=${tgt}`,
    mergeAction: "/api/clusters/merge",
    buildFormData: (src, tgt) => ({ sourceClusterId: src.toString(), targetClusterId: tgt.toString() }),
  });

  const fetcher = useFetcher<typeof loader>();

  // Reset state when initial data changes (e.g., navigation)
  useEffect(() => {
    setItems(initialItems);
    setPage(initialPage);
    setHasMore(initialHasMore);
  }, [initialItems, initialPage, initialHasMore]);

  // Append new items when fetcher returns data
  useEffect(() => {
    if (fetcher.data && fetcher.data.items.length > 0) {
      setItems((prev) => {
        const existingKeys = new Set(prev.map((item) => `${item.item_type}:${item.id}`));
        const newItems = fetcher.data?.items.filter((item) => !existingKeys.has(`${item.item_type}:${item.id}`)) ?? [];
        return [...prev, ...newItems];
      });
      setPage(fetcher.data.page);
      setHasMore(fetcher.data.hasMore);
    }
  }, [fetcher.data]);

  const loadMore = useCallback(() => {
    if (fetcher.state === "idle" && hasMore) {
      fetcher.load(`/clusters/grid?page=${page + 1}`);
    }
  }, [fetcher, hasMore, page]);

  const loadMoreRef = useInfiniteScroll({
    onLoadMore: loadMore,
    hasMore,
    isLoading: fetcher.state === "loading",
  });

  const isLoading = fetcher.state === "loading";

  // Context menu handlers
  const handleRename = (item: Item) => {
    setContextItem(item);
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

  const handleRenameSuccess = (newFirstName: string, newLastName: string) => {
    if (contextItem) {
      const newName = newLastName ? `${newFirstName} ${newLastName}` : newFirstName;
      setItems((prev) =>
        prev.map((item) =>
          item.item_type === contextItem.item_type && item.id === contextItem.id
            ? { ...item, person_name: newName }
            : item,
        ),
      );
      setContextItem(null);
    }
  };

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
          <SecondaryControls variant="grid">
            {hiddenCount > 0 && (
              <Link to="/clusters/hidden">
                <Button variant="outline" size="sm">
                  <EyeOff className="h-4 w-4 mr-1" />
                  Hidden ({hiddenCount})
                </Button>
              </Link>
            )}
            <ControlsCount count={totalItems} singular="item" plural="items" variant="grid" />
          </SecondaryControls>
        </div>

        <SearchBox
          open={searchOpen}
          onOpenChange={setSearchOpen}
          query={searchQuery}
          onQueryChange={setSearchQuery}
          placeholder="Search by name or cluster ID..."
          resultCount={searchQuery ? filteredItems.length : undefined}
        />

        {items.length > 0 ? (
          searchQuery && filteredItems.length === 0 ? (
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
                  const isSelected = selectedItemKeys.has(itemKey);
                  const isPerson = item.item_type === "person";
                  const linkTo = isPerson ? `/person/${item.id}` : `/cluster/${item.id}`;

                  return (
                    <ContextMenu key={itemKey}>
                      <ContextMenuTrigger asChild>
                        <button
                          type="button"
                          draggable
                          onClick={(e) => toggleItemSelection(item, e)}
                          onKeyDown={(e) => {
                            if (e.key === "Enter" || e.key === " ") {
                              e.preventDefault();
                              toggleItemSelection(item, e as unknown as React.MouseEvent);
                            }
                          }}
                          onDragStart={(e) => drag.handleDragStart(e, item)}
                          onDragEnd={drag.handleDragEnd}
                          onDragOver={(e) => drag.handleDragOver(e, item)}
                          onDragLeave={drag.handleDragLeave}
                          onDrop={(e) => drag.handleDrop(e, item)}
                          className={`transition-all text-left ${drag.isDragging(item) ? "opacity-50" : ""}`}
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
                                  : drag.isDropTarget(item)
                                    ? "ring-2 ring-blue-500 ring-offset-2 bg-blue-50"
                                    : ""
                              }`}
                            >
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
                                      title={
                                        isPerson && item.auto_created
                                          ? "Auto-grouped"
                                          : item.person_name || `Cluster #${item.id}`
                                      }
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
                        </button>
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
                        <ContextMenuSeparator />
                        <ContextMenuItem onClick={() => handleLink(item)}>
                          <Link2 className="h-4 w-4 mr-2" />
                          Same Person...
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
                    <span>Loading more...</span>
                  </div>
                )}
                {!hasMore && items.length > 0 && !searchQuery && (
                  <span className="text-gray-400 text-sm">All items loaded</span>
                )}
              </div>
            </>
          )
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
        <MergeConfirmationDialog
          open={drag.pendingMerge !== null}
          sourceName={drag.sourceName}
          targetName={drag.targetName}
          preview={drag.mergePreview}
          isLoadingPreview={drag.isLoadingPreview}
          isSubmitting={drag.isSubmitting}
          onConfirm={drag.confirmMerge}
          onCancel={drag.cancelMerge}
        />

        {/* Rename Dialog */}
        {contextItem && (
          <RenamePersonDialog
            open={renameDialogOpen}
            onOpenChange={setRenameDialogOpen}
            personId={contextItem.id.toString()}
            currentFirstName={contextItem.person_name?.split(" ")[0] || ""}
            currentLastName={contextItem.person_name?.split(" ").slice(1).join(" ") || ""}
            title={
              contextItem.person_name
                ? `Rename ${contextItem.item_type === "person" ? "Person" : "Cluster"}`
                : "Set Name"
            }
            description="Enter a name for this person."
            apiType={contextItem.item_type === "person" ? "person" : "cluster"}
            onSuccess={handleRenameSuccess}
          />
        )}

        {/* Context Menu Link Dialog */}
        {contextItem && (
          <ClusterLinkDialog
            open={linkDialogOpen}
            onOpenChange={setLinkDialogOpen}
            sourceClusterId={contextItem.primary_cluster_id.toString()}
            sourceClusterName={contextItem.person_name || `Cluster #${contextItem.id}`}
            onLinkComplete={handleLinkComplete}
          />
        )}
      </div>
    </Layout>
  );
}
