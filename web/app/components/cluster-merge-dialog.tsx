import { GitMerge, Loader2, Search, Users } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { useFetcher } from "react-router";
import { Button } from "~/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "~/components/ui/dialog";
import { Input } from "~/components/ui/input";
import { cn } from "~/lib/utils";

interface SearchCluster {
  id: string;
  face_count: number;
  person_name?: string;
  photo_id?: string;
  bbox_x?: number;
  bbox_y?: number;
  bbox_width?: number;
  bbox_height?: number;
  normalized_width?: number;
  normalized_height?: number;
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
  containerSize = 48,
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

interface ClusterMergeDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  sourceClusterId: string;
  sourceClusterName: string;
  excludeClusterId?: string;
  onMergeComplete?: () => void;
}

export function ClusterMergeDialog({
  open,
  onOpenChange,
  sourceClusterId,
  sourceClusterName,
  excludeClusterId,
  onMergeComplete,
}: ClusterMergeDialogProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<SearchCluster[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [pendingMergeClusterId, setPendingMergeClusterId] = useState<string | null>(null);
  const mergeFetcher = useFetcher();

  const isMerging = mergeFetcher.state !== "idle";

  // Reset state when dialog opens/closes
  useEffect(() => {
    if (open) {
      setSearchQuery("");
      setSearchResults([]);
      setPendingMergeClusterId(null);
    }
  }, [open]);

  // Handle merge completion
  useEffect(() => {
    if (mergeFetcher.data?.success) {
      onOpenChange(false);
      onMergeComplete?.();
    }
  }, [mergeFetcher.data, onOpenChange, onMergeComplete]);

  // Debounced search for clusters
  const searchClusters = useCallback(
    async (query: string) => {
      setIsSearching(true);
      try {
        const excludeId = excludeClusterId || sourceClusterId;
        const response = await fetch(`/api/clusters/search?q=${encodeURIComponent(query)}&exclude=${excludeId}`);
        const data = await response.json();
        setSearchResults(data.clusters || []);
      } catch (error) {
        console.error("Failed to search clusters:", error);
        setSearchResults([]);
      } finally {
        setIsSearching(false);
      }
    },
    [sourceClusterId, excludeClusterId],
  );

  useEffect(() => {
    if (open) {
      const timer = setTimeout(() => {
        searchClusters(searchQuery);
      }, 300);
      return () => clearTimeout(timer);
    }
  }, [searchQuery, open, searchClusters]);

  const handleMergeClick = (targetClusterId: string) => {
    setPendingMergeClusterId(targetClusterId);
  };

  const confirmMerge = (targetClusterId: string) => {
    mergeFetcher.submit(
      {
        sourceClusterId,
        targetClusterId,
      },
      { method: "post", action: "/api/clusters/merge" },
    );
    setPendingMergeClusterId(null);
  };

  const cancelMerge = () => {
    setPendingMergeClusterId(null);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <GitMerge className="h-5 w-5" />
            Merge "{sourceClusterName}"
          </DialogTitle>
          <DialogDescription>
            Search for a cluster to merge this one into. All faces will be moved to the target cluster.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
            <Input
              placeholder="Search by cluster ID or person name..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9"
              autoFocus
            />
          </div>
          <div className="max-h-80 overflow-y-auto space-y-2">
            {isSearching ? (
              <div className="text-center py-4 text-gray-500">Searching...</div>
            ) : searchResults.length > 0 ? (
              searchResults.map((result) => {
                const isPending = pendingMergeClusterId === result.id;
                const targetName = result.person_name || `Cluster ${result.id}`;

                return (
                  <div
                    key={result.id}
                    className={cn(
                      "w-full flex items-center justify-between p-3 border rounded-lg transition-colors",
                      isPending ? "bg-amber-50 border-amber-200" : "hover:bg-gray-50",
                    )}
                  >
                    <div className="flex items-center space-x-3">
                      {result.photo_id && result.bbox_x !== undefined && result.normalized_width ? (
                        <div className="relative w-12 h-12 bg-gray-100 rounded border overflow-hidden flex-shrink-0">
                          <img
                            src={`/api/image/${result.photo_id}`}
                            alt={`Cluster ${result.id}`}
                            className="absolute max-w-none max-h-none"
                            style={getFaceCropStyle(
                              {
                                bbox_x: result.bbox_x,
                                bbox_y: result.bbox_y || 0,
                                bbox_width: result.bbox_width || 0.1,
                                bbox_height: result.bbox_height || 0.1,
                              },
                              result.normalized_width,
                              result.normalized_height || result.normalized_width,
                            )}
                          />
                        </div>
                      ) : (
                        <div className="w-12 h-12 bg-gray-200 rounded flex items-center justify-center flex-shrink-0">
                          <Users className="h-5 w-5 text-gray-400" />
                        </div>
                      )}
                      <div>
                        {isPending ? (
                          <div className="font-medium text-amber-800">Merge into "{targetName}"?</div>
                        ) : (
                          <>
                            <div className="font-medium">{targetName}</div>
                            <div className="text-sm text-gray-500">
                              {result.face_count} face{result.face_count !== 1 ? "s" : ""}
                            </div>
                          </>
                        )}
                      </div>
                    </div>
                    {isPending ? (
                      <div className="flex items-center space-x-2">
                        <Button variant="outline" size="sm" onClick={cancelMerge} disabled={isMerging}>
                          Cancel
                        </Button>
                        <Button size="sm" onClick={() => confirmMerge(result.id)} disabled={isMerging}>
                          {isMerging ? (
                            <>
                              <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                              Merging...
                            </>
                          ) : (
                            "Confirm"
                          )}
                        </Button>
                      </div>
                    ) : (
                      <button
                        type="button"
                        className="text-sm text-blue-600 hover:text-blue-800"
                        onClick={() => handleMergeClick(result.id)}
                      >
                        Merge into
                      </button>
                    )}
                  </div>
                );
              })
            ) : searchQuery ? (
              <div className="text-center py-4 text-gray-500">No clusters found</div>
            ) : (
              <div className="text-center py-4 text-gray-500">Type to search for clusters</div>
            )}
          </div>
        </div>
        {mergeFetcher.data && !mergeFetcher.data.success && (
          <DialogFooter>
            <p className="text-sm text-red-600">{mergeFetcher.data.message}</p>
          </DialogFooter>
        )}
      </DialogContent>
    </Dialog>
  );
}
