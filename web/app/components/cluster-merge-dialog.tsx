import { AlertTriangle, Link2, Loader2, Search, User, Users } from "lucide-react";
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
  id: string | null;
  item_type?: "person" | "cluster";
  person_id?: string;
  face_count: number;
  person_name?: string;
  detection_id?: number;
  cluster_count?: number;
}

interface LinkPreview {
  found: boolean;
  source?: {
    clusterId: string;
    personId: number | null;
    personName: string | null;
    personClusterCount: number;
  };
  target?: {
    clusterId: string;
    personId: number | null;
    personName: string | null;
    personClusterCount: number;
  };
  willMergePersons?: boolean;
  sourcePersonWillBeDeleted?: boolean;
}

interface PendingTarget {
  key: string;
  clusterId: string | null;
  personId: string | null;
  targetName: string;
}

interface ClusterLinkDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  sourceClusterId: string;
  sourceClusterName: string;
  excludeClusterId?: string;
  onLinkComplete?: (personId?: number) => void;
}

/**
 * Dialog for linking two clusters as the same person.
 * This does NOT merge clusters - both clusters are preserved
 * and assigned to the same person_id.
 *
 * If both clusters belong to different persons, the source person
 * will be merged into the target person (all clusters moved, source deleted).
 */
export function ClusterLinkDialog({
  open,
  onOpenChange,
  sourceClusterId,
  sourceClusterName,
  excludeClusterId,
  onLinkComplete,
}: ClusterLinkDialogProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<SearchCluster[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [pendingTarget, setPendingTarget] = useState<PendingTarget | null>(null);
  const [linkPreview, setLinkPreview] = useState<LinkPreview | null>(null);
  const [isLoadingPreview, setIsLoadingPreview] = useState(false);
  const linkFetcher = useFetcher();

  const isLinking = linkFetcher.state !== "idle";

  // Reset state when dialog opens/closes
  useEffect(() => {
    if (open) {
      setSearchQuery("");
      setSearchResults([]);
      setPendingTarget(null);
      setLinkPreview(null);
    }
  }, [open]);

  // Handle link completion
  useEffect(() => {
    if (linkFetcher.data?.success) {
      onOpenChange(false);
      onLinkComplete?.(linkFetcher.data.personId);
    }
  }, [linkFetcher.data, onOpenChange, onLinkComplete]);

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

  const getResultKey = (result: SearchCluster) => {
    const type = result.item_type || "cluster";
    return `${type}:${result.person_id || result.id}`;
  };

  // Select a target and fetch preview if applicable
  const handleLinkClick = async (result: SearchCluster) => {
    const key = getResultKey(result);
    const clusterId = result.id || null;
    const personId = result.item_type === "person" ? result.person_id || null : null;
    const targetName = result.person_name || `Cluster ${result.id}`;

    setPendingTarget({ key, clusterId, personId, targetName });
    setLinkPreview(null);

    if (clusterId) {
      // Has a cluster to preview against
      setIsLoadingPreview(true);
      try {
        const response = await fetch(`/api/clusters/link-preview?source=${sourceClusterId}&target=${clusterId}`);
        const preview = await response.json();
        setLinkPreview(preview);
      } catch (error) {
        console.error("Failed to fetch link preview:", error);
      } finally {
        setIsLoadingPreview(false);
      }
    }
  };

  const confirmLink = () => {
    if (!pendingTarget) return;

    if (pendingTarget.clusterId) {
      linkFetcher.submit(
        { sourceClusterId, targetClusterId: pendingTarget.clusterId },
        { method: "post", action: "/api/clusters/merge" },
      );
    } else if (pendingTarget.personId) {
      linkFetcher.submit(
        { sourceClusterId, targetPersonId: pendingTarget.personId },
        { method: "post", action: "/api/clusters/merge" },
      );
    }
    setPendingTarget(null);
    setLinkPreview(null);
  };

  const cancelLink = () => {
    setPendingTarget(null);
    setLinkPreview(null);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Link2 className="h-5 w-5" />
            Link "{sourceClusterName}" as Same Person
          </DialogTitle>
          <DialogDescription>Search for another cluster or person to link to the same identity.</DialogDescription>
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
                const resultKey = getResultKey(result);
                const isPending = pendingTarget?.key === resultKey;
                const isPerson = result.item_type === "person";
                const targetName = result.person_name || `Cluster ${result.id}`;

                return (
                  <div
                    key={resultKey}
                    className={cn(
                      "w-full flex flex-col p-3 border rounded-lg transition-colors",
                      isPending ? "bg-amber-50 border-amber-200" : "hover:bg-gray-50",
                    )}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        {result.detection_id ? (
                          <div className="w-12 h-12 bg-gray-100 rounded border overflow-hidden flex-shrink-0">
                            <img
                              src={`/api/face/${result.detection_id}`}
                              alt={targetName}
                              className="w-full h-full object-cover"
                            />
                          </div>
                        ) : (
                          <div className="w-12 h-12 bg-gray-200 rounded flex items-center justify-center flex-shrink-0">
                            {isPerson ? (
                              <User className="h-5 w-5 text-gray-400" />
                            ) : (
                              <Users className="h-5 w-5 text-gray-400" />
                            )}
                          </div>
                        )}
                        <div>
                          <div className="font-medium">{targetName}</div>
                          <div className="text-sm text-gray-500">
                            {result.face_count} face{result.face_count !== 1 ? "s" : ""}
                            {isPerson && (result.cluster_count ?? 0) > 1 && (
                              <span className="text-gray-400"> · {result.cluster_count} clusters</span>
                            )}
                          </div>
                        </div>
                      </div>
                      {!isPending && (
                        <button
                          type="button"
                          className="text-sm text-blue-600 hover:text-blue-800"
                          onClick={() => handleLinkClick(result)}
                        >
                          Same Person
                        </button>
                      )}
                    </div>

                    {/* Confirmation section when pending */}
                    {isPending && (
                      <div className="mt-3 pt-3 border-t border-amber-200">
                        {isLoadingPreview ? (
                          <div className="flex items-center justify-center py-2 text-gray-500">
                            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                            Loading...
                          </div>
                        ) : linkPreview?.willMergePersons ? (
                          <div className="space-y-3">
                            <div className="flex items-start gap-2 p-2 bg-amber-100 rounded text-amber-800 text-sm">
                              <AlertTriangle className="h-4 w-4 mt-0.5 flex-shrink-0" />
                              <div>
                                <strong>Person records will be merged.</strong>
                                <br />"{linkPreview.source?.personName}" will be merged into "
                                {linkPreview.target?.personName}".
                                {(linkPreview.source?.personClusterCount ?? 0) > 1 && (
                                  <>
                                    <br />
                                    All {linkPreview.source?.personClusterCount} clusters of "
                                    {linkPreview.source?.personName}" will be reassigned.
                                  </>
                                )}
                              </div>
                            </div>
                            <div className="flex items-center justify-end space-x-2">
                              <Button variant="outline" size="sm" onClick={cancelLink} disabled={isLinking}>
                                Cancel
                              </Button>
                              <Button size="sm" onClick={confirmLink} disabled={isLinking}>
                                {isLinking ? (
                                  <>
                                    <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                                    Merging...
                                  </>
                                ) : (
                                  "Merge Persons"
                                )}
                              </Button>
                            </div>
                          </div>
                        ) : (
                          <div className="flex items-center justify-between">
                            <div className="text-sm text-amber-800">Link with "{targetName}"?</div>
                            <div className="flex items-center space-x-2">
                              <Button variant="outline" size="sm" onClick={cancelLink} disabled={isLinking}>
                                Cancel
                              </Button>
                              <Button size="sm" onClick={confirmLink} disabled={isLinking}>
                                {isLinking ? (
                                  <>
                                    <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                                    Linking...
                                  </>
                                ) : (
                                  "Confirm"
                                )}
                              </Button>
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })
            ) : searchQuery ? (
              <div className="text-center py-4 text-gray-500">No results found</div>
            ) : (
              <div className="text-center py-4 text-gray-500">Type to search for clusters or persons</div>
            )}
          </div>
        </div>
        {linkFetcher.data && !linkFetcher.data.success && (
          <DialogFooter>
            <p className="text-sm text-red-600">{linkFetcher.data.message}</p>
          </DialogFooter>
        )}
      </DialogContent>
    </Dialog>
  );
}
