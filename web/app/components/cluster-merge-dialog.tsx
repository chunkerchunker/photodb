import { AlertTriangle, Link2, Loader2, Search, Users } from "lucide-react";
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
  detection_id?: number;
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

interface ClusterLinkDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  sourceClusterId: string;
  sourceClusterName: string;
  excludeClusterId?: string;
  onLinkComplete?: () => void;
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
  const [pendingLinkClusterId, setPendingLinkClusterId] = useState<string | null>(null);
  const [linkPreview, setLinkPreview] = useState<LinkPreview | null>(null);
  const [isLoadingPreview, setIsLoadingPreview] = useState(false);
  const linkFetcher = useFetcher();

  const isLinking = linkFetcher.state !== "idle";

  // Reset state when dialog opens/closes
  useEffect(() => {
    if (open) {
      setSearchQuery("");
      setSearchResults([]);
      setPendingLinkClusterId(null);
      setLinkPreview(null);
    }
  }, [open]);

  // Handle link completion
  useEffect(() => {
    if (linkFetcher.data?.success) {
      onOpenChange(false);
      onLinkComplete?.();
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

  // Fetch preview when a cluster is selected
  const handleLinkClick = async (targetClusterId: string) => {
    setPendingLinkClusterId(targetClusterId);
    setIsLoadingPreview(true);
    setLinkPreview(null);

    try {
      const response = await fetch(`/api/clusters/link-preview?source=${sourceClusterId}&target=${targetClusterId}`);
      const preview = await response.json();
      setLinkPreview(preview);
    } catch (error) {
      console.error("Failed to fetch link preview:", error);
    } finally {
      setIsLoadingPreview(false);
    }
  };

  const confirmLink = (targetClusterId: string) => {
    linkFetcher.submit(
      {
        sourceClusterId,
        targetClusterId,
      },
      { method: "post", action: "/api/clusters/merge" },
    );
    setPendingLinkClusterId(null);
    setLinkPreview(null);
  };

  const cancelLink = () => {
    setPendingLinkClusterId(null);
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
          <DialogDescription>
            Search for another cluster of the same person. Both clusters will be linked to the same identity.
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
                const isPending = pendingLinkClusterId === result.id;
                const targetName = result.person_name || `Cluster ${result.id}`;

                return (
                  <div
                    key={result.id}
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
                              alt={`Cluster ${result.id}`}
                              className="w-full h-full object-cover"
                            />
                          </div>
                        ) : (
                          <div className="w-12 h-12 bg-gray-200 rounded flex items-center justify-center flex-shrink-0">
                            <Users className="h-5 w-5 text-gray-400" />
                          </div>
                        )}
                        <div>
                          <div className="font-medium">{targetName}</div>
                          <div className="text-sm text-gray-500">
                            {result.face_count} face{result.face_count !== 1 ? "s" : ""}
                          </div>
                        </div>
                      </div>
                      {!isPending && (
                        <button
                          type="button"
                          className="text-sm text-blue-600 hover:text-blue-800"
                          onClick={() => handleLinkClick(result.id)}
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
                              <Button size="sm" onClick={() => confirmLink(result.id)} disabled={isLinking}>
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
                              <Button size="sm" onClick={() => confirmLink(result.id)} disabled={isLinking}>
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
              <div className="text-center py-4 text-gray-500">No clusters found</div>
            ) : (
              <div className="text-center py-4 text-gray-500">Type to search for clusters</div>
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
