import {
  Calendar,
  Check,
  Eye,
  EyeOff,
  GitMerge,
  Loader2,
  Pencil,
  Scissors,
  Search,
  Star,
  Trash2,
  User,
  UserMinus,
  Users,
  XCircle,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Link, redirect, useFetcher } from "react-router";
import { Breadcrumb } from "~/components/breadcrumb";
import { Layout } from "~/components/layout";
import { Badge } from "~/components/ui/badge";
import { Button } from "~/components/ui/button";
import { Card, CardContent } from "~/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "~/components/ui/dialog";
import { Input } from "~/components/ui/input";
import {
  deleteCluster,
  dissociateFacesFromCluster,
  dissociateFaceWithConfidenceCutoff,
  getClusterDetails,
  getClusterFaces,
  getClusterFacesCount,
  mergeClusters,
  setClusterHidden,
  setClusterPersonName,
  setClusterRepresentative,
} from "~/lib/db.server";
import { cn } from "~/lib/utils";
import type { Route } from "./+types/cluster.$id";

// Helper to format gender display
function formatGender(gender?: string): string | null {
  if (!gender) return null;
  switch (gender) {
    case "M":
      return "Male";
    case "F":
      return "Female";
    case "U":
      return "Unknown";
    default:
      return null;
  }
}

// Helper to format age estimate
function formatAge(age?: number): string | null {
  if (age === undefined || age === null) return null;
  return `~${Math.round(age)}`;
}

export function meta({ params }: Route.MetaArgs) {
  return [
    { title: `PhotoDB - Cluster ${params.id}` },
    {
      name: "description",
      content: `View all faces in cluster ${params.id}`,
    },
  ];
}

export async function action({ request, params }: Route.ActionArgs) {
  const formData = await request.formData();
  const intent = formData.get("intent");
  const clusterId = params.id;

  if (intent === "dissociate") {
    const faceIdsStr = formData.get("faceIds") as string;
    if (!faceIdsStr || !clusterId) {
      return { success: false, message: "Invalid request" };
    }

    const faceIds = faceIdsStr
      .split(",")
      .map((id) => parseInt(id, 10))
      .filter((id) => !Number.isNaN(id));
    if (faceIds.length === 0) {
      return { success: false, message: "No faces selected" };
    }

    const result = await dissociateFacesFromCluster(clusterId, faceIds);
    return result;
  }

  if (intent === "set-representative") {
    const faceId = parseInt(formData.get("faceId") as string, 10);

    if (faceId && clusterId) {
      const result = await setClusterRepresentative(clusterId, faceId);
      return result;
    }
    return { success: false, message: "Invalid face ID" };
  }

  if (intent === "hide" || intent === "unhide") {
    if (clusterId) {
      const result = await setClusterHidden(clusterId, intent === "hide");
      return result;
    }
    return { success: false, message: "Invalid cluster ID" };
  }

  if (intent === "delete") {
    if (clusterId) {
      const result = await deleteCluster(clusterId);
      if (result.success) {
        return redirect("/clusters");
      }
      return result;
    }
    return { success: false, message: "Invalid cluster ID" };
  }

  if (intent === "cutoff") {
    const faceId = parseInt(formData.get("faceId") as string, 10);

    if (faceId && clusterId) {
      const result = await dissociateFaceWithConfidenceCutoff(clusterId, faceId);
      return result;
    }
    return { success: false, message: "Invalid face ID" };
  }

  if (intent === "merge") {
    const targetClusterId = formData.get("targetClusterId") as string;
    if (!targetClusterId || !clusterId) {
      return { success: false, message: "Invalid cluster IDs" };
    }

    const result = await mergeClusters(clusterId, targetClusterId);
    if (result.success) {
      return redirect(`/cluster/${targetClusterId}`);
    }
    return result;
  }

  if (intent === "set-name") {
    const firstName = (formData.get("firstName") as string)?.trim();
    const lastName = (formData.get("lastName") as string)?.trim();
    if (!firstName || !clusterId) {
      return { success: false, message: "First name is required" };
    }

    const result = await setClusterPersonName(clusterId, firstName, lastName || undefined);
    return result;
  }

  return { success: false, message: "Unknown action" };
}

const LIMIT = 24; // 4x6 grid

export async function loader({ params, request }: Route.LoaderArgs) {
  const clusterId = params.id;
  const url = new URL(request.url);
  const page = parseInt(url.searchParams.get("page") || "1", 10);
  const offset = (page - 1) * LIMIT;

  try {
    const cluster = await getClusterDetails(clusterId);
    if (!cluster) {
      throw new Response("Cluster not found", { status: 404 });
    }

    const faces = await getClusterFaces(clusterId, LIMIT, offset);
    const totalFaces = await getClusterFacesCount(clusterId);
    const hasMore = offset + faces.length < totalFaces;

    return {
      cluster,
      faces,
      totalFaces,
      hasMore,
      page,
    };
  } catch (error) {
    console.error(`Failed to load cluster ${clusterId}:`, error);
    if (error instanceof Response && error.status === 404) {
      throw error;
    }
    return {
      cluster: null,
      faces: [],
      totalFaces: 0,
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
  containerSize = 128,
) {
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

type Face = Route.ComponentProps["loaderData"]["faces"][number];

export default function ClusterDetailView({ loaderData }: Route.ComponentProps) {
  const { cluster, faces: initialFaces, totalFaces, hasMore: initialHasMore, page: initialPage } = loaderData;
  const [selectedFaces, setSelectedFaces] = useState<number[]>([]);
  const [mergeModalOpen, setMergeModalOpen] = useState(false);
  const [nameModalOpen, setNameModalOpen] = useState(false);
  const [editFirstName, setEditFirstName] = useState("");
  const [editLastName, setEditLastName] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<SearchCluster[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const fetcher = useFetcher();

  // Infinite scroll state
  const [faces, setFaces] = useState<Face[]>(initialFaces);
  const [page, setPage] = useState(initialPage);
  const [hasMore, setHasMore] = useState(initialHasMore);
  const scrollFetcher = useFetcher<typeof loader>();
  const loadMoreRef = useRef<HTMLDivElement>(null);

  // Reset state when initial data changes (e.g., navigation or action)
  useEffect(() => {
    setFaces(initialFaces);
    setPage(initialPage);
    setHasMore(initialHasMore);
    setSelectedFaces([]);
  }, [initialFaces, initialPage, initialHasMore]);

  // Append new faces when scroll fetcher returns data
  useEffect(() => {
    if (scrollFetcher.data?.faces && scrollFetcher.data.faces.length > 0) {
      setFaces((prev) => {
        const existingIds = new Set(prev.map((f) => f.id));
        const newFaces = scrollFetcher.data!.faces.filter((f) => !existingIds.has(f.id));
        return [...prev, ...newFaces];
      });
      setPage(scrollFetcher.data.page);
      setHasMore(scrollFetcher.data.hasMore);
    }
  }, [scrollFetcher.data]);

  const loadMore = useCallback(() => {
    if (scrollFetcher.state === "idle" && hasMore && cluster) {
      scrollFetcher.load(`/cluster/${cluster.id}?page=${page + 1}`);
    }
  }, [scrollFetcher, hasMore, page, cluster]);

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

  const isLoadingMore = scrollFetcher.state === "loading";

  // Initialize edit name when modal opens
  useEffect(() => {
    if (nameModalOpen && cluster) {
      setEditFirstName(cluster.first_name || "");
      setEditLastName(cluster.last_name || "");
    }
  }, [nameModalOpen, cluster]);

  // Debounced search for clusters
  const searchClusters = useCallback(
    async (query: string) => {
      if (!cluster) return;
      setIsSearching(true);
      try {
        const response = await fetch(`/api/clusters/search?q=${encodeURIComponent(query)}&exclude=${cluster.id}`);
        const data = await response.json();
        setSearchResults(data.clusters || []);
      } catch (error) {
        console.error("Failed to search clusters:", error);
        setSearchResults([]);
      } finally {
        setIsSearching(false);
      }
    },
    [cluster],
  );

  useEffect(() => {
    if (mergeModalOpen) {
      const timer = setTimeout(() => {
        searchClusters(searchQuery);
      }, 300);
      return () => clearTimeout(timer);
    }
  }, [searchQuery, mergeModalOpen, searchClusters]);

  const handleMerge = (targetClusterId: string) => {
    if (confirm(`Merge this cluster into cluster ${targetClusterId}? This cluster will be deleted.`)) {
      fetcher.submit({ intent: "merge", targetClusterId }, { method: "post" });
      setMergeModalOpen(false);
    }
  };

  const handleSaveName = () => {
    if (editFirstName.trim()) {
      fetcher.submit(
        { intent: "set-name", firstName: editFirstName.trim(), lastName: editLastName.trim() },
        { method: "post" },
      );
      setNameModalOpen(false);
    }
  };

  const toggleFaceSelection = (faceId: number) => {
    setSelectedFaces((prev) => (prev.includes(faceId) ? prev.filter((id) => id !== faceId) : [...prev, faceId]));
  };

  const handleDissociate = () => {
    if (selectedFaces.length > 0) {
      fetcher.submit(
        {
          intent: "dissociate",
          faceIds: selectedFaces.join(","),
        },
        { method: "post" },
      );
      setSelectedFaces([]);
    }
  };

  const handleSetRepresentative = () => {
    if (selectedFaces.length === 1) {
      fetcher.submit(
        {
          intent: "set-representative",
          faceId: selectedFaces[0].toString(),
        },
        { method: "post" },
      );
      setSelectedFaces([]);
    }
  };

  const handleToggleHidden = () => {
    fetcher.submit({ intent: cluster.hidden ? "unhide" : "hide" }, { method: "post" });
  };

  const handleDelete = () => {
    if (confirm("Delete this cluster? All faces will be moved back to the unassigned pool.")) {
      fetcher.submit({ intent: "delete" }, { method: "post" });
    }
  };

  const handleCutoff = () => {
    if (selectedFaces.length === 1) {
      fetcher.submit(
        {
          intent: "cutoff",
          faceId: selectedFaces[0].toString(),
        },
        { method: "post" },
      );
      setSelectedFaces([]);
    }
  };

  const isSubmitting = fetcher.state === "submitting";

  if (!cluster) {
    return (
      <Layout>
        <div className="text-center py-12">
          <div className="text-red-500 text-lg">Cluster not found</div>
        </div>
      </Layout>
    );
  }

  const displayName = cluster.person_name || `Cluster ${cluster.id}`;
  const breadcrumbItems = [{ label: "Clusters", href: "/clusters" }, { label: displayName }];

  return (
    <Layout>
      <div className="space-y-6">
        <Breadcrumb items={breadcrumbItems} />

        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Users className="h-8 w-8 text-gray-700" />
            <h1 className="text-3xl font-bold text-gray-900">{displayName}</h1>
            {cluster.person_name && (
              <Badge variant="secondary" className="text-sm">
                #{cluster.id}
              </Badge>
            )}
            <Dialog open={nameModalOpen} onOpenChange={setNameModalOpen}>
              <DialogTrigger asChild>
                <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                  <Pencil className="h-4 w-4" />
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-sm">
                <DialogHeader>
                  <DialogTitle>{cluster.person_name ? "Edit Name" : "Set Name"}</DialogTitle>
                  <DialogDescription>Enter a name for this person.</DialogDescription>
                </DialogHeader>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-3">
                    <div className="space-y-1">
                      <label htmlFor="personFirstName" className="text-sm font-medium text-gray-700">
                        First Name
                      </label>
                      <Input
                        id="personFirstName"
                        name="personFirstName"
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
                      <label htmlFor="personLastName" className="text-sm font-medium text-gray-700">
                        Last Name
                      </label>
                      <Input
                        id="personLastName"
                        name="personLastName"
                        placeholder="Last name"
                        value={editLastName}
                        onChange={(e) => setEditLastName(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") {
                            handleSaveName();
                          }
                        }}
                        autoComplete="off"
                        data-form-type="other"
                        data-1p-ignore
                        data-lpignore="true"
                      />
                    </div>
                  </div>
                  <div className="flex justify-end space-x-2">
                    <Button variant="outline" onClick={() => setNameModalOpen(false)}>
                      Cancel
                    </Button>
                    <Button onClick={handleSaveName} disabled={!editFirstName.trim()}>
                      Save
                    </Button>
                  </div>
                </div>
              </DialogContent>
            </Dialog>
          </div>
          <div className="flex items-center space-x-4">
            <span className="text-gray-600">
              {totalFaces} face{totalFaces !== 1 ? "s" : ""}
            </span>
            <Dialog open={mergeModalOpen} onOpenChange={setMergeModalOpen}>
              <DialogTrigger asChild>
                <Button variant="outline" size="sm" disabled={isSubmitting}>
                  <GitMerge className="h-4 w-4 mr-1" />
                  Merge
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-lg">
                <DialogHeader>
                  <DialogTitle>Merge into another cluster</DialogTitle>
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
                    />
                  </div>
                  <div className="max-h-80 overflow-y-auto space-y-2">
                    {isSearching ? (
                      <div className="text-center py-4 text-gray-500">Searching...</div>
                    ) : searchResults.length > 0 ? (
                      searchResults.map((result) => (
                        <button
                          type="button"
                          key={result.id}
                          className="w-full flex items-center justify-between p-3 border rounded-lg hover:bg-gray-50 cursor-pointer text-left"
                          onClick={() => handleMerge(result.id)}
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
                                    48,
                                  )}
                                />
                              </div>
                            ) : (
                              <div className="w-12 h-12 bg-gray-200 rounded flex items-center justify-center flex-shrink-0">
                                <Users className="h-5 w-5 text-gray-400" />
                              </div>
                            )}
                            <div>
                              <div className="font-medium">{result.person_name || `Cluster ${result.id}`}</div>
                              <div className="text-sm text-gray-500">
                                {result.face_count} face{result.face_count !== 1 ? "s" : ""}
                              </div>
                            </div>
                          </div>
                          <span className="text-sm text-blue-600 hover:text-blue-800">Merge into</span>
                        </button>
                      ))
                    ) : searchQuery ? (
                      <div className="text-center py-4 text-gray-500">No clusters found</div>
                    ) : (
                      <div className="text-center py-4 text-gray-500">Type to search for clusters</div>
                    )}
                  </div>
                </div>
              </DialogContent>
            </Dialog>
            <Button variant="outline" size="sm" onClick={handleToggleHidden} disabled={isSubmitting}>
              {cluster.hidden ? (
                <>
                  <Eye className="h-4 w-4 mr-1" />
                  Unhide
                </>
              ) : (
                <>
                  <EyeOff className="h-4 w-4 mr-1" />
                  Hide
                </>
              )}
            </Button>
            <Button variant="destructive" size="sm" onClick={handleDelete} disabled={isSubmitting}>
              <Trash2 className="h-4 w-4 mr-1" />
              Delete
            </Button>
          </div>
        </div>

        {cluster.hidden && (
          <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 text-amber-800 text-sm">
            This cluster is hidden and won't appear in the main clusters list.
          </div>
        )}

        {/* Face selection controls - sticky toolbar */}
        <div className="bg-gray-50 border rounded-lg p-4 space-y-3 sticky top-0 z-10 shadow-sm">
          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <p className="text-sm font-medium text-gray-700">Face actions</p>
              <p className="text-xs text-gray-500">
                Click faces to select. Select 1 for representative or cutoff. Select multiple to remove (similar faces
                also removed).
              </p>
            </div>
            <div className="flex items-center space-x-2">
              {selectedFaces.length > 0 && (
                <Button variant="ghost" size="sm" onClick={() => setSelectedFaces([])}>
                  <XCircle className="h-4 w-4 mr-1" />
                  Clear ({selectedFaces.length})
                </Button>
              )}
              <Button
                onClick={handleSetRepresentative}
                disabled={selectedFaces.length !== 1 || isSubmitting}
                size="sm"
                variant="outline"
              >
                <Star className="h-4 w-4 mr-1" />
                Representative
              </Button>
              <Button
                onClick={handleCutoff}
                disabled={selectedFaces.length !== 1 || isSubmitting}
                size="sm"
                variant="outline"
                title="Remove this face (constrained) and all faces with lower confidence (unconstrained)"
              >
                <Scissors className="h-4 w-4 mr-1" />
                Cutoff
              </Button>
              <Button
                onClick={handleDissociate}
                disabled={selectedFaces.length === 0 || isSubmitting}
                size="sm"
                variant="destructive"
              >
                <UserMinus className="h-4 w-4 mr-1" />
                {isSubmitting ? "Removing..." : `Remove ${selectedFaces.length > 0 ? `(${selectedFaces.length})` : ""}`}
              </Button>
            </div>
          </div>
          {fetcher.data?.message && (
            <p className={`text-sm ${fetcher.data.success ? "text-green-600" : "text-red-600"}`}>
              {fetcher.data.message}
            </p>
          )}
        </div>

        {faces.length > 0 ? (
          <>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-4">
              {faces.map((face) => {
                const isSelected = selectedFaces.includes(face.id);
                const isRepresentative = face.id === cluster.representative_detection_id;

                return (
                  <Card
                    key={face.id}
                    className={`hover:shadow-lg transition-all cursor-pointer ${
                      isRepresentative && !isSelected ? "ring-2 ring-amber-400" : ""
                    } ${isSelected ? "ring-2 ring-blue-500 bg-blue-50" : "hover:ring-1 hover:ring-gray-300"}`}
                    onClick={() => toggleFaceSelection(face.id)}
                  >
                    <CardContent className="p-4 relative">
                      {isRepresentative && (
                        <div className="absolute top-2 left-2 z-10 w-6 h-6 bg-amber-400 text-white rounded-full flex items-center justify-center">
                          <Star className="h-3 w-3 fill-current" />
                        </div>
                      )}
                      {isSelected && (
                        <div className="absolute top-2 right-2 z-10 w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center">
                          <Check className="h-4 w-4" />
                        </div>
                      )}
                      <div className="text-center space-y-3">
                        {face.photo_id && face.bbox_x !== null && face.normalized_width && face.normalized_height ? (
                          <div className="relative w-32 h-32 mx-auto bg-gray-100 rounded-lg border overflow-hidden">
                            <Link to={`/photo/${face.photo_id}`} onClick={(e) => e.stopPropagation()}>
                              <img
                                src={`/api/image/${face.photo_id}`}
                                alt={`Face ${face.id}`}
                                className="absolute max-w-none max-h-none"
                                style={getFaceCropStyle(
                                  {
                                    bbox_x: face.bbox_x,
                                    bbox_y: face.bbox_y,
                                    bbox_width: face.bbox_width,
                                    bbox_height: face.bbox_height,
                                  },
                                  face.normalized_width,
                                  face.normalized_height,
                                )}
                                loading="lazy"
                              />
                            </Link>
                          </div>
                        ) : (
                          <div className="w-full h-32 bg-gray-200 rounded-lg flex items-center justify-center">
                            <Users className="h-8 w-8 text-gray-400" />
                          </div>
                        )}

                        <div className="space-y-1">
                          <div className="text-sm text-gray-600">Photo #{face.photo_id}</div>
                          <div className="text-xs text-gray-500">
                            {Math.round(face.cluster_confidence * 100)}% confidence
                          </div>
                          {/* Age/Gender badges */}
                          {(face.age_estimate || face.gender) && (
                            <div className="flex items-center justify-center space-x-1 mt-1">
                              {formatAge(face.age_estimate) && (
                                <Badge variant="outline" className="text-xs bg-blue-50 border-blue-200 px-1.5 py-0">
                                  <Calendar className="h-2.5 w-2.5 mr-0.5" />
                                  {formatAge(face.age_estimate)}
                                </Badge>
                              )}
                              {formatGender(face.gender) && (
                                <Badge
                                  variant="outline"
                                  className={cn(
                                    "text-xs px-1.5 py-0",
                                    face.gender === "M"
                                      ? "bg-sky-50 border-sky-200"
                                      : face.gender === "F"
                                        ? "bg-pink-50 border-pink-200"
                                        : "bg-gray-50 border-gray-200",
                                  )}
                                >
                                  <User className="h-2.5 w-2.5 mr-0.5" />
                                  {face.gender === "M" ? "M" : face.gender === "F" ? "F" : "?"}
                                </Badge>
                              )}
                            </div>
                          )}
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>

            {/* Infinite scroll trigger */}
            <div ref={loadMoreRef} className="flex justify-center py-8">
              {isLoadingMore && (
                <div className="flex items-center space-x-2 text-gray-500">
                  <Loader2 className="h-5 w-5 animate-spin" />
                  <span>Loading more faces...</span>
                </div>
              )}
              {!hasMore && faces.length > 0 && <span className="text-gray-400 text-sm">All faces loaded</span>}
            </div>
          </>
        ) : (
          <div className="text-center py-12">
            <Users className="h-16 w-16 text-gray-400 mx-auto mb-4" />
            <div className="text-gray-500 text-lg">No faces found in this cluster.</div>
          </div>
        )}
      </div>
    </Layout>
  );
}
