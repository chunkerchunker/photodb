import { Check, FolderPlus, Plus, ScanFace, Search, UserPlus, Users, XCircle, ZoomIn } from "lucide-react";
import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, redirect, useFetcher, useNavigate } from "react-router";
import { Breadcrumb } from "~/components/breadcrumb";
import { Layout } from "~/components/layout";
import { Badge } from "~/components/ui/badge";
import { Button } from "~/components/ui/button";
import { Card, CardContent } from "~/components/ui/card";
import { Checkbox } from "~/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "~/components/ui/dialog";
import { Input } from "~/components/ui/input";
import { Slider } from "~/components/ui/slider";
import { addFacesToCluster, createClusterFromFaces, getFaceDetails, getSimilarFaces } from "~/lib/db.server";
import type { Route } from "./+types/face.$id.similar";

export function meta({ params }: Route.MetaArgs) {
  return [
    { title: `PhotoDB - Similar Faces` },
    {
      name: "description",
      content: `Find faces similar to face ${params.id}`,
    },
  ];
}

interface SimilarFace {
  id: string;
  bbox_x: number;
  bbox_y: number;
  bbox_width: number;
  bbox_height: number;
  photo_id: string;
  normalized_width: number;
  normalized_height: number;
  cluster_id?: string;
  cluster_face_count?: number;
  person_name?: string;
  similarity: number;
  // Age/gender fields
  age_estimate?: number;
  gender?: string;
  gender_confidence?: number;
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

// Memoized face card to prevent re-renders when other faces' selection changes
const SimilarFaceCard = memo(function SimilarFaceCard({
  face,
  isSelected,
  isClustered,
  isPreviewVisible,
  onToggleSelection,
  onZoomEnter,
  onZoomLeave,
}: {
  face: SimilarFace;
  isSelected: boolean;
  isClustered: boolean;
  isPreviewVisible: boolean;
  onToggleSelection: (faceId: number) => void;
  onZoomEnter: (faceId: number) => void;
  onZoomLeave: () => void;
}) {
  const faceIdNum = parseInt(face.id, 10);

  return (
    <Card
      className={`transition-all cursor-default select-none ${
        isClustered
          ? "opacity-60"
          : isSelected
            ? "ring-2 ring-blue-500 bg-blue-50"
            : "hover:ring-1 hover:ring-gray-300"
      }`}
      onClick={() => !isClustered && onToggleSelection(faceIdNum)}
    >
      <CardContent className="p-0 relative">
        {isSelected && (
          <div className="absolute -top-2 -right-2 z-10 w-5 h-5 bg-blue-500 text-white rounded-full flex items-center justify-center">
            <Check className="h-3 w-3" />
          </div>
        )}
        <div className="text-center space-y-1">
          <div className="relative w-28 h-28 mx-auto">
            <div className="group relative w-full h-full bg-gray-100 rounded-lg border overflow-hidden">
              <Link
                to={`/photo/${face.photo_id}`}
                onClick={(e) => e.stopPropagation()}
                className="cursor-pointer"
              >
                <img
                  src={`/api/image/${face.photo_id}`}
                  alt={`Similar face ${face.id}`}
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
                    112,
                  )}
                  loading="lazy"
                />
              </Link>
              {/* Zoom icon */}
              <div
                className="absolute top-1 right-1 z-10 w-5 h-5 bg-black/50 text-white rounded flex items-center justify-center cursor-zoom-in opacity-0 group-hover:opacity-100 transition-opacity duration-200"
                onMouseEnter={() => onZoomEnter(faceIdNum)}
                onMouseLeave={onZoomLeave}
                onClick={(e) => e.stopPropagation()}
              >
                <ZoomIn className="h-3 w-3" />
              </div>
            </div>
            {/* Preview overlay - shows full image with face aligned over card */}
            {isPreviewVisible && (() => {
              const thumbSize = 112;
              const faceScale = thumbSize / face.bbox_width;
              const scaledW = face.normalized_width * faceScale;
              const scaledH = face.normalized_height * faceScale;
              const imgLeft = -face.bbox_x * faceScale;
              const imgTop = -face.bbox_y * faceScale;
              const originX = -imgLeft + thumbSize / 2;
              const originY = -imgTop + thumbSize / 2;

              return (
                <img
                  src={`/api/image/${face.photo_id}`}
                  alt={`Face ${face.id} preview`}
                  className="absolute z-50 max-w-none max-h-none pointer-events-none rounded-xl shadow-2xl animate-in fade-in zoom-in-95 duration-150"
                  style={{
                    width: `${scaledW}px`,
                    height: `${scaledH}px`,
                    left: `${imgLeft}px`,
                    top: `${imgTop}px`,
                    transformOrigin: `${originX}px ${originY}px`,
                  }}
                />
              );
            })()}
          </div>

          <div className="flex items-center justify-center gap-2 text-xs text-gray-500">
            <span className="inline-flex items-center font-medium">
              <ScanFace className="h-3 w-3 mr-0.5" />
              {Math.round(face.similarity * 100)}%
            </span>
            {isClustered && (
              <Link
                to={`/cluster/${face.cluster_id}`}
                onClick={(e) => e.stopPropagation()}
                className="text-blue-600 hover:text-blue-800 truncate max-w-[60px]"
                title={face.person_name || `Cluster ${face.cluster_id}`}
              >
                {face.person_name || `#${face.cluster_id}`}
              </Link>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
});

export async function action({ request, params }: Route.ActionArgs) {
  const formData = await request.formData();
  const intent = formData.get("intent");
  const faceId = params.id;

  if (intent === "create-cluster") {
    const faceIdsStr = formData.get("faceIds") as string;
    if (!faceIdsStr) {
      return { success: false, message: "No faces selected" };
    }

    const faceIds = faceIdsStr
      .split(",")
      .map((id) => parseInt(id, 10))
      .filter((id) => !Number.isNaN(id));

    // Include the source face in the cluster
    faceIds.unshift(parseInt(faceId, 10));

    const result = await createClusterFromFaces(faceIds);
    if (result.success && result.clusterId) {
      return redirect(`/cluster/${result.clusterId}`);
    }
    return result;
  }

  if (intent === "add-to-cluster") {
    const faceIdsStr = formData.get("faceIds") as string;
    const targetClusterId = formData.get("targetClusterId") as string;

    if (!targetClusterId) {
      return { success: false, message: "No cluster selected" };
    }

    const faceIds = faceIdsStr
      ? faceIdsStr
          .split(",")
          .map((id) => parseInt(id, 10))
          .filter((id) => !Number.isNaN(id))
      : [];

    // Include the source face
    faceIds.unshift(parseInt(faceId, 10));

    const result = await addFacesToCluster(targetClusterId, faceIds);
    if (result.success) {
      return redirect(`/cluster/${targetClusterId}`);
    }
    return result;
  }

  return { success: false, message: "Unknown action" };
}

export async function loader({ params, request }: Route.LoaderArgs) {
  const faceId = parseInt(params.id, 10);
  const url = new URL(request.url);
  const threshold = parseFloat(url.searchParams.get("threshold") || "0.6");

  try {
    const face = await getFaceDetails(faceId);
    if (!face) {
      throw new Response("Face not found", { status: 404 });
    }

    const similarFaces = await getSimilarFaces(faceId, 50, threshold);

    return {
      face,
      similarFaces,
      threshold,
    };
  } catch (error) {
    console.error(`Failed to load similar faces for ${faceId}:`, error);
    if (error instanceof Response && error.status === 404) {
      throw error;
    }
    return {
      face: null,
      similarFaces: [],
      threshold,
    };
  }
}

export default function SimilarFacesPage({ loaderData }: Route.ComponentProps) {
  const { face, similarFaces, threshold } = loaderData;
  const [selectedFaces, setSelectedFaces] = useState<number[]>([]);
  const [addToClusterModalOpen, setAddToClusterModalOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<SearchCluster[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [sliderValue, setSliderValue] = useState(threshold);
  const [hideClustered, setHideClustered] = useState(false);
  const [previewFaceId, setPreviewFaceId] = useState<number | null>(null);
  const previewTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const fetcher = useFetcher();
  const navigate = useNavigate();

  // Memoized Set for O(1) selection lookup
  const selectedFacesSet = useMemo(() => new Set(selectedFaces), [selectedFaces]);

  const handleZoomMouseEnter = useCallback((faceId: number) => {
    previewTimeoutRef.current = setTimeout(() => {
      setPreviewFaceId(faceId);
    }, 400);
  }, []);

  const handleZoomMouseLeave = useCallback(() => {
    if (previewTimeoutRef.current) {
      clearTimeout(previewTimeoutRef.current);
      previewTimeoutRef.current = null;
    }
    setPreviewFaceId(null);
  }, []);

  // Load hideClustered preference from localStorage
  useEffect(() => {
    const saved = localStorage.getItem("similarFaces.hideClustered");
    if (saved !== null) {
      setHideClustered(saved === "true");
    }
  }, []);

  const updateHideClustered = (checked: boolean) => {
    setHideClustered(checked);
    localStorage.setItem("similarFaces.hideClustered", String(checked));
  };

  // Update slider when threshold changes (e.g., from URL)
  useEffect(() => {
    setSliderValue(threshold);
  }, [threshold]);

  const handleThresholdChange = useCallback((value: number[]) => {
    setSliderValue(value[0]);
  }, []);

  const handleThresholdCommit = useCallback(
    (value: number[]) => {
      const newThreshold = value[0];
      if (newThreshold !== threshold) {
        navigate(`/face/${face?.id}/similar?threshold=${newThreshold}`);
      }
    },
    [navigate, face?.id, threshold],
  );

  const unclusteredFaces = similarFaces.filter((f: SimilarFace) => !f.cluster_id);
  const displayedFaces = hideClustered ? unclusteredFaces : similarFaces;

  // Debounced search for clusters
  const searchClusters = useCallback(async (query: string) => {
    setIsSearching(true);
    try {
      const response = await fetch(`/api/clusters/search?q=${encodeURIComponent(query)}`);
      const data = await response.json();
      setSearchResults(data.clusters || []);
    } catch (error) {
      console.error("Failed to search clusters:", error);
      setSearchResults([]);
    } finally {
      setIsSearching(false);
    }
  }, []);

  useEffect(() => {
    if (addToClusterModalOpen) {
      const timer = setTimeout(() => {
        searchClusters(searchQuery);
      }, 300);
      return () => clearTimeout(timer);
    }
  }, [searchQuery, addToClusterModalOpen, searchClusters]);

  const toggleFaceSelection = useCallback((faceId: number) => {
    setSelectedFaces((prev) => (prev.includes(faceId) ? prev.filter((id) => id !== faceId) : [...prev, faceId]));
  }, []);

  const selectAll = () => {
    setSelectedFaces(unclusteredFaces.map((f: SimilarFace) => parseInt(f.id, 10)));
  };

  const clearSelection = () => {
    setSelectedFaces([]);
  };

  const handleCreateCluster = () => {
    if (selectedFaces.length > 0) {
      fetcher.submit(
        {
          intent: "create-cluster",
          faceIds: selectedFaces.join(","),
        },
        { method: "post" },
      );
    }
  };

  const handleAddToCluster = (targetClusterId: string) => {
    fetcher.submit(
      {
        intent: "add-to-cluster",
        faceIds: selectedFaces.join(","),
        targetClusterId,
      },
      { method: "post" },
    );
    setAddToClusterModalOpen(false);
  };

  const isSubmitting = fetcher.state === "submitting";

  if (!face) {
    return (
      <Layout>
        <div className="text-center py-12">
          <div className="text-red-500 text-lg">Face not found</div>
        </div>
      </Layout>
    );
  }

  const breadcrumbItems = [
    { label: "Photo", href: `/photo/${face.photo_id}` },
    { label: `Face ${face.id}` },
    { label: "Similar Faces" },
  ];

  return (
    <Layout>
      <div className="space-y-6">
        <Breadcrumb items={breadcrumbItems} />

        <div className="flex items-center space-x-4">
          <h1 className="text-2xl font-bold text-gray-900">Similar Faces</h1>
          <Badge variant="secondary">{similarFaces.length} found</Badge>
          <div className="flex items-center space-x-2">
            <span className="text-xs text-gray-400">≥</span>
            <Slider
              value={[sliderValue]}
              onValueChange={handleThresholdChange}
              onValueCommit={handleThresholdCommit}
              min={0.3}
              max={0.95}
              step={0.05}
              className="w-24"
            />
            <span className="text-xs text-gray-500 w-8">{Math.round(sliderValue * 100)}%</span>
          </div>
          <div className="flex items-center space-x-2">
            <Checkbox
              id="hide-clustered"
              checked={hideClustered}
              onCheckedChange={(checked) => updateHideClustered(checked === true)}
            />
            <label htmlFor="hide-clustered" className="text-sm text-gray-600 cursor-pointer">
              Hide clustered
            </label>
          </div>
        </div>

        {/* Source face */}
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-4">
              <div className="relative w-24 h-24 bg-gray-100 rounded-lg border overflow-hidden">
                <Link to={`/photo/${face.photo_id}`}>
                  <img
                    src={`/api/image/${face.photo_id}`}
                    alt={`Source face ${face.id}`}
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
                      96,
                    )}
                  />
                </Link>
              </div>
              <div>
                <h2 className="font-medium">{face.person_name || "Source Face"}</h2>
                <p className="text-sm text-gray-500">
                  From{" "}
                  <Link to={`/photo/${face.photo_id}`} className="text-blue-600 hover:underline">
                    Photo #{face.photo_id}
                  </Link>
                </p>
                <p className="text-xs text-gray-400 mt-1">
                  Face size: {Math.round(face.bbox_width)}×{Math.round(face.bbox_height)}px
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Selection controls */}
        <div className="bg-gray-50 border rounded-lg p-4 space-y-3">
          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <p className="text-sm font-medium text-gray-700">Cluster actions</p>
              <p className="text-xs text-gray-500">
                Select similar faces to group with the source face into a new or existing cluster.
              </p>
            </div>
            <div className="flex items-center space-x-2">
              {unclusteredFaces.length > 0 && (
                <>
                  <div className="flex items-center space-x-2 mr-4">
                    <Checkbox
                      id="select-all"
                      checked={selectedFaces.length === unclusteredFaces.length && unclusteredFaces.length > 0}
                      onCheckedChange={(checked) => {
                        if (checked) {
                          selectAll();
                        } else {
                          clearSelection();
                        }
                      }}
                    />
                    <label htmlFor="select-all" className="text-sm cursor-pointer">
                      Select all unclustered ({unclusteredFaces.length})
                    </label>
                  </div>
                  {selectedFaces.length > 0 && (
                    <Button variant="ghost" size="sm" onClick={clearSelection}>
                      <XCircle className="h-4 w-4 mr-1" />
                      Clear ({selectedFaces.length})
                    </Button>
                  )}
                </>
              )}
              {face.cluster_id && (
                <Button
                  size="sm"
                  disabled={selectedFaces.length === 0 || isSubmitting}
                  onClick={() => handleAddToCluster(face.cluster_id)}
                >
                  <UserPlus className="h-4 w-4 mr-1" />
                  Add to {face.person_name || `Cluster ${face.cluster_id}`} ({selectedFaces.length})
                </Button>
              )}
              {!face.cluster_id && (
                <Dialog open={addToClusterModalOpen} onOpenChange={setAddToClusterModalOpen}>
                  <DialogTrigger asChild>
                    <Button variant="outline" size="sm" disabled={isSubmitting}>
                      <FolderPlus className="h-4 w-4 mr-1" />
                      Add to Cluster ({selectedFaces.length + 1})
                    </Button>
                  </DialogTrigger>
                <DialogContent className="max-w-lg">
                  <DialogHeader>
                    <DialogTitle>Add to existing cluster</DialogTitle>
                    <DialogDescription>
                      Search for a cluster to add {selectedFaces.length + 1} face
                      {selectedFaces.length === 0 ? "" : "s"} to.
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
                            onClick={() => handleAddToCluster(result.id)}
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
                                  {result.face_count} face
                                  {result.face_count !== 1 ? "s" : ""}
                                </div>
                              </div>
                            </div>
                            <span className="text-sm text-blue-600 hover:text-blue-800">Add to</span>
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
              )}
              {!face.cluster_id && (
                <Button onClick={handleCreateCluster} disabled={selectedFaces.length === 0 || isSubmitting} size="sm">
                  <Plus className="h-4 w-4 mr-1" />
                  {isSubmitting ? "Creating..." : `Create Cluster (${selectedFaces.length + 1})`}
                </Button>
              )}
            </div>
          </div>
          {fetcher.data?.message && (
            <p className={`text-sm ${fetcher.data.success ? "text-green-600" : "text-red-600"}`}>
              {fetcher.data.message}
            </p>
          )}
        </div>

        {/* Similar faces grid */}
        {displayedFaces.length > 0 ? (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 xl:grid-cols-8 gap-4">
            {displayedFaces.map((similarFace: SimilarFace) => (
              <SimilarFaceCard
                key={similarFace.id}
                face={similarFace}
                isSelected={selectedFacesSet.has(parseInt(similarFace.id, 10))}
                isClustered={!!similarFace.cluster_id}
                isPreviewVisible={previewFaceId === parseInt(similarFace.id, 10)}
                onToggleSelection={toggleFaceSelection}
                onZoomEnter={handleZoomMouseEnter}
                onZoomLeave={handleZoomMouseLeave}
              />
            ))}
          </div>
        ) : (
          <div className="text-center py-12">
            <Users className="h-16 w-16 text-gray-400 mx-auto mb-4" />
            {hideClustered && similarFaces.length > 0 ? (
              <>
                <div className="text-gray-500 text-lg">All similar faces are already clustered</div>
                <p className="text-sm text-gray-400 mt-2">Uncheck "Hide clustered" to see them</p>
              </>
            ) : (
              <>
                <div className="text-gray-500 text-lg">No similar faces found above {Math.round(threshold * 100)}%</div>
                <p className="text-sm text-gray-400 mt-2">Try lowering the similarity threshold</p>
              </>
            )}
          </div>
        )}
      </div>
    </Layout>
  );
}
