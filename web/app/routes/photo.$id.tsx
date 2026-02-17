import { useMeasure } from "@react-hookz/web";
import {
  Ban,
  Bot,
  Calendar,
  Camera,
  ChevronDown,
  Code,
  ExternalLink,
  Info,
  MapPin,
  ScanFace,
  Sparkles,
  Tag,
  User,
  Users,
  X,
} from "lucide-react";
import { displaySettings } from "~/lib/settings";

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
  return `~${Math.round(age)} years`;
}

// Helper to get consistent color for category
function getCategoryColor(category: string): string {
  const colors = [
    "bg-red-50 border-red-200",
    "bg-orange-50 border-orange-200",
    "bg-amber-50 border-amber-200",
    "bg-yellow-50 border-yellow-200",
    "bg-lime-50 border-lime-200",
    "bg-green-50 border-green-200",
    "bg-emerald-50 border-emerald-200",
    "bg-teal-50 border-teal-200",
    "bg-cyan-50 border-cyan-200",
    "bg-sky-50 border-sky-200",
    "bg-blue-50 border-blue-200",
    "bg-indigo-50 border-indigo-200",
    "bg-violet-50 border-violet-200",
    "bg-purple-50 border-purple-200",
    "bg-fuchsia-50 border-fuchsia-200",
    "bg-pink-50 border-pink-200",
    "bg-rose-50 border-rose-200",
  ];

  // Simple hash function to map string to index
  let hash = 0;
  for (let i = 0; i < category.length; i++) {
    hash = category.charCodeAt(i) + ((hash << 5) - hash);
  }

  const index = Math.abs(hash) % colors.length;
  return colors[index];
}

import { useEffect, useState } from "react";
import { Link, useFetcher, useLocation, useNavigate } from "react-router";
import { Breadcrumb } from "~/components/breadcrumb";
import { Layout } from "~/components/layout";
import { Badge } from "~/components/ui/badge";
import { Button } from "~/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { Checkbox } from "~/components/ui/checkbox";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "~/components/ui/collapsible";
import { requireCollectionId } from "~/lib/auth.server";
import { getPhotoDetails, removeFaceFromClusterWithConstraint } from "~/lib/db.server";
import { cn } from "~/lib/utils";
import type { Route } from "./+types/photo.$id";

interface FaceMatchCandidate {
  cluster_id: string;
  similarity: number;
  status: string;
  person_id?: string;
  person_name?: string;
  face_count: number;
}

interface SceneTag {
  confidence: number;
  rank_in_category?: number;
  label: string;
  display_name?: string;
  prompt_text: string;
  category_name: string;
  target: string;
}

interface SceneTaxonomyLabel {
  label: string;
  confidence: number;
}

interface Face {
  id: string;
  bbox_x: number;
  bbox_y: number;
  bbox_width: number;
  bbox_height: number;
  confidence: number;
  person_name?: string;
  cluster_id?: string;
  cluster_confidence?: number;
  match_candidates?: FaceMatchCandidate[];
  // Age/gender fields
  age_estimate?: number;
  gender?: string; // 'M', 'F', 'U'
  gender_confidence?: number;
  // Face tags (expression, emotion, gaze)
  tags?: SceneTag[];
}

interface FaceOverlayProps {
  faces: Face[];
  originalWidth: number;
  originalHeight: number;
  displayWidth: number;
  displayHeight: number;
  hoveredFaceId: string | null;
  onFaceHover: (faceId: string | null) => void;
  onFaceClick: (face: Face, event: React.MouseEvent) => void;
  opacity?: number;
}

function FaceOverlay({
  faces,
  originalWidth,
  originalHeight,
  displayWidth,
  displayHeight,
  hoveredFaceId,
  onFaceHover,
  onFaceClick,
  opacity = 1,
}: FaceOverlayProps) {
  // Calculate scaling factors based on displayed vs original dimensions
  const scaleX = displayWidth / originalWidth;
  const scaleY = displayHeight / originalHeight;

  return (
    <div
      className="absolute inset-0 pointer-events-none"
      style={{
        width: displayWidth,
        height: displayHeight,
      }}
    >
      {faces.map((face, index) => {
        // Calculate scaled positions
        const left = face.bbox_x * scaleX;
        const top = face.bbox_y * scaleY;
        const width = face.bbox_width * scaleX;
        const height = face.bbox_height * scaleY;

        // Determine if label should be above the box (when there's enough space)
        const labelAbove = top > 20;
        const isHovered = hoveredFaceId === face.id;

        return (
          <button
            type="button"
            key={face.id}
            className={cn(
              "absolute border-2 transition-all duration-200 pointer-events-auto cursor-pointer",
              isHovered ? "border-blue-500 shadow-lg shadow-blue-500/50 z-20" : "border-red-500",
            )}
            style={{
              left: `${left}px`,
              top: `${top}px`,
              width: `${width}px`,
              height: `${height}px`,
              opacity: isHovered ? 1 : opacity,
            }}
            onMouseEnter={() => onFaceHover(face.id)}
            onMouseLeave={() => onFaceHover(null)}
            onClick={(e) => onFaceClick(face, e)}
          >
            <div
              className={cn(
                "absolute text-white px-2 py-0.5 text-xs rounded whitespace-nowrap -left-0.5 z-10 transition-all duration-200",
                isHovered ? "bg-blue-500" : "bg-red-500",
                labelAbove ? "top-[-20px] rounded-bl-none" : "bottom-[-20px] rounded-tl-none",
              )}
            >
              <div>Face {index + 1}</div>
              {face.person_name && <div>{face.person_name}</div>}
            </div>
          </button>
        );
      })}
    </div>
  );
}

export function meta({ params }: Route.MetaArgs) {
  return [
    { title: `Storyteller - Photo ${params.id}` },
    { name: "description", content: `View details for photo ${params.id}` },
  ];
}

export async function action({ request }: Route.ActionArgs) {
  const { collectionId } = await requireCollectionId(request);
  const formData = await request.formData();
  const intent = formData.get("intent");

  if (intent === "remove-from-cluster") {
    const faceId = parseInt(formData.get("faceId") as string, 10);
    if (!faceId || Number.isNaN(faceId)) {
      return { success: false, message: "Invalid face ID" };
    }
    return removeFaceFromClusterWithConstraint(collectionId, faceId);
  }

  return { success: false, message: "Unknown action" };
}

export async function loader({ request, params }: Route.LoaderArgs) {
  const { collectionId } = await requireCollectionId(request);
  const photoId = parseInt(params.id, 10);

  try {
    const photo = await getPhotoDetails(collectionId, photoId);
    return { photo };
  } catch (error) {
    console.error(`Failed to load photo ${photoId}:`, error);
    return { photo: null };
  }
}

export default function PhotoDetail({ loaderData }: Route.ComponentProps) {
  const { photo } = loaderData;
  const navigate = useNavigate();
  const location = useLocation();
  const fetcher = useFetcher();
  const fromWall = (location.state as { fromWall?: boolean } | null)?.fromWall === true;
  const [showFaces, setShowFaces] = useState(false);
  const [showLowConfidenceFaceTags, setShowLowConfidenceFaceTags] = useState(false);
  const [showLowConfidenceSceneTags, setShowLowConfidenceSceneTags] = useState(false);
  const [hoveredFaceId, setHoveredFaceId] = useState<string | null>(null);
  const [isToggleHovered, setIsToggleHovered] = useState(false);
  const [isZoomed, setIsZoomed] = useState(false);
  const [imageMeasures, imageMeasureRef] = useMeasure<HTMLImageElement>();
  const [zoomedImageMeasures, zoomedImageMeasureRef] = useMeasure<HTMLImageElement>();
  const [sectionStates, setSectionStates] = useState({
    basicInfo: true,
    location: true,
    faces: true,
    sceneTags: true,
    aiAnalysis: true,
  });

  // Close zoom overlay on Escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isZoomed) {
        setIsZoomed(false);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isZoomed]);

  useEffect(() => {
    // Load section states from localStorage
    const savedStates = localStorage.getItem("photoDetailSections");
    if (savedStates) {
      setSectionStates(JSON.parse(savedStates));
    }

    // Load face overlay preference
    const savedFaceState = localStorage.getItem("showFaceBoundingBoxes");
    if (savedFaceState) {
      setShowFaces(savedFaceState === "true");
    }

    // Load low confidence tags preferences
    const savedLowConfidenceFaceState = localStorage.getItem("showLowConfidenceFaceTags");
    if (savedLowConfidenceFaceState) {
      setShowLowConfidenceFaceTags(savedLowConfidenceFaceState === "true");
    }
    const savedLowConfidenceSceneState = localStorage.getItem("showLowConfidenceSceneTags");
    if (savedLowConfidenceSceneState) {
      setShowLowConfidenceSceneTags(savedLowConfidenceSceneState === "true");
    }
  }, []);

  const updateSectionState = (section: string, isOpen: boolean) => {
    const newStates = { ...sectionStates, [section]: isOpen };
    setSectionStates(newStates);
    localStorage.setItem("photoDetailSections", JSON.stringify(newStates));
  };

  const updateFaceState = (checked: boolean | "indeterminate") => {
    if (checked === "indeterminate") return;
    setShowFaces(checked);
    localStorage.setItem("showFaceBoundingBoxes", checked.toString());
  };

  const updateLowConfidenceFaceTagsState = (checked: boolean | "indeterminate") => {
    if (checked === "indeterminate") return;
    setShowLowConfidenceFaceTags(checked);
    localStorage.setItem("showLowConfidenceFaceTags", checked.toString());
  };

  const updateLowConfidenceSceneTagsState = (checked: boolean | "indeterminate") => {
    if (checked === "indeterminate") return;
    setShowLowConfidenceSceneTags(checked);
    localStorage.setItem("showLowConfidenceSceneTags", checked.toString());
  };

  if (!photo) {
    return (
      <Layout>
        <div className="text-center py-12">
          <div className="text-red-500 text-lg">Photo not found</div>
        </div>
      </Layout>
    );
  }

  const breadcrumbItems = [];
  if (photo.year && photo.month && photo.month_name) {
    const wallSuffix = fromWall ? "/wall" : "";
    breadcrumbItems.push(
      { label: photo.year.toString(), href: `/year/${photo.year}${wallSuffix}` },
      {
        label: photo.month_name,
        href: `/year/${photo.year}/month/${photo.month}${wallSuffix}`,
      },
      { label: photo.filename_only },
    );
  } else {
    breadcrumbItems.push({ label: photo.filename_only });
  }

  return (
    <Layout>
      {/* Fullscreen zoom overlay */}
      {isZoomed && (
        <button
          type="button"
          className="fixed inset-0 z-50 bg-black/95 flex items-center justify-center"
          onClick={() => setIsZoomed(false)}
        >
          <button
            type="button"
            className="absolute top-4 right-4 text-white/70 hover:text-white z-10"
            onClick={() => setIsZoomed(false)}
          >
            <X className="h-8 w-8" />
          </button>
          <div className="overflow-auto max-h-screen max-w-screen p-4">
            <div className="relative inline-block">
              <img
                ref={zoomedImageMeasureRef}
                src={`/api/image/${photo.id}?full=true`}
                alt={photo.filename_only}
                className="max-w-none cursor-zoom-out"
                onClick={(e) => {
                  e.stopPropagation();
                  setIsZoomed(false);
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.stopPropagation();
                    setIsZoomed(false);
                  }
                }}
              />
              {/* Face Overlay for zoomed view */}
              {showFaces &&
                photo.faces &&
                photo.faces.length > 0 &&
                photo.image_width &&
                photo.image_height &&
                zoomedImageMeasures &&
                zoomedImageMeasures.width &&
                zoomedImageMeasures.height && (
                  <FaceOverlay
                    faces={photo.faces}
                    originalWidth={photo.image_width}
                    originalHeight={photo.image_height}
                    displayWidth={zoomedImageMeasures.width}
                    displayHeight={zoomedImageMeasures.height}
                    hoveredFaceId={hoveredFaceId}
                    onFaceHover={setHoveredFaceId}
                    opacity={1}
                    onFaceClick={(face, event) => {
                      event.stopPropagation();
                      const url = face.cluster_id ? `/cluster/${face.cluster_id}` : `/face/${face.id}/similar`;
                      if (event.metaKey || event.ctrlKey) {
                        window.open(url, "_blank");
                      } else {
                        setIsZoomed(false);
                        navigate(url);
                      }
                    }}
                  />
                )}
            </div>
          </div>
        </button>
      )}

      <div className="space-y-6">
        <Breadcrumb items={breadcrumbItems} />

        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
          {/* Image Column */}
          <div className="lg:col-span-3 flex flex-col">
            <div className="flex items-start gap-2">
              <button type="button" className="relative inline-block cursor-zoom-in" onClick={() => setIsZoomed(true)}>
                <img
                  ref={imageMeasureRef}
                  src={`/api/image/${photo.id}`}
                  alt={photo.filename_only}
                  className="rounded-lg shadow-lg max-w-full max-h-[85vh] w-auto h-auto object-contain"
                />

                {/* Face Overlay */}
                {(showFaces || isToggleHovered || hoveredFaceId) &&
                  photo.faces &&
                  photo.faces.length > 0 &&
                  photo.image_width &&
                  photo.image_height &&
                  imageMeasures &&
                  imageMeasures.width &&
                  imageMeasures.height && (
                    <FaceOverlay
                      faces={
                        showFaces || isToggleHovered
                          ? photo.faces
                          : photo.faces.filter((f: Face) => f.id === hoveredFaceId)
                      }
                      originalWidth={photo.image_width}
                      originalHeight={photo.image_height}
                      displayWidth={imageMeasures.width}
                      displayHeight={imageMeasures.height}
                      hoveredFaceId={hoveredFaceId}
                      onFaceHover={setHoveredFaceId}
                      opacity={showFaces ? 1 : isToggleHovered ? 0.25 : 1}
                      onFaceClick={(face, event) => {
                        event.stopPropagation();
                        const url = face.cluster_id ? `/cluster/${face.cluster_id}` : `/face/${face.id}/similar`;
                        if (event.metaKey || event.ctrlKey) {
                          window.open(url, "_blank");
                        } else {
                          navigate(url);
                        }
                      }}
                    />
                  )}
              </button>

              {/* Face toggle button */}
              {photo.face_count > 0 && (
                <Button
                  variant="ghost"
                  size="icon"
                  className={cn(
                    "h-8 w-8 flex-shrink-0",
                    showFaces ? "text-blue-600 bg-blue-50 hover:bg-blue-100" : "text-gray-400 hover:text-gray-600",
                  )}
                  onClick={(e) => {
                    e.stopPropagation();
                    updateFaceState(!showFaces);
                  }}
                  onMouseEnter={() => setIsToggleHovered(true)}
                  onMouseLeave={() => setIsToggleHovered(false)}
                  title={showFaces ? "Hide face bounding boxes" : "Show face bounding boxes"}
                >
                  <ScanFace className="h-5 w-5" />
                </Button>
              )}
            </div>
          </div>

          {/* Metadata Column */}
          <div className="lg:col-span-2 space-y-4">
            {/* Basic Information */}
            <Card>
              <Collapsible
                open={sectionStates.basicInfo}
                onOpenChange={(open) => updateSectionState("basicInfo", open)}
              >
                <CollapsibleTrigger asChild>
                  <CardHeader className="cursor-pointer hover:bg-gray-50">
                    <CardTitle className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <Info className="h-5 w-5" />
                        <span>Basic Information</span>
                      </div>
                      <ChevronDown className="h-4 w-4 transition-transform data-[state=open]:rotate-180" />
                    </CardTitle>
                  </CardHeader>
                </CollapsibleTrigger>
                <CollapsibleContent>
                  <CardContent>
                    <div className="space-y-3 text-sm">
                      <div>
                        <span className="font-medium">Filename:</span> {photo.filename_only}
                      </div>
                      <div>
                        <span className="font-medium">Photo ID:</span> <code className="text-xs">{photo.id}</code>
                      </div>
                      <div>
                        <span className="font-medium">Original Path:</span>
                        <div className="text-xs text-gray-600 break-all">{photo.orig_path}</div>
                      </div>
                      {photo.full_path && (
                        <div>
                          <span className="font-medium">Full Size Path:</span>
                          <div className="text-xs text-gray-600 break-all">{photo.full_path}</div>
                        </div>
                      )}
                      <div>
                        <span className="font-medium">Medium Path:</span>
                        <div className="text-xs text-gray-600 break-all">{photo.med_path}</div>
                      </div>
                      {photo.captured_at && (
                        <div>
                          <span className="font-medium">Captured:</span> {new Date(photo.captured_at).toLocaleString()}
                        </div>
                      )}
                      {photo.photo_created_at && (
                        <div>
                          <span className="font-medium">Added to DB:</span>{" "}
                          {new Date(photo.photo_created_at).toLocaleString()}
                        </div>
                      )}
                    </div>
                  </CardContent>
                </CollapsibleContent>
              </Collapsible>
            </Card>

            {/* Location */}
            {photo.latitude && photo.longitude && (
              <Card>
                <Collapsible
                  open={sectionStates.location}
                  onOpenChange={(open) => updateSectionState("location", open)}
                >
                  <CollapsibleTrigger asChild>
                    <CardHeader className="cursor-pointer hover:bg-gray-50">
                      <CardTitle className="flex items-center justify-between">
                        <div className="flex items-center space-x-2">
                          <MapPin className="h-5 w-5" />
                          <span>Location</span>
                        </div>
                        <ChevronDown className="h-4 w-4 transition-transform data-[state=open]:rotate-180" />
                      </CardTitle>
                    </CardHeader>
                  </CollapsibleTrigger>
                  <CollapsibleContent>
                    <CardContent>
                      <div className="space-y-3 text-sm">
                        <div>Latitude: {photo.latitude}</div>
                        <div>Longitude: {photo.longitude}</div>
                        <Button size="sm" variant="outline" asChild>
                          <a
                            href={`https://www.google.com/maps?q=${photo.latitude},${photo.longitude}`}
                            target="_blank"
                            rel="noopener noreferrer"
                          >
                            <MapPin className="h-4 w-4 mr-1" />
                            View on Map
                            <ExternalLink className="h-3 w-3 ml-1" />
                          </a>
                        </Button>
                      </div>
                    </CardContent>
                  </CollapsibleContent>
                </Collapsible>
              </Card>
            )}

            {/* Face Detection */}
            {photo.face_count > 0 && (
              <Card>
                <Collapsible open={sectionStates.faces} onOpenChange={(open) => updateSectionState("faces", open)}>
                  <CollapsibleTrigger asChild>
                    <CardHeader className="cursor-pointer hover:bg-gray-50">
                      <CardTitle className="flex items-center justify-between">
                        <div className="flex items-center space-x-2">
                          <Users className="h-5 w-5" />
                          <span>Face Detection</span>
                        </div>
                        <ChevronDown className="h-4 w-4 transition-transform data-[state=open]:rotate-180" />
                      </CardTitle>
                    </CardHeader>
                  </CollapsibleTrigger>
                  <CollapsibleContent>
                    <CardContent className="space-y-4">
                      <div>
                        <span className="font-medium">Faces Detected:</span> {photo.face_count}
                      </div>

                      <div className="flex items-center gap-6">
                        <div className="flex items-center space-x-2">
                          <Checkbox name="show-faces" checked={showFaces} onCheckedChange={updateFaceState} />
                          <label htmlFor="show-faces" className="text-sm cursor-pointer">
                            Show face bounding boxes
                          </label>
                        </div>
                        <div className="flex items-center space-x-2">
                          <Checkbox
                            name="show-low-confidence-face"
                            checked={showLowConfidenceFaceTags}
                            onCheckedChange={updateLowConfidenceFaceTagsState}
                          />
                          <label htmlFor="show-low-confidence-face" className="text-sm cursor-pointer">
                            Show low-confidence face tags
                          </label>
                        </div>
                      </div>

                      {photo.faces && photo.faces.length > 0 && (
                        <div className="space-y-2">
                          <h6 className="font-medium">Face Details:</h6>
                          {photo.faces.map((face: Face, index: number) => {
                            const handleFaceClick = (event: React.MouseEvent | React.KeyboardEvent) => {
                              const url = face.cluster_id ? `/cluster/${face.cluster_id}` : `/face/${face.id}/similar`;
                              if (event.metaKey || event.ctrlKey) {
                                window.open(url, "_blank");
                              } else {
                                navigate(url);
                              }
                            };

                            return (
                              // biome-ignore lint/a11y/useSemanticElements: This is an interactive highlight, not a button action
                              <div
                                key={face.id}
                                role="button"
                                tabIndex={0}
                                className={cn(
                                  "flex items-center justify-between p-2 border rounded transition-all duration-200 cursor-pointer",
                                  hoveredFaceId === face.id
                                    ? "border-blue-500 bg-blue-50 shadow-md"
                                    : "border-gray-300 hover:border-blue-400 hover:bg-blue-50/50",
                                )}
                                onClick={handleFaceClick}
                                onKeyDown={(e) => {
                                  if (e.key === "Enter" || e.key === " ") {
                                    handleFaceClick(e);
                                  }
                                }}
                                onMouseEnter={() => setHoveredFaceId(face.id)}
                                onMouseLeave={() => setHoveredFaceId(null)}
                                onFocus={() => setHoveredFaceId(face.id)}
                                onBlur={() => setHoveredFaceId(null)}
                              >
                                <div className="flex flex-col space-y-2">
                                  <div className="flex items-center space-x-2">
                                    <span className="font-medium">Face {index + 1}:</span>
                                    {face.person_name ? (
                                      <>
                                        <Badge variant="default">{face.person_name}</Badge>
                                        {face.cluster_id && (
                                          <Button
                                            variant="ghost"
                                            size="sm"
                                            className="h-6 w-6 p-0 text-gray-400 hover:text-red-500"
                                            title="Remove from cluster (adds constraint)"
                                            onClick={(e) => {
                                              e.stopPropagation();
                                              fetcher.submit(
                                                { intent: "remove-from-cluster", faceId: face.id },
                                                { method: "post" },
                                              );
                                            }}
                                            disabled={fetcher.state === "submitting"}
                                          >
                                            <Ban className="h-4 w-4" />
                                          </Button>
                                        )}
                                      </>
                                    ) : face.cluster_id ? (
                                      <>
                                        <Badge variant="secondary">
                                          Cluster {face.cluster_id}
                                          {face.cluster_confidence && (
                                            <span> ({Math.round(face.cluster_confidence * 100)}%)</span>
                                          )}
                                        </Badge>
                                        <Button
                                          variant="ghost"
                                          size="sm"
                                          className="h-6 w-6 p-0 text-gray-400 hover:text-red-500"
                                          title="Remove from cluster (adds constraint)"
                                          onClick={(e) => {
                                            e.stopPropagation();
                                            fetcher.submit(
                                              { intent: "remove-from-cluster", faceId: face.id },
                                              { method: "post" },
                                            );
                                          }}
                                          disabled={fetcher.state === "submitting"}
                                        >
                                          <Ban className="h-4 w-4" />
                                        </Button>
                                      </>
                                    ) : (
                                      <Badge variant="outline">
                                        <Users className="h-3 w-3 mr-1" />
                                        Find Similar
                                      </Badge>
                                    )}
                                    {/* Age/Gender badges */}
                                    {(face.age_estimate || face.gender) && (
                                      <div className="flex items-center space-x-1 ml-2">
                                        {formatAge(face.age_estimate) && (
                                          <Badge variant="outline" className="text-xs bg-blue-50 border-blue-200">
                                            <Calendar className="h-3 w-3 mr-1" />
                                            {formatAge(face.age_estimate)}
                                          </Badge>
                                        )}
                                        {formatGender(face.gender) && (
                                          <Badge
                                            variant="outline"
                                            className={cn(
                                              "text-xs",
                                              face.gender === "M"
                                                ? "bg-sky-50 border-sky-200"
                                                : face.gender === "F"
                                                  ? "bg-pink-50 border-pink-200"
                                                  : "bg-gray-50 border-gray-200",
                                            )}
                                          >
                                            <User className="h-3 w-3 mr-1" />
                                            {formatGender(face.gender)}
                                            {face.gender_confidence && (
                                              <span className="ml-1 text-gray-500">
                                                ({Math.round(face.gender_confidence * 100)}%)
                                              </span>
                                            )}
                                          </Badge>
                                        )}
                                      </div>
                                    )}
                                  </div>
                                  {/* Face tags (expression, emotion, gaze) - only show if face detection confidence is high enough */}
                                  {face.tags &&
                                    (showLowConfidenceFaceTags ||
                                      face.confidence >= displaySettings.minFaceConfidenceForTags) &&
                                    face.tags.filter(
                                      (t: SceneTag) =>
                                        showLowConfidenceFaceTags || t.confidence > displaySettings.minTagConfidence,
                                    ).length > 0 && (
                                      <div className="flex flex-wrap gap-1 mt-1">
                                        {face.tags
                                          .filter(
                                            (t: SceneTag) =>
                                              showLowConfidenceFaceTags ||
                                              t.confidence > displaySettings.minTagConfidence,
                                          )
                                          .map((tag: SceneTag, tagIndex: number) => (
                                            <Badge
                                              // biome-ignore lint/suspicious/noArrayIndexKey: index is fine
                                              key={tagIndex}
                                              variant="outline"
                                              className={cn("text-xs", getCategoryColor(tag.category_name))}
                                              title={`${tag.category_name}: ${tag.prompt_text}`}
                                            >
                                              {tag.display_name || tag.label}
                                              <span className="ml-1 text-gray-400">
                                                {Math.round(tag.confidence * 100)}%
                                              </span>
                                            </Badge>
                                          ))}
                                      </div>
                                    )}
                                  {face.match_candidates && face.match_candidates.length > 0 && (
                                    <div className="flex flex-col space-y-1">
                                      <span className="text-xs text-gray-500">Potential matches:</span>
                                      <div className="flex flex-wrap gap-1">
                                        {face.match_candidates.map((candidate) => (
                                          <Link
                                            key={candidate.cluster_id}
                                            to={`/cluster/${candidate.cluster_id}`}
                                            onClick={(e) => e.stopPropagation()}
                                          >
                                            <Badge
                                              variant="outline"
                                              className="hover:bg-gray-100 cursor-pointer text-xs"
                                            >
                                              {candidate.person_name || `Cluster ${candidate.cluster_id}`}
                                              <span className="ml-1 text-gray-500">
                                                ({Math.round(candidate.similarity * 100)}%)
                                              </span>
                                            </Badge>
                                          </Link>
                                        ))}
                                      </div>
                                    </div>
                                  )}
                                </div>
                                <span className="text-sm text-gray-600">{Math.round(face.confidence * 100)}%</span>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </CardContent>
                  </CollapsibleContent>
                </Collapsible>
              </Card>
            )}

            {/* Scene & Face Tags */}
            {(photo.scene_tags?.some(
              (t: SceneTag) => showLowConfidenceSceneTags || t.confidence > displaySettings.minTagConfidence,
            ) ||
              photo.scene_taxonomy?.top_labels?.some(
                (t: SceneTaxonomyLabel) =>
                  showLowConfidenceSceneTags || t.confidence > displaySettings.minTaxonomyConfidence,
              )) && (
              <Card>
                <Collapsible
                  open={sectionStates.sceneTags}
                  onOpenChange={(open) => updateSectionState("sceneTags", open)}
                >
                  <CollapsibleTrigger asChild>
                    <CardHeader className="cursor-pointer hover:bg-gray-50">
                      <CardTitle className="flex items-center justify-between">
                        <div className="flex items-center space-x-2">
                          <Sparkles className="h-5 w-5" />
                          <span>Scene Analysis</span>
                        </div>
                        <ChevronDown className="h-4 w-4 transition-transform data-[state=open]:rotate-180" />
                      </CardTitle>
                    </CardHeader>
                  </CollapsibleTrigger>
                  <CollapsibleContent>
                    <CardContent className="space-y-4">
                      <div className="flex items-center space-x-2">
                        <Checkbox
                          name="show-low-confidence-scene"
                          checked={showLowConfidenceSceneTags}
                          onCheckedChange={updateLowConfidenceSceneTagsState}
                        />
                        <label htmlFor="show-low-confidence-scene" className="text-sm cursor-pointer">
                          Show low-confidence tags
                        </label>
                      </div>

                      {/* Apple Vision Taxonomy */}
                      {photo.scene_taxonomy?.top_labels &&
                        photo.scene_taxonomy.top_labels.filter(
                          (t: SceneTaxonomyLabel) =>
                            showLowConfidenceSceneTags || t.confidence > displaySettings.minTaxonomyConfidence,
                        ).length > 0 && (
                          <div>
                            <h6 className="font-medium text-sm mb-2 flex items-center">
                              <Tag className="h-4 w-4 mr-1" />
                              Scene Classification
                            </h6>
                            <div className="flex flex-wrap gap-1">
                              {photo.scene_taxonomy.top_labels
                                .filter(
                                  (t: SceneTaxonomyLabel) =>
                                    showLowConfidenceSceneTags || t.confidence > displaySettings.minTaxonomyConfidence,
                                )
                                .map((item: SceneTaxonomyLabel, index: number) => (
                                  <Badge
                                    // biome-ignore lint/suspicious/noArrayIndexKey: label might not be unique
                                    key={index}
                                    variant="secondary"
                                    className="text-xs"
                                  >
                                    {item.label}
                                    <span className="ml-1 text-gray-500">({Math.round(item.confidence * 100)}%)</span>
                                  </Badge>
                                ))}
                            </div>
                          </div>
                        )}

                      {/* Scene Tags by Category */}
                      {photo.scene_tags &&
                        photo.scene_tags.filter(
                          (t: SceneTag) =>
                            showLowConfidenceSceneTags || t.confidence > displaySettings.minTagConfidence,
                        ).length > 0 && (
                          <div className="space-y-3">
                            {/* Group tags by category */}
                            {Object.entries(
                              photo.scene_tags
                                .filter(
                                  (t: SceneTag) =>
                                    showLowConfidenceSceneTags || t.confidence > displaySettings.minTagConfidence,
                                )
                                .reduce(
                                  (acc: Record<string, SceneTag[]>, tag: SceneTag) => {
                                    const category = tag.category_name;
                                    if (!acc[category]) acc[category] = [];
                                    acc[category].push(tag);
                                    return acc;
                                  },
                                  {} as Record<string, SceneTag[]>,
                                ),
                            ).map(([category, tags]) => (
                              <div key={category}>
                                <h6 className="font-medium text-sm mb-1 capitalize">{category.replace(/_/g, " ")}</h6>
                                <div className="flex flex-wrap gap-1">
                                  {(tags as SceneTag[]).map((tag: SceneTag, index: number) => (
                                    <Badge
                                      // biome-ignore lint/suspicious/noArrayIndexKey: index is fine here
                                      key={index}
                                      variant="outline"
                                      className={cn("text-xs", getCategoryColor(tag.category_name))}
                                      title={tag.prompt_text}
                                    >
                                      {tag.display_name || tag.label}
                                      <span className="ml-1 text-gray-500">({Math.round(tag.confidence * 100)}%)</span>
                                    </Badge>
                                  ))}
                                </div>
                              </div>
                            ))}
                          </div>
                        )}
                    </CardContent>
                  </CollapsibleContent>
                </Collapsible>
              </Card>
            )}

            {/* AI Analysis */}
            {(photo.description || photo.emotional_tone || photo.objects) && (
              <Card>
                <Collapsible
                  open={sectionStates.aiAnalysis}
                  onOpenChange={(open) => updateSectionState("aiAnalysis", open)}
                >
                  <CollapsibleTrigger asChild>
                    <CardHeader className="cursor-pointer hover:bg-gray-50">
                      <CardTitle className="flex items-center justify-between">
                        <div className="flex items-center space-x-2">
                          <Bot className="h-5 w-5" />
                          <span>AI Analysis Summary</span>
                        </div>
                        <ChevronDown className="h-4 w-4 transition-transform data-[state=open]:rotate-180" />
                      </CardTitle>
                    </CardHeader>
                  </CollapsibleTrigger>
                  <CollapsibleContent>
                    <CardContent className="space-y-3 text-sm">
                      {photo.description && (
                        <div>
                          <span className="font-medium">Description:</span> {photo.description}
                        </div>
                      )}
                      {photo.emotional_tone && (
                        <div>
                          <span className="font-medium">Emotional Tone:</span> {photo.emotional_tone}
                        </div>
                      )}
                      {photo.location_description && (
                        <div>
                          <span className="font-medium">Location Description:</span> {photo.location_description}
                        </div>
                      )}
                      {photo.people_count && (
                        <div>
                          <span className="font-medium">People Count:</span> {photo.people_count}
                        </div>
                      )}
                      {photo.objects && photo.objects.length > 0 && (
                        <div>
                          <span className="font-medium">Objects Detected:</span>
                          <div className="flex flex-wrap gap-1 mt-1">
                            {photo.objects.map((obj: string, index: number) => (
                              // biome-ignore lint/suspicious/noArrayIndexKey: ok
                              <Badge key={index} variant="secondary">
                                {obj}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}
                      {photo.model_name && <div className="text-xs text-gray-500">Model: {photo.model_name}</div>}
                    </CardContent>
                  </CollapsibleContent>
                </Collapsible>
              </Card>
            )}
          </div>
        </div>

        {/* Full Analysis & Metadata */}
        {photo.analysis_formatted && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Code className="h-5 w-5" />
                <span>Full AI Analysis</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <pre className="bg-gray-50 p-4 rounded text-xs overflow-x-auto">
                <code>{photo.analysis_formatted}</code>
              </pre>
            </CardContent>
          </Card>
        )}

        {photo.metadata_formatted && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Camera className="h-5 w-5" />
                <span>EXIF Metadata</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <details>
                <summary className="cursor-pointer text-sm font-medium mb-2">
                  Click to expand technical metadata
                </summary>
                <pre className="bg-gray-50 p-4 rounded text-xs overflow-x-auto">
                  <code>{photo.metadata_formatted}</code>
                </pre>
              </details>
            </CardContent>
          </Card>
        )}
      </div>
    </Layout>
  );
}
