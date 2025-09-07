import { useMeasure } from "@react-hookz/web";
import { Bot, Camera, ChevronDown, Code, ExternalLink, Info, MapPin, Users } from "lucide-react";
import { useEffect, useState } from "react";
import { Link } from "react-router";
import { Breadcrumb } from "~/components/breadcrumb";
import { Layout } from "~/components/layout";
import { Badge } from "~/components/ui/badge";
import { Button } from "~/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { Checkbox } from "~/components/ui/checkbox";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "~/components/ui/collapsible";
import { getPhotoDetails } from "~/lib/db.server";
import type { Route } from "./+types/photo";

interface Face {
  id: string;
  bbox_x: number;
  bbox_y: number;
  bbox_width: number;
  bbox_height: number;
  person_name?: string;
  cluster_id?: string;
  confidence: number;
}

interface FaceOverlayProps {
  faces: Face[];
  originalWidth: number;
  originalHeight: number;
  displayWidth: number;
  displayHeight: number;
}

function FaceOverlay({ faces, originalWidth, originalHeight, displayWidth, displayHeight }: FaceOverlayProps) {
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
        const labelAbove = top > 28;

        return (
          <div
            key={face.id}
            className="absolute border-2 border-red-500 bg-red-500/10"
            style={{
              left: `${left}px`,
              top: `${top}px`,
              width: `${width}px`,
              height: `${height}px`,
            }}
          >
            <div
              className="absolute bg-red-500 text-white px-2 py-1 text-xs rounded whitespace-nowrap left-0 z-10"
              style={{
                top: labelAbove ? "-28px" : "0",
              }}
            >
              <div>Face {index + 1}</div>
              {face.person_name && <div>{face.person_name}</div>}
              <div>{Math.round(face.confidence * 100)}%</div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function meta({ params }: Route.MetaArgs) {
  return [
    { title: `PhotoDB - Photo ${params.id}` },
    { name: "description", content: `View details for photo ${params.id}` },
  ];
}

export async function loader({ params }: Route.LoaderArgs) {
  const photoId = parseInt(params.id, 10);

  try {
    const photo = await getPhotoDetails(photoId);
    return { photo };
  } catch (error) {
    console.error(`Failed to load photo ${photoId}:`, error);
    return { photo: null };
  }
}

export default function PhotoDetail({ loaderData }: Route.ComponentProps) {
  const { photo } = loaderData;
  const [showFaces, setShowFaces] = useState(false);
  const [imageMeasures, imageMeasureRef] = useMeasure<HTMLImageElement>();
  const [sectionStates, setSectionStates] = useState({
    basicInfo: true,
    location: true,
    faces: true,
    aiAnalysis: true,
  });

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
  }, []);

  const updateSectionState = (section: string, isOpen: boolean) => {
    const newStates = { ...sectionStates, [section]: isOpen };
    setSectionStates(newStates);
    localStorage.setItem("photoDetailSections", JSON.stringify(newStates));
  };

  const updateFaceState = (checked: boolean) => {
    setShowFaces(checked);
    localStorage.setItem("showFaceBoundingBoxes", checked.toString());
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
    breadcrumbItems.push(
      { label: photo.year.toString(), href: `/year/${photo.year}` },
      {
        label: photo.month_name,
        href: `/year/${photo.year}/month/${photo.month}`,
      },
      { label: photo.filename_only },
    );
  } else {
    breadcrumbItems.push({ label: photo.filename_only });
  }

  return (
    <Layout>
      <div className="space-y-6">
        <Breadcrumb items={breadcrumbItems} />

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Image Column */}
          <div className="flex flex-col">
            <div className="relative inline-block size-fit">
              <img
                ref={imageMeasureRef}
                src={`/api/image/${photo.id}`}
                alt={photo.filename_only}
                className="w-full h-auto rounded-lg shadow-lg"
              />

              {/* Face Overlay */}
              {showFaces &&
                photo.faces &&
                photo.faces.length > 0 &&
                photo.image_width &&
                photo.image_height &&
                imageMeasures &&
                imageMeasures.width &&
                imageMeasures.height && (
                  <FaceOverlay
                    faces={photo.faces}
                    originalWidth={photo.image_width}
                    originalHeight={photo.image_height}
                    displayWidth={imageMeasures.width}
                    displayHeight={imageMeasures.height}
                  />
                )}
            </div>
          </div>

          {/* Metadata Column */}
          <div className="space-y-6">
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
                        <div className="text-xs text-gray-600 break-all">{photo.filename}</div>
                      </div>
                      <div>
                        <span className="font-medium">Normalized Path:</span>
                        <div className="text-xs text-gray-600 break-all">{photo.normalized_path}</div>
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
            {photo.face_count && photo.face_count > 0 && (
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

                      <div className="flex items-center space-x-2">
                        <Checkbox name="show-faces" checked={showFaces} onCheckedChange={updateFaceState} />
                        <label htmlFor="show-faces" className="text-sm cursor-pointer">
                          Show face bounding boxes
                        </label>
                      </div>

                      {photo.faces && photo.faces.length > 0 && (
                        <div className="space-y-2">
                          <h6 className="font-medium">Face Details:</h6>
                          {photo.faces.map((face: Face, index: number) => (
                            <div key={face.id} className="flex items-center justify-between p-2 border rounded">
                              <div className="flex items-center space-x-2">
                                <span className="font-medium">Face {index + 1}:</span>
                                {face.person_name ? (
                                  <Badge variant="default">{face.person_name}</Badge>
                                ) : face.cluster_id ? (
                                  <Link to={`/cluster/${face.cluster_id}`}>
                                    <Badge variant="secondary" className="hover:bg-secondary/80 cursor-pointer">
                                      Cluster {face.cluster_id}
                                    </Badge>
                                  </Link>
                                ) : (
                                  <Badge variant="outline">Unknown</Badge>
                                )}
                              </div>
                              <span className="text-sm text-gray-600">{Math.round(face.confidence * 100)}%</span>
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
                      {photo.model_name && (
                        <div className="text-xs text-gray-500">
                          Model: {photo.model_name}
                        </div>
                      )}
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
