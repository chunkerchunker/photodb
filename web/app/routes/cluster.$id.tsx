import { Users } from "lucide-react";
import { Link } from "react-router";
import { Breadcrumb } from "~/components/breadcrumb";
import { Layout } from "~/components/layout";
import { Pagination } from "~/components/pagination";
import { Badge } from "~/components/ui/badge";
import { Card, CardContent } from "~/components/ui/card";
import { getClusterDetails, getClusterFaces, getClusterFacesCount } from "~/lib/db.server";
import type { Route } from "./+types/cluster.$id";

export function meta({ params }: Route.MetaArgs) {
  return [
    { title: `PhotoDB - Cluster ${params.id}` },
    {
      name: "description",
      content: `View all faces in cluster ${params.id}`,
    },
  ];
}

export async function loader({ params, request }: Route.LoaderArgs) {
  const clusterId = params.id;
  const url = new URL(request.url);
  const page = parseInt(url.searchParams.get("page") || "1", 10);
  const limit = 24; // 4x6 grid
  const offset = (page - 1) * limit;

  try {
    const cluster = await getClusterDetails(clusterId);
    if (!cluster) {
      throw new Response("Cluster not found", { status: 404 });
    }

    const faces = await getClusterFaces(clusterId, limit, offset);
    const totalFaces = await getClusterFacesCount(clusterId);
    const totalPages = Math.ceil(totalFaces / limit);

    return {
      cluster,
      faces,
      totalFaces,
      totalPages,
      currentPage: page,
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
      totalPages: 0,
      currentPage: page,
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

export default function ClusterDetailView({ loaderData }: Route.ComponentProps) {
  const { cluster, faces, totalFaces, totalPages, currentPage } = loaderData;

  if (!cluster) {
    return (
      <Layout>
        <div className="text-center py-12">
          <div className="text-red-500 text-lg">Cluster not found</div>
        </div>
      </Layout>
    );
  }

  const breadcrumbItems = [{ label: "Clusters", href: "/clusters" }, { label: `Cluster ${cluster.id}` }];

  return (
    <Layout>
      <div className="space-y-6">
        <Breadcrumb items={breadcrumbItems} />

        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Users className="h-8 w-8 text-gray-700" />
            <h1 className="text-3xl font-bold text-gray-900">
              Cluster {cluster.id}
              {cluster.person_name && (
                <Badge variant="default" className="ml-3 text-base">
                  {cluster.person_name}
                </Badge>
              )}
            </h1>
          </div>
          <span className="text-gray-600">
            {totalFaces} face{totalFaces !== 1 ? "s" : ""}
          </span>
        </div>

        {faces.length > 0 ? (
          <>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-4">
              {faces.map((face) => (
                <Card key={face.id} className="hover:shadow-lg transition-shadow">
                  <CardContent className="p-4">
                    <div className="text-center space-y-3">
                      {face.photo_id && face.bbox_x !== null && face.normalized_width && face.normalized_height ? (
                        <Link to={`/photo/${face.photo_id}`}>
                          <div className="relative w-32 h-32 mx-auto bg-gray-100 rounded-lg border overflow-hidden hover:ring-2 hover:ring-blue-500 transition-all">
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
                          </div>
                        </Link>
                      ) : (
                        <div className="w-full h-32 bg-gray-200 rounded-lg flex items-center justify-center">
                          <Users className="h-8 w-8 text-gray-400" />
                        </div>
                      )}

                      <div className="space-y-1">
                        <div className="text-sm text-gray-600">Photo #{face.photo_id}</div>
                        <div className="text-xs text-gray-500">{Math.round(face.confidence * 100)}% confidence</div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            <Pagination currentPage={currentPage} totalPages={totalPages} baseUrl={`/cluster/${cluster.id}`} />
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
