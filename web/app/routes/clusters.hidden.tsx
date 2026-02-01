import { Eye, Users } from "lucide-react";
import { Link, useFetcher } from "react-router";
import { Breadcrumb } from "~/components/breadcrumb";
import { Layout } from "~/components/layout";
import { Pagination } from "~/components/pagination";
import { Button } from "~/components/ui/button";
import { Card, CardContent } from "~/components/ui/card";
import { getHiddenClusters, getHiddenClustersCount, setClusterHidden } from "~/lib/db.server";
import type { Route } from "./+types/clusters.hidden";

export function meta() {
  return [
    { title: "PhotoDB - Hidden Clusters" },
    {
      name: "description",
      content: "View and manage hidden face clusters",
    },
  ];
}

export async function action({ request }: Route.ActionArgs) {
  const formData = await request.formData();
  const intent = formData.get("intent");

  if (intent === "unhide") {
    const clusterId = formData.get("clusterId") as string;
    if (clusterId) {
      const result = await setClusterHidden(clusterId, false);
      return result;
    }
    return { success: false, message: "Invalid cluster ID" };
  }

  return { success: false, message: "Unknown action" };
}

export async function loader({ request }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const page = parseInt(url.searchParams.get("page") || "1", 10);
  const limit = 24;
  const offset = (page - 1) * limit;

  try {
    const clusters = await getHiddenClusters(limit, offset);
    const totalClusters = await getHiddenClustersCount();
    const totalPages = Math.ceil(totalClusters / limit);

    return {
      clusters,
      totalClusters,
      totalPages,
      currentPage: page,
    };
  } catch (error) {
    console.error("Failed to load hidden clusters:", error);
    return {
      clusters: [],
      totalClusters: 0,
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
  const containerSize = 128;
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

export default function HiddenClustersView({ loaderData }: Route.ComponentProps) {
  const { clusters, totalClusters, totalPages, currentPage } = loaderData;
  const fetcher = useFetcher();

  const handleUnhide = (clusterId: string) => {
    fetcher.submit({ intent: "unhide", clusterId }, { method: "post" });
  };

  const breadcrumbItems = [{ label: "Clusters", href: "/clusters" }, { label: "Hidden" }];

  return (
    <Layout>
      <div className="space-y-6">
        <Breadcrumb items={breadcrumbItems} />

        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Users className="h-8 w-8 text-gray-700" />
            <h1 className="text-3xl font-bold text-gray-900">Hidden Clusters</h1>
          </div>
          <span className="text-gray-600">
            {totalClusters} hidden cluster{totalClusters !== 1 ? "s" : ""}
          </span>
        </div>

        {clusters.length > 0 ? (
          <>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-4">
              {clusters.map((cluster) => (
                <Card key={cluster.id} className="h-full">
                  <CardContent className="p-4">
                    <div className="text-center space-y-3">
                      <Link to={`/cluster/${cluster.id}`}>
                        {cluster.photo_id &&
                        cluster.bbox_x !== null &&
                        cluster.normalized_width &&
                        cluster.normalized_height ? (
                          <div className="relative w-32 h-32 mx-auto bg-gray-100 rounded-lg border overflow-hidden opacity-60 hover:opacity-100 transition-opacity">
                            <img
                              src={`/api/image/${cluster.photo_id}`}
                              alt={`Cluster ${cluster.id}`}
                              className="absolute max-w-none max-h-none"
                              style={getFaceCropStyle(
                                {
                                  bbox_x: cluster.bbox_x,
                                  bbox_y: cluster.bbox_y,
                                  bbox_width: cluster.bbox_width,
                                  bbox_height: cluster.bbox_height,
                                },
                                cluster.normalized_width,
                                cluster.normalized_height,
                              )}
                              loading="lazy"
                            />
                          </div>
                        ) : (
                          <div className="w-full h-32 bg-gray-200 rounded-lg flex items-center justify-center opacity-60">
                            <Users className="h-8 w-8 text-gray-400" />
                          </div>
                        )}
                      </Link>

                      <div className="space-y-1">
                        <div className="font-semibold text-gray-900">Cluster #{cluster.id}</div>
                        <div className="text-sm text-gray-600">
                          {cluster.face_count} face
                          {cluster.face_count !== 1 ? "s" : ""}
                        </div>
                        {cluster.person_name && (
                          <div className="text-sm font-medium text-blue-600 truncate" title={cluster.person_name}>
                            {cluster.person_name}
                          </div>
                        )}
                      </div>

                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleUnhide(cluster.id.toString())}
                        disabled={fetcher.state === "submitting"}
                      >
                        <Eye className="h-4 w-4 mr-1" />
                        Unhide
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            <Pagination currentPage={currentPage} totalPages={totalPages} baseUrl="/clusters/hidden" />
          </>
        ) : (
          <div className="text-center py-12">
            <Users className="h-16 w-16 text-gray-400 mx-auto mb-4" />
            <div className="text-gray-500 text-lg">No hidden clusters.</div>
            <div className="text-gray-400 text-sm mt-2">
              <Link to="/clusters" className="text-blue-500 hover:underline">
                View all clusters
              </Link>
            </div>
          </div>
        )}
      </div>
    </Layout>
  );
}
