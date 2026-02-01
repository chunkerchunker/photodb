import { EyeOff, Loader2, Users } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useFetcher } from "react-router";
import { Layout } from "~/components/layout";
import { Button } from "~/components/ui/button";
import { Card, CardContent } from "~/components/ui/card";
import { getClusters, getClustersCount, getHiddenClustersCount } from "~/lib/db.server";
import type { Route } from "./+types/clusters";

export function meta() {
  return [
    { title: "PhotoDB - Face Clusters" },
    {
      name: "description",
      content: "Browse face clusters sorted by face count",
    },
  ];
}

const LIMIT = 24; // 4x6 grid

export async function loader({ request }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const page = parseInt(url.searchParams.get("page") || "1", 10);
  const offset = (page - 1) * LIMIT;

  try {
    const clusters = await getClusters(LIMIT, offset);
    const totalClusters = await getClustersCount();
    const hiddenCount = await getHiddenClustersCount();
    const hasMore = offset + clusters.length < totalClusters;

    return {
      clusters,
      totalClusters,
      hiddenCount,
      hasMore,
      page,
    };
  } catch (error) {
    console.error("Failed to load clusters:", error);
    return {
      clusters: [],
      totalClusters: 0,
      hiddenCount: 0,
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

type Cluster = Route.ComponentProps["loaderData"]["clusters"][number];

export default function ClustersView({ loaderData }: Route.ComponentProps) {
  const { clusters: initialClusters, totalClusters, hiddenCount, hasMore: initialHasMore, page: initialPage } = loaderData;

  const [clusters, setClusters] = useState<Cluster[]>(initialClusters);
  const [page, setPage] = useState(initialPage);
  const [hasMore, setHasMore] = useState(initialHasMore);

  const fetcher = useFetcher<typeof loader>();
  const loadMoreRef = useRef<HTMLDivElement>(null);

  // Reset state when initial data changes (e.g., navigation)
  useEffect(() => {
    setClusters(initialClusters);
    setPage(initialPage);
    setHasMore(initialHasMore);
  }, [initialClusters, initialPage, initialHasMore]);

  // Append new clusters when fetcher returns data
  useEffect(() => {
    if (fetcher.data && fetcher.data.clusters.length > 0) {
      setClusters((prev) => {
        const existingIds = new Set(prev.map((c) => c.id));
        const newClusters = fetcher.data!.clusters.filter((c) => !existingIds.has(c.id));
        return [...prev, ...newClusters];
      });
      setPage(fetcher.data.page);
      setHasMore(fetcher.data.hasMore);
    }
  }, [fetcher.data]);

  const loadMore = useCallback(() => {
    if (fetcher.state === "idle" && hasMore) {
      fetcher.load(`/clusters?page=${page + 1}`);
    }
  }, [fetcher, hasMore, page]);

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

  const isLoading = fetcher.state === "loading";

  return (
    <Layout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Users className="h-8 w-8 text-gray-700" />
            <h1 className="text-3xl font-bold text-gray-900">Face Clusters</h1>
          </div>
          <div className="flex items-center space-x-4">
            {hiddenCount > 0 && (
              <Link to="/clusters/hidden">
                <Button variant="outline" size="sm">
                  <EyeOff className="h-4 w-4 mr-1" />
                  Hidden ({hiddenCount})
                </Button>
              </Link>
            )}
            <span className="text-gray-600">
              {totalClusters} cluster{totalClusters !== 1 ? "s" : ""}
            </span>
          </div>
        </div>

        {clusters.length > 0 ? (
          <>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-4">
              {clusters.map((cluster) => (
                <Link to={`/cluster/${cluster.id}`} key={cluster.id}>
                  <Card className="hover:shadow-lg transition-shadow h-full">
                    <CardContent className="p-4">
                      <div className="text-center space-y-3">
                        {cluster.photo_id &&
                        cluster.bbox_x !== null &&
                        cluster.normalized_width &&
                        cluster.normalized_height ? (
                          <div className="relative w-32 h-32 mx-auto bg-gray-100 rounded-lg border overflow-hidden">
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
                          <div className="w-full h-32 bg-gray-200 rounded-lg flex items-center justify-center">
                            <Users className="h-8 w-8 text-gray-400" />
                          </div>
                        )}

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
                      </div>
                    </CardContent>
                  </Card>
                </Link>
              ))}
            </div>

            {/* Infinite scroll trigger */}
            <div ref={loadMoreRef} className="flex justify-center py-8">
              {isLoading && (
                <div className="flex items-center space-x-2 text-gray-500">
                  <Loader2 className="h-5 w-5 animate-spin" />
                  <span>Loading more clusters...</span>
                </div>
              )}
              {!hasMore && clusters.length > 0 && (
                <span className="text-gray-400 text-sm">All clusters loaded</span>
              )}
            </div>
          </>
        ) : (
          <div className="text-center py-12">
            <Users className="h-16 w-16 text-gray-400 mx-auto mb-4" />
            <div className="text-gray-500 text-lg">No face clusters found in the database.</div>
            <div className="text-gray-400 text-sm mt-2">
              Clusters are created when faces are detected and grouped together.
            </div>
          </div>
        )}
      </div>
    </Layout>
  );
}
