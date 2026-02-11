import { Eye, Grid, Loader2, Users } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useFetcher } from "react-router";
import { Breadcrumb } from "~/components/breadcrumb";
import { CoverflowIcon } from "~/components/coverflow-icon";
import { Layout } from "~/components/layout";
import { ControlsCount, SecondaryControls } from "~/components/secondary-controls";
import { Button } from "~/components/ui/button";
import { Card, CardContent } from "~/components/ui/card";
import { ViewSwitcher } from "~/components/view-switcher";
import { requireCollectionId } from "~/lib/auth.server";
import { dataWithViewMode } from "~/lib/cookies.server";
import { getHiddenClusters, getHiddenClustersCount, setClusterHidden } from "~/lib/db.server";
import type { Route } from "./+types/clusters.hidden.grid";

export function meta() {
  return [
    { title: "Storyteller - Hidden Clusters" },
    {
      name: "description",
      content: "View and manage hidden face clusters",
    },
  ];
}

export async function action({ request }: Route.ActionArgs) {
  const { collectionId } = await requireCollectionId(request);
  const formData = await request.formData();
  const intent = formData.get("intent");

  if (intent === "unhide") {
    const clusterId = formData.get("clusterId") as string;
    if (clusterId) {
      const result = await setClusterHidden(collectionId, clusterId, false);
      return result;
    }
    return { success: false, message: "Invalid cluster ID" };
  }

  return { success: false, message: "Unknown action" };
}

const LIMIT = 24;

export async function loader({ request }: Route.LoaderArgs) {
  const { collectionId } = await requireCollectionId(request);
  const url = new URL(request.url);
  const page = parseInt(url.searchParams.get("page") || "1", 10);
  const offset = (page - 1) * LIMIT;

  try {
    const clusters = await getHiddenClusters(collectionId, LIMIT, offset);
    const totalClusters = await getHiddenClustersCount(collectionId);
    const hasMore = offset + clusters.length < totalClusters;

    return dataWithViewMode(
      {
        clusters,
        totalClusters,
        hasMore,
        page,
      },
      "grid",
    );
  } catch (error) {
    console.error("Failed to load hidden clusters:", error);
    return dataWithViewMode(
      {
        clusters: [],
        totalClusters: 0,
        hasMore: false,
        page,
      },
      "grid",
    );
  }
}

type Cluster = Route.ComponentProps["loaderData"]["clusters"][number];

export default function HiddenClustersView({ loaderData }: Route.ComponentProps) {
  const { clusters: initialClusters, totalClusters, hasMore: initialHasMore, page: initialPage } = loaderData;
  const fetcher = useFetcher();

  // Infinite scroll state
  const [clusters, setClusters] = useState<Cluster[]>(initialClusters);
  const [page, setPage] = useState(initialPage);
  const [hasMore, setHasMore] = useState(initialHasMore);
  const scrollFetcher = useFetcher<typeof loader>();
  const loadMoreRef = useRef<HTMLDivElement>(null);

  // Reset state when initial data changes (e.g., navigation or action)
  useEffect(() => {
    setClusters(initialClusters);
    setPage(initialPage);
    setHasMore(initialHasMore);
  }, [initialClusters, initialPage, initialHasMore]);

  // Append new clusters when scroll fetcher returns data
  useEffect(() => {
    if (scrollFetcher.data?.clusters && scrollFetcher.data.clusters.length > 0) {
      setClusters((prev) => {
        const existingIds = new Set(prev.map((c) => c.id));
        const newClusters = scrollFetcher.data!.clusters.filter((c) => !existingIds.has(c.id));
        return [...prev, ...newClusters];
      });
      setPage(scrollFetcher.data.page);
      setHasMore(scrollFetcher.data.hasMore);
    }
  }, [scrollFetcher.data]);

  const loadMore = useCallback(() => {
    if (scrollFetcher.state === "idle" && hasMore) {
      scrollFetcher.load(`/clusters/hidden/grid?page=${page + 1}`);
    }
  }, [scrollFetcher, hasMore, page]);

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
          <SecondaryControls variant="grid">
            <ViewSwitcher
              variant="light"
              modes={[
                { key: "grid", label: "Grid View", icon: <Grid className="h-4 w-4" />, isActive: true },
                {
                  key: "wall",
                  label: "3D Wall",
                  icon: <CoverflowIcon className="size-4" />,
                  to: "/clusters/hidden/wall",
                  isActive: false,
                },
              ]}
            />
            <ControlsCount count={totalClusters} singular="hidden cluster" plural="hidden clusters" variant="grid" />
          </SecondaryControls>
        </div>

        {clusters.length > 0 ? (
          <>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-4">
              {clusters.map((cluster) => (
                <Card key={cluster.id} className="h-full">
                  <CardContent className="p-4">
                    <div className="text-center space-y-3">
                      <Link to={`/cluster/${cluster.id}`}>
                        {cluster.detection_id ? (
                          <div className="w-32 h-32 mx-auto bg-gray-100 rounded-lg border overflow-hidden opacity-60 hover:opacity-100 transition-opacity">
                            <img
                              src={`/api/face/${cluster.detection_id}`}
                              alt={`Cluster ${cluster.id}`}
                              className="w-full h-full object-cover"
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

            {/* Infinite scroll trigger */}
            <div ref={loadMoreRef} className="flex justify-center py-8">
              {isLoadingMore && (
                <div className="flex items-center space-x-2 text-gray-500">
                  <Loader2 className="h-5 w-5 animate-spin" />
                  <span>Loading more clusters...</span>
                </div>
              )}
              {!hasMore && clusters.length > 0 && (
                <span className="text-gray-400 text-sm">All hidden clusters loaded</span>
              )}
            </div>
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
