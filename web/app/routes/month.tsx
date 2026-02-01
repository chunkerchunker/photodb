import { Loader2 } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useFetcher } from "react-router";
import { Breadcrumb } from "~/components/breadcrumb";
import { Layout } from "~/components/layout";
import { Card, CardContent } from "~/components/ui/card";
import { getPhotoCountByMonth, getPhotosByMonth } from "~/lib/db.server";
import type { Route } from "./+types/month";

export function meta({ params }: Route.MetaArgs) {
  const monthNames = [
    "",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
  ];
  const monthName = monthNames[parseInt(params.month, 10)] || params.month;

  return [
    { title: `PhotoDB - ${monthName} ${params.year}` },
    {
      name: "description",
      content: `Browse photos from ${monthName} ${params.year}`,
    },
  ];
}

const LIMIT = 48; // 6x8 grid

export async function loader({ params, request }: Route.LoaderArgs) {
  const year = parseInt(params.year, 10);
  const month = parseInt(params.month, 10);
  const url = new URL(request.url);
  const page = parseInt(url.searchParams.get("page") || "1", 10);
  const offset = (page - 1) * LIMIT;

  const monthNames = [
    "",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
  ];
  const monthName = monthNames[month] || `Month ${month}`;

  try {
    const photos = await getPhotosByMonth(year, month, LIMIT, offset);
    const totalPhotos = await getPhotoCountByMonth(year, month);
    const hasMore = offset + photos.length < totalPhotos;

    return {
      photos,
      totalPhotos,
      hasMore,
      page,
      year: params.year,
      month: params.month,
      monthName,
    };
  } catch (error) {
    console.error(`Failed to load photos for ${year}-${month}:`, error);
    return {
      photos: [],
      totalPhotos: 0,
      hasMore: false,
      page,
      year: params.year,
      month: params.month,
      monthName,
    };
  }
}

type Photo = Route.ComponentProps["loaderData"]["photos"][number];

export default function MonthView({ loaderData }: Route.ComponentProps) {
  const { photos: initialPhotos, totalPhotos, hasMore: initialHasMore, page: initialPage, year, month, monthName } = loaderData;

  const [photos, setPhotos] = useState<Photo[]>(initialPhotos);
  const [page, setPage] = useState(initialPage);
  const [hasMore, setHasMore] = useState(initialHasMore);

  const fetcher = useFetcher<typeof loader>();
  const loadMoreRef = useRef<HTMLDivElement>(null);

  // Reset state when initial data changes (e.g., navigation)
  useEffect(() => {
    setPhotos(initialPhotos);
    setPage(initialPage);
    setHasMore(initialHasMore);
  }, [initialPhotos, initialPage, initialHasMore]);

  // Append new photos when fetcher returns data
  useEffect(() => {
    if (fetcher.data?.photos && fetcher.data.photos.length > 0) {
      setPhotos((prev) => {
        const existingIds = new Set(prev.map((p) => p.id));
        const newPhotos = fetcher.data!.photos.filter((p) => !existingIds.has(p.id));
        return [...prev, ...newPhotos];
      });
      setPage(fetcher.data.page);
      setHasMore(fetcher.data.hasMore);
    }
  }, [fetcher.data]);

  const loadMore = useCallback(() => {
    if (fetcher.state === "idle" && hasMore) {
      fetcher.load(`/year/${year}/month/${month}?page=${page + 1}`);
    }
  }, [fetcher, hasMore, page, year, month]);

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
        <Breadcrumb items={[{ label: year, href: `/year/${year}` }, { label: monthName }]} />

        <div className="flex items-center justify-between">
          <h1 className="text-3xl font-bold text-gray-900">
            {monthName} {year}
          </h1>
          <span className="text-gray-600">
            {totalPhotos} photo{totalPhotos !== 1 ? "s" : ""}
          </span>
        </div>

        {photos.length > 0 ? (
          <>
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 xl:grid-cols-8 gap-4">
              {photos.map((photo) => (
                <Link key={photo.id} to={`/photo/${photo.id}`} className="block transition-transform hover:scale-105">
                  <Card className="overflow-hidden hover:shadow-lg transition-shadow">
                    <div className="relative">
                      <img
                        src={`/api/image/${photo.id}`}
                        alt={photo.filename_only}
                        className="w-full h-48 object-cover"
                        loading="lazy"
                      />
                    </div>
                    <CardContent className="p-3">
                      <div className="text-sm font-medium text-gray-900 truncate" title={photo.filename_only}>
                        {photo.filename_only}
                      </div>
                      <div className="text-xs text-gray-500 mt-1">ID: {photo.id}</div>
                      {photo.short_description && (
                        <div className="text-xs text-gray-600 mt-1 line-clamp-2" title={photo.description}>
                          {photo.short_description}
                        </div>
                      )}
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
                  <span>Loading more photos...</span>
                </div>
              )}
              {!hasMore && photos.length > 0 && (
                <span className="text-gray-400 text-sm">All photos loaded</span>
              )}
            </div>
          </>
        ) : (
          <div className="text-center py-12">
            <div className="text-gray-500 text-lg">
              No photos found for {monthName} {year}.
            </div>
          </div>
        )}
      </div>
    </Layout>
  );
}
