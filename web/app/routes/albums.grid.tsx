import { Grid, Images, Loader2 } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useFetcher } from "react-router";
import { CoverflowIcon } from "~/components/coverflow-icon";
import { Header } from "~/components/header";
import { Card, CardContent } from "~/components/ui/card";
import { ViewSwitcher } from "~/components/view-switcher";
import { requireCollectionId } from "~/lib/auth.server";
import { dataWithViewMode } from "~/lib/cookies.server";
import { getAlbums, getAlbumsCount } from "~/lib/db.server";
import type { Route } from "./+types/albums.grid";

const LIMIT = 24;

export function meta() {
  return [{ title: "Storyteller - Albums" }, { name: "description", content: "Browse your photo albums" }];
}

export async function loader({ request }: Route.LoaderArgs) {
  const { collectionId } = await requireCollectionId(request);
  const url = new URL(request.url);
  const page = parseInt(url.searchParams.get("page") || "1", 10);
  const offset = (page - 1) * LIMIT;

  try {
    const albums = await getAlbums(collectionId, LIMIT, offset);
    const totalAlbums = await getAlbumsCount(collectionId);
    const hasMore = offset + albums.length < totalAlbums;
    return dataWithViewMode({ albums, totalAlbums, page, hasMore }, "grid");
  } catch (error) {
    console.error("Failed to load albums:", error);
    return dataWithViewMode({ albums: [], totalAlbums: 0, page: 1, hasMore: false }, "grid");
  }
}

type Album = Route.ComponentProps["loaderData"]["albums"][number];

export default function AlbumsGridView({ loaderData }: Route.ComponentProps) {
  const { albums: initialAlbums, totalAlbums, page: initialPage, hasMore: initialHasMore } = loaderData;

  const [albums, setAlbums] = useState<Album[]>(initialAlbums);
  const [page, setPage] = useState(initialPage);
  const [hasMore, setHasMore] = useState(initialHasMore);

  const fetcher = useFetcher<typeof loader>();
  const loadMoreRef = useRef<HTMLDivElement>(null);

  // Reset state when initial data changes (e.g., navigation)
  useEffect(() => {
    setAlbums(initialAlbums);
    setPage(initialPage);
    setHasMore(initialHasMore);
  }, [initialAlbums, initialPage, initialHasMore]);

  // Append new albums when fetcher returns data
  useEffect(() => {
    if (fetcher.data?.albums && fetcher.data.albums.length > 0) {
      setAlbums((prev) => {
        const existingIds = new Set(prev.map((a) => a.id));
        const newAlbums = fetcher.data!.albums.filter((a) => !existingIds.has(a.id));
        return [...prev, ...newAlbums];
      });
      setPage(fetcher.data.page);
      setHasMore(fetcher.data.hasMore);
    }
  }, [fetcher.data]);

  const loadMore = useCallback(() => {
    if (fetcher.state === "idle" && hasMore) {
      fetcher.load(`/albums/grid?page=${page + 1}`);
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

  const headerContent = (
    <Header
      breadcrumbs={[{ label: "Albums" }]}
      viewAction={
        <ViewSwitcher
          modes={[
            {
              key: "grid",
              label: "Grid View",
              icon: <Grid className="h-4 w-4" />,
              isActive: true,
            },
            {
              key: "wall",
              label: "Wall",
              icon: <CoverflowIcon className="size-4" />,
              to: "/albums/wall",
              isActive: false,
            },
          ]}
        />
      }
    />
  );

  if (albums.length === 0) {
    return (
      <div className="min-h-screen bg-gray-900">
        {headerContent}
        <div className="h-16" />
        <div className="container mx-auto px-4 py-12 text-center">
          <Images className="h-16 w-16 text-gray-600 mx-auto mb-4" />
          <p className="text-gray-400 text-lg">No albums found.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900">
      {headerContent}
      <div className="h-16" />
      <main className="container mx-auto px-4 py-6">
        <div className="text-gray-400 text-sm mb-4">
          {totalAlbums} album{totalAlbums !== 1 ? "s" : ""}
        </div>

        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-4">
          {albums.map((album) => (
            <Link key={album.id} to={`/album/${album.id}/grid`} className="group h-full">
              <Card className="h-full bg-gray-800 border-gray-700 hover:border-gray-600 transition-colors overflow-hidden flex flex-col">
                <div className="aspect-square relative bg-gray-700 overflow-hidden flex-shrink-0">
                  {album.display_photo_id ? (
                    <img
                      src={`/api/image/${album.display_photo_id}`}
                      alt={album.name}
                      className="absolute inset-0 w-full h-full object-cover group-hover:scale-105 transition-transform"
                    />
                  ) : (
                    <div className="absolute inset-0 flex items-center justify-center">
                      <Images className="h-12 w-12 text-gray-500" />
                    </div>
                  )}
                </div>
                <CardContent className="px-3 py-0 flex-shrink-0">
                  <div className="text-white font-medium truncate text-sm">{album.name}</div>
                  <div className="text-gray-400 text-xs">
                    {album.photo_count} photo{album.photo_count !== 1 ? "s" : ""}
                  </div>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>

        {/* Infinite scroll trigger */}
        <div ref={loadMoreRef} className="flex justify-center py-8">
          {isLoading && (
            <div className="flex items-center space-x-2 text-gray-400">
              <Loader2 className="h-5 w-5 animate-spin" />
              <span>Loading more albums...</span>
            </div>
          )}
          {!hasMore && albums.length > 0 && <span className="text-gray-500 text-sm">All albums loaded</span>}
        </div>
      </main>
    </div>
  );
}
