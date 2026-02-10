import { Grid, Images, Loader2 } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useFetcher } from "react-router";
import { CoverflowIcon } from "~/components/coverflow-icon";
import { Header } from "~/components/header";
import { ViewSwitcher } from "~/components/view-switcher";
import { requireCollectionId } from "~/lib/auth.server";
import { dataWithViewMode } from "~/lib/cookies.server";
import { getAlbumById, getAlbumPhotos, getAlbumPhotosCount } from "~/lib/db.server";
import type { Route } from "./+types/album.$id.grid";

const LIMIT = 48;

export function meta({ data }: Route.MetaArgs) {
  const albumName = data?.album?.name || "Album";
  return [{ title: `PhotoDB - ${albumName}` }, { name: "description", content: `Browse ${albumName}` }];
}

export async function loader({ request, params }: Route.LoaderArgs) {
  const { collectionId } = await requireCollectionId(request);
  const albumId = params.id;
  const url = new URL(request.url);
  const page = parseInt(url.searchParams.get("page") || "1", 10);
  const offset = (page - 1) * LIMIT;

  try {
    const album = await getAlbumById(collectionId, albumId);
    if (!album) {
      throw new Response("Album not found", { status: 404 });
    }
    const photos = await getAlbumPhotos(collectionId, albumId, LIMIT, offset);
    const totalPhotos = await getAlbumPhotosCount(collectionId, albumId);
    const hasMore = offset + photos.length < totalPhotos;
    return dataWithViewMode({ album, photos, totalPhotos, page, hasMore }, "grid");
  } catch (error) {
    if (error instanceof Response) throw error;
    console.error("Failed to load album:", error);
    throw new Response("Failed to load album", { status: 500 });
  }
}

type Photo = Route.ComponentProps["loaderData"]["photos"][number];

export default function AlbumGridView({ loaderData }: Route.ComponentProps) {
  const { album, photos: initialPhotos, totalPhotos, page: initialPage, hasMore: initialHasMore } = loaderData;

  const [photos, setPhotos] = useState<Photo[]>(initialPhotos);
  const [page, setPage] = useState(initialPage);
  const [hasMore, setHasMore] = useState(initialHasMore);

  const fetcher = useFetcher<typeof loader>();
  const loadMoreRef = useRef<HTMLDivElement>(null);

  // Reset state when initial data changes (e.g., navigation to different album)
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
      fetcher.load(`/album/${album.id}/grid?page=${page + 1}`);
    }
  }, [fetcher, hasMore, page, album.id]);

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
      breadcrumbs={[{ label: "Albums", to: "/albums" }, { label: album.name }]}
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
              to: `/album/${album.id}/wall`,
              isActive: false,
            },
          ]}
        />
      }
    />
  );

  if (photos.length === 0) {
    return (
      <div className="min-h-screen bg-gray-900">
        {headerContent}
        <div className="h-16" />
        <div className="container mx-auto px-4 py-12 text-center">
          <Images className="h-16 w-16 text-gray-600 mx-auto mb-4" />
          <p className="text-gray-400 text-lg">No photos in this album.</p>
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
          {totalPhotos} photo{totalPhotos !== 1 ? "s" : ""}
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 xl:grid-cols-8 gap-2">
          {photos.map((photo) => (
            <Link
              key={photo.id}
              to={`/photo/${photo.id}`}
              className="aspect-square relative group overflow-hidden rounded bg-gray-800"
            >
              <img
                src={`/api/image/${photo.id}`}
                alt={photo.description || photo.orig_path}
                className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                loading="lazy"
              />
            </Link>
          ))}
        </div>

        {/* Infinite scroll trigger */}
        <div ref={loadMoreRef} className="flex justify-center py-8">
          {isLoading && (
            <div className="flex items-center space-x-2 text-gray-400">
              <Loader2 className="h-5 w-5 animate-spin" />
              <span>Loading more photos...</span>
            </div>
          )}
          {!hasMore && photos.length > 0 && <span className="text-gray-500 text-sm">All photos loaded</span>}
        </div>
      </main>
    </div>
  );
}
