import { Images, Grid } from "lucide-react";
import { Link } from "react-router";
import { CoverflowIcon } from "~/components/coverflow-icon";
import { Header } from "~/components/header";
import { Pagination } from "~/components/pagination";
import { ViewSwitcher } from "~/components/view-switcher";
import { requireCollectionId } from "~/lib/auth.server";
import { dataWithViewMode } from "~/lib/cookies.server";
import { getAlbumById, getAlbumPhotos, getAlbumPhotosCount } from "~/lib/db.server";
import type { Route } from "./+types/album.$id.grid";

const LIMIT = 48;

export function meta({ data }: Route.MetaArgs) {
  const albumName = data?.album?.name || "Album";
  return [
    { title: `PhotoDB - ${albumName}` },
    { name: "description", content: `Browse ${albumName}` },
  ];
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
    const totalPages = Math.ceil(totalPhotos / LIMIT);
    return dataWithViewMode({ album, photos, totalPhotos, page, totalPages }, "grid");
  } catch (error) {
    if (error instanceof Response) throw error;
    console.error("Failed to load album:", error);
    throw new Response("Failed to load album", { status: 500 });
  }
}

export default function AlbumGridView({ loaderData }: Route.ComponentProps) {
  const { album, photos, totalPhotos, page, totalPages } = loaderData;

  const headerContent = (
    <Header
      breadcrumbs={[
        { label: "Albums", to: "/albums" },
        { label: album.name },
      ]}
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
                alt={photo.description || photo.filename}
                className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                loading="lazy"
              />
            </Link>
          ))}
        </div>

        {totalPages > 1 && (
          <div className="mt-8">
            <Pagination
              currentPage={page}
              totalPages={totalPages}
              baseUrl={`/album/${album.id}/grid`}
            />
          </div>
        )}
      </main>
    </div>
  );
}
