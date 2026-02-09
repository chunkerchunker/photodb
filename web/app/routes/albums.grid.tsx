import { Images, Grid } from "lucide-react";
import { Link } from "react-router";
import { CoverflowIcon } from "~/components/coverflow-icon";
import { Header } from "~/components/header";
import { Pagination } from "~/components/pagination";
import { Card, CardContent } from "~/components/ui/card";
import { ViewSwitcher } from "~/components/view-switcher";
import { requireCollectionId } from "~/lib/auth.server";
import { dataWithViewMode } from "~/lib/cookies.server";
import { getAlbums, getAlbumsCount } from "~/lib/db.server";
import type { Route } from "./+types/albums.grid";

const LIMIT = 24;

export function meta() {
  return [
    { title: "PhotoDB - Albums" },
    { name: "description", content: "Browse your photo albums" },
  ];
}

export async function loader({ request }: Route.LoaderArgs) {
  const { collectionId } = await requireCollectionId(request);
  const url = new URL(request.url);
  const page = parseInt(url.searchParams.get("page") || "1", 10);
  const offset = (page - 1) * LIMIT;

  try {
    const albums = await getAlbums(collectionId, LIMIT, offset);
    const totalAlbums = await getAlbumsCount(collectionId);
    const totalPages = Math.ceil(totalAlbums / LIMIT);
    return dataWithViewMode({ albums, totalAlbums, page, totalPages }, "grid");
  } catch (error) {
    console.error("Failed to load albums:", error);
    return dataWithViewMode({ albums: [], totalAlbums: 0, page: 1, totalPages: 0 }, "grid");
  }
}

export default function AlbumsGridView({ loaderData }: Route.ComponentProps) {
  const { albums, totalAlbums, page, totalPages } = loaderData;

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
            <Link key={album.id} to={`/album/${album.id}/grid`} className="group">
              <Card className="bg-gray-800 border-gray-700 hover:border-gray-600 transition-colors overflow-hidden">
                <div className="aspect-square relative bg-gray-700">
                  {album.display_photo_id ? (
                    <img
                      src={`/api/image/${album.display_photo_id}`}
                      alt={album.name}
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center">
                      <Images className="h-12 w-12 text-gray-500" />
                    </div>
                  )}
                </div>
                <CardContent className="p-3">
                  <div className="text-white font-medium truncate">{album.name}</div>
                  <div className="text-gray-400 text-sm">
                    {album.photo_count} photo{album.photo_count !== 1 ? "s" : ""}
                  </div>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>

        {totalPages > 1 && (
          <div className="mt-8">
            <Pagination
              currentPage={page}
              totalPages={totalPages}
              baseUrl="/albums/grid"
            />
          </div>
        )}
      </main>
    </div>
  );
}
