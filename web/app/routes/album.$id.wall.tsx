import { Grid, Images } from "lucide-react";
import { useMemo } from "react";
import { useLocation } from "react-router";
import { CoverflowIcon } from "~/components/coverflow-icon";
import { Header } from "~/components/header";
import { PhotoWall, type WallTile } from "~/components/photo-wall";
import { ViewSwitcher } from "~/components/view-switcher";
import { useRootData } from "~/hooks/use-root-data";
import { requireCollectionId } from "~/lib/auth.server";
import { dataWithViewMode } from "~/lib/cookies.server";
import { getAlbumById, getAlbumPhotos, getAlbumPhotosCount } from "~/lib/db.server";
import type { Route } from "./+types/album.$id.wall";

export function meta({ data }: Route.MetaArgs) {
  const albumName = data?.album?.name || "Album";
  return [
    { title: `Storyteller - ${albumName} - Wall` },
    { name: "description", content: `Browse ${albumName} in wall view` },
  ];
}

export async function loader({ request, params }: Route.LoaderArgs) {
  const { collectionId } = await requireCollectionId(request);
  const albumId = params.id;

  try {
    const album = await getAlbumById(collectionId, albumId);
    if (!album) {
      throw new Response("Album not found", { status: 404 });
    }
    const photos = await getAlbumPhotos(collectionId, albumId, 500, 0);
    const totalPhotos = await getAlbumPhotosCount(collectionId, albumId);
    return dataWithViewMode({ album, photos, totalPhotos }, "wall");
  } catch (error) {
    if (error instanceof Response) throw error;
    console.error("Failed to load album:", error);
    throw new Response("Failed to load album", { status: 500 });
  }
}

export default function AlbumWallView({ loaderData }: Route.ComponentProps) {
  const rootData = useRootData();
  const { album, photos, totalPhotos } = loaderData;
  const location = useLocation();

  const tiles: WallTile[] = useMemo(
    () =>
      photos.map((photo) => ({
        id: `photo:${photo.id}`,
        imageUrls: [`/api/image/${photo.id}`],
        label: "",
        navigateTo: `/photo/${photo.id}`,
        metadata: {
          subtitle: photo.description || undefined,
        },
      })),
    [photos],
  );

  if (photos.length === 0) {
    return (
      <div className="h-screen w-screen bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <Images className="h-16 w-16 text-gray-600 mx-auto mb-4" />
          <p className="text-gray-400 text-lg mb-4">No photos in this album.</p>
        </div>
      </div>
    );
  }

  const headerContent = (
    <Header
      homeTo="/albums/wall"
      breadcrumbs={[{ label: "Albums", to: "/albums" }, { label: album.name }]}
      user={rootData?.userAvatar}
      isAdmin={rootData?.user?.isAdmin}
      isImpersonating={rootData?.impersonation?.isImpersonating}
      viewAction={
        <ViewSwitcher
          modes={[
            {
              key: "grid",
              label: "Grid View",
              icon: <Grid className="h-4 w-4" />,
              to: `/album/${album.id}/grid`,
              isActive: false,
            },
            {
              key: "wall",
              label: "Wall",
              icon: <CoverflowIcon className="size-4" />,
              isActive: true,
            },
          ]}
        />
      }
    />
  );

  return (
    <PhotoWall key={location.key} tiles={tiles} sessionKey={`album-${album.id}-wall`} headerContent={headerContent} />
  );
}
