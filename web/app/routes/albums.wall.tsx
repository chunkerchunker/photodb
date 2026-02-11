import { Images } from "lucide-react";
import { useMemo } from "react";
import { useLocation } from "react-router";
import { Header } from "~/components/header";
import { PhotoWall, type WallTile } from "~/components/photo-wall";
import { WallViewSwitcher } from "~/components/wall-view-switcher";
import { useRootData } from "~/hooks/use-root-data";
import { requireCollectionId } from "~/lib/auth.server";
import { dataWithViewMode } from "~/lib/cookies.server";
import { getAlbums, getAlbumsCount } from "~/lib/db.server";
import type { Route } from "./+types/albums.wall";

export function meta() {
  return [{ title: "Storyteller - Albums - Wall" }, { name: "description", content: "Browse albums in wall view" }];
}

export async function loader({ request }: Route.LoaderArgs) {
  const { collectionId } = await requireCollectionId(request);
  try {
    const albums = await getAlbums(collectionId, 500, 0);
    const totalAlbums = await getAlbumsCount(collectionId);
    return dataWithViewMode({ albums, totalAlbums }, "wall");
  } catch (error) {
    console.error("Failed to load albums:", error);
    return dataWithViewMode({ albums: [], totalAlbums: 0 }, "wall");
  }
}

export default function AlbumsWallView({ loaderData }: Route.ComponentProps) {
  const rootData = useRootData();
  const { albums, totalAlbums } = loaderData;
  const location = useLocation();

  const tiles: WallTile[] = useMemo(
    () =>
      albums.map((album) => {
        const imageUrl = album.display_photo_id ? `/api/image/${album.display_photo_id}` : null;

        return {
          id: `album:${album.id}`,
          imageUrls: imageUrl ? [imageUrl] : [],
          label: album.name,
          navigateTo: `/album/${album.id}/wall`,
          metadata: {
            subtitle: `${album.photo_count} photo${album.photo_count !== 1 ? "s" : ""}`,
            count: album.photo_count,
          },
        };
      }),
    [albums],
  );

  if (albums.length === 0) {
    return (
      <div className="h-screen w-screen bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <Images className="h-16 w-16 text-gray-600 mx-auto mb-4" />
          <p className="text-gray-400 text-lg mb-4">No albums found.</p>
        </div>
      </div>
    );
  }

  const headerContent = (
    <Header
      homeTo="/wall"
      breadcrumbs={[{ label: "Albums" }]}
      user={rootData?.userAvatar}
      isAdmin={rootData?.user?.isAdmin}
      isImpersonating={rootData?.impersonation?.isImpersonating}
      viewAction={<WallViewSwitcher />}
    />
  );

  return <PhotoWall key={location.key} tiles={tiles} sessionKey="albums-wall" headerContent={headerContent} />;
}
