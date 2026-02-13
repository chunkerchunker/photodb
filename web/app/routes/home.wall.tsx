import { useMemo } from "react";
import { useLocation } from "react-router";
import { Header } from "~/components/header";
import { PhotoWall, type WallTile } from "~/components/photo-wall";
import { WallViewSwitcher } from "~/components/wall-view-switcher";
import { useRootData } from "~/hooks/use-root-data";
import { requireCollectionId } from "~/lib/auth.server";
import { getYearsWithPhotos } from "~/lib/db.server";
import type { Route } from "./+types/home.wall";

export function meta() {
  return [{ title: "Storyteller - Years - Photo Wall" }];
}

import { dataWithViewMode } from "~/lib/cookies.server";

export async function loader({ request }: Route.LoaderArgs) {
  const { collectionId } = await requireCollectionId(request);

  try {
    const years = await getYearsWithPhotos(collectionId);
    return dataWithViewMode({ years }, "wall");
  } catch (error) {
    console.error("Failed to load years:", error);
    return dataWithViewMode({ years: [] }, "wall");
  }
}

export default function HomeWallView({ loaderData }: Route.ComponentProps) {
  const { years } = loaderData;
  const location = useLocation();
  const rootData = useRootData();

  // Convert years to wall tiles (memoized to prevent effect re-runs)
  const tiles: WallTile[] = useMemo(
    () =>
      years.map((yearData) => ({
        id: yearData.year,
        imageUrls: (yearData.sample_photo_ids || []).map((id: number) => `/api/image/${id}`),
        label: String(yearData.year),
        navigateTo: `/year/${yearData.year}/wall`,
        metadata: {
          subtitle: `${yearData.photo_count} photo${yearData.photo_count !== 1 ? "s" : ""}`,
          count: yearData.photo_count,
        },
      })),
    [years],
  );

  const headerContent = (
    <Header
      user={rootData?.userAvatar}
      isAdmin={rootData?.user?.isAdmin}
      isImpersonating={rootData?.impersonation?.isImpersonating}
      homeTo="/wall"
      viewAction={<WallViewSwitcher />}
    />
  );

  if (years.length === 0) {
    return (
      <div className="h-screen w-screen bg-gray-900 flex flex-col">
        <div className="p-4">{headerContent}</div>
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <p className="text-gray-400 text-lg mb-4">No photos with date information found.</p>
          </div>
        </div>
      </div>
    );
  }

  return <PhotoWall key={location.key} tiles={tiles} sessionKey="home-wall" headerContent={headerContent} />;
}
