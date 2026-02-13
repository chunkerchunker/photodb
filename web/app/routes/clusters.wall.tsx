import { EyeOff, Users } from "lucide-react";
import { useMemo } from "react";
import { Link, useLocation } from "react-router";
import { Header } from "~/components/header";
import { PhotoWall, type WallTile } from "~/components/photo-wall";
import { ControlsCount, ControlsDivider, SecondaryControls } from "~/components/secondary-controls";
import { WallViewSwitcher } from "~/components/wall-view-switcher";
import { useRootData } from "~/hooks/use-root-data";
import { requireCollectionId } from "~/lib/auth.server";
import { dataWithViewMode } from "~/lib/cookies.server";
import { getClustersGroupedByPerson, getClustersGroupedCount, getHiddenClustersCount } from "~/lib/db.server";
import type { Route } from "./+types/clusters.wall";

export function meta() {
  return [
    { title: "Storyteller - Face Clusters - 3D Wall" },
    { name: "description", content: "Browse face clusters in 3D wall view" },
  ];
}

export async function loader({ request }: Route.LoaderArgs) {
  const { collectionId } = await requireCollectionId(request);
  try {
    // Load all items for wall view (no pagination)
    const items = await getClustersGroupedByPerson(collectionId, 500, 0);
    const totalItems = await getClustersGroupedCount(collectionId);
    const hiddenCount = await getHiddenClustersCount(collectionId);
    return dataWithViewMode({ items, totalItems, hiddenCount }, "wall");
  } catch (error) {
    console.error("Failed to load clusters:", error);
    return dataWithViewMode({ items: [], totalItems: 0, hiddenCount: 0 }, "wall");
  }
}

export default function ClustersWallView({ loaderData }: Route.ComponentProps) {
  const rootData = useRootData();
  const { items, totalItems, hiddenCount } = loaderData;
  const location = useLocation();

  // Convert items to wall tiles
  const tiles: WallTile[] = useMemo(
    () =>
      items.map((item) => {
        const isPerson = item.item_type === "person";
        // Use direct face image URL (pre-extracted face crops)
        const imageUrl = item.detection_id ? `/api/face/${item.detection_id}` : null;
        const navigateTo = isPerson ? `/person/${item.id}/wall` : `/cluster/${item.id}/wall`;

        return {
          id: `${item.item_type}:${item.id}`,
          imageUrls: imageUrl ? [imageUrl] : [],
          // For people: show name only. For clusters: show photo count only (as subtitle)
          label: isPerson ? item.person_name || "" : "",
          navigateTo,
          metadata: {
            subtitle: isPerson ? undefined : `${item.face_count} photo${item.face_count !== 1 ? "s" : ""}`,
            count: item.face_count,
          },
        };
      }),
    [items],
  );

  const headerContent = (
    <Header
      homeTo="/wall"
      breadcrumbs={[{ label: "Clusters" }]}
      user={rootData?.userAvatar}
      isAdmin={rootData?.user?.isAdmin}
      isImpersonating={rootData?.impersonation?.isImpersonating}
      viewAction={<WallViewSwitcher />}
    />
  );

  if (items.length === 0) {
    return (
      <div className="h-screen w-screen bg-gray-900 flex flex-col">
        <div className="p-4">{headerContent}</div>
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <Users className="h-16 w-16 text-gray-600 mx-auto mb-4" />
            <p className="text-gray-400 text-lg mb-4">No face clusters found.</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <>
      <PhotoWall key={location.key} tiles={tiles} sessionKey="clusters-wall" headerContent={headerContent} />
      <SecondaryControls variant="wall">
        {hiddenCount > 0 && (
          <>
            <Link to="/clusters/hidden" className="flex items-center gap-1.5 hover:text-white transition-colors">
              <EyeOff className="h-4 w-4" />
              <span>Hidden ({hiddenCount})</span>
            </Link>
            <ControlsDivider variant="wall" />
          </>
        )}
        <ControlsCount count={totalItems} singular="item" plural="items" variant="wall" />
      </SecondaryControls>
    </>
  );
}
