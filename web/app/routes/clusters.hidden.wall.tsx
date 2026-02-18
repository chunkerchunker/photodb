import { Users } from "lucide-react";
import { useMemo } from "react";
import { useLocation } from "react-router";
import { Header } from "~/components/header";
import { PhotoWall, type WallTile } from "~/components/photo-wall";
import { ControlsCount, SecondaryControls } from "~/components/secondary-controls";
import { WallViewSwitcher } from "~/components/wall-view-switcher";
import { useRootData } from "~/hooks/use-root-data";
import { requireCollectionId } from "~/lib/auth.server";
import { dataWithViewMode } from "~/lib/cookies.server";
import { getHiddenClusters, getHiddenClustersCount } from "~/lib/db.server";
import type { Route } from "./+types/clusters.hidden.wall";

export function meta() {
  return [
    { title: "Storyteller - Hidden Clusters - 3D Wall" },
    { name: "description", content: "View hidden clusters in 3D wall view" },
  ];
}

export async function loader({ request }: Route.LoaderArgs) {
  const { collectionId } = await requireCollectionId(request);

  try {
    // Load all hidden clusters for wall view (no pagination)
    const clusters = await getHiddenClusters(collectionId, 500, 0);
    const totalClusters = await getHiddenClustersCount(collectionId);
    return dataWithViewMode({ clusters, totalClusters }, "wall");
  } catch (error) {
    console.error("Failed to load hidden clusters:", error);
    return dataWithViewMode({ clusters: [], totalClusters: 0 }, "wall");
  }
}

export default function HiddenClustersWallView({ loaderData }: Route.ComponentProps) {
  const rootData = useRootData();
  const { clusters, totalClusters } = loaderData;
  const location = useLocation();

  // Convert clusters to wall tiles
  const tiles: WallTile[] = useMemo(
    () =>
      clusters.map((cluster) => {
        const imageUrl = cluster.detection_id ? `/api/face/${cluster.detection_id}` : null;

        return {
          id: cluster.id,
          imageUrls: imageUrl ? [imageUrl] : [],
          label: cluster.auto_created ? "✨" : cluster.person_name || "",
          navigateTo: `/cluster/${cluster.id}/wall`,
          metadata: {
            subtitle: `${cluster.face_count} photo${cluster.face_count !== 1 ? "s" : ""}`,
            count: cluster.face_count,
          },
        };
      }),
    [clusters],
  );

  const headerContent = (
    <Header
      homeTo="/wall"
      breadcrumbs={[{ label: "Clusters", to: "/clusters" }, { label: "Hidden" }]}
      user={rootData?.userAvatar}
      isAdmin={rootData?.user?.isAdmin}
      isImpersonating={rootData?.impersonation?.isImpersonating}
      viewAction={<WallViewSwitcher />}
    />
  );

  if (clusters.length === 0) {
    return (
      <div className="h-screen w-screen bg-gray-900 flex flex-col">
        <div className="p-4">{headerContent}</div>
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <Users className="h-16 w-16 text-gray-600 mx-auto mb-4" />
            <p className="text-gray-400 text-lg mb-4">No hidden clusters.</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <>
      <PhotoWall key={location.key} tiles={tiles} sessionKey="clusters-hidden-wall" headerContent={headerContent} />
      <SecondaryControls variant="wall">
        <ControlsCount count={totalClusters} singular="hidden cluster" plural="hidden clusters" variant="wall" />
      </SecondaryControls>
    </>
  );
}
