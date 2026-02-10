import { Grid, Users } from "lucide-react";
import { useMemo } from "react";
import { useLocation } from "react-router";
import { CoverflowIcon } from "~/components/coverflow-icon";
import { Header } from "~/components/header";
import { PhotoWall, type WallTile } from "~/components/photo-wall";
import { ViewSwitcher } from "~/components/view-switcher";
import { requireCollectionId } from "~/lib/auth.server";
import { dataWithViewMode } from "~/lib/cookies.server";
import { getClustersGroupedByPerson, getClustersGroupedCount } from "~/lib/db.server";
import type { Route } from "./+types/clusters.wall";

export function meta() {
  return [
    { title: "PhotoDB - Face Clusters - 3D Wall" },
    { name: "description", content: "Browse face clusters in 3D wall view" },
  ];
}

export async function loader({ request }: Route.LoaderArgs) {
  const { collectionId } = await requireCollectionId(request);
  try {
    // Load all items for wall view (no pagination)
    const items = await getClustersGroupedByPerson(collectionId, 500, 0);
    const totalItems = await getClustersGroupedCount(collectionId);
    return dataWithViewMode({ items, totalItems }, "wall");
  } catch (error) {
    console.error("Failed to load clusters:", error);
    return dataWithViewMode({ items: [], totalItems: 0 }, "wall");
  }
}

export default function ClustersWallView({ loaderData }: Route.ComponentProps) {
  const { items, totalItems } = loaderData;
  const location = useLocation();

  // Convert items to wall tiles
  const tiles: WallTile[] = useMemo(
    () =>
      items.map((item) => {
        const isPerson = item.item_type === "person";
        const imageUrl = item.photo_id ? `/api/image/${item.photo_id}` : null;
        const displayName = item.person_name || `Cluster #${item.id}`;
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
            // Pass bbox data for face cropping (rectangular tiles)
            bbox:
              item.bbox_x !== null
                ? {
                    x: item.bbox_x,
                    y: item.bbox_y,
                    width: item.bbox_width,
                    height: item.bbox_height,
                    imageWidth: item.med_width,
                    imageHeight: item.med_height,
                  }
                : undefined,
            // Set isCircular: true to use circular tiles instead
          },
        };
      }),
    [items],
  );

  if (items.length === 0) {
    return (
      <div className="h-screen w-screen bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <Users className="h-16 w-16 text-gray-600 mx-auto mb-4" />
          <p className="text-gray-400 text-lg mb-4">No face clusters found.</p>
        </div>
      </div>
    );
  }

  const headerContent = (
    <Header
      homeTo="/wall"
      breadcrumbs={[{ label: "Clusters" }]}
      viewAction={
        <ViewSwitcher
          modes={[
            {
              key: "grid",
              label: "Grid View",
              icon: <Grid className="h-4 w-4" />,
              to: "/clusters/grid",
              isActive: false,
            },
            {
              key: "wall",
              label: "3D Wall",
              icon: <CoverflowIcon className="size-4" />,
              isActive: true,
            },
          ]}
        />
      }
    />
  );

  return <PhotoWall key={location.key} tiles={tiles} sessionKey="clusters-wall" headerContent={headerContent} />;
}
