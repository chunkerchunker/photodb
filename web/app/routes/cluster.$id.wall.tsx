import { Grid, Users } from "lucide-react";
import { useMemo } from "react";
import { useLocation } from "react-router";
import { CoverflowIcon } from "~/components/coverflow-icon";
import { Header } from "~/components/header";
import { PhotoWall, type WallTile } from "~/components/photo-wall";
import { ViewSwitcher } from "~/components/view-switcher";
import { dataWithViewMode } from "~/lib/cookies.server";
import { getClusterDetails, getClusterFaces, getClusterFacesCount } from "~/lib/db.server";
import type { Route } from "./+types/cluster.$id.wall";

export function meta({ params }: Route.MetaArgs) {
  return [
    { title: `PhotoDB - Cluster ${params.id} - 3D Wall` },
    { name: "description", content: `View faces in cluster ${params.id} in 3D wall view` },
  ];
}

export async function loader({ params }: Route.LoaderArgs) {
  const clusterId = params.id;

  try {
    const cluster = await getClusterDetails(clusterId);
    if (!cluster) {
      throw new Response("Cluster not found", { status: 404 });
    }

    // Load all faces for wall view (no pagination)
    const faces = await getClusterFaces(clusterId, 500, 0);
    const totalFaces = await getClusterFacesCount(clusterId);

    return dataWithViewMode({ cluster, faces, totalFaces }, "wall");
  } catch (error) {
    console.error(`Failed to load cluster ${clusterId}:`, error);
    if (error instanceof Response && error.status === 404) {
      throw error;
    }
    return dataWithViewMode({ cluster: null, faces: [], totalFaces: 0 }, "wall");
  }
}

export default function ClusterWallView({ loaderData }: Route.ComponentProps) {
  const { cluster, faces, totalFaces } = loaderData;
  const location = useLocation();

  // Convert faces to wall tiles
  const tiles: WallTile[] = useMemo(
    () =>
      faces.map((face) => {
        const imageUrl = face.photo_id ? `/api/image/${face.photo_id}` : null;

        return {
          id: face.id,
          imageUrls: imageUrl ? [imageUrl] : [],
          label: `${Math.round(face.cluster_confidence * 100)}%`,
          navigateTo: `/photo/${face.photo_id}`,
          metadata: {
            subtitle: `Face #${face.id}`,
            count: 1,
            // Pass bbox data for face cropping (rectangular tiles)
            bbox:
              face.bbox_x !== null
                ? {
                    x: face.bbox_x,
                    y: face.bbox_y,
                    width: face.bbox_width,
                    height: face.bbox_height,
                    imageWidth: face.normalized_width,
                    imageHeight: face.normalized_height,
                  }
                : undefined,
            // Set isCircular: true to use circular tiles instead
          },
        };
      }),
    [faces],
  );

  if (!cluster) {
    return (
      <div className="h-screen w-screen bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <Users className="h-16 w-16 text-gray-600 mx-auto mb-4" />
          <p className="text-gray-400 text-lg mb-4">Cluster not found.</p>
        </div>
      </div>
    );
  }

  if (faces.length === 0) {
    return (
      <div className="h-screen w-screen bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <Users className="h-16 w-16 text-gray-600 mx-auto mb-4" />
          <p className="text-gray-400 text-lg mb-4">No faces in this cluster.</p>
        </div>
      </div>
    );
  }

  const displayName = cluster.person_name || `Cluster #${cluster.id}`;

  const headerContent = (
    <Header
      homeTo="/wall"
      breadcrumbs={[{ label: "Clusters", to: "/clusters" }, { label: displayName }]}
      viewAction={
        <ViewSwitcher
          modes={[
            {
              key: "grid",
              label: "Grid View",
              icon: <Grid className="h-4 w-4" />,
              to: `/cluster/${cluster.id}/grid`,
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

  return (
    <PhotoWall
      key={location.key}
      tiles={tiles}
      sessionKey={`cluster-${cluster.id}-wall`}
      headerContent={headerContent}
    />
  );
}
