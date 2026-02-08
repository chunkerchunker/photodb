import { Grid, User, Users } from "lucide-react";
import { useMemo } from "react";
import { useLocation } from "react-router";
import { Header } from "~/components/header";
import { PhotoWall, type WallTile } from "~/components/photo-wall";
import { ViewSwitcher } from "~/components/view-switcher";
import { CoverflowIcon } from "~/components/coverflow-icon";
import { dataWithViewMode } from "~/lib/cookies.server";
import { getClustersByPerson, getPersonById } from "~/lib/db.server";
import type { Route } from "./+types/person.$id.wall";

export function meta({ data }: Route.MetaArgs) {
  const personName = data?.person?.person_name || "Person";
  return [
    { title: `PhotoDB - ${personName} - 3D Wall` },
    { name: "description", content: `View ${personName}'s clusters in 3D wall view` },
  ];
}

export async function loader({ params }: Route.LoaderArgs) {
  const personId = params.id;
  if (!personId) {
    throw new Response("Person ID required", { status: 400 });
  }

  const person = await getPersonById(personId);
  if (!person) {
    throw new Response("Person not found", { status: 404 });
  }

  const clusters = await getClustersByPerson(personId);

  return dataWithViewMode({ person, clusters }, "wall");
}

export default function PersonWallView({ loaderData }: Route.ComponentProps) {
  const { person, clusters } = loaderData;
  const location = useLocation();

  // Convert clusters to wall tiles
  const tiles: WallTile[] = useMemo(
    () =>
      clusters
        .filter((cluster) => !cluster.hidden)
        .map((cluster) => {
          const imageUrl = cluster.photo_id ? `/api/image/${cluster.photo_id}` : null;

          return {
            id: cluster.id,
            imageUrls: imageUrl ? [imageUrl] : [],
            label: `Cluster #${cluster.id}`,
            navigateTo: `/cluster/${cluster.id}/wall`,
            metadata: {
              subtitle: `${cluster.face_count} photo${cluster.face_count !== 1 ? "s" : ""}`,
              count: cluster.face_count,
              // Pass bbox data for face cropping (rectangular tiles)
              bbox:
                cluster.bbox_x !== null
                  ? {
                      x: cluster.bbox_x,
                      y: cluster.bbox_y,
                      width: cluster.bbox_width,
                      height: cluster.bbox_height,
                      imageWidth: cluster.normalized_width,
                      imageHeight: cluster.normalized_height,
                    }
                  : undefined,
              // Set isCircular: true to use circular tiles instead
            },
          };
        }),
    [clusters],
  );

  if (clusters.length === 0) {
    return (
      <div className="h-screen w-screen bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <Users className="h-16 w-16 text-gray-600 mx-auto mb-4" />
          <p className="text-gray-400 text-lg mb-4">No clusters found for this person.</p>
        </div>
      </div>
    );
  }

  const displayName = person.person_name || `Person ${person.id}`;

  const headerContent = (
    <Header
      homeTo="/wall"
      breadcrumbs={[{ label: "People", to: "/people" }, { label: displayName }]}
      viewAction={
        <ViewSwitcher
          modes={[
            {
              key: "grid",
              label: "Grid View",
              icon: <Grid className="h-4 w-4" />,
              to: `/person/${person.id}/grid`,
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
    <PhotoWall key={location.key} tiles={tiles} sessionKey={`person-${person.id}-wall`} headerContent={headerContent} />
  );
}
