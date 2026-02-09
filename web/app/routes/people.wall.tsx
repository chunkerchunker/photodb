import { Grid, User } from "lucide-react";
import { useMemo } from "react";
import { useLocation } from "react-router";
import { CoverflowIcon } from "~/components/coverflow-icon";
import { Header } from "~/components/header";
import { PhotoWall, type WallTile } from "~/components/photo-wall";
import { ViewSwitcher } from "~/components/view-switcher";
import { requireCollectionId } from "~/lib/auth.server";
import { dataWithViewMode } from "~/lib/cookies.server";
import { getPeople, getPeopleCount } from "~/lib/db.server";
import type { Route } from "./+types/people.wall";

export function meta() {
  return [
    { title: "PhotoDB - People - 3D Wall" },
    { name: "description", content: "Browse identified people in 3D wall view" },
  ];
}

export async function loader({ request }: Route.LoaderArgs) {
  const { collectionId } = await requireCollectionId(request);
  const url = new URL(request.url);
  const sortParam = url.searchParams.get("sort");
  const sort: "photos" | "name" = sortParam === "photos" ? "photos" : "name";

  try {
    // Load all people for wall view (no pagination)
    const people = await getPeople(collectionId, 500, 0, sort);
    const totalPeople = await getPeopleCount(collectionId);
    return dataWithViewMode({ people, totalPeople, sort }, "wall");
  } catch (error) {
    console.error("Failed to load people:", error);
    return dataWithViewMode({ people: [], totalPeople: 0, sort }, "wall");
  }
}

export default function PeopleWallView({ loaderData }: Route.ComponentProps) {
  const { people, totalPeople, sort } = loaderData;
  const location = useLocation();

  // Convert people to wall tiles
  const tiles: WallTile[] = useMemo(
    () =>
      people.map((person) => {
        const imageUrl = person.photo_id ? `/api/image/${person.photo_id}` : null;
        const displayName = person.person_name || `Person ${person.id}`;

        return {
          id: person.id,
          imageUrls: imageUrl ? [imageUrl] : [],
          // Show only person name, no photo/cluster counts
          label: displayName,
          navigateTo: `/person/${person.id}/wall`,
          metadata: {
            count: person.total_face_count,
            // Pass bbox data for face cropping (rectangular tiles)
            bbox:
              person.bbox_x !== null
                ? {
                    x: person.bbox_x,
                    y: person.bbox_y,
                    width: person.bbox_width,
                    height: person.bbox_height,
                    imageWidth: person.normalized_width,
                    imageHeight: person.normalized_height,
                  }
                : undefined,
            // Set isCircular: true to use circular tiles instead
          },
        };
      }),
    [people],
  );

  if (people.length === 0) {
    return (
      <div className="h-screen w-screen bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <User className="h-16 w-16 text-gray-600 mx-auto mb-4" />
          <p className="text-gray-400 text-lg mb-4">No identified people yet.</p>
        </div>
      </div>
    );
  }

  const headerContent = (
    <Header
      homeTo="/wall"
      breadcrumbs={[{ label: "People" }]}
      viewAction={
        <ViewSwitcher
          modes={[
            {
              key: "grid",
              label: "Grid View",
              icon: <Grid className="h-4 w-4" />,
              to: `/people/grid${sort !== "name" ? `?sort=${sort}` : ""}`,
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

  return <PhotoWall key={location.key} tiles={tiles} sessionKey="people-wall" headerContent={headerContent} />;
}
