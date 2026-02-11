import { EyeOff, User } from "lucide-react";
import { useMemo } from "react";
import { Link, useLocation, useNavigate } from "react-router";
import { Header } from "~/components/header";
import { PhotoWall, type WallTile } from "~/components/photo-wall";
import { ControlsCount, ControlsDivider, SecondaryControls, SortToggle } from "~/components/secondary-controls";
import { WallViewSwitcher } from "~/components/wall-view-switcher";
import { useRootData } from "~/hooks/use-root-data";
import { requireCollectionId } from "~/lib/auth.server";
import { dataWithViewMode } from "~/lib/cookies.server";
import { getHiddenPeopleCount, getPeople, getPeopleCount } from "~/lib/db.server";
import type { Route } from "./+types/people.wall";

export function meta() {
  return [
    { title: "Storyteller - People - 3D Wall" },
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
    const hiddenCount = await getHiddenPeopleCount(collectionId);
    return dataWithViewMode({ people, totalPeople, hiddenCount, sort }, "wall");
  } catch (error) {
    console.error("Failed to load people:", error);
    return dataWithViewMode({ people: [], totalPeople: 0, hiddenCount: 0, sort }, "wall");
  }
}

export default function PeopleWallView({ loaderData }: Route.ComponentProps) {
  const rootData = useRootData();
  const { people, totalPeople, hiddenCount, sort } = loaderData;
  const location = useLocation();
  const navigate = useNavigate();

  // Convert people to wall tiles
  const tiles: WallTile[] = useMemo(
    () =>
      people.map((person) => {
        // Use direct face image URL (pre-extracted face crops)
        const imageUrl = person.detection_id ? `/api/face/${person.detection_id}` : null;
        const displayName = person.person_name || `Person ${person.id}`;

        return {
          id: person.id,
          imageUrls: imageUrl ? [imageUrl] : [],
          // Show only person name, no photo/cluster counts
          label: displayName,
          navigateTo: `/person/${person.id}/wall`,
          metadata: {
            count: person.total_face_count,
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
      user={rootData?.userAvatar}
      isAdmin={rootData?.user?.isAdmin}
      isImpersonating={rootData?.impersonation?.isImpersonating}
      viewAction={<WallViewSwitcher />}
    />
  );

  return (
    <>
      <PhotoWall key={location.key} tiles={tiles} sessionKey="people-wall" headerContent={headerContent} />
      <SecondaryControls variant="wall">
        <SortToggle sort={sort} onSortChange={(newSort) => navigate(`/people/wall?sort=${newSort}`)} variant="wall" />
        <ControlsDivider variant="wall" />
        {hiddenCount > 0 && (
          <>
            <Link to="/people/hidden" className="flex items-center gap-1.5 hover:text-white transition-colors">
              <EyeOff className="h-4 w-4" />
              <span>Hidden ({hiddenCount})</span>
            </Link>
            <ControlsDivider variant="wall" />
          </>
        )}
        <ControlsCount count={totalPeople} singular="person" plural="people" variant="wall" />
      </SecondaryControls>
    </>
  );
}
