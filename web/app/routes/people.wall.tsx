import { EyeOff, User } from "lucide-react";
import { useMemo } from "react";
import { Link, useLocation, useNavigate, useRevalidator } from "react-router";
import { Header } from "~/components/header";
import { PhotoWall, type WallTile } from "~/components/photo-wall";
import {
  ControlsCount,
  ControlsDivider,
  SecondaryControls,
  SortToggle,
  WithoutImagesToggle,
} from "~/components/secondary-controls";
import { WallViewSwitcher } from "~/components/wall-view-switcher";
import { useRootData } from "~/hooks/use-root-data";
import { requireCollectionId } from "~/lib/auth.server";
import { dataWithPreferences, getShowWithoutImagesCookie } from "~/lib/cookies.server";
import { getHiddenPeopleCount, getPeople, getPeopleCount, getPeopleWithoutImagesCount } from "~/lib/db.server";
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
  const showWithoutImages = getShowWithoutImagesCookie(request);

  try {
    // Load all people for wall view (no pagination)
    const people = await getPeople(collectionId, 500, 0, sort, showWithoutImages);
    const totalPeople = await getPeopleCount(collectionId, showWithoutImages);
    const hiddenCount = await getHiddenPeopleCount(collectionId);
    const withoutImagesCount = await getPeopleWithoutImagesCount(collectionId);
    return dataWithPreferences(
      { people, totalPeople, hiddenCount, withoutImagesCount, sort, showWithoutImages },
      "wall",
      showWithoutImages,
    );
  } catch (error) {
    console.error("Failed to load people:", error);
    return dataWithPreferences(
      { people: [], totalPeople: 0, hiddenCount: 0, withoutImagesCount: 0, sort, showWithoutImages },
      "wall",
      showWithoutImages,
    );
  }
}

export default function PeopleWallView({ loaderData }: Route.ComponentProps) {
  const rootData = useRootData();
  const { people, totalPeople, hiddenCount, withoutImagesCount, sort, showWithoutImages } = loaderData;
  const location = useLocation();
  const navigate = useNavigate();
  const revalidator = useRevalidator();

  // Convert people to wall tiles
  const tiles: WallTile[] = useMemo(
    () =>
      people.map((person) => {
        // Use direct face image URL (pre-extracted face crops)
        const imageUrl = person.detection_id ? `/api/face/${person.detection_id}` : null;
        const displayName = person.auto_created ? "✨" : person.person_name || `Person ${person.id}`;

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

  // Show empty state only if there are truly no people at all
  if (people.length === 0 && withoutImagesCount === 0) {
    return (
      <div className="h-screen w-screen bg-gray-900 flex flex-col">
        <div className="p-4">{headerContent}</div>
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <User className="h-16 w-16 text-gray-600 mx-auto mb-4" />
            <p className="text-gray-400 text-lg mb-4">No identified people yet.</p>
          </div>
        </div>
      </div>
    );
  }

  // Empty state when filtering hides all people (but toggle can show more)
  if (people.length === 0) {
    return (
      <div className="h-screen w-screen bg-gray-900 flex flex-col">
        <div className="p-4">{headerContent}</div>
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <User className="h-16 w-16 text-gray-600 mx-auto mb-4" />
            <p className="text-gray-400 text-lg mb-4">No people with images.</p>
            <p className="text-gray-500 text-sm">
              Use the "No image" toggle to show {withoutImagesCount} {withoutImagesCount === 1 ? "person" : "people"}{" "}
              without linked photos.
            </p>
          </div>
        </div>
        <SecondaryControls variant="wall">
          <WithoutImagesToggle
            showWithoutImages={showWithoutImages}
            onToggle={(show) => {
              cookieStore.set({ name: "showWithoutImages", value: String(show), path: "/", sameSite: "lax" });
              revalidator.revalidate();
            }}
            withoutImagesCount={withoutImagesCount}
            variant="wall"
          />
          <ControlsDivider variant="wall" />
          <ControlsCount count={totalPeople} singular="person" plural="people" variant="wall" />
        </SecondaryControls>
      </div>
    );
  }

  return (
    <>
      <PhotoWall key={location.key} tiles={tiles} sessionKey="people-wall" headerContent={headerContent} />
      <SecondaryControls variant="wall">
        <SortToggle sort={sort} onSortChange={(newSort) => navigate(`/people/wall?sort=${newSort}`)} variant="wall" />
        <ControlsDivider variant="wall" />
        <WithoutImagesToggle
          showWithoutImages={showWithoutImages}
          onToggle={(show) => {
            cookieStore.set({ name: "showWithoutImages", value: String(show), path: "/", sameSite: "lax" });
            revalidator.revalidate();
          }}
          withoutImagesCount={withoutImagesCount}
          variant="wall"
        />
        {withoutImagesCount > 0 && <ControlsDivider variant="wall" />}
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
