import { Camera, Grid, User, Users } from "lucide-react";
import { useMemo } from "react";
import { data, Link, useLocation } from "react-router";
import { CoverflowIcon } from "~/components/coverflow-icon";
import { Header } from "~/components/header";
import { PhotoWall, type WallTile } from "~/components/photo-wall";
import { ViewSwitcher } from "~/components/view-switcher";
import { getYearsWithPhotos } from "~/lib/db.server";
import type { Route } from "./+types/home.wall";

export function meta() {
  return [{ title: "PhotoDB - Years - Photo Wall" }];
}

import { dataWithViewMode } from "~/lib/cookies.server";

export async function loader() {
  try {
    const years = await getYearsWithPhotos();
    return dataWithViewMode({ years }, "wall");
  } catch (error) {
    console.error("Failed to load years:", error);
    return dataWithViewMode({ years: [] }, "wall");
  }
}

export default function HomeWallView({ loaderData }: Route.ComponentProps) {
  const { years } = loaderData;
  const location = useLocation();

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

  if (years.length === 0) {
    return (
      <div className="h-screen w-screen bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <p className="text-gray-400 text-lg mb-4">No photos with date information found.</p>
        </div>
      </div>
    );
  }

  const headerContent = (
    <Header
      homeTo="/wall"
      viewAction={
        <ViewSwitcher
          modes={[
            {
              key: "grid",
              label: "Grid View",
              icon: <Grid className="h-4 w-4" />,
              to: "/grid",
              isActive: false,
            },
            {
              key: "wall",
              label: "Photo Wall",
              icon: <CoverflowIcon className="size-4" />,
              isActive: true,
            },
          ]}
        />
      }
    />
  );

  return <PhotoWall key={location.key} tiles={tiles} sessionKey="home-wall" headerContent={headerContent} />;
}
