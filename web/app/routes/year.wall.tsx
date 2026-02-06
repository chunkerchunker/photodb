import { ArrowLeft, Camera, Grid, User, Users } from "lucide-react";
import { useMemo } from "react";
import { data, Link, useLocation } from "react-router";
import { CoverflowIcon } from "~/components/coverflow-icon";
import { Header } from "~/components/header";
import { PhotoWall, type WallTile } from "~/components/photo-wall";
import { ViewSwitcher } from "~/components/view-switcher";
import { getMonthsInYear } from "~/lib/db.server";
import type { Route } from "./+types/year.wall";

export function meta({ params }: Route.MetaArgs) {
  return [{ title: `PhotoDB - ${params.year} - Photo Wall` }];
}

import { dataWithViewMode } from "~/lib/cookies.server";

export async function loader({ params }: Route.LoaderArgs) {
  const year = parseInt(params.year, 10);

  try {
    const months = await getMonthsInYear(year);
    return dataWithViewMode({ months, year: params.year }, "wall");
  } catch (error) {
    console.error(`Failed to load months for year ${year}:`, error);
    return dataWithViewMode({ months: [], year: params.year }, "wall");
  }
}

export default function YearWallView({ loaderData }: Route.ComponentProps) {
  const { months, year } = loaderData;
  const location = useLocation();

  // Convert months to wall tiles (memoized to prevent effect re-runs)
  const tiles: WallTile[] = useMemo(
    () =>
      months.map((monthData) => ({
        id: `${year}-${monthData.month}`,
        imageUrls: (monthData.sample_photo_ids || []).map((id: number) => `/api/image/${id}`),
        label: monthData.month_name,
        navigateTo: `/year/${year}/month/${monthData.month}/wall`,
        metadata: {
          subtitle: `${monthData.photo_count} photo${monthData.photo_count !== 1 ? "s" : ""}`,
          count: monthData.photo_count,
        },
      })),
    [months, year],
  );

  if (months.length === 0) {
    return (
      <div className="h-screen w-screen bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <p className="text-gray-400 text-lg mb-4">No photos found for {year}.</p>
          <Link to="/wall" className="text-blue-400 hover:text-blue-300 inline-flex items-center gap-2">
            <ArrowLeft className="h-4 w-4" />
            Back to Years
          </Link>
        </div>
      </div>
    );
  }

  const headerContent = (
    <Header
      homeTo="/wall"
      breadcrumbs={[{ label: year.toString() }]}
      viewAction={
        <ViewSwitcher
          modes={[
            {
              key: "grid",
              label: "Grid View",
              icon: <Grid className="h-4 w-4" />,
              to: `/year/${year}/grid`,
              isActive: false,
            },
            {
              key: "wall",
              label: "Photo Wall",
              icon: <CoverflowIcon className="h-4 w-4" />,
              isActive: true,
            },
          ]}
        />
      }
    />
  );

  return <PhotoWall key={location.key} tiles={tiles} sessionKey={`year-wall-${year}`} headerContent={headerContent} />;
}
