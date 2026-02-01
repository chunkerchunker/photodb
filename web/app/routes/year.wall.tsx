import { ArrowLeft, Grid } from "lucide-react";
import { useMemo } from "react";
import { Link, useLocation } from "react-router";
import { PhotoWall, type WallTile } from "~/components/photo-wall";
import { getMonthsInYear } from "~/lib/db.server";
import type { Route } from "./+types/year.wall";

export function meta({ params }: Route.MetaArgs) {
  return [
    { title: `PhotoDB - ${params.year} - 3D Wall` },
    { name: "description", content: `Browse photos from ${params.year} in 3D wall view` },
  ];
}

export async function loader({ params }: Route.LoaderArgs) {
  const year = parseInt(params.year, 10);

  try {
    const months = await getMonthsInYear(year);
    return { months, year: params.year };
  } catch (error) {
    console.error(`Failed to load months for year ${year}:`, error);
    return { months: [], year: params.year };
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
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-4">
        <Link
          to={`/year/${year}`}
          className="text-white/80 hover:text-white transition-colors p-2 rounded-lg hover:bg-white/10"
          title="Grid view"
        >
          <Grid className="h-5 w-5" />
        </Link>
        <div>
          <h1 className="text-2xl font-bold text-white">{year}</h1>
          <p className="text-white/60 text-sm">
            {months.length} month{months.length !== 1 ? "s" : ""} - 3D Wall View
          </p>
        </div>
      </div>
      <Link to="/wall" className="text-white/80 hover:text-white transition-colors flex items-center gap-2">
        <ArrowLeft className="h-4 w-4" />
        Back to Years
      </Link>
    </div>
  );

  return <PhotoWall key={location.key} tiles={tiles} sessionKey={`year-wall-${year}`} headerContent={headerContent} />;
}
