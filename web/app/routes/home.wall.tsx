import { Grid } from "lucide-react";
import { useMemo } from "react";
import { Link, useLocation } from "react-router";
import { PhotoWall, type WallTile } from "~/components/photo-wall";
import { getYearsWithPhotos } from "~/lib/db.server";
import type { Route } from "./+types/home.wall";

export function meta() {
  return [
    { title: "PhotoDB - Years - 3D Wall" },
    { name: "description", content: "Browse your photo collection by year in 3D wall view" },
  ];
}

export async function loader() {
  try {
    const years = await getYearsWithPhotos();
    return { years };
  } catch (error) {
    console.error("Failed to load years:", error);
    return { years: [] };
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
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-4">
        <Link
          to="/"
          className="text-white/80 hover:text-white transition-colors p-2 rounded-lg hover:bg-white/10"
          title="Grid view"
        >
          <Grid className="h-5 w-5" />
        </Link>
        <div>
          <h1 className="text-2xl font-bold text-white">Photo Collection</h1>
          <p className="text-white/60 text-sm">
            {years.length} year{years.length !== 1 ? "s" : ""} - 3D Wall View
          </p>
        </div>
      </div>
    </div>
  );

  return <PhotoWall key={location.key} tiles={tiles} sessionKey="home-wall" headerContent={headerContent} />;
}
