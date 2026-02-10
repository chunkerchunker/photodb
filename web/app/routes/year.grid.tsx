import { Box } from "lucide-react";
import { data, Link } from "react-router";
import { Breadcrumb } from "~/components/breadcrumb";
import { Layout } from "~/components/layout";
import { Card, CardContent } from "~/components/ui/card";
import { requireCollectionId } from "~/lib/auth.server";
import { getMonthsInYear } from "~/lib/db.server";
import type { Route } from "./+types/year.grid";

export function meta({ params }: Route.MetaArgs) {
  return [{ title: `PhotoDB - ${params.year}` }, { name: "description", content: `Browse photos from ${params.year}` }];
}

import { dataWithViewMode } from "~/lib/cookies.server";

export async function loader({ request, params }: Route.LoaderArgs) {
  const { collectionId } = await requireCollectionId(request);
  const year = parseInt(params.year, 10);

  try {
    const months = await getMonthsInYear(collectionId, year);
    return dataWithViewMode({ months, year: params.year }, "grid");
  } catch (error) {
    console.error(`Failed to load months for year ${year}:`, error);
    return dataWithViewMode({ months: [], year: params.year }, "grid");
  }
}

export default function YearView({ loaderData }: Route.ComponentProps) {
  const { months, year } = loaderData;

  return (
    <Layout>
      <div className="space-y-6">
        <Breadcrumb items={[{ label: year }]} />

        <div className="flex items-center justify-between">
          <h1 className="text-3xl font-bold text-gray-900">{year}</h1>
          <Link
            to={`/year/${year}/wall`}
            className="inline-flex items-center gap-2 px-3 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 hover:border-gray-400 transition-colors"
            title="Photo Wall"
          >
            <Box className="h-4 w-4" />
            Photo Wall
          </Link>
        </div>

        {months.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {months.map((monthData) => (
              <Link
                key={monthData.month}
                to={`/year/${year}/month/${monthData.month}`}
                className="block transition-transform hover:scale-105"
              >
                <Card className="h-full hover:shadow-lg transition-shadow">
                  <CardContent className="p-6 text-center">
                    <div className="text-2xl font-bold text-gray-800 mb-2">{monthData.month_name}</div>
                    <div className="text-gray-600 mb-4">
                      {monthData.photo_count} photo
                      {monthData.photo_count !== 1 ? "s" : ""}
                    </div>

                    {monthData.sample_photo_ids && monthData.sample_photo_ids.length > 0 && (
                      <div className="mt-4">
                        <div className="grid grid-cols-2 gap-1 rounded overflow-hidden">
                          {monthData.sample_photo_ids.slice(0, 4).map((photoId: number) => (
                            <img
                              key={photoId}
                              src={`/api/image/${photoId}`}
                              alt="Preview"
                              className="w-full aspect-square object-cover"
                            />
                          ))}
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </Link>
            ))}
          </div>
        ) : (
          <div className="text-center py-12">
            <div className="text-gray-500 text-lg">No photos found for {year}.</div>
          </div>
        )}
      </div>
    </Layout>
  );
}
