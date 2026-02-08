import { data, Link } from "react-router";
import { Layout } from "~/components/layout";
import { Card, CardContent } from "~/components/ui/card";
import { getYearsWithPhotos } from "~/lib/db.server";
import type { Route } from "./+types/home.grid";

export function meta() {
  return [{ title: "PhotoDB - Years" }, { name: "description", content: "Browse your photo collection by year" }];
}

// Middleware to set cookie needs to be handled via headers in React Router usually,
// or by returning a Response object.
// React Router 7 loaders can return Response objects.

import { dataWithViewMode } from "~/lib/cookies.server";

export async function loader() {
  try {
    const years = await getYearsWithPhotos();
    return dataWithViewMode({ years }, "grid");
  } catch (error) {
    console.error("Failed to load years:", error);
    return dataWithViewMode({ years: [] }, "grid");
  }
}

export default function Home({ loaderData }: Route.ComponentProps) {
  const { years } = loaderData;

  return (
    <Layout>
      <div className="mt-4 space-y-6">
        {years.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {years.map((yearData) => (
              <Link
                key={yearData.year}
                to={`/year/${yearData.year}`}
                className="block transition-transform hover:scale-105"
              >
                <Card className="h-full hover:shadow-lg transition-shadow">
                  <CardContent className="p-6 text-center">
                    <div className="text-4xl font-bold text-gray-800 mb-2">{yearData.year}</div>
                    <div className="text-gray-600 mb-4">
                      {yearData.photo_count} photo
                      {yearData.photo_count !== 1 ? "s" : ""}
                    </div>
                    {yearData.sample_photo_id && (
                      <div className="mt-4">
                        <img
                          src={`/api/image/${yearData.sample_photo_id}`}
                          alt={`Sample from ${yearData.year}`}
                          className="w-full h-32 object-cover rounded"
                        />
                      </div>
                    )}
                  </CardContent>
                </Card>
              </Link>
            ))}
          </div>
        ) : (
          <div className="text-center py-12">
            <div className="text-gray-500 text-lg">No photos with date information found in the database.</div>
          </div>
        )}
      </div>
    </Layout>
  );
}
