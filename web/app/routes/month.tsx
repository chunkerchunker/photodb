import type { Route } from "./+types/month";
import { Layout } from "~/components/layout";
import { Breadcrumb } from "~/components/breadcrumb";
import { Pagination } from "~/components/pagination";
import { Card, CardContent } from "~/components/ui/card";
import { Link } from "react-router";
import { getPhotosByMonth, getPhotoCountByMonth } from "~/lib/db.server";

export function meta({ params }: Route.MetaArgs) {
	const monthNames = [
		"",
		"January",
		"February",
		"March",
		"April",
		"May",
		"June",
		"July",
		"August",
		"September",
		"October",
		"November",
		"December",
	];
	const monthName = monthNames[parseInt(params.month)] || params.month;

	return [
		{ title: `PhotoDB - ${monthName} ${params.year}` },
		{
			name: "description",
			content: `Browse photos from ${monthName} ${params.year}`,
		},
	];
}

export async function loader({ params, request }: Route.LoaderArgs) {
	const year = parseInt(params.year);
	const month = parseInt(params.month);
	const url = new URL(request.url);
	const page = parseInt(url.searchParams.get("page") || "1");
	const limit = 48; // 6x8 grid
	const offset = (page - 1) * limit;

	const monthNames = [
		"",
		"January",
		"February",
		"March",
		"April",
		"May",
		"June",
		"July",
		"August",
		"September",
		"October",
		"November",
		"December",
	];
	const monthName = monthNames[month] || `Month ${month}`;

	try {
		const photos = await getPhotosByMonth(year, month, limit, offset);
		const totalPhotos = await getPhotoCountByMonth(year, month);
		const totalPages = Math.ceil(totalPhotos / limit);

		return {
			photos,
			totalPhotos,
			totalPages,
			currentPage: page,
			year: params.year,
			month: params.month,
			monthName,
		};
	} catch (error) {
		console.error(`Failed to load photos for ${year}-${month}:`, error);
		return {
			photos: [],
			totalPhotos: 0,
			totalPages: 0,
			currentPage: page,
			year: params.year,
			month: params.month,
			monthName,
		};
	}
}

export default function MonthView({ loaderData }: Route.ComponentProps) {
	const {
		photos,
		totalPhotos,
		totalPages,
		currentPage,
		year,
		month,
		monthName,
	} = loaderData;

	return (
		<Layout>
			<div className="space-y-6">
				<Breadcrumb
					items={[{ label: year, href: `/year/${year}` }, { label: monthName }]}
				/>

				<div className="flex items-center justify-between">
					<h1 className="text-3xl font-bold text-gray-900">
						{monthName} {year}
					</h1>
					<span className="text-gray-600">
						{totalPhotos} photo{totalPhotos !== 1 ? "s" : ""}
					</span>
				</div>

				{photos.length > 0 ? (
					<>
						<div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 xl:grid-cols-8 gap-4">
							{photos.map((photo) => (
								<Link
									key={photo.id}
									to={`/photo/${photo.id}`}
									className="block transition-transform hover:scale-105"
								>
									<Card className="overflow-hidden hover:shadow-lg transition-shadow">
										<div className="relative">
											<img
												src={`/api/image/${photo.id}`}
												alt={photo.filename_only}
												className="w-full h-48 object-cover"
												loading="lazy"
											/>
										</div>
										<CardContent className="p-3">
											<div
												className="text-sm font-medium text-gray-900 truncate"
												title={photo.filename_only}
											>
												{photo.filename_only}
											</div>
											<div className="text-xs text-gray-500 mt-1">
												ID: {photo.id}
											</div>
											{photo.short_description && (
												<div
													className="text-xs text-gray-600 mt-1 line-clamp-2"
													title={photo.description}
												>
													{photo.short_description}
												</div>
											)}
										</CardContent>
									</Card>
								</Link>
							))}
						</div>

						<Pagination
							currentPage={currentPage}
							totalPages={totalPages}
							baseUrl={`/year/${year}/month/${month}`}
						/>
					</>
				) : (
					<div className="text-center py-12">
						<div className="text-gray-500 text-lg">
							No photos found for {monthName} {year}.
						</div>
					</div>
				)}
			</div>
		</Layout>
	);
}
