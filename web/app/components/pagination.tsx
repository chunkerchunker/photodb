import { Link } from "react-router";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "~/components/ui/button";

interface PaginationProps {
	currentPage: number;
	totalPages: number;
	baseUrl: string;
}

export function Pagination({
	currentPage,
	totalPages,
	baseUrl,
}: PaginationProps) {
	if (totalPages <= 1) return null;

	const getPageNumbers = () => {
		const pages: (number | string)[] = [];
		const showEllipsis = totalPages > 7;

		if (!showEllipsis) {
			for (let i = 1; i <= totalPages; i++) {
				pages.push(i);
			}
		} else {
			// Always show first 3 pages
			for (let i = 1; i <= Math.min(3, totalPages); i++) {
				pages.push(i);
			}

			// Show ellipsis if there's a gap
			if (currentPage > 5) {
				pages.push("...");
			}

			// Show pages around current page
			const start = Math.max(4, currentPage - 1);
			const end = Math.min(totalPages - 3, currentPage + 1);

			if (start <= end) {
				for (let i = start; i <= end; i++) {
					if (!pages.includes(i)) {
						pages.push(i);
					}
				}
			}

			// Show ellipsis if there's a gap
			if (currentPage < totalPages - 4) {
				pages.push("...");
			}

			// Always show last 3 pages
			for (let i = Math.max(totalPages - 2, 4); i <= totalPages; i++) {
				if (!pages.includes(i)) {
					pages.push(i);
				}
			}
		}

		return pages;
	};

	const pageNumbers = getPageNumbers();

	return (
		<nav className="flex items-center justify-center space-x-2 mt-8">
			{/* Previous button */}
			{currentPage > 1 && (
				<Button variant="outline" size="sm" asChild>
					<Link to={`${baseUrl}?page=${currentPage - 1}`}>
						<ChevronLeft className="h-4 w-4 mr-1" />
						Previous
					</Link>
				</Button>
			)}

			{/* Page numbers */}
			<div className="flex items-center space-x-1">
				{pageNumbers.map((page, index) => (
					<div key={index}>
						{page === "..." ? (
							<span className="px-3 py-2 text-gray-400">...</span>
						) : (
							<Button
								variant={page === currentPage ? "default" : "outline"}
								size="sm"
								asChild
							>
								<Link to={`${baseUrl}?page=${page}`}>{page}</Link>
							</Button>
						)}
					</div>
				))}
			</div>

			{/* Next button */}
			{currentPage < totalPages && (
				<Button variant="outline" size="sm" asChild>
					<Link to={`${baseUrl}?page=${currentPage + 1}`}>
						Next
						<ChevronRight className="h-4 w-4 ml-1" />
					</Link>
				</Button>
			)}
		</nav>
	);
}
