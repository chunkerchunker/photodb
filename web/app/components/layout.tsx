import { Link } from "react-router";
import { Camera } from "lucide-react";

interface LayoutProps {
	children: React.ReactNode;
}

export function Layout({ children }: LayoutProps) {
	return (
		<div className="min-h-screen bg-gray-50">
			<nav className="bg-gray-900 text-white">
				<div className="container mx-auto px-4">
					<div className="flex items-center h-16">
						<Link
							to="/"
							className="flex items-center space-x-2 text-xl font-bold hover:text-gray-300"
						>
							<Camera className="h-6 w-6" />
							<span>PhotoDB</span>
						</Link>
					</div>
				</div>
			</nav>
			<main className="container mx-auto px-4 py-6">{children}</main>
		</div>
	);
}
