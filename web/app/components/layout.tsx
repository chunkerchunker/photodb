import { Camera, User, Users } from "lucide-react";
import { Link } from "react-router";

interface LayoutProps {
  children: React.ReactNode;
}

export function Layout({ children }: LayoutProps) {
  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-gray-900 text-white">
        <div className="container mx-auto px-4">
          <div className="flex items-center justify-between h-16">
            <Link to="/" className="flex items-center space-x-2 text-xl font-bold hover:text-gray-300">
              <Camera className="h-6 w-6" />
              <span>PhotoDB</span>
            </Link>
            <div className="flex items-center space-x-6">
              <Link to="/" className="flex items-center space-x-1 hover:text-gray-300 transition-colors">
                <Camera className="h-4 w-4" />
                <span>Photos</span>
              </Link>
              <Link to="/people" className="flex items-center space-x-1 hover:text-gray-300 transition-colors">
                <User className="h-4 w-4" />
                <span>People</span>
              </Link>
              <Link to="/clusters" className="flex items-center space-x-1 hover:text-gray-300 transition-colors">
                <Users className="h-4 w-4" />
                <span>Clusters</span>
              </Link>
              <form method="post" action="/logout">
                <button type="submit" className="text-sm text-gray-300 hover:text-white transition-colors">
                  Sign out
                </button>
              </form>
            </div>
          </div>
        </div>
      </nav>
      <main className="container mx-auto px-4 py-6">{children}</main>
    </div>
  );
}
