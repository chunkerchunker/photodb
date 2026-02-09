import { Camera, Images, User, Users } from "lucide-react";
import { Link } from "react-router";

interface HeaderProps {
  viewAction?: React.ReactNode;
  breadcrumbs?: { label: string; to?: string }[];
  homeTo?: string;
}

export function Header({ viewAction, breadcrumbs = [], homeTo = "/" }: HeaderProps) {
  return (
    <nav className="absolute top-0 left-0 right-0 z-50 bg-gradient-to-b from-black/80 to-transparent pb-4 pointer-events-none">
      <div className="container mx-auto px-4 pointer-events-auto">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center">
            <Link
              to={homeTo}
              className="flex items-center space-x-2 text-3xl hover:text-white/80 font-[family-name:--font-display] text-white transition-opacity"
            >
              <img src="/family-tree-2.svg" alt="Storyteller" className="size-10 mt-1 mr-1" />
              <span>Storyteller</span>
            </Link>
            {breadcrumbs.map((crumb, index) => (
              <div key={index} className="flex items-center text-3xl font-[family-name:--font-display] text-white">
                <span className="mx-2 opacity-50">/</span>
                {crumb.to ? (
                  <Link to={crumb.to} className="hover:text-white/80 transition-colors">
                    {crumb.label}
                  </Link>
                ) : (
                  <span>{crumb.label}</span>
                )}
              </div>
            ))}
          </div>

          <div className="flex items-center space-x-6 text-white">
            {viewAction}

            <Link to="/" className="flex items-center space-x-1 hover:text-white/80 transition-colors">
              <Camera className="h-4 w-4" />
              <span>Photos</span>
            </Link>

            <Link to="/people" className="flex items-center space-x-1 hover:text-white/80 transition-colors">
              <User className="h-4 w-4" />
              <span>People</span>
            </Link>

            <Link to="/albums" className="flex items-center space-x-1 hover:text-white/80 transition-colors">
              <Images className="h-4 w-4" />
              <span>Albums</span>
            </Link>

            <Link to="/clusters" className="flex items-center space-x-1 hover:text-white/80 transition-colors">
              <Users className="h-4 w-4" />
              <span>Clusters</span>
            </Link>

            <form method="post" action="/logout">
              <button type="submit" className="text-sm text-white/80 hover:text-white transition-colors">
                Sign out
              </button>
            </form>
          </div>
        </div>
      </div>
    </nav>
  );
}
