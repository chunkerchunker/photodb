import { Camera, FolderOpen, Images, LogOut, Shield, User, UserX, Users } from "lucide-react";
import { Link } from "react-router";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "~/components/ui/dropdown-menu";
import { getFaceCropStyle } from "~/lib/face-crop";

export interface UserAvatarInfo {
  firstName: string;
  lastName?: string | null;
  avatarPhotoId: number | null;
  avatarBboxX: number | null;
  avatarBboxY: number | null;
  avatarBboxWidth: number | null;
  avatarBboxHeight: number | null;
  avatarMedWidth: number | null;
  avatarMedHeight: number | null;
}

interface HeaderProps {
  viewAction?: React.ReactNode;
  breadcrumbs?: { label: string; to?: string }[];
  homeTo?: string;
  user?: UserAvatarInfo;
  isAdmin?: boolean;
  isImpersonating?: boolean;
}

function UserAvatar({ user }: { user?: UserAvatarInfo }) {
  const hasAvatar =
    user?.avatarPhotoId &&
    user.avatarBboxX !== null &&
    user.avatarBboxY !== null &&
    user.avatarBboxWidth !== null &&
    user.avatarBboxHeight !== null &&
    user.avatarMedWidth !== null &&
    user.avatarMedHeight !== null;

  if (hasAvatar) {
    return (
      <div className="relative w-8 h-8 bg-white/20 rounded-full overflow-hidden ring-2 ring-white/30 hover:ring-white/50 transition-all">
        <img
          src={`/api/image/${user.avatarPhotoId}`}
          alt={user.firstName}
          className="absolute max-w-none max-h-none"
          style={getFaceCropStyle(
            {
              bbox_x: user.avatarBboxX!,
              bbox_y: user.avatarBboxY!,
              bbox_width: user.avatarBboxWidth!,
              bbox_height: user.avatarBboxHeight!,
            },
            user.avatarMedWidth!,
            user.avatarMedHeight!,
            32,
          )}
        />
      </div>
    );
  }

  return (
    <div className="w-8 h-8 bg-white/20 rounded-full flex items-center justify-center ring-2 ring-white/30 hover:ring-white/50 transition-all">
      <User className="h-4 w-4 text-white" />
    </div>
  );
}

export function Header({ viewAction, breadcrumbs = [], homeTo = "/", user, isAdmin, isImpersonating }: HeaderProps) {
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

            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button type="button" className="focus:outline-none cursor-pointer">
                  <UserAvatar user={user} />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-48">
                {user && (
                  <>
                    <div className="px-2 py-1.5 text-sm font-medium text-gray-900">
                      {user.firstName} {user.lastName || ""}
                    </div>
                    <DropdownMenuSeparator />
                  </>
                )}
                <DropdownMenuItem asChild>
                  <Link to="/collections" className="flex items-center cursor-pointer">
                    <FolderOpen className="h-4 w-4 mr-2" />
                    Collections
                  </Link>
                </DropdownMenuItem>
                {isAdmin && (
                  <DropdownMenuItem asChild>
                    <Link to="/admin/users" className="flex items-center cursor-pointer">
                      <Shield className="h-4 w-4 mr-2" />
                      Admin
                    </Link>
                  </DropdownMenuItem>
                )}
                <DropdownMenuSeparator />
                <DropdownMenuItem asChild>
                  <form method="post" action={isImpersonating ? "/api/admin/stop-impersonate" : "/logout"} className="w-full">
                    <button
                      type="submit"
                      className={`flex items-center w-full text-left cursor-pointer ${isImpersonating ? "text-amber-600" : "text-red-600"}`}
                    >
                      {isImpersonating ? (
                        <>
                          <UserX className="h-4 w-4 mr-2" />
                          Stop Impersonating
                        </>
                      ) : (
                        <>
                          <LogOut className="h-4 w-4 mr-2" />
                          Sign out
                        </>
                      )}
                    </button>
                  </form>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
      </div>
    </nav>
  );
}
