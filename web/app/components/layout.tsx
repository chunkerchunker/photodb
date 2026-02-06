import { Camera, Grid, User, Users } from "lucide-react";
import { Link } from "react-router";
import { Header } from "./header";
import { ViewSwitcher } from "./view-switcher";
import { CoverflowIcon } from "./coverflow-icon";

interface LayoutProps {
  children: React.ReactNode;
}

export function Layout({ children }: LayoutProps) {
  return (
    <div className="min-h-screen bg-gray-50">
      <Header
        viewAction={
          <ViewSwitcher
            modes={[
              {
                key: "grid",
                label: "Grid View",
                icon: <Grid className="h-4 w-4" />,
                isActive: true,
              },
              {
                key: "wall",
                label: "3D Wall",
                icon: <CoverflowIcon className="h-4 w-4" />,
                to: "/wall",
                isActive: false,
              },
            ]}
          />
        }
      />
      {/* Spacer for fixed/absolute header if needed, but since Wall header overlays, maybe we want that effect here too? 
          However, for standard grid pages, we usually want the content to start below.
          The prompt says: "unify the header presentation... matching Wall style".
          Wall style is transparent/gradient overlay.
          If we make it absolute overlay on Grid view, content will be covered.
          Let's try to make it look like the standard header but with the gradient style, 
          OR we add padding to top of main content to account for it.
      */}
      <div className="h-16 bg-gray-900" />{" "}
      {/* shim to hold space if we want the dark branding behind it, or we just render it normally without absolute */}
      <main className="container mx-auto px-4 py-6">{children}</main>
    </div>
  );
}
