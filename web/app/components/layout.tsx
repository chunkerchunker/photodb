import { Grid } from "lucide-react";
import { useRootData } from "~/hooks/use-root-data";
import { useViewToggle } from "~/hooks/use-view-toggle";
import { CoverflowIcon } from "./coverflow-icon";
import { Header } from "./header";
import { ViewSwitcher } from "./view-switcher";

interface LayoutProps {
  children: React.ReactNode;
}

export function Layout({ children }: LayoutProps) {
  const rootData = useRootData();
  const { wallPath } = useViewToggle();

  return (
    <div className="min-h-screen bg-gray-50">
      <Header
        user={rootData?.userAvatar}
        isAdmin={rootData?.user?.isAdmin}
        isImpersonating={rootData?.impersonation?.isImpersonating}
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
                label: "Photo Wall",
                icon: <CoverflowIcon className="h-4 w-4" />,
                to: wallPath,
                isActive: false,
              },
            ]}
          />
        }
      />
      <div className="h-16 bg-gray-900" />
      <main className="container mx-auto px-4 py-6">{children}</main>
    </div>
  );
}
