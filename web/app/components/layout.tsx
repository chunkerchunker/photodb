import { Grid } from "lucide-react";
import { useRootData } from "~/hooks/use-root-data";
import { CoverflowIcon } from "./coverflow-icon";
import { Header } from "./header";
import { ImpersonationBanner } from "./impersonation-banner";
import { ViewSwitcher } from "./view-switcher";

interface LayoutProps {
  children: React.ReactNode;
}

export function Layout({ children }: LayoutProps) {
  const rootData = useRootData();
  const isImpersonating = rootData?.impersonation?.isImpersonating ?? false;

  return (
    <div className="min-h-screen bg-gray-50">
      {isImpersonating && rootData?.impersonation?.realAdminName && (
        <ImpersonationBanner
          realAdminName={rootData.impersonation.realAdminName}
          impersonatedUserName={`${rootData.userAvatar.firstName} ${rootData.userAvatar.lastName || ""}`.trim()}
        />
      )}
      <Header
        user={rootData?.userAvatar}
        isAdmin={rootData?.user?.isAdmin}
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
      <div className={`h-16 bg-gray-900 ${isImpersonating ? "mt-10" : ""}`} />{" "}
      {/* shim to hold space if we want the dark branding behind it, or we just render it normally without absolute */}
      <main className="container mx-auto px-4 py-6">{children}</main>
    </div>
  );
}
