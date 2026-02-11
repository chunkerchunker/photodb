import { Grid } from "lucide-react";
import { useViewToggle } from "~/hooks/use-view-toggle";
import { CoverflowIcon } from "./coverflow-icon";
import { ViewSwitcher } from "./view-switcher";

/**
 * Pre-configured ViewSwitcher for wall views with dynamic path switching.
 * Use this in Header's viewAction prop for wall pages.
 */
export function WallViewSwitcher() {
  const { gridPath } = useViewToggle();

  return (
    <ViewSwitcher
      modes={[
        {
          key: "grid",
          label: "Grid View",
          icon: <Grid className="h-4 w-4" />,
          to: gridPath,
          isActive: false,
        },
        {
          key: "wall",
          label: "Photo Wall",
          icon: <CoverflowIcon className="h-4 w-4" />,
          isActive: true,
        },
      ]}
    />
  );
}
