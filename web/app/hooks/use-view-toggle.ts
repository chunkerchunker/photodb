import { useLocation } from "react-router";

/**
 * Hook to get view toggle paths based on current location.
 * Automatically computes grid <-> wall paths by replacing the view segment.
 */
export function useViewToggle() {
  const location = useLocation();
  const pathname = location.pathname;

  // Determine current view and compute alternate path
  const isGridView = pathname.endsWith("/grid") || pathname === "/";
  const isWallView = pathname.endsWith("/wall");

  let gridPath: string;
  let wallPath: string;

  if (pathname === "/" || pathname === "/grid" || pathname === "/wall") {
    // Home page special cases
    gridPath = "/grid";
    wallPath = "/wall";
  } else if (isGridView) {
    gridPath = pathname;
    wallPath = pathname.replace(/\/grid$/, "/wall");
  } else if (isWallView) {
    gridPath = pathname.replace(/\/wall$/, "/grid");
    wallPath = pathname;
  } else {
    // Default fallback - append /grid or /wall
    gridPath = pathname + "/grid";
    wallPath = pathname + "/wall";
  }

  return {
    isGridView,
    isWallView,
    gridPath,
    wallPath,
    currentView: isWallView ? "wall" : "grid",
  };
}
