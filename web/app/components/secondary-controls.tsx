import { ArrowDownAZ, ImageOff, Users } from "lucide-react";
import { cn } from "~/lib/utils";

interface SecondaryControlsProps {
  variant: "grid" | "wall";
  children: React.ReactNode;
  className?: string;
}

/**
 * Container for secondary page controls (hidden button, sort, counts, etc.)
 *
 * - `grid` variant: Inline flex container for use in page headers
 * - `wall` variant: Floating semi-transparent toolbar anchored to top-right
 */
export function SecondaryControls({ variant, children, className }: SecondaryControlsProps) {
  if (variant === "grid") {
    return <div className={cn("flex items-center space-x-4", className)}>{children}</div>;
  }

  // Wall variant: floating toolbar
  return (
    <div
      className={cn(
        "fixed top-20 right-6 z-40",
        "flex items-center gap-3 px-4 py-2.5",
        "bg-black/60 backdrop-blur-md rounded-full",
        "border border-white/10 shadow-lg",
        "text-white text-sm",
        "opacity-60 hover:opacity-100 transition-opacity duration-200",
        className,
      )}
    >
      {children}
    </div>
  );
}

/**
 * Styled count display for use within SecondaryControls
 */
export function ControlsCount({
  count,
  singular,
  plural,
  variant = "grid",
}: {
  count: number;
  singular: string;
  plural: string;
  variant?: "grid" | "wall";
}) {
  const label = count === 1 ? singular : plural;

  if (variant === "wall") {
    return (
      <span className="text-white/70">
        {count} {label}
      </span>
    );
  }

  return (
    <span className="text-gray-600">
      {count} {label}
    </span>
  );
}

/**
 * Divider for separating control groups in wall variant
 */
export function ControlsDivider({ variant = "grid" }: { variant?: "grid" | "wall" }) {
  if (variant === "wall") {
    return <div className="w-px h-4 bg-white/20" />;
  }
  return null;
}

/**
 * Sort toggle for switching between photo count and alphabetical sorting
 */
export function SortToggle({
  sort,
  onSortChange,
  variant = "grid",
}: {
  sort: "photos" | "name";
  onSortChange: (sort: "photos" | "name") => void;
  variant?: "grid" | "wall";
}) {
  if (variant === "wall") {
    return (
      <div className="flex items-center gap-1">
        <button
          type="button"
          onClick={() => onSortChange("photos")}
          className={cn(
            "p-1.5 rounded transition-colors",
            sort === "photos" ? "bg-white/20 text-white" : "text-white/50 hover:text-white/80",
          )}
          title="Sort by most photos"
        >
          <Users className="h-4 w-4" />
        </button>
        <button
          type="button"
          onClick={() => onSortChange("name")}
          className={cn(
            "p-1.5 rounded transition-colors",
            sort === "name" ? "bg-white/20 text-white" : "text-white/50 hover:text-white/80",
          )}
          title="Sort alphabetically"
        >
          <ArrowDownAZ className="h-4 w-4" />
        </button>
      </div>
    );
  }

  return (
    <div className="flex items-center rounded-lg border bg-gray-50 p-1" title="Sort order">
      <button
        type="button"
        onClick={() => onSortChange("photos")}
        className={cn(
          "flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors",
          sort === "photos" ? "bg-white text-gray-900 shadow-sm" : "text-gray-500 hover:text-gray-700",
        )}
        title="Sort by most photos"
      >
        <Users className="h-4 w-4" />
      </button>
      <button
        type="button"
        onClick={() => onSortChange("name")}
        className={cn(
          "flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors",
          sort === "name" ? "bg-white text-gray-900 shadow-sm" : "text-gray-500 hover:text-gray-700",
        )}
        title="Sort alphabetically"
      >
        <ArrowDownAZ className="h-4 w-4" />
      </button>
    </div>
  );
}

/**
 * Toggle for showing/hiding people without images (not linked to clusters)
 */
export function WithoutImagesToggle({
  showWithoutImages,
  onToggle,
  withoutImagesCount,
  variant = "grid",
}: {
  showWithoutImages: boolean;
  onToggle: (show: boolean) => void;
  withoutImagesCount: number;
  variant?: "grid" | "wall";
}) {
  if (withoutImagesCount === 0) {
    return null;
  }

  if (variant === "wall") {
    return (
      <button
        type="button"
        onClick={() => onToggle(!showWithoutImages)}
        className={cn(
          "flex items-center gap-1.5 transition-colors",
          showWithoutImages ? "text-white" : "text-white/50 hover:text-white/80",
        )}
        title={showWithoutImages ? "Hide people without images" : "Show people without images"}
      >
        <ImageOff className="h-4 w-4" />
        <span>No image ({withoutImagesCount})</span>
      </button>
    );
  }

  return (
    <button
      type="button"
      onClick={() => onToggle(!showWithoutImages)}
      className={cn(
        "flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors border",
        showWithoutImages
          ? "bg-white text-gray-900 shadow-sm border-gray-200"
          : "text-gray-500 hover:text-gray-700 border-transparent hover:border-gray-200",
      )}
      title={showWithoutImages ? "Hide people without images" : "Show people without images"}
    >
      <ImageOff className="h-4 w-4" />
      <span>No image ({withoutImagesCount})</span>
    </button>
  );
}
