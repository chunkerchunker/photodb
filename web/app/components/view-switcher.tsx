import { Link } from "react-router";
import { cn } from "~/lib/utils";

export interface ViewMode {
  key: string;
  label: string;
  icon: React.ReactNode;
  to?: string;
  isActive: boolean;
}

interface ViewSwitcherProps {
  modes: ViewMode[];
  /** Visual variant - "dark" for dark backgrounds (default), "light" for light backgrounds */
  variant?: "dark" | "light";
}

export function ViewSwitcher({ modes, variant = "dark" }: ViewSwitcherProps): React.ReactElement {
  const containerClass =
    variant === "light"
      ? "flex items-center rounded-lg border bg-gray-50 p-1"
      : "flex items-center bg-white/10 rounded-lg p-1 space-x-1";

  const getItemClass = (isActive: boolean): string => {
    if (variant === "light") {
      return cn(
        "flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors",
        isActive ? "bg-white text-gray-900 shadow-sm" : "text-gray-500 hover:text-gray-700",
      );
    }
    return cn(
      "p-1.5 rounded-md transition-all flex items-center justify-center",
      isActive ? "bg-white text-black shadow-sm" : "text-white hover:bg-white/10 text-white/60",
    );
  };

  return (
    <div className={containerClass}>
      {modes.map((mode) => {
        const content = (
          <div className={getItemClass(mode.isActive)} title={mode.label}>
            {mode.icon}
          </div>
        );

        if (mode.isActive || !mode.to) {
          return (
            <div key={mode.key} title={mode.label}>
              {content}
            </div>
          );
        }

        return (
          <Link key={mode.key} to={mode.to} aria-label={mode.label}>
            {content}
          </Link>
        );
      })}
    </div>
  );
}
