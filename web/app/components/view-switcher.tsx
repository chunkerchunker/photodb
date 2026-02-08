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
}

export function ViewSwitcher({ modes }: ViewSwitcherProps) {
  return (
    <div className="flex items-center bg-white/10 rounded-lg p-1 space-x-1">
      {modes.map((mode) => {
        const content = (
          <div
            className={cn(
              "p-1.5 rounded-md transition-all flex items-center justify-center",
              mode.isActive ? "bg-white text-black shadow-sm" : "text-white hover:bg-white/10 text-white/60",
            )}
            title={mode.label}
          >
            {mode.icon}
          </div>
        );

        if (mode.isActive || !mode.to) {
          return (
            <div key={mode.key} aria-label={mode.label}>
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
