import { UserX } from "lucide-react";
import { useFetcher } from "react-router";

interface ImpersonationBannerProps {
  realAdminName: string;
  impersonatedUserName: string;
}

export function ImpersonationBanner({ realAdminName, impersonatedUserName }: ImpersonationBannerProps) {
  const fetcher = useFetcher();
  const isLoading = fetcher.state !== "idle";

  return (
    <div className="fixed top-0 left-0 right-0 z-[100] bg-amber-500 text-amber-950 px-4 py-2">
      <div className="container mx-auto flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <UserX className="h-4 w-4" />
          <span className="text-sm font-medium">
            {realAdminName} is viewing as <strong>{impersonatedUserName}</strong>
          </span>
        </div>
        <fetcher.Form method="post" action="/api/admin/stop-impersonate">
          <button
            type="submit"
            disabled={isLoading}
            className="text-sm font-medium px-3 py-1 bg-amber-600 hover:bg-amber-700 rounded transition-colors disabled:opacity-50"
          >
            {isLoading ? "Stopping..." : "Stop Impersonating"}
          </button>
        </fetcher.Form>
      </div>
    </div>
  );
}
