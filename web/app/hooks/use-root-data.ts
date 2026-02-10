import { useOutletContext } from "react-router";
import type { RootLoaderData } from "~/root";

/**
 * Hook to access root loader data from child routes.
 * Returns user info and avatar data for the header.
 */
export function useRootData(): RootLoaderData {
  return useOutletContext<RootLoaderData>();
}
