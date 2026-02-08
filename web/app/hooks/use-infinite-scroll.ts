import { useCallback, useEffect, useRef } from "react";

interface UseInfiniteScrollOptions {
  /** Callback to load more items */
  onLoadMore: () => void;
  /** Whether there are more items to load */
  hasMore: boolean;
  /** Whether currently loading */
  isLoading: boolean;
  /** Root margin for IntersectionObserver (default: "200px") */
  rootMargin?: string;
}

/**
 * Hook for implementing infinite scroll using IntersectionObserver.
 * Returns a ref to attach to a sentinel element at the bottom of the list.
 */
export function useInfiniteScroll({
  onLoadMore,
  hasMore,
  isLoading,
  rootMargin = "200px",
}: UseInfiniteScrollOptions): React.RefObject<HTMLDivElement | null> {
  const loadMoreRef = useRef<HTMLDivElement>(null);

  const handleIntersect = useCallback(() => {
    if (!isLoading && hasMore) {
      onLoadMore();
    }
  }, [onLoadMore, hasMore, isLoading]);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          handleIntersect();
        }
      },
      { rootMargin },
    );

    const currentRef = loadMoreRef.current;
    if (currentRef) {
      observer.observe(currentRef);
    }

    return () => {
      if (currentRef) {
        observer.unobserve(currentRef);
      }
    };
  }, [handleIntersect, rootMargin]);

  return loadMoreRef;
}
