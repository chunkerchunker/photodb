import { useCallback, useEffect, useState } from "react";
import { useFetcher, useRevalidator } from "react-router";

export interface MergePreview {
  willMergePersons: boolean;
  source?: { personName: string | null; personClusterCount: number };
  target?: { personName: string | null; personClusterCount: number };
}

interface UseDragToMergeOptions<T> {
  items: T[];
  getItemKey: (item: T) => string;
  getMergeId: (item: T) => number;
  getItemName: (item: T) => string;
  previewUrl: (sourceId: number, targetId: number) => string;
  mergeAction: string;
  buildFormData: (sourceId: number, targetId: number) => Record<string, string>;
}

export function useDragToMerge<T>({
  items,
  getItemKey,
  getMergeId,
  getItemName,
  previewUrl,
  mergeAction,
  buildFormData,
}: UseDragToMergeOptions<T>) {
  const [draggingItemKey, setDraggingItemKey] = useState<string | null>(null);
  const [dropTargetItemKey, setDropTargetItemKey] = useState<string | null>(null);
  const [pendingMerge, setPendingMerge] = useState<{ sourceId: number; targetId: number } | null>(null);
  const [mergePreview, setMergePreview] = useState<MergePreview | null>(null);
  const [isLoadingPreview, setIsLoadingPreview] = useState(false);

  const mergeFetcher = useFetcher();
  const revalidator = useRevalidator();

  const isSubmitting = mergeFetcher.state !== "idle";

  // Revalidate after merge completes
  // pendingMerge intentionally omitted from deps — including it causes stale mergeFetcher.data
  // to immediately clear the next pendingMerge before the user can confirm
  // biome-ignore lint/correctness/useExhaustiveDependencies: pendingMerge intentionally omitted
  useEffect(() => {
    if (mergeFetcher.state === "idle" && mergeFetcher.data?.success && pendingMerge) {
      setPendingMerge(null);
      setMergePreview(null);
      revalidator.revalidate();
    }
  }, [mergeFetcher.state, mergeFetcher.data, revalidator]);

  const handleDragStart = useCallback(
    (e: React.DragEvent, item: T) => {
      const itemKey = getItemKey(item);
      setDraggingItemKey(itemKey);
      e.dataTransfer.effectAllowed = "move";
      e.dataTransfer.setData("text/plain", `${itemKey}|${getMergeId(item)}`);
    },
    [getItemKey, getMergeId],
  );

  const handleDragEnd = useCallback(() => {
    setDraggingItemKey(null);
    setDropTargetItemKey(null);
  }, []);

  const handleDragOver = useCallback(
    (e: React.DragEvent, item: T) => {
      e.preventDefault();
      const itemKey = getItemKey(item);
      if (draggingItemKey && draggingItemKey !== itemKey) {
        e.dataTransfer.dropEffect = "move";
        setDropTargetItemKey(itemKey);
      }
    },
    [draggingItemKey, getItemKey],
  );

  const handleDragLeave = useCallback(() => {
    setDropTargetItemKey(null);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent, targetItem: T) => {
      e.preventDefault();
      const targetItemKey = getItemKey(targetItem);
      const data = e.dataTransfer.getData("text/plain");
      const [sourceItemKey, sourceIdStr] = data.split("|");
      const sourceId = parseInt(sourceIdStr, 10);
      const targetId = getMergeId(targetItem);

      if (sourceItemKey && sourceItemKey !== targetItemKey && sourceId && targetId) {
        setPendingMerge({ sourceId, targetId });
        setIsLoadingPreview(true);
        setMergePreview(null);
        fetch(previewUrl(sourceId, targetId))
          .then((res) => {
            if (!res.ok) throw new Error(`Preview failed: ${res.status}`);
            return res.json();
          })
          .then((preview) => setMergePreview(preview))
          .catch((err) => console.error("Failed to fetch merge preview:", err))
          .finally(() => setIsLoadingPreview(false));
      }
      setDraggingItemKey(null);
      setDropTargetItemKey(null);
    },
    [getItemKey, getMergeId, previewUrl],
  );

  const getNameByMergeId = useCallback(
    (mergeId: number) => {
      const item = items.find((i) => getMergeId(i) === mergeId);
      return item ? getItemName(item) : `#${mergeId}`;
    },
    [items, getMergeId, getItemName],
  );

  const confirmMerge = useCallback(() => {
    if (pendingMerge) {
      mergeFetcher.submit(buildFormData(pendingMerge.sourceId, pendingMerge.targetId), {
        method: "post",
        action: mergeAction,
      });
    }
  }, [pendingMerge, mergeFetcher, buildFormData, mergeAction]);

  const cancelMerge = useCallback(() => {
    setPendingMerge(null);
    setMergePreview(null);
  }, []);

  return {
    handleDragStart,
    handleDragEnd,
    handleDragOver,
    handleDragLeave,
    handleDrop,
    isDragging: (item: T) => draggingItemKey === getItemKey(item),
    isDropTarget: (item: T) => dropTargetItemKey === getItemKey(item),
    pendingMerge,
    mergePreview,
    isLoadingPreview,
    isSubmitting,
    sourceName: pendingMerge ? getNameByMergeId(pendingMerge.sourceId) : "",
    targetName: pendingMerge ? getNameByMergeId(pendingMerge.targetId) : "",
    confirmMerge,
    cancelMerge,
  };
}
