import type { NodeProps } from "@xyflow/react";
import { useViewport } from "@xyflow/react";

/** Renders a generation label that sticks to the left edge of the viewport
 *  while scrolling vertically with the flow. Uses zIndex: -1 (set in layout)
 *  so it renders behind person/placeholder nodes. */
export function GenerationLabelNode({ data, positionAbsoluteX }: NodeProps) {
  const d = data as Record<string, unknown>;
  const { x: vx, zoom } = useViewport();

  // Shift the label content so it always appears 12px from the viewport's left edge.
  // The node itself lives at (0, genY) in flow coordinates; this CSS transform
  // moves the visual content to the correct screen-x without changing the node position.
  const dx = (12 - vx) / zoom - positionAbsoluteX;

  return (
    <div
      className="flex items-center h-[80px] pointer-events-none select-none"
      style={{ transform: `translateX(${dx}px)` }}
    >
      <span className="text-xs font-medium text-gray-500 whitespace-nowrap">{d.label as string}</span>
    </div>
  );
}
