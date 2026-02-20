import type { NodeProps } from "@xyflow/react";
import { Unlink } from "lucide-react";
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuTrigger,
} from "~/components/ui/context-menu";
import { InvisibleHandles } from "./invisible-handles";

export interface PlaceholderNodeData {
  name: string;
  personId: number;
  relation: string;
  isCenter: boolean;
  centerId: number;
  onRecenter: (personId: number) => void;
  onRemoveRelationship: (personId: number, relation: string) => void;
}

function PlaceholderNodeInner({ d }: { d: PlaceholderNodeData }) {
  return (
    // biome-ignore lint/a11y/useSemanticElements: React Flow custom node requires div container for Handle children
    <div
      role="button"
      tabIndex={0}
      className="flex flex-col items-center justify-center w-[140px] h-[80px] rounded-xl border border-dashed border-gray-500 bg-gray-800/80 cursor-pointer"
      onDoubleClick={() => d.onRecenter(d.personId)}
      onKeyDown={(e) => {
        if (e.key === "Enter") d.onRecenter(d.personId);
      }}
    >
      <div className="w-9 h-9 rounded-full bg-gray-600/60" />
      <span className="text-xs mt-1 text-gray-400 text-center truncate max-w-[120px] italic">{d.name}</span>
      <InvisibleHandles />
    </div>
  );
}

export function PlaceholderNode({ data }: NodeProps) {
  const d = data as unknown as PlaceholderNodeData;

  if (d.isCenter) {
    return <PlaceholderNodeInner d={d} />;
  }

  return (
    <ContextMenu>
      <ContextMenuTrigger asChild>
        <PlaceholderNodeInner d={d} />
      </ContextMenuTrigger>
      <ContextMenuContent>
        <ContextMenuItem onClick={() => d.onRecenter(d.personId)}>
          View family tree
        </ContextMenuItem>
        <ContextMenuItem onClick={() => d.onRemoveRelationship(d.personId, d.relation)}>
          <Unlink className="h-4 w-4 mr-2" />
          Remove relationship
        </ContextMenuItem>
      </ContextMenuContent>
    </ContextMenu>
  );
}
