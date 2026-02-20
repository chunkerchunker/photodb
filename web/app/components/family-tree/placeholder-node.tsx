import { Handle, type NodeProps, Position } from "@xyflow/react";
import { Unlink } from "lucide-react";
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuTrigger,
} from "~/components/ui/context-menu";

export interface PlaceholderNodeData {
  name: string;
  personId: number;
  relation: string;
  isCenter: boolean;
  centerId: number;
  onRecenter: (personId: number) => void;
  onRemoveRelationship: (personId: number, relation: string) => void;
}

export function PlaceholderNode({ data }: NodeProps) {
  const d = data as unknown as PlaceholderNodeData;
  return (
    <ContextMenu>
      <ContextMenuTrigger asChild>
        {/* biome-ignore lint/a11y/useSemanticElements: React Flow custom node requires div container for Handle children */}
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
          <Handle type="target" position={Position.Top} className="!opacity-0 !w-0 !h-0 !min-w-0 !min-h-0 !border-0" />
          <Handle type="source" position={Position.Bottom} className="!opacity-0 !w-0 !h-0 !min-w-0 !min-h-0 !border-0" />
          <Handle id="left" type="target" position={Position.Left} className="!opacity-0 !w-0 !h-0 !min-w-0 !min-h-0 !border-0" />
          <Handle id="right" type="source" position={Position.Right} className="!opacity-0 !w-0 !h-0 !min-w-0 !min-h-0 !border-0" />
        </div>
      </ContextMenuTrigger>
      <ContextMenuContent>
        {!d.isCenter && (
          <ContextMenuItem onClick={() => d.onRecenter(d.personId)}>
            View family tree
          </ContextMenuItem>
        )}
        {!d.isCenter && (
          <ContextMenuItem onClick={() => d.onRemoveRelationship(d.personId, d.relation)}>
            <Unlink className="h-4 w-4 mr-2" />
            Remove relationship
          </ContextMenuItem>
        )}
      </ContextMenuContent>
    </ContextMenu>
  );
}
