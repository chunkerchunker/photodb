import { Handle, type NodeProps, Position } from "@xyflow/react";
import { Unlink } from "lucide-react";
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuTrigger,
} from "~/components/ui/context-menu";

export interface PersonNodeData {
  name: string;
  detectionId: number | null;
  isCenter: boolean;
  relation: string;
  personId: number;
  centerId: number;
  onRecenter: (personId: number) => void;
  onRemoveRelationship: (personId: number, relation: string) => void;
}

export function PersonNode({ data }: NodeProps) {
  const d = data as unknown as PersonNodeData;
  return (
    <ContextMenu>
      <ContextMenuTrigger asChild>
        {/* biome-ignore lint/a11y/useSemanticElements: React Flow custom node requires div container for Handle children */}
        <div
          role="button"
          tabIndex={0}
          className={`flex flex-col items-center justify-center w-[140px] h-[80px] rounded-xl border ${
            d.isCenter ? "border-blue-500 border-2 shadow-[0_0_16px_rgba(59,130,246,0.2)]" : "border-gray-600"
          } bg-gray-800 cursor-pointer`}
          onDoubleClick={() => d.onRecenter(d.personId)}
          onKeyDown={(e) => {
            if (e.key === "Enter") d.onRecenter(d.personId);
          }}
        >
          {d.detectionId ? (
            <img src={`/api/face/${d.detectionId}`} alt={d.name} className="w-9 h-9 rounded-full object-cover pointer-events-none" />
          ) : (
            <div className="w-9 h-9 rounded-full bg-gray-600" />
          )}
          <span
            className={`text-xs mt-1 text-gray-200 text-center truncate max-w-[120px] ${
              d.isCenter ? "font-semibold" : "font-medium"
            }`}
          >
            {d.name}
          </span>
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
