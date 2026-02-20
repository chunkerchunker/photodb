import { Handle, type NodeProps, Position } from "@xyflow/react";
import { Plus } from "lucide-react";
import { useState } from "react";

export interface DropZoneNodeData {
  label: string;
  relationshipType: "parent" | "child" | "sibling" | "partner";
  targetPersonId: number;
  onDrop: (personId: number, relationshipType: string, targetPersonId: number) => void;
}

export function DropZoneNode({ data }: NodeProps) {
  const d = data as unknown as DropZoneNodeData;
  const [isOver, setIsOver] = useState(false);

  return (
    // biome-ignore lint/a11y/useSemanticElements: React Flow custom node requires div container for Handle children
    <div
      role="region"
      aria-label={d.label}
      className={`flex flex-col items-center justify-center w-[140px] h-[80px] rounded-xl border-2 border-dashed transition-colors ${
        isOver ? "border-blue-500 bg-blue-500/10" : "border-gray-600 bg-gray-800/40"
      }`}
      onDragOver={(e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = "move";
        setIsOver(true);
      }}
      onDragLeave={() => setIsOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setIsOver(false);
        const droppedPersonId = e.dataTransfer.getData("application/person-id");
        if (droppedPersonId) {
          d.onDrop(Number(droppedPersonId), d.relationshipType, d.targetPersonId);
        }
      }}
    >
      <Plus className={`w-6 h-6 ${isOver ? "text-blue-400" : "text-gray-500"}`} />
      <span className={`text-xs mt-1 ${isOver ? "text-blue-400" : "text-gray-500"}`}>{d.label}</span>
      <Handle type="target" position={Position.Top} className="!opacity-0 !w-0 !h-0 !min-w-0 !min-h-0 !border-0" />
      <Handle type="source" position={Position.Bottom} className="!opacity-0 !w-0 !h-0 !min-w-0 !min-h-0 !border-0" />
      <Handle id="left" type="target" position={Position.Left} className="!opacity-0 !w-0 !h-0 !min-w-0 !min-h-0 !border-0" />
      <Handle id="right" type="source" position={Position.Right} className="!opacity-0 !w-0 !h-0 !min-w-0 !min-h-0 !border-0" />
    </div>
  );
}
