import { Handle, type NodeProps, Position } from "@xyflow/react";

export interface PlaceholderNodeData {
  name: string;
  personId: number;
  onRecenter: (personId: number) => void;
}

export function PlaceholderNode({ data }: NodeProps) {
  const d = data as PlaceholderNodeData;
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
      <Handle type="target" position={Position.Top} className="!bg-gray-500 !w-2 !h-2 !border-0" />
      <Handle type="source" position={Position.Bottom} className="!bg-gray-500 !w-2 !h-2 !border-0" />
      <Handle id="left" type="target" position={Position.Left} className="!bg-gray-500 !w-2 !h-2 !border-0" />
      <Handle id="right" type="source" position={Position.Right} className="!bg-gray-500 !w-2 !h-2 !border-0" />
    </div>
  );
}
