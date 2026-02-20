import { Handle, type NodeProps, Position } from "@xyflow/react";

export interface PersonNodeData {
  name: string;
  detectionId: number | null;
  isCenter: boolean;
  relation: string;
  personId: number;
  onRecenter: (personId: number) => void;
}

export function PersonNode({ data }: NodeProps) {
  const d = data as PersonNodeData;
  return (
    // biome-ignore lint/a11y/useSemanticElements: React Flow custom node requires div container for Handle children
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
        <img src={`/api/face/${d.detectionId}`} alt={d.name} className="w-9 h-9 rounded-full object-cover" />
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
      <Handle type="target" position={Position.Top} className="!bg-gray-500 !w-2 !h-2 !border-0" />
      <Handle type="source" position={Position.Bottom} className="!bg-gray-500 !w-2 !h-2 !border-0" />
      <Handle id="left" type="target" position={Position.Left} className="!bg-gray-500 !w-2 !h-2 !border-0" />
      <Handle id="right" type="source" position={Position.Right} className="!bg-gray-500 !w-2 !h-2 !border-0" />
    </div>
  );
}
