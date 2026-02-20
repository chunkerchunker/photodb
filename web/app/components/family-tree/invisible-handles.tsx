import { Handle, Position } from "@xyflow/react";

const HANDLE_CLS = "!opacity-0 !w-0 !h-0 !min-w-0 !min-h-0 !border-0";

export function InvisibleHandles() {
  return (
    <>
      <Handle type="target" position={Position.Top} className={HANDLE_CLS} />
      <Handle type="source" position={Position.Bottom} className={HANDLE_CLS} />
      <Handle id="left" type="target" position={Position.Left} className={HANDLE_CLS} />
      <Handle id="right" type="source" position={Position.Right} className={HANDLE_CLS} />
    </>
  );
}
