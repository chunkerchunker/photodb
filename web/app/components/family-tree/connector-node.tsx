import { Handle, Position } from "@xyflow/react";
import type React from "react";

const HANDLE_CLS = "!opacity-0 !w-0 !h-0 !min-w-0 !min-h-0 !border-0";

// Force all handles to the exact center of the node, overriding React Flow's
// default position-based offsets so the partner line and children lines meet
// at a single point with no visible notch.
const CENTER: React.CSSProperties = {
  left: "50%",
  top: "50%",
  right: "auto",
  bottom: "auto",
  transform: "translate(-50%, -50%)",
};

/** Invisible junction node placed on the partner line; children connect from its bottom handle. */
export function ConnectorNode() {
  return (
    <div style={{ width: 2, height: 2 }}>
      <Handle id="left" type="target" position={Position.Left} className={HANDLE_CLS} style={CENTER} />
      <Handle id="right" type="source" position={Position.Right} className={HANDLE_CLS} style={CENTER} />
      <Handle id="bottom" type="source" position={Position.Bottom} className={HANDLE_CLS} style={CENTER} />
    </div>
  );
}
