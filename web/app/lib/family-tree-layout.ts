import type { Edge, Node } from "@xyflow/react";
import type { FamilyMember, PersonParentRow, PersonPartnershipRow } from "./db.server";

const NODE_W = 140;
const NODE_H = 80;
const H_GAP = 40;
const V_GAP = 100;

interface LayoutInput {
  centerId: number;
  familyMembers: FamilyMember[];
  parents: PersonParentRow[];
  partnerships: PersonPartnershipRow[];
}

export function computeFamilyTreeLayout(input: LayoutInput): {
  nodes: Node[];
  edges: Edge[];
} {
  const { centerId, familyMembers, parents, partnerships } = input;
  const nodes: Node[] = [];
  const edges: Edge[] = [];

  // Group family members by generation
  const byGeneration = new Map<number, FamilyMember[]>();
  for (const fm of familyMembers) {
    const gen = fm.generation_offset;
    if (!byGeneration.has(gen)) byGeneration.set(gen, []);
    byGeneration.get(gen)?.push(fm);
  }

  // Build membership set for edge validation
  const memberIds = new Set(familyMembers.map((fm) => fm.person_id));

  // Position nodes by generation
  const sortedGens = [...byGeneration.keys()].sort((a, b) => a - b);

  for (const gen of sortedGens) {
    const members = byGeneration.get(gen) ?? [];
    const y = gen * (NODE_H + V_GAP);
    const totalWidth = members.length * NODE_W + (members.length - 1) * H_GAP;
    const startX = -totalWidth / 2;

    for (let i = 0; i < members.length; i++) {
      const fm = members[i];
      const x = startX + i * (NODE_W + H_GAP);
      const isCenter = fm.person_id === centerId;

      const nodeType = fm.is_placeholder ? "placeholder" : "person";
      nodes.push({
        id: `person-${fm.person_id}`,
        type: nodeType,
        position: { x, y },
        data: {
          name: fm.display_name,
          detectionId: fm.detection_id,
          isCenter,
          relation: fm.relation,
          personId: fm.person_id,
        },
      });
    }
  }

  // Create edges from parent relationships
  for (const p of parents) {
    if (memberIds.has(p.parent_id) && memberIds.has(p.person_id)) {
      edges.push({
        id: `parent-${p.parent_id}-${p.person_id}`,
        source: `person-${p.parent_id}`,
        target: `person-${p.person_id}`,
        type: "smoothstep",
        style: { stroke: "#6b7280", strokeWidth: 2 },
      });
    }
  }

  // Create edges for partnerships (horizontal)
  for (const p of partnerships) {
    if (memberIds.has(p.person1_id) && memberIds.has(p.person2_id)) {
      edges.push({
        id: `partner-${p.person1_id}-${p.person2_id}`,
        source: `person-${p.person1_id}`,
        sourceHandle: "right",
        target: `person-${p.person2_id}`,
        targetHandle: "left",
        type: "straight",
        style: { stroke: "#6b7280", strokeWidth: 2 },
      });
    }
  }

  // Add drop zones for missing relationships of center person
  const centerMember = familyMembers.find((fm) => fm.person_id === centerId);
  if (centerMember) {
    const centerParents = parents.filter((p) => p.person_id === centerId);
    // Drop zone for missing parents (up to 2)
    if (centerParents.length < 2) {
      const parentGen = -1;
      const existingParentNodes = nodes.filter((n) => n.position.y === parentGen * (NODE_H + V_GAP));
      const dropX =
        existingParentNodes.length > 0 ? Math.max(...existingParentNodes.map((n) => n.position.x)) + NODE_W + H_GAP : 0;

      nodes.push({
        id: `drop-parent-${centerId}`,
        type: "dropZone",
        position: { x: dropX, y: parentGen * (NODE_H + V_GAP) },
        data: {
          label: "Add parent",
          relationshipType: "parent",
          targetPersonId: centerId,
        },
      });
      edges.push({
        id: `drop-edge-parent-${centerId}`,
        source: `drop-parent-${centerId}`,
        target: `person-${centerId}`,
        type: "smoothstep",
        style: { stroke: "#6b7280", strokeWidth: 1, strokeDasharray: "6 4" },
      });
    }

    // Drop zone for adding a child
    const childGen = 1;
    const existingChildNodes = nodes.filter((n) => n.position.y === childGen * (NODE_H + V_GAP));
    const childDropX =
      existingChildNodes.length > 0 ? Math.max(...existingChildNodes.map((n) => n.position.x)) + NODE_W + H_GAP : 0;

    nodes.push({
      id: `drop-child-${centerId}`,
      type: "dropZone",
      position: { x: childDropX, y: childGen * (NODE_H + V_GAP) },
      data: {
        label: "Add child",
        relationshipType: "child",
        targetPersonId: centerId,
      },
    });
    edges.push({
      id: `drop-edge-child-${centerId}`,
      source: `person-${centerId}`,
      target: `drop-child-${centerId}`,
      type: "smoothstep",
      style: { stroke: "#6b7280", strokeWidth: 1, strokeDasharray: "6 4" },
    });

    // Drop zone for adding a partner (if none exists)
    const hasPartner = partnerships.some((p) => p.person1_id === centerId || p.person2_id === centerId);
    if (!hasPartner) {
      const centerNode = nodes.find((n) => n.id === `person-${centerId}`);
      if (centerNode) {
        nodes.push({
          id: `drop-partner-${centerId}`,
          type: "dropZone",
          position: {
            x: centerNode.position.x + NODE_W + H_GAP,
            y: centerNode.position.y,
          },
          data: {
            label: "Add partner",
            relationshipType: "partner",
            targetPersonId: centerId,
          },
        });
        edges.push({
          id: `drop-edge-partner-${centerId}`,
          source: `person-${centerId}`,
          sourceHandle: "right",
          target: `drop-partner-${centerId}`,
          targetHandle: "left",
          type: "straight",
          style: { stroke: "#6b7280", strokeWidth: 1, strokeDasharray: "6 4" },
        });
      }
    }
  }

  return { nodes, edges };
}
