import type { Edge, Node } from "@xyflow/react";
import type { FamilyMember, PersonParentRow, PersonPartnershipRow } from "./db.server";

export const NODE_W = 140;
export const NODE_H = 80;
export const H_GAP = 40;
export const V_GAP = 100;

const GENERATION_LABELS: Record<number, string> = {
  "-3": "Great-Grandparents",
  "-2": "Grandparents",
  "-1": "Parents",
  0: "You + Siblings",
  1: "Children",
  2: "Grandchildren",
  3: "Great-Grandchildren",
};

interface LayoutInput {
  centerId: number;
  centerName: string;
  centerDetectionId: number | null;
  familyMembers: FamilyMember[];
  parents: PersonParentRow[];
  centerParentCount: number;
  partnerships: PersonPartnershipRow[];
}

export interface GenerationInfo {
  gen: number;
  label: string;
  y: number;
}

export function computeFamilyTreeLayout(input: LayoutInput): {
  nodes: Node[];
  edges: Edge[];
  generations: GenerationInfo[];
} {
  const { centerId, centerName, centerDetectionId, familyMembers, parents, centerParentCount, partnerships } = input;
  const nodes: Node[] = [];
  const edges: Edge[] = [];

  // Deduplicate family members by person_id (SQL function can return dupes for multi-relation people)
  // Prefer 'self' relation, then first occurrence
  // Use Number() because pg may return person_id as string from SQL functions
  const seen = new Map<number, FamilyMember>();
  for (const fm of familyMembers) {
    const pid = Number(fm.person_id);
    const existing = seen.get(pid);
    if (!existing || fm.relation === "self") {
      seen.set(pid, { ...fm, person_id: pid });
    }
  }

  // Ensure center person is in familyMembers (get_family_tree may return empty if no closure rows exist)
  if (!seen.has(centerId)) {
    seen.set(centerId, {
      person_id: centerId,
      display_name: centerName,
      relation: "self",
      generation_offset: 0,
      is_placeholder: false,
      detection_id: centerDetectionId,
    });
  }

  const members = [...seen.values()];

  // Group family members by generation
  const byGeneration = new Map<number, FamilyMember[]>();
  for (const fm of members) {
    const gen = fm.generation_offset;
    if (!byGeneration.has(gen)) byGeneration.set(gen, []);
    byGeneration.get(gen)?.push(fm);
  }

  // Build membership set for edge validation
  const memberIds = new Set(members.map((fm) => fm.person_id));

  // Determine generation range — always include at least -1, 0, 1 for drop zones
  const allGens = [...byGeneration.keys()];
  const minGen = Math.min(-1, ...allGens);
  const maxGen = Math.max(1, ...allGens);
  const generations: GenerationInfo[] = [];

  // Position nodes by generation
  for (let gen = minGen; gen <= maxGen; gen++) {
    const genMembers = byGeneration.get(gen) ?? [];
    const y = gen * (NODE_H + V_GAP);

    if (gen === 0) {
      // Gen 0: center person at x=0, partners to the right, siblings to the left
      const center = genMembers.find((fm) => Number(fm.person_id) === centerId);
      const partnerIds = new Set(
        partnerships.flatMap((p) => {
          if (Number(p.person1_id) === centerId) return [Number(p.person2_id)];
          if (Number(p.person2_id) === centerId) return [Number(p.person1_id)];
          return [];
        }),
      );
      const partners = genMembers.filter((fm) => partnerIds.has(Number(fm.person_id)));
      const siblings = genMembers.filter(
        (fm) => Number(fm.person_id) !== centerId && !partnerIds.has(Number(fm.person_id)),
      );

      // Center person at x = -NODE_W/2 (visual center at 0)
      if (center) {
        nodes.push({
          id: `person-${center.person_id}`,
          type: center.is_placeholder ? "placeholder" : "person",
          position: { x: -NODE_W / 2, y },
          data: {
            name: center.display_name,
            detectionId: center.detection_id,
            isCenter: true,
            relation: center.relation,
            personId: center.person_id,
          },
        });
      }

      // Partners to the right of center person
      for (let i = 0; i < partners.length; i++) {
        const fm = partners[i];
        const x = -NODE_W / 2 + (i + 1) * (NODE_W + H_GAP);
        nodes.push({
          id: `person-${fm.person_id}`,
          type: fm.is_placeholder ? "placeholder" : "person",
          position: { x, y },
          data: {
            name: fm.display_name,
            detectionId: fm.detection_id,
            isCenter: false,
            relation: fm.relation,
            personId: fm.person_id,
          },
        });
      }

      // Siblings to the left of center person
      for (let i = 0; i < siblings.length; i++) {
        const fm = siblings[i];
        const x = -NODE_W / 2 - (i + 1) * (NODE_W + H_GAP);
        nodes.push({
          id: `person-${fm.person_id}`,
          type: fm.is_placeholder ? "placeholder" : "person",
          position: { x, y },
          data: {
            name: fm.display_name,
            detectionId: fm.detection_id,
            isCenter: false,
            relation: fm.relation,
            personId: fm.person_id,
          },
        });
      }
    } else {
      // Other generations: center around x=0
      const totalWidth = genMembers.length * NODE_W + (genMembers.length - 1) * H_GAP;
      const startX = -totalWidth / 2;

      for (let i = 0; i < genMembers.length; i++) {
        const fm = genMembers[i];
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

    // Collect generation info for sticky labels
    const label = GENERATION_LABELS[gen] ?? `Generation ${gen > 0 ? "+" : ""}${gen}`;
    generations.push({ gen, label, y });
  }

  // Create edges from parent relationships
  for (const p of parents) {
    if (memberIds.has(Number(p.parent_id)) && memberIds.has(Number(p.person_id))) {
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
    if (memberIds.has(Number(p.person1_id)) && memberIds.has(Number(p.person2_id))) {
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

  // --- Drop zones for center person's missing relationships ---

  // Parents: add drop zones for missing parents (up to 2 total)
  // parents array is already scoped to centerId by the loader query
  // centerParentCount is passed from the loader (scoped to center person's direct parents)
  const dropZonesNeeded = Math.max(0, 2 - centerParentCount);
  const parentGen = -1;
  const parentY = parentGen * (NODE_H + V_GAP);

  if (dropZonesNeeded > 0) {
    // Reposition entire parent row (existing parents + drop zones) centered at x=0
    const totalSlots = centerParentCount + dropZonesNeeded;
    const totalWidth = totalSlots * NODE_W + (totalSlots - 1) * H_GAP;
    const startX = -totalWidth / 2;

    // Reposition existing parent nodes to centered positions
    const existingParentNodes = nodes.filter((n) => n.position.y === parentY && n.type !== "dropZone");
    for (let i = 0; i < existingParentNodes.length; i++) {
      existingParentNodes[i].position.x = startX + i * (NODE_W + H_GAP);
    }

    // Add drop zones in the remaining slots
    for (let i = 0; i < dropZonesNeeded; i++) {
      const slotIndex = centerParentCount + i;
      const dropX = startX + slotIndex * (NODE_W + H_GAP);
      const dropId = `drop-parent-${centerId}-${i}`;
      nodes.push({
        id: dropId,
        type: "dropZone",
        position: { x: dropX, y: parentY },
        data: {
          label: "Add parent",
          relationshipType: "parent",
          targetPersonId: centerId,
        },
      });
      edges.push({
        id: `drop-edge-${dropId}`,
        source: dropId,
        target: `person-${centerId}`,
        type: "smoothstep",
        style: { stroke: "#6b7280", strokeWidth: 1, strokeDasharray: "6 4" },
      });
    }
  }

  // Children: always show a drop zone, center entire row (children + drop zone) at x=0
  const childGen = 1;
  const childY = childGen * (NODE_H + V_GAP);
  const existingChildNodes = nodes.filter((n) => n.position.y === childY && n.type !== "dropZone");
  const totalChildSlots = existingChildNodes.length + 1; // existing children + 1 drop zone
  const childTotalWidth = totalChildSlots * NODE_W + (totalChildSlots - 1) * H_GAP;
  const childStartX = -childTotalWidth / 2;

  // Reposition existing child nodes to centered positions
  for (let i = 0; i < existingChildNodes.length; i++) {
    existingChildNodes[i].position.x = childStartX + i * (NODE_W + H_GAP);
  }

  // Drop zone in the last slot
  const childDropX = childStartX + existingChildNodes.length * (NODE_W + H_GAP);

  nodes.push({
    id: `drop-child-${centerId}`,
    type: "dropZone",
    position: { x: childDropX, y: childY },
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

  // Partner: drop zone if no partner
  const hasPartner = partnerships.some((p) => Number(p.person1_id) === centerId || Number(p.person2_id) === centerId);
  if (!hasPartner) {
    // Place partner drop zone to the right of all gen-0 nodes
    const gen0Nodes = nodes.filter((n) => n.position.y === 0 && n.type !== "generationLabel");
    const partnerDropX =
      gen0Nodes.length > 0 ? Math.max(...gen0Nodes.map((n) => n.position.x)) + NODE_W + H_GAP : NODE_W + H_GAP;

    nodes.push({
      id: `drop-partner-${centerId}`,
      type: "dropZone",
      position: {
        x: partnerDropX,
        y: 0,
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

  // Sibling: drop zone — place to the left of leftmost gen-0 node
  const centerY = 0;
  const gen0NonLabelNodes = nodes.filter(
    (n) => n.position.y === centerY && n.type !== "generationLabel" && n.id !== `drop-partner-${centerId}`,
  );
  const siblingDropX =
    gen0NonLabelNodes.length > 0
      ? Math.min(...gen0NonLabelNodes.map((n) => n.position.x)) - NODE_W - H_GAP
      : -(NODE_W + H_GAP);

  nodes.push({
    id: `drop-sibling-${centerId}`,
    type: "dropZone",
    position: { x: siblingDropX, y: centerY },
    data: {
      label: "Add sibling",
      relationshipType: "sibling",
      targetPersonId: centerId,
    },
  });

  return { nodes, edges, generations };
}
