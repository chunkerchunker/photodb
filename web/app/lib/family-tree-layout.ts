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

  // Build adjacency maps from parent links
  const childrenOf = new Map<number, number[]>(); // parent_id → child person_ids
  const parentsOf = new Map<number, number[]>(); // person_id → parent_ids
  for (const p of parents) {
    const parentId = Number(p.parent_id);
    const childId = Number(p.person_id);
    if (!childrenOf.has(parentId)) childrenOf.set(parentId, []);
    childrenOf.get(parentId)?.push(childId);
    if (!parentsOf.has(childId)) parentsOf.set(childId, []);
    parentsOf.get(childId)?.push(parentId);
  }

  // Track positioned x for each person_id (center of node = x + NODE_W/2)
  const positionOf = new Map<number, number>();

  // Compute the horizontal width a person's subtree needs (memoized)
  const subtreeCache = new Map<string, number>();
  function getSubtreeWidth(personId: number, direction: "down" | "up"): number {
    const cacheKey = `${personId}-${direction}`;
    const cached = subtreeCache.get(cacheKey);
    if (cached !== undefined) return cached;

    const linked =
      direction === "down"
        ? (childrenOf.get(personId) ?? []).filter((id) => memberIds.has(id))
        : (parentsOf.get(personId) ?? []).filter((id) => memberIds.has(id));

    if (linked.length === 0) {
      subtreeCache.set(cacheKey, NODE_W);
      return NODE_W;
    }

    const childWidths = linked.map((id) => getSubtreeWidth(id, direction));
    const width = Math.max(NODE_W, childWidths.reduce((sum, w) => sum + w, 0) + (linked.length - 1) * H_GAP);
    subtreeCache.set(cacheKey, width);
    return width;
  }

  // Determine generation range — always include at least -1, 0, 1 for drop zones
  const allGens = [...byGeneration.keys()];
  const minGen = Math.min(-1, ...allGens);
  const maxGen = Math.max(1, ...allGens);
  const generations: GenerationInfo[] = [];

  // --- Gen 0 positioning ---
  const gen0Members = byGeneration.get(0) ?? [];
  const gen0Y = 0;

  const center = gen0Members.find((fm) => Number(fm.person_id) === centerId);
  const partnerIds = new Set(
    partnerships.flatMap((p) => {
      if (Number(p.person1_id) === centerId) return [Number(p.person2_id)];
      if (Number(p.person2_id) === centerId) return [Number(p.person1_id)];
      return [];
    }),
  );
  const gen0Partners = gen0Members.filter((fm) => partnerIds.has(Number(fm.person_id)));
  const gen0Siblings = gen0Members.filter(
    (fm) => Number(fm.person_id) !== centerId && !partnerIds.has(Number(fm.person_id)),
  );

  // Center person at x = -NODE_W/2 (visual center at 0)
  if (center) {
    const x = -NODE_W / 2;
    nodes.push({
      id: `person-${center.person_id}`,
      type: center.is_placeholder ? "placeholder" : "person",
      position: { x, y: gen0Y },
      data: {
        name: center.display_name,
        detectionId: center.detection_id,
        isCenter: true,
        relation: center.relation,
        personId: center.person_id,
      },
    });
    positionOf.set(centerId, x);
  }

  // Partners to the right of center person
  for (let i = 0; i < gen0Partners.length; i++) {
    const fm = gen0Partners[i];
    const x = -NODE_W / 2 + (i + 1) * (NODE_W + H_GAP);
    nodes.push({
      id: `person-${fm.person_id}`,
      type: fm.is_placeholder ? "placeholder" : "person",
      position: { x, y: gen0Y },
      data: {
        name: fm.display_name,
        detectionId: fm.detection_id,
        isCenter: false,
        relation: fm.relation,
        personId: fm.person_id,
      },
    });
    positionOf.set(Number(fm.person_id), x);
  }

  // Siblings to the left of center person
  for (let i = 0; i < gen0Siblings.length; i++) {
    const fm = gen0Siblings[i];
    const x = -NODE_W / 2 - (i + 1) * (NODE_W + H_GAP);
    nodes.push({
      id: `person-${fm.person_id}`,
      type: fm.is_placeholder ? "placeholder" : "person",
      position: { x, y: gen0Y },
      data: {
        name: fm.display_name,
        detectionId: fm.detection_id,
        isCenter: false,
        relation: fm.relation,
        personId: fm.person_id,
      },
    });
    positionOf.set(Number(fm.person_id), x);
  }

  // Gen 0 label — use center person's first/preferred name
  const centerFirstName = centerName.split(" ")[0] || centerName;
  generations.push({ gen: 0, label: `${centerFirstName} + Siblings`, y: gen0Y });

  // --- Helper: position a non-gen-0 generation relative to the adjacent generation ---
  function positionGeneration(gen: number, adjacentGen: number) {
    const genMembers = byGeneration.get(gen) ?? [];
    if (genMembers.length === 0) return;

    const y = gen * (NODE_H + V_GAP);
    const isDescendant = gen > adjacentGen; // walking down (children)

    // Group members by their connection key (sorted parent/child ids in adjacent gen)
    const groups = new Map<string, FamilyMember[]>();
    const ungrouped: FamilyMember[] = [];

    for (const fm of genMembers) {
      const pid = Number(fm.person_id);
      // Find connected person(s) in the adjacent generation
      let connectedIds: number[];
      if (isDescendant) {
        // This person's parents should be in adjacentGen
        connectedIds = (parentsOf.get(pid) ?? []).filter((id) => positionOf.has(id));
      } else {
        // This person's children should be in adjacentGen
        connectedIds = (childrenOf.get(pid) ?? []).filter((id) => positionOf.has(id));
      }

      if (connectedIds.length > 0) {
        const key = connectedIds.sort((a, b) => a - b).join(",");
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key)?.push(fm);
      } else {
        ungrouped.push(fm);
      }
    }

    // Compute target x for each group (average x of connected persons)
    const groupEntries: { key: string; members: FamilyMember[]; targetX: number }[] = [];
    for (const [key, members] of groups) {
      const connectedIds = key.split(",").map(Number);
      const avgX =
        connectedIds.reduce((sum, id) => sum + ((positionOf.get(id) ?? 0) + NODE_W / 2), 0) / connectedIds.length;
      groupEntries.push({ key, members, targetX: avgX });
    }

    // Sort groups by their target x position
    groupEntries.sort((a, b) => a.targetX - b.targetX);

    // Add ungrouped members as a separate group centered at x=0
    if (ungrouped.length > 0) {
      groupEntries.push({ key: "ungrouped", members: ungrouped, targetX: 0 });
      groupEntries.sort((a, b) => a.targetX - b.targetX);
    }

    // Position each group centered at its target x, using subtree widths for spacing
    const subtreeDir = isDescendant ? "down" : "up";
    interface PlacedGroup {
      members: FamilyMember[];
      positions: number[]; // x positions for each member
      allocLeft: number; // leftmost subtree allocation boundary
      allocRight: number; // rightmost subtree allocation boundary
    }
    const placedGroups: PlacedGroup[] = [];

    for (const group of groupEntries) {
      const n = group.members.length;
      const memberWidths = group.members.map((fm) => getSubtreeWidth(Number(fm.person_id), subtreeDir));
      const groupWidth = memberWidths.reduce((sum, w) => sum + w, 0) + (n - 1) * H_GAP;
      const startX = group.targetX - groupWidth / 2;

      // Center each node within its subtree-width allocation
      const positions: number[] = [];
      let x = startX;
      for (let i = 0; i < n; i++) {
        positions.push(x + memberWidths[i] / 2 - NODE_W / 2);
        x += memberWidths[i] + H_GAP;
      }
      placedGroups.push({ members: group.members, positions, allocLeft: startX, allocRight: startX + groupWidth });
    }

    // Resolve overlaps between adjacent groups using allocation boundaries
    for (let i = 1; i < placedGroups.length; i++) {
      const prev = placedGroups[i - 1];
      const curr = placedGroups[i];
      const overlap = prev.allocRight + H_GAP - curr.allocLeft;
      if (overlap > 0) {
        for (let j = 0; j < curr.positions.length; j++) {
          curr.positions[j] += overlap;
        }
        curr.allocLeft += overlap;
        curr.allocRight += overlap;
      }
    }

    // Create nodes and record positions
    for (const group of placedGroups) {
      for (let i = 0; i < group.members.length; i++) {
        const fm = group.members[i];
        const x = group.positions[i];
        nodes.push({
          id: `person-${fm.person_id}`,
          type: fm.is_placeholder ? "placeholder" : "person",
          position: { x, y },
          data: {
            name: fm.display_name,
            detectionId: fm.detection_id,
            isCenter: fm.person_id === centerId,
            relation: fm.relation,
            personId: fm.person_id,
          },
        });
        positionOf.set(Number(fm.person_id), x);
      }
    }
  }

  // --- Position descendant generations (gen 1, 2, 3, ...) walking down from gen 0 ---
  for (let gen = 1; gen <= maxGen; gen++) {
    positionGeneration(gen, gen - 1);
    const y = gen * (NODE_H + V_GAP);
    const label = GENERATION_LABELS[gen] ?? `Generation +${gen}`;
    generations.push({ gen, label, y });
  }

  // --- Position ancestor generations (gen -1, -2, -3, ...) walking up from gen 0 ---
  for (let gen = -1; gen >= minGen; gen--) {
    positionGeneration(gen, gen + 1);
    const y = gen * (NODE_H + V_GAP);
    const label = GENERATION_LABELS[gen] ?? `Generation ${gen}`;
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
  // centerParentCount is passed from the loader (scoped to center person's direct parents)
  const dropZonesNeeded = Math.max(0, 2 - centerParentCount);
  const parentY = -1 * (NODE_H + V_GAP);

  if (dropZonesNeeded > 0) {
    // Find center's existing parent nodes (already positioned by connection-based layout)
    const centerParentIds = parentsOf.get(centerId) ?? [];
    const centerParentNodes = nodes.filter(
      (n) => n.position.y === parentY && centerParentIds.some((pid) => n.id === `person-${pid}`),
    );

    // Place drop zones next to existing parents, or centered above center if none
    let dropBaseX: number;
    if (centerParentNodes.length > 0) {
      dropBaseX = Math.max(...centerParentNodes.map((n) => n.position.x)) + NODE_W + H_GAP;
    } else {
      // No existing parents — center the drop zones above center person
      const totalWidth = dropZonesNeeded * NODE_W + (dropZonesNeeded - 1) * H_GAP;
      dropBaseX = -totalWidth / 2;
    }

    for (let i = 0; i < dropZonesNeeded; i++) {
      const dropX = dropBaseX + i * (NODE_W + H_GAP);
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

  // Children: always show a drop zone next to center's existing children
  const childY = 1 * (NODE_H + V_GAP);
  const centerChildIds = childrenOf.get(centerId) ?? [];
  const centerChildNodes = nodes.filter(
    (n) => n.position.y === childY && centerChildIds.some((cid) => n.id === `person-${cid}`),
  );

  let childDropX: number;
  if (centerChildNodes.length > 0) {
    childDropX = Math.max(...centerChildNodes.map((n) => n.position.x)) + NODE_W + H_GAP;
  } else {
    // No children — place below center person
    childDropX = positionOf.get(centerId) ?? -NODE_W / 2;
  }

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
