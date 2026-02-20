# Family Tree Page Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an interactive family tree page at `person/:id/family` using React Flow with drag-and-drop relationship management, placeholder persons, and re-centering navigation.

**Architecture:** New React Router route with a React Flow canvas rendering family tree data from the existing `get_family_tree()` SQL function. Custom node types for persons, placeholders, and drop zones. Right sidebar with draggable person cards. API routes for relationship CRUD that delegate to existing Python repository methods via direct PostgreSQL queries.

**Tech Stack:** React Flow (`@xyflow/react`), React Router v7, Tailwind v4, shadcn/ui, PostgreSQL (direct via `pg`), Lucide icons

**Design doc:** `docs/plans/2026-02-19-family-tree-page-design.md`

---

## Task 1: Install React Flow and Register Route

**Files:**
- Modify: `web/package.json`
- Modify: `web/app/routes.ts:51-63` (add new route alongside person routes)
- Create: `web/app/routes/person.$id.family-tree.tsx` (minimal placeholder)

**Step 1: Install React Flow**

```bash
cd web && pnpm add @xyflow/react
```

**Step 2: Register the route**

In `web/app/routes.ts`, add after line 60 (`person/:id/wall` route):

```typescript
route("person/:id/family-tree", "routes/person.$id.family-tree.tsx"),
```

**Step 3: Create placeholder route file**

Create `web/app/routes/person.$id.family-tree.tsx`:

```tsx
import { requireCollectionId } from "~/lib/auth.server";
import { getPersonById } from "~/lib/db.server";
import type { Route } from "./+types/person.$id.family-tree";

export function meta({ data }: Route.MetaArgs) {
  const personName = data?.person?.person_name || "Person";
  return [
    { title: `Storyteller - ${personName} - Family Tree` },
    { name: "description", content: `${personName}'s family tree` },
  ];
}

export async function loader({ request, params }: Route.LoaderArgs) {
  const { collectionId } = await requireCollectionId(request);
  const personId = params.id;
  if (!personId) {
    throw new Response("Person ID required", { status: 400 });
  }
  const person = await getPersonById(collectionId, personId);
  if (!person) {
    throw new Response("Person not found", { status: 404 });
  }
  return { person };
}

export default function FamilyTreePage({ loaderData }: Route.ComponentProps) {
  const { person } = loaderData;
  return (
    <div className="h-screen bg-gray-900 text-white flex items-center justify-center">
      <p>Family tree for {person.person_name} — coming soon</p>
    </div>
  );
}
```

**Step 4: Verify it builds**

```bash
cd web && pnpm build
```

Expected: Build succeeds with no errors.

**Step 5: Commit**

```bash
git add web/package.json web/pnpm-lock.yaml web/app/routes.ts web/app/routes/person.\$id.family-tree.tsx
git commit -m "feat: scaffold family tree route with React Flow dependency"
```

---

## Task 2: Add Genealogy Queries to db.server.ts

**Files:**
- Modify: `web/app/lib/db.server.ts` (add new functions after existing person queries)

**Context:** The PostgreSQL function `get_family_tree(center_id, max_generations, include_placeholders)` already exists (migration 019). It returns rows with: `person_id`, `display_name`, `relation`, `generation_offset`, `is_placeholder`. The `person_parent`, `person_partnership`, and `person_siblings` tables/views also exist. We need TypeScript wrappers.

**Step 1: Add TypeScript types and query functions**

Add to the end of `web/app/lib/db.server.ts`:

```typescript
// --- Genealogy queries ---

export interface FamilyMember {
  person_id: number;
  display_name: string;
  relation: string;
  generation_offset: number;
  is_placeholder: boolean;
  detection_id: number | null;
}

export interface PersonParentRow {
  id: number;
  person_id: number;
  parent_id: number;
  parent_role: string;
  is_biological: boolean;
}

export interface PersonPartnershipRow {
  id: number;
  person1_id: number;
  person2_id: number;
  partnership_type: string;
  start_year: number | null;
  end_year: number | null;
  is_current: boolean;
}

export async function getFamilyTree(
  collectionId: number,
  personId: string,
  maxGenerations = 3,
): Promise<FamilyMember[]> {
  const result = await pool.query(
    `SELECT ft.person_id, ft.display_name, ft.relation, ft.generation_offset, ft.is_placeholder,
            COALESCE(p.representative_detection_id, (
              SELECT pd.id FROM person_detection pd
              JOIN cluster c ON c.id = pd.cluster_id
              WHERE c.person_id = ft.person_id AND c.collection_id = $1
              ORDER BY pd.face_confidence DESC NULLS LAST LIMIT 1
            )) AS detection_id
     FROM get_family_tree($2::int, $3::int, true) ft
     LEFT JOIN person p ON p.id = ft.person_id
     WHERE p.collection_id = $1 OR ft.is_placeholder = true`,
    [collectionId, personId, maxGenerations],
  );
  return result.rows;
}

export async function getPersonParents(
  collectionId: number,
  personId: string,
): Promise<PersonParentRow[]> {
  const result = await pool.query(
    `SELECT pp.id, pp.person_id, pp.parent_id, pp.parent_role, pp.is_biological
     FROM person_parent pp
     JOIN person p ON p.id = pp.parent_id
     WHERE pp.person_id = $2 AND (p.collection_id = $1 OR p.is_placeholder = true)
     ORDER BY pp.parent_role, pp.parent_id`,
    [collectionId, personId],
  );
  return result.rows;
}

export async function getPersonPartnerships(
  collectionId: number,
  personId: string,
): Promise<PersonPartnershipRow[]> {
  const result = await pool.query(
    `SELECT pp.id, pp.person1_id, pp.person2_id, pp.partnership_type, pp.start_year, pp.end_year, pp.is_current
     FROM person_partnership pp
     WHERE pp.person1_id = $2::int OR pp.person2_id = $2::int
     ORDER BY pp.is_current DESC, pp.start_year DESC NULLS LAST`,
    [collectionId, personId],
  );
  return result.rows;
}

export async function addPersonParent(
  collectionId: number,
  personId: string,
  parentId: string,
  parentRole: string = "parent",
): Promise<{ success: boolean }> {
  await pool.query(
    `INSERT INTO person_parent (person_id, parent_id, parent_role, source)
     VALUES ($1::int, $2::int, $3, 'user')
     ON CONFLICT (person_id, parent_id) DO UPDATE SET parent_role = $3`,
    [personId, parentId, parentRole],
  );
  await pool.query(`SELECT refresh_genealogy_closures()`);
  return { success: true };
}

export async function removePersonParent(
  personId: string,
  parentId: string,
): Promise<{ success: boolean }> {
  await pool.query(
    `DELETE FROM person_parent WHERE person_id = $1::int AND parent_id = $2::int`,
    [personId, parentId],
  );
  await pool.query(`SELECT refresh_genealogy_closures()`);
  return { success: true };
}

export async function addPersonPartnership(
  personId: string,
  partnerId: string,
  partnershipType: string = "partner",
): Promise<{ success: boolean }> {
  const p1 = Math.min(Number(personId), Number(partnerId));
  const p2 = Math.max(Number(personId), Number(partnerId));
  await pool.query(
    `INSERT INTO person_partnership (person1_id, person2_id, partnership_type, is_current)
     VALUES ($1, $2, $3, true)
     ON CONFLICT (person1_id, person2_id, COALESCE(start_year, 0)) DO UPDATE SET partnership_type = $3`,
    [p1, p2, partnershipType],
  );
  await pool.query(`SELECT refresh_genealogy_closures()`);
  return { success: true };
}

export async function removePersonPartnership(
  personId: string,
  partnerId: string,
): Promise<{ success: boolean }> {
  const p1 = Math.min(Number(personId), Number(partnerId));
  const p2 = Math.max(Number(personId), Number(partnerId));
  await pool.query(
    `DELETE FROM person_partnership WHERE person1_id = $1 AND person2_id = $2`,
    [p1, p2],
  );
  await pool.query(`SELECT refresh_genealogy_closures()`);
  return { success: true };
}

export async function addPersonChild(
  collectionId: number,
  parentId: string,
  childId: string,
  parentRole: string = "parent",
): Promise<{ success: boolean }> {
  return addPersonParent(collectionId, childId, parentId, parentRole);
}

export async function createPlaceholderPerson(
  collectionId: number,
  name: string | null,
  gender: string = "U",
  description: string | null = null,
): Promise<{ id: number }> {
  const result = await pool.query(
    `INSERT INTO person (collection_id, first_name, is_placeholder, placeholder_description, gender, auto_created)
     VALUES ($1, $2, true, $3, $4, false)
     RETURNING id`,
    [collectionId, name, description, gender],
  );
  return { id: result.rows[0].id };
}
```

**Step 2: Verify it builds**

```bash
cd web && pnpm build
```

Expected: Build succeeds. (No runtime test yet — queries validated in later tasks.)

**Step 3: Commit**

```bash
git add web/app/lib/db.server.ts
git commit -m "feat: add genealogy query functions to db.server.ts"
```

---

## Task 3: Create Custom React Flow Node Components

**Files:**
- Create: `web/app/components/family-tree/person-node.tsx`
- Create: `web/app/components/family-tree/placeholder-node.tsx`
- Create: `web/app/components/family-tree/drop-zone-node.tsx`

**Context:** React Flow custom nodes receive `{ data, id }` props and must be wrapped in a container. Each node type renders differently. Reference the existing face avatar pattern: `<img src={/api/face/${detectionId}} />`.

**Step 1: Create PersonNode**

Create `web/app/components/family-tree/person-node.tsx`:

```tsx
import { Handle, Position, type NodeProps } from "@xyflow/react";

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
    <div
      className={`flex flex-col items-center justify-center w-[140px] h-[80px] rounded-xl border ${
        d.isCenter
          ? "border-blue-500 border-2 shadow-[0_0_16px_rgba(59,130,246,0.2)]"
          : "border-gray-600"
      } bg-gray-800 cursor-pointer`}
      onDoubleClick={() => d.onRecenter(d.personId)}
    >
      {d.detectionId ? (
        <img
          src={`/api/face/${d.detectionId}`}
          alt={d.name}
          className="w-9 h-9 rounded-full object-cover"
        />
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
```

**Step 2: Create PlaceholderNode**

Create `web/app/components/family-tree/placeholder-node.tsx`:

```tsx
import { Handle, Position, type NodeProps } from "@xyflow/react";

export interface PlaceholderNodeData {
  name: string;
  personId: number;
  onRecenter: (personId: number) => void;
}

export function PlaceholderNode({ data }: NodeProps) {
  const d = data as PlaceholderNodeData;
  return (
    <div
      className="flex flex-col items-center justify-center w-[140px] h-[80px] rounded-xl border border-dashed border-gray-500 bg-gray-800/80 cursor-pointer"
      onDoubleClick={() => d.onRecenter(d.personId)}
    >
      <div className="w-9 h-9 rounded-full bg-gray-600/60" />
      <span className="text-xs mt-1 text-gray-400 text-center truncate max-w-[120px] italic">
        {d.name}
      </span>
      <Handle type="target" position={Position.Top} className="!bg-gray-500 !w-2 !h-2 !border-0" />
      <Handle type="source" position={Position.Bottom} className="!bg-gray-500 !w-2 !h-2 !border-0" />
      <Handle id="left" type="target" position={Position.Left} className="!bg-gray-500 !w-2 !h-2 !border-0" />
      <Handle id="right" type="source" position={Position.Right} className="!bg-gray-500 !w-2 !h-2 !border-0" />
    </div>
  );
}
```

**Step 3: Create DropZoneNode**

Create `web/app/components/family-tree/drop-zone-node.tsx`:

```tsx
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { Plus } from "lucide-react";
import { useState } from "react";

export interface DropZoneNodeData {
  label: string;
  relationshipType: "parent" | "child" | "sibling" | "partner";
  targetPersonId: number;
  onDrop: (personId: number, relationshipType: string, targetPersonId: number) => void;
}

export function DropZoneNode({ data }: NodeProps) {
  const d = data as DropZoneNodeData;
  const [isOver, setIsOver] = useState(false);

  return (
    <div
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
      <span className={`text-xs mt-1 ${isOver ? "text-blue-400" : "text-gray-500"}`}>
        {d.label}
      </span>
      <Handle type="target" position={Position.Top} className="!bg-transparent !w-0 !h-0" />
      <Handle type="source" position={Position.Bottom} className="!bg-transparent !w-0 !h-0" />
      <Handle id="left" type="target" position={Position.Left} className="!bg-transparent !w-0 !h-0" />
      <Handle id="right" type="source" position={Position.Right} className="!bg-transparent !w-0 !h-0" />
    </div>
  );
}
```

**Step 4: Verify build**

```bash
cd web && pnpm build
```

**Step 5: Commit**

```bash
git add web/app/components/family-tree/
git commit -m "feat: add custom React Flow node components for family tree"
```

---

## Task 4: Tree Layout Algorithm

**Files:**
- Create: `web/app/lib/family-tree-layout.ts`

**Context:** The `get_family_tree()` function returns flat rows with `person_id`, `relation`, `generation_offset`, and `is_placeholder`. We need to convert this into React Flow nodes and edges with `(x, y)` positions. The layout groups by generation (vertical), centers horizontally, and inserts drop zone nodes for missing relationships.

**Step 1: Create the layout module**

Create `web/app/lib/family-tree-layout.ts`:

```typescript
import type { Node, Edge } from "@xyflow/react";
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
    byGeneration.get(gen)!.push(fm);
  }

  // Build a set of parent-child relationships for edge creation
  const parentChildSet = new Set<string>();
  for (const p of parents) {
    parentChildSet.add(`${p.parent_id}->${p.person_id}`);
  }

  // Build partnership set
  const partnershipSet = new Set<string>();
  for (const p of partnerships) {
    partnershipSet.add(`${p.person1_id}<->${p.person2_id}`);
  }

  // Find all parent relationships across the tree (not just for the center person)
  // We'll derive edges from generation_offset: if person A at gen -1 and person B at gen 0,
  // and A's relation contains "parent"/"mother"/"father", connect them

  // Position nodes by generation
  const sortedGens = [...byGeneration.keys()].sort((a, b) => a - b);

  for (const gen of sortedGens) {
    const members = byGeneration.get(gen)!;
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
  // For each person in the tree, check if their parents are also in the tree
  const memberIds = new Set(familyMembers.map((fm) => fm.person_id));
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

  // Add drop zones for missing relationships
  const centerMember = familyMembers.find((fm) => fm.person_id === centerId);
  if (centerMember) {
    const centerParents = parents.filter((p) => p.person_id === centerId);
    // Drop zone for missing parents (center person can have up to 2)
    if (centerParents.length < 2) {
      const parentGen = -1;
      const existingParentNodes = nodes.filter(
        (n) => n.position.y === parentGen * (NODE_H + V_GAP),
      );
      const dropX =
        existingParentNodes.length > 0
          ? Math.max(...existingParentNodes.map((n) => n.position.x)) + NODE_W + H_GAP
          : 0;

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
    const existingChildNodes = nodes.filter(
      (n) => n.position.y === childGen * (NODE_H + V_GAP),
    );
    const childDropX =
      existingChildNodes.length > 0
        ? Math.max(...existingChildNodes.map((n) => n.position.x)) + NODE_W + H_GAP
        : 0;

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

    // Drop zone for adding a partner
    const hasPartner = partnerships.some(
      (p) => p.person1_id === centerId || p.person2_id === centerId,
    );
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
```

**Step 2: Verify build**

```bash
cd web && pnpm build
```

**Step 3: Commit**

```bash
git add web/app/lib/family-tree-layout.ts
git commit -m "feat: add family tree layout algorithm for React Flow"
```

---

## Task 5: Build the Full Family Tree Page

**Files:**
- Modify: `web/app/routes/person.$id.family-tree.tsx` (replace placeholder with full implementation)

**Context:** This task wires together the loader (fetching family tree + parents + partnerships + person list), the React Flow canvas with custom nodes, and the right sidebar with draggable person cards. Reference the drag-and-drop pattern from React Flow docs: `onDrop` + `onDragOver` on the `<ReactFlow>` component, `draggable` + `onDragStart` on sidebar cards.

**Step 1: Update the loader to fetch all needed data**

Replace the loader in `web/app/routes/person.$id.family-tree.tsx`:

```tsx
import { requireCollectionId } from "~/lib/auth.server";
import {
  getFamilyTree,
  getPersonById,
  getPersonParents,
  getPersonPartnerships,
  getPersonsForCollection,
} from "~/lib/db.server";
import type { Route } from "./+types/person.$id.family-tree";

export async function loader({ request, params }: Route.LoaderArgs) {
  const { collectionId } = await requireCollectionId(request);
  const personId = params.id;
  if (!personId) {
    throw new Response("Person ID required", { status: 400 });
  }
  const person = await getPersonById(collectionId, personId);
  if (!person) {
    throw new Response("Person not found", { status: 404 });
  }

  const [familyMembers, parents, partnerships, persons] = await Promise.all([
    getFamilyTree(collectionId, personId, 3),
    getPersonParents(collectionId, personId),
    getPersonPartnerships(collectionId, personId),
    getPersonsForCollection(collectionId),
  ]);

  return { person, familyMembers, parents, partnerships, persons, collectionId };
}
```

**Step 2: Build the full component**

Replace the default export component in the same file:

```tsx
import { useCallback, useMemo, useState } from "react";
import { useNavigate, useFetcher } from "react-router";
import {
  ReactFlow,
  ReactFlowProvider,
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  useReactFlow,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { Search } from "lucide-react";
import { PersonNode } from "~/components/family-tree/person-node";
import { PlaceholderNode } from "~/components/family-tree/placeholder-node";
import { DropZoneNode } from "~/components/family-tree/drop-zone-node";
import { computeFamilyTreeLayout } from "~/lib/family-tree-layout";

const nodeTypes = {
  person: PersonNode,
  placeholder: PlaceholderNode,
  dropZone: DropZoneNode,
};

function FamilyTreeCanvas({ loaderData }: Route.ComponentProps) {
  const { person, familyMembers, parents, partnerships, persons, collectionId } = loaderData;
  const navigate = useNavigate();
  const fetcher = useFetcher();
  const { fitView } = useReactFlow();
  const [searchQuery, setSearchQuery] = useState("");

  const handleRecenter = useCallback(
    (personId: number) => {
      navigate(`/person/${personId}/family-tree`);
    },
    [navigate],
  );

  const handleDrop = useCallback(
    (droppedPersonId: number, relationshipType: string, targetPersonId: number) => {
      const action =
        relationshipType === "parent"
          ? `/api/person/${targetPersonId}/add-parent`
          : relationshipType === "child"
            ? `/api/person/${targetPersonId}/add-child`
            : `/api/person/${targetPersonId}/add-partner`;

      fetcher.submit(
        { relatedPersonId: String(droppedPersonId), role: relationshipType },
        { method: "post", action },
      );
    },
    [fetcher],
  );

  const { nodes: initialNodes, edges: initialEdges } = useMemo(() => {
    const layout = computeFamilyTreeLayout({
      centerId: Number(person.id),
      familyMembers,
      parents,
      partnerships,
    });
    // Inject callbacks into node data
    for (const node of layout.nodes) {
      if (node.type === "person" || node.type === "placeholder") {
        (node.data as any).onRecenter = handleRecenter;
      }
      if (node.type === "dropZone") {
        (node.data as any).onDrop = handleDrop;
      }
    }
    return layout;
  }, [person.id, familyMembers, parents, partnerships, handleRecenter, handleDrop]);

  const [nodes, , onNodesChange] = useNodesState(initialNodes);
  const [edges, , onEdgesChange] = useEdgesState(initialEdges);

  // Filter sidebar persons
  const filteredPersons = useMemo(() => {
    const q = searchQuery.toLowerCase();
    return persons.filter((p: any) => p.person_name?.toLowerCase().includes(q));
  }, [persons, searchQuery]);

  return (
    <div className="h-screen flex flex-col bg-gray-900">
      {/* Header */}
      <div className="h-14 bg-gray-950 flex items-center justify-between px-6 shrink-0">
        <div className="flex items-center gap-2 text-sm text-gray-400">
          <a href="/people/grid" className="hover:text-gray-200">People</a>
          <span>/</span>
          <a href={`/person/${person.id}/grid`} className="hover:text-gray-200">
            {person.person_name}
          </a>
          <span>/</span>
          <span className="text-gray-200">Family</span>
        </div>
        <div className="flex items-center gap-4 text-sm">
          <a href={`/person/${person.id}/grid`} className="text-gray-500 hover:text-gray-300">Grid</a>
          <a href={`/person/${person.id}/wall`} className="text-gray-500 hover:text-gray-300">Wall</a>
          <span className="text-gray-200 font-medium">Family</span>
        </div>
      </div>

      <div className="flex flex-1 min-h-0">
        {/* React Flow Canvas */}
        <div className="flex-1">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            nodeTypes={nodeTypes}
            fitView
            fitViewOptions={{ padding: 0.3 }}
            nodesDraggable={false}
            nodesConnectable={false}
            proOptions={{ hideAttribution: true }}
          >
            <MiniMap
              nodeColor={(n) => (n.type === "dropZone" ? "transparent" : "#6b7280")}
              maskColor="rgba(0,0,0,0.7)"
              className="!bg-gray-800/80 !border-gray-700"
            />
            <Controls className="!bg-gray-800 !border-gray-700 !text-gray-300" />
            <Background color="#333" gap={32} />
          </ReactFlow>
        </div>

        {/* Right Sidebar */}
        <div className="w-[280px] bg-gray-50 flex flex-col shrink-0 border-l border-gray-200">
          <div className="p-4 pb-2">
            <h2 className="text-sm font-semibold text-gray-900 mb-3">People</h2>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                placeholder="Search people..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-9 pr-3 py-2 text-sm bg-white border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>

          <div className="flex-1 overflow-y-auto px-4 pb-4 space-y-1">
            {filteredPersons.map((p: any) => (
              <div
                key={p.id}
                draggable
                onDragStart={(e) => {
                  e.dataTransfer.setData("application/person-id", String(p.id));
                  e.dataTransfer.effectAllowed = "move";
                }}
                className="flex items-center gap-3 p-2 rounded-lg bg-white hover:bg-blue-50 cursor-grab active:cursor-grabbing transition-colors"
              >
                {p.detection_id ? (
                  <img
                    src={`/api/face/${p.detection_id}`}
                    alt={p.person_name}
                    className="w-8 h-8 rounded-full object-cover shrink-0"
                  />
                ) : (
                  <div className="w-8 h-8 rounded-full bg-gray-300 shrink-0" />
                )}
                <span className="text-sm font-medium text-gray-900 truncate">
                  {p.person_name}
                </span>
              </div>
            ))}
          </div>

          <div className="p-4 pt-2 border-t border-gray-200">
            <button
              onClick={() => {
                fetcher.submit(
                  { name: "", gender: "U" },
                  { method: "post", action: `/api/person/create-placeholder` },
                );
              }}
              className="w-full py-2 text-sm font-medium text-gray-500 border border-dashed border-gray-300 rounded-lg hover:border-gray-400 hover:text-gray-700 transition-colors"
            >
              + New Placeholder Person
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function FamilyTreePage(props: Route.ComponentProps) {
  return (
    <ReactFlowProvider>
      <FamilyTreeCanvas {...props} />
    </ReactFlowProvider>
  );
}
```

**Step 3: Verify build**

```bash
cd web && pnpm build
```

**Step 4: Commit**

```bash
git add web/app/routes/person.\$id.family-tree.tsx
git commit -m "feat: build full family tree page with React Flow canvas and sidebar"
```

---

## Task 6: Create API Routes for Relationship CRUD

**Files:**
- Create: `web/app/routes/api.person.$id.add-parent.tsx`
- Create: `web/app/routes/api.person.$id.add-child.tsx`
- Create: `web/app/routes/api.person.$id.add-partner.tsx`
- Create: `web/app/routes/api.person.$id.remove-parent.tsx`
- Create: `web/app/routes/api.person.$id.remove-partner.tsx`
- Create: `web/app/routes/api.person.create-placeholder.tsx`
- Modify: `web/app/routes.ts` (register new API routes)

**Step 1: Create add-parent API route**

Create `web/app/routes/api.person.$id.add-parent.tsx`:

```tsx
import { requireCollectionId } from "~/lib/auth.server";
import { addPersonParent } from "~/lib/db.server";
import type { Route } from "./+types/api.person.$id.add-parent";

export async function action({ request, params }: Route.ActionArgs) {
  if (request.method !== "POST") {
    return Response.json({ success: false, message: "Method not allowed" }, { status: 405 });
  }
  const { collectionId } = await requireCollectionId(request);
  const personId = params.id;
  if (!personId) {
    return Response.json({ success: false, message: "Person ID required" }, { status: 400 });
  }
  const formData = await request.formData();
  const relatedPersonId = formData.get("relatedPersonId") as string;
  const role = (formData.get("role") as string) || "parent";
  if (!relatedPersonId) {
    return Response.json({ success: false, message: "Related person ID required" }, { status: 400 });
  }
  const result = await addPersonParent(collectionId, personId, relatedPersonId, role);
  return Response.json(result);
}
```

**Step 2: Create add-child API route**

Create `web/app/routes/api.person.$id.add-child.tsx`:

```tsx
import { requireCollectionId } from "~/lib/auth.server";
import { addPersonChild } from "~/lib/db.server";
import type { Route } from "./+types/api.person.$id.add-child";

export async function action({ request, params }: Route.ActionArgs) {
  if (request.method !== "POST") {
    return Response.json({ success: false, message: "Method not allowed" }, { status: 405 });
  }
  const { collectionId } = await requireCollectionId(request);
  const parentId = params.id;
  if (!parentId) {
    return Response.json({ success: false, message: "Person ID required" }, { status: 400 });
  }
  const formData = await request.formData();
  const relatedPersonId = formData.get("relatedPersonId") as string;
  const role = (formData.get("role") as string) || "parent";
  if (!relatedPersonId) {
    return Response.json({ success: false, message: "Related person ID required" }, { status: 400 });
  }
  const result = await addPersonChild(collectionId, parentId, relatedPersonId, role);
  return Response.json(result);
}
```

**Step 3: Create add-partner API route**

Create `web/app/routes/api.person.$id.add-partner.tsx`:

```tsx
import { requireCollectionId } from "~/lib/auth.server";
import { addPersonPartnership } from "~/lib/db.server";
import type { Route } from "./+types/api.person.$id.add-partner";

export async function action({ request, params }: Route.ActionArgs) {
  if (request.method !== "POST") {
    return Response.json({ success: false, message: "Method not allowed" }, { status: 405 });
  }
  await requireCollectionId(request);
  const personId = params.id;
  if (!personId) {
    return Response.json({ success: false, message: "Person ID required" }, { status: 400 });
  }
  const formData = await request.formData();
  const relatedPersonId = formData.get("relatedPersonId") as string;
  if (!relatedPersonId) {
    return Response.json({ success: false, message: "Related person ID required" }, { status: 400 });
  }
  const result = await addPersonPartnership(personId, relatedPersonId);
  return Response.json(result);
}
```

**Step 4: Create remove-parent API route**

Create `web/app/routes/api.person.$id.remove-parent.tsx`:

```tsx
import { requireCollectionId } from "~/lib/auth.server";
import { removePersonParent } from "~/lib/db.server";
import type { Route } from "./+types/api.person.$id.remove-parent";

export async function action({ request, params }: Route.ActionArgs) {
  if (request.method !== "POST") {
    return Response.json({ success: false, message: "Method not allowed" }, { status: 405 });
  }
  await requireCollectionId(request);
  const personId = params.id;
  if (!personId) {
    return Response.json({ success: false, message: "Person ID required" }, { status: 400 });
  }
  const formData = await request.formData();
  const parentId = formData.get("parentId") as string;
  if (!parentId) {
    return Response.json({ success: false, message: "Parent ID required" }, { status: 400 });
  }
  const result = await removePersonParent(personId, parentId);
  return Response.json(result);
}
```

**Step 5: Create remove-partner API route**

Create `web/app/routes/api.person.$id.remove-partner.tsx`:

```tsx
import { requireCollectionId } from "~/lib/auth.server";
import { removePersonPartnership } from "~/lib/db.server";
import type { Route } from "./+types/api.person.$id.remove-partner";

export async function action({ request, params }: Route.ActionArgs) {
  if (request.method !== "POST") {
    return Response.json({ success: false, message: "Method not allowed" }, { status: 405 });
  }
  await requireCollectionId(request);
  const personId = params.id;
  if (!personId) {
    return Response.json({ success: false, message: "Person ID required" }, { status: 400 });
  }
  const formData = await request.formData();
  const partnerId = formData.get("partnerId") as string;
  if (!partnerId) {
    return Response.json({ success: false, message: "Partner ID required" }, { status: 400 });
  }
  const result = await removePersonPartnership(personId, partnerId);
  return Response.json(result);
}
```

**Step 6: Create create-placeholder API route**

Create `web/app/routes/api.person.create-placeholder.tsx`:

```tsx
import { requireCollectionId } from "~/lib/auth.server";
import { createPlaceholderPerson } from "~/lib/db.server";
import type { Route } from "./+types/api.person.create-placeholder";

export async function action({ request }: Route.ActionArgs) {
  if (request.method !== "POST") {
    return Response.json({ success: false, message: "Method not allowed" }, { status: 405 });
  }
  const { collectionId } = await requireCollectionId(request);
  const formData = await request.formData();
  const name = (formData.get("name") as string) || null;
  const gender = (formData.get("gender") as string) || "U";
  const description = (formData.get("description") as string) || null;
  const result = await createPlaceholderPerson(collectionId, name, gender, description);
  return Response.json({ success: true, ...result });
}
```

**Step 7: Register all new API routes**

Add to `web/app/routes.ts` after the existing `api/person` routes:

```typescript
route("api/person/:id/add-parent", "routes/api.person.$id.add-parent.tsx"),
route("api/person/:id/add-child", "routes/api.person.$id.add-child.tsx"),
route("api/person/:id/add-partner", "routes/api.person.$id.add-partner.tsx"),
route("api/person/:id/remove-parent", "routes/api.person.$id.remove-parent.tsx"),
route("api/person/:id/remove-partner", "routes/api.person.$id.remove-partner.tsx"),
route("api/person/create-placeholder", "routes/api.person.create-placeholder.tsx"),
```

**Step 8: Verify build**

```bash
cd web && pnpm build
```

**Step 9: Commit**

```bash
git add web/app/routes/api.person.*.tsx web/app/routes.ts
git commit -m "feat: add API routes for family tree relationship CRUD"
```

---

## Task 7: Update Person Redirect to Include Family Tree Option

**Files:**
- Modify: `web/app/routes/person.$id.redirect.tsx` (add "family-tree" as a valid view mode)

**Context:** The redirect route at `person/:id` redirects to the user's preferred view (`grid` or `wall`). We need to add `family-tree` as a valid option but keep the default as-is. Check the existing redirect logic to understand how to extend it.

**Step 1: Read the redirect file and add family-tree support**

Read `web/app/routes/person.$id.redirect.tsx` and add `"family-tree"` to the valid view modes list, so links to `/person/:id` with `?view=family-tree` work. No other changes needed — the family tree is accessed via direct links from the grid/wall views.

**Step 2: Verify build**

```bash
cd web && pnpm build
```

**Step 3: Commit**

```bash
git add web/app/routes/person.\$id.redirect.tsx
git commit -m "feat: support family-tree view mode in person redirect"
```

---

## Task 8: Integration Test and Polish

**Files:**
- All files from previous tasks (no new files)

**Step 1: Start the dev server and test manually**

```bash
cd web && pnpm dev
```

- Navigate to a person page (e.g., `/person/1/grid`)
- Click the "Family" tab → should load the family tree page
- Verify: React Flow canvas renders with the center person
- Verify: Right sidebar shows person list with search
- Verify: Drag a person from sidebar → drop zone highlights on hover
- Verify: Drop onto "Add parent" creates the relationship
- Verify: Double-click a relative → page navigates and re-centers

**Step 2: Fix any TypeScript errors**

```bash
cd web && pnpm build
```

Address any type errors that surface from the full build.

**Step 3: Run linting**

```bash
cd web && pnpm check
```

Fix any Biome issues.

**Step 4: Commit**

```bash
git add -A
git commit -m "fix: polish family tree page types and lint"
```

---

## Summary

| Task | Description | Est. Complexity |
|------|-------------|----------------|
| 1 | Install React Flow, register route, scaffold page | Low |
| 2 | Add genealogy DB queries to db.server.ts | Medium |
| 3 | Create custom React Flow node components | Medium |
| 4 | Tree layout algorithm | Medium |
| 5 | Build full page (canvas + sidebar + wiring) | High |
| 6 | API routes for relationship CRUD | Medium |
| 7 | Update person redirect | Low |
| 8 | Integration test and polish | Medium |

Tasks 1-4 can proceed in parallel. Task 5 depends on 1-4. Task 6 can proceed in parallel with 5. Tasks 7-8 depend on 5+6.
