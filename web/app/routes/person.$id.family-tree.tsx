import {
  Background,
  Controls,
  MiniMap,
  Panel,
  ReactFlow,
  ReactFlowProvider,
  useEdgesState,
  useNodesState,
  useReactFlow,
  useViewport,
} from "@xyflow/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useFetcher, useNavigate, useRevalidator } from "react-router";
import "@xyflow/react/dist/style.css";
import { ChevronRight, Search } from "lucide-react";
import { subsequenceMatch } from "~/lib/utils";
import { DropZoneNode } from "~/components/family-tree/drop-zone-node";
import { Header } from "~/components/header";
import { PersonNode } from "~/components/family-tree/person-node";
import { PlaceholderNode } from "~/components/family-tree/placeholder-node";
import { useRootData } from "~/hooks/use-root-data";
import { type GenerationInfo, H_GAP, NODE_H, NODE_W, V_GAP } from "~/lib/family-tree-layout";
import { requireCollectionId } from "~/lib/auth.server";
import {
  getFamilyParentLinks,
  getFamilyTree,
  getPersonById,
  getPersonParents,
  getPersonPartnerships,
  getPersonsForCollection,
} from "~/lib/db.server";
import { computeFamilyTreeLayout } from "~/lib/family-tree-layout";
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

  const [familyMembers, centerParents, partnerships, persons] = await Promise.all([
    getFamilyTree(collectionId, personId, 3),
    getPersonParents(collectionId, personId),
    getPersonPartnerships(collectionId, personId),
    getPersonsForCollection(collectionId),
  ]);

  // Get all parent-child links among family members for edge drawing
  const familyIds = familyMembers.map((fm) => Number(fm.person_id));
  if (!familyIds.includes(Number(personId))) familyIds.push(Number(personId));
  const allParentLinks = await getFamilyParentLinks(familyIds);

  return { person, familyMembers, centerParents, allParentLinks, partnerships, persons };
}

function GenerationLabels({ generations }: { generations: GenerationInfo[] }) {
  const { y: vy, zoom } = useViewport();
  return (
    <Panel position="top-left" className="pointer-events-none !m-0 !p-0" style={{ zIndex: 1 }}>
      <div className="relative">
        {generations.map((g) => (
          <div
            key={g.gen}
            className="absolute left-3 text-xs font-medium text-gray-500 whitespace-nowrap select-none"
            style={{ top: g.y * zoom + vy - 20 * zoom }}
          >
            {g.label}
          </div>
        ))}
      </div>
    </Panel>
  );
}

const nodeTypes = {
  person: PersonNode,
  placeholder: PlaceholderNode,
  dropZone: DropZoneNode,
};

function FamilyTreeCanvas({ loaderData }: Route.ComponentProps) {
  const { person, familyMembers, centerParents, allParentLinks, partnerships, persons } = loaderData;
  const navigate = useNavigate();
  const rootData = useRootData();
  const fetcher = useFetcher();
  const revalidator = useRevalidator();
  const [searchQuery, setSearchQuery] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // Revalidate loader data after mutation completes
  const prevFetcherData = useRef(fetcher.data);
  useEffect(() => {
    if (fetcher.state === "idle" && fetcher.data && fetcher.data !== prevFetcherData.current) {
      prevFetcherData.current = fetcher.data;
      revalidator.revalidate();
    }
  }, [fetcher.state, fetcher.data, revalidator]);

  const handleRecenter = useCallback(
    (personId: number) => {
      navigate(`/person/${personId}/family-tree`);
    },
    [navigate],
  );

  const handleDrop = useCallback(
    (droppedPersonId: number, relationshipType: string, targetPersonId: number) => {
      const actionMap: Record<string, string> = {
        parent: `/api/person/${targetPersonId}/add-parent`,
        child: `/api/person/${targetPersonId}/add-child`,
        partner: `/api/person/${targetPersonId}/add-partner`,
        sibling: `/api/person/${targetPersonId}/add-sibling`,
      };
      const action = actionMap[relationshipType] ?? `/api/person/${targetPersonId}/add-parent`;

      fetcher.submit({ relatedPersonId: String(droppedPersonId), role: relationshipType }, { method: "post", action });
    },
    [fetcher],
  );

  const handleRemoveRelationship = useCallback(
    (relatedPersonId: number, relation: string) => {
      const centerId = Number(person.id);
      if (relation === "parent" || relation === "grandparent" || relation.includes("parent")) {
        fetcher.submit(
          { parentId: String(relatedPersonId) },
          { method: "post", action: `/api/person/${centerId}/remove-parent` },
        );
      } else if (relation === "child" || relation === "grandchild" || relation.includes("child")) {
        fetcher.submit(
          { parentId: String(centerId) },
          { method: "post", action: `/api/person/${relatedPersonId}/remove-parent` },
        );
      } else if (relation === "partner" || relation === "spouse" || relation === "married") {
        fetcher.submit(
          { partnerId: String(relatedPersonId) },
          { method: "post", action: `/api/person/${centerId}/remove-partner` },
        );
      } else if (relation.includes("sibling")) {
        fetcher.submit(
          { siblingId: String(relatedPersonId) },
          { method: "post", action: `/api/person/${centerId}/remove-sibling` },
        );
      }
    },
    [fetcher, person.id],
  );

  const { nodes: initialNodes, edges: initialEdges, generations } = useMemo(() => {
    const layout = computeFamilyTreeLayout({
      centerId: Number(person.id),
      centerName: person.person_name || `Person ${person.id}`,
      centerDetectionId: person.detection_id ?? null,
      familyMembers,
      parents: allParentLinks,
      centerParentCount: centerParents.length,
      partnerships,
    });
    for (const node of layout.nodes) {
      if (node.type === "person" || node.type === "placeholder") {
        const d = node.data as Record<string, unknown>;
        d.onRecenter = handleRecenter;
        d.onRemoveRelationship = handleRemoveRelationship;
        d.centerId = Number(person.id);
      }
      if (node.type === "dropZone") {
        (node.data as Record<string, unknown>).onDrop = handleDrop;
      }
    }
    return layout;
  }, [person.id, familyMembers, centerParents, allParentLinks, partnerships, handleRecenter, handleDrop, handleRemoveRelationship]);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  // Reset nodes/edges when loader data changes (after revalidation)
  useEffect(() => {
    setNodes(initialNodes);
    setEdges(initialEdges);
  }, [initialNodes, initialEdges, setNodes, setEdges]);

  // Dynamic drop zones: appear when dragging over parent row (2+ parents) or partner area (1+ partners)
  const reactFlow = useReactFlow();
  const extraParentDropRef = useRef(false);
  const extraPartnerDropRef = useRef(false);
  const centerParentCount = centerParents.length;
  const parentRowY = -1 * (NODE_H + V_GAP);
  const extraParentDropId = `drop-parent-extra-${person.id}`;
  const extraPartnerDropId = `drop-partner-extra-${person.id}`;
  const hasPartner = partnerships.some(
    (p) => Number(p.person1_id) === Number(person.id) || Number(p.person2_id) === Number(person.id),
  );

  const handleCanvasDragOver = useCallback(
    (e: React.DragEvent) => {
      const pos = reactFlow.screenToFlowPosition({ x: e.clientX, y: e.clientY });

      // Dynamic parent drop zone
      if (centerParentCount >= 2) {
        const inParentRow = pos.y >= parentRowY - NODE_H && pos.y <= parentRowY + NODE_H * 2;
        if (inParentRow && !extraParentDropRef.current) {
          extraParentDropRef.current = true;
          setNodes((prev) => {
            if (prev.some((n) => n.id === extraParentDropId)) return prev;
            const parentNodes = prev.filter((n) => n.position.y === parentRowY);
            const dropX =
              parentNodes.length > 0
                ? Math.max(...parentNodes.map((n) => n.position.x)) + NODE_W + H_GAP
                : 0;
            return [
              ...prev,
              {
                id: extraParentDropId,
                type: "dropZone",
                position: { x: dropX, y: parentRowY },
                data: {
                  label: "Add parent",
                  relationshipType: "parent",
                  targetPersonId: Number(person.id),
                  onDrop: handleDrop,
                },
              },
            ];
          });
          setEdges((prev) => {
            const edgeId = `drop-edge-${extraParentDropId}`;
            if (prev.some((e) => e.id === edgeId)) return prev;
            return [
              ...prev,
              {
                id: edgeId,
                source: extraParentDropId,
                target: `person-${person.id}`,
                type: "smoothstep",
                style: { stroke: "#6b7280", strokeWidth: 1, strokeDasharray: "6 4" },
              },
            ];
          });
        } else if (!inParentRow && extraParentDropRef.current) {
          extraParentDropRef.current = false;
          setNodes((prev) => prev.filter((n) => n.id !== extraParentDropId));
          setEdges((prev) => prev.filter((e) => e.id !== `drop-edge-${extraParentDropId}`));
        }
      }

      // Dynamic partner drop zone
      if (hasPartner) {
        const inGen0Row = pos.y >= -NODE_H && pos.y <= NODE_H * 2;
        const centerX = -NODE_W / 2;
        const toRight = pos.x > centerX + NODE_W;
        if (inGen0Row && toRight && !extraPartnerDropRef.current) {
          extraPartnerDropRef.current = true;
          setNodes((prev) => {
            if (prev.some((n) => n.id === extraPartnerDropId)) return prev;
            const gen0Nodes = prev.filter(
              (n) => n.position.y === 0 && n.id !== `drop-sibling-${person.id}`,
            );
            const dropX =
              gen0Nodes.length > 0
                ? Math.max(...gen0Nodes.map((n) => n.position.x)) + NODE_W + H_GAP
                : NODE_W + H_GAP;
            return [
              ...prev,
              {
                id: extraPartnerDropId,
                type: "dropZone",
                position: { x: dropX, y: 0 },
                data: {
                  label: "Add partner",
                  relationshipType: "partner",
                  targetPersonId: Number(person.id),
                  onDrop: handleDrop,
                },
              },
            ];
          });
          setEdges((prev) => {
            const edgeId = `drop-edge-${extraPartnerDropId}`;
            if (prev.some((e) => e.id === edgeId)) return prev;
            return [
              ...prev,
              {
                id: edgeId,
                source: `person-${person.id}`,
                sourceHandle: "right",
                target: extraPartnerDropId,
                targetHandle: "left",
                type: "straight",
                style: { stroke: "#6b7280", strokeWidth: 1, strokeDasharray: "6 4" },
              },
            ];
          });
        } else if ((!inGen0Row || !toRight) && extraPartnerDropRef.current) {
          extraPartnerDropRef.current = false;
          setNodes((prev) => prev.filter((n) => n.id !== extraPartnerDropId));
          setEdges((prev) => prev.filter((e) => e.id !== `drop-edge-${extraPartnerDropId}`));
        }
      }
    },
    [
      centerParentCount,
      hasPartner,
      reactFlow,
      parentRowY,
      extraParentDropId,
      extraPartnerDropId,
      person.id,
      handleDrop,
      setNodes,
      setEdges,
    ],
  );

  const handleCanvasDragLeave = useCallback(() => {
    if (extraParentDropRef.current) {
      extraParentDropRef.current = false;
      setNodes((prev) => prev.filter((n) => n.id !== extraParentDropId));
      setEdges((prev) => prev.filter((e) => e.id !== `drop-edge-${extraParentDropId}`));
    }
    if (extraPartnerDropRef.current) {
      extraPartnerDropRef.current = false;
      setNodes((prev) => prev.filter((n) => n.id !== extraPartnerDropId));
      setEdges((prev) => prev.filter((e) => e.id !== `drop-edge-${extraPartnerDropId}`));
    }
  }, [extraParentDropId, extraPartnerDropId, setNodes, setEdges]);

  const filteredPersons = useMemo(() => {
    const treeIds = new Set(familyMembers.map((fm) => Number(fm.person_id)));
    treeIds.add(Number(person.id));
    return persons.filter(
      (p: (typeof persons)[number]) =>
        !treeIds.has(Number(p.id)) && (!searchQuery || subsequenceMatch(p.person_name ?? "", searchQuery)),
    );
  }, [persons, searchQuery, familyMembers, person.id]);

  return (
    <div className="h-screen flex flex-col bg-gray-900 overflow-hidden">
      <Header
        user={rootData?.userAvatar}
        isAdmin={rootData?.user?.isAdmin}
        isImpersonating={rootData?.impersonation?.isImpersonating}
        breadcrumbs={[
          { label: "People", to: "/people" },
          { label: person.person_name || `Person ${person.id}`, to: `/person/${person.id}/grid` },
          { label: "Family" },
        ]}
      />

      <div className="flex flex-1 min-h-0 pt-16 relative">
        {/* React Flow Canvas */}
        <div className="flex-1" onDragOver={handleCanvasDragOver} onDragLeave={handleCanvasDragLeave}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            nodeTypes={nodeTypes}
            fitView
            fitViewOptions={{ padding: 0.3 }}
            zoomOnDoubleClick={false}
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
            <GenerationLabels generations={generations} />
          </ReactFlow>
        </div>

        {/* Right Sidebar Toggle (visible when collapsed) */}
        <button
          type="button"
          onClick={() => setSidebarOpen(true)}
          className={`absolute right-0 top-1/2 -translate-y-1/2 z-10 bg-gray-50 border border-r-0 border-gray-200 rounded-l-lg p-1.5 hover:bg-gray-100 transition-all duration-300 ${
            sidebarOpen ? "opacity-0 pointer-events-none translate-x-2" : "opacity-100 translate-x-0"
          }`}
        >
          <ChevronRight className="h-4 w-4 text-gray-500 rotate-180" />
        </button>

        {/* Right Sidebar */}
        <div
          className={`bg-gray-50 flex flex-col shrink-0 border-l border-gray-200 overflow-hidden transition-all duration-300 ease-in-out ${
            sidebarOpen ? "w-70" : "w-0 border-l-0"
          }`}
        >
          <div className="w-70 flex flex-col h-full">
            <div className="p-4 pb-2">
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-sm font-semibold text-gray-900">People</h2>
                <button
                  type="button"
                  onClick={() => setSidebarOpen(false)}
                  className="p-0.5 rounded hover:bg-gray-200 transition-colors"
                >
                  <ChevronRight className="h-4 w-4 text-gray-400" />
                </button>
              </div>
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

            <ul className="flex-1 overflow-y-auto px-4 pb-4 space-y-1 list-none">
              {filteredPersons.map((p) => (
                <li
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
                  <span className="text-sm font-medium text-gray-900 truncate">{p.person_name}</span>
                </li>
              ))}
            </ul>

            <div className="p-4 pt-2 border-t border-gray-200">
              <button
                type="button"
                onClick={() => {
                  fetcher.submit({ name: "", gender: "U" }, { method: "post", action: "/api/person/create-placeholder" });
                }}
                className="w-full py-2 text-sm font-medium text-gray-500 border border-dashed border-gray-300 rounded-lg hover:border-gray-400 hover:text-gray-700 transition-colors"
              >
                + New Placeholder Person
              </button>
            </div>
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
