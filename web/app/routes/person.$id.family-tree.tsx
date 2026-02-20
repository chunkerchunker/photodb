import { useCallback, useMemo, useState } from "react";
import { useNavigate, useFetcher, Link } from "react-router";
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
import { requireCollectionId } from "~/lib/auth.server";
import {
  getFamilyTree,
  getPersonById,
  getPersonParents,
  getPersonPartnerships,
  getPersonsForCollection,
} from "~/lib/db.server";
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

  const [familyMembers, parents, partnerships, persons] = await Promise.all([
    getFamilyTree(collectionId, personId, 3),
    getPersonParents(collectionId, personId),
    getPersonPartnerships(collectionId, personId),
    getPersonsForCollection(collectionId),
  ]);

  return { person, familyMembers, parents, partnerships, persons, collectionId };
}

const nodeTypes = {
  person: PersonNode,
  placeholder: PlaceholderNode,
  dropZone: DropZoneNode,
};

function FamilyTreeCanvas({ loaderData }: Route.ComponentProps) {
  const { person, familyMembers, parents, partnerships, persons } = loaderData;
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
    for (const node of layout.nodes) {
      if (node.type === "person" || node.type === "placeholder") {
        (node.data as Record<string, unknown>).onRecenter = handleRecenter;
      }
      if (node.type === "dropZone") {
        (node.data as Record<string, unknown>).onDrop = handleDrop;
      }
    }
    return layout;
  }, [person.id, familyMembers, parents, partnerships, handleRecenter, handleDrop]);

  const [nodes, , onNodesChange] = useNodesState(initialNodes);
  const [edges, , onEdgesChange] = useEdgesState(initialEdges);

  const filteredPersons = useMemo(() => {
    const q = searchQuery.toLowerCase();
    return persons.filter(
      (p: (typeof persons)[number]) => p.person_name?.toLowerCase().includes(q),
    );
  }, [persons, searchQuery]);

  return (
    <div className="h-screen flex flex-col bg-gray-900">
      {/* Header */}
      <div className="h-14 bg-gray-950 flex items-center justify-between px-6 shrink-0">
        <div className="flex items-center gap-2 text-sm text-gray-400">
          <Link to="/people/grid" className="hover:text-gray-200">
            People
          </Link>
          <span>/</span>
          <Link to={`/person/${person.id}/grid`} className="hover:text-gray-200">
            {person.person_name}
          </Link>
          <span>/</span>
          <span className="text-gray-200">Family</span>
        </div>
        <div className="flex items-center gap-4 text-sm">
          <Link to={`/person/${person.id}/grid`} className="text-gray-500 hover:text-gray-300">
            Grid
          </Link>
          <Link to={`/person/${person.id}/wall`} className="text-gray-500 hover:text-gray-300">
            Wall
          </Link>
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
            {filteredPersons.map((p) => (
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
              type="button"
              onClick={() => {
                fetcher.submit(
                  { name: "", gender: "U" },
                  { method: "post", action: "/api/person/create-placeholder" },
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
