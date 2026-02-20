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
