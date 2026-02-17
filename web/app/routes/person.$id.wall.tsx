import { EyeOff, Pencil, Trash2, Users } from "lucide-react";
import { useMemo, useState } from "react";
import { useFetcher, useLocation, useNavigate } from "react-router";
import { DeletePersonDialog } from "~/components/delete-person-dialog";
import { Header } from "~/components/header";
import { PhotoWall, type WallTile } from "~/components/photo-wall";
import { RenamePersonDialog } from "~/components/rename-person-dialog";
import { ControlsDivider, SecondaryControls } from "~/components/secondary-controls";
import { WallViewSwitcher } from "~/components/wall-view-switcher";
import { useRootData } from "~/hooks/use-root-data";
import { requireCollectionId } from "~/lib/auth.server";
import { dataWithViewMode } from "~/lib/cookies.server";
import { getClustersByPerson, getPersonById } from "~/lib/db.server";
import type { Route } from "./+types/person.$id.wall";

export function meta({ data }: Route.MetaArgs) {
  const personName = data?.person?.person_name || "Person";
  return [
    { title: `Storyteller - ${personName} - 3D Wall` },
    { name: "description", content: `View ${personName}'s clusters in 3D wall view` },
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

  const clusters = await getClustersByPerson(collectionId, personId);

  return dataWithViewMode({ person, clusters }, "wall");
}

export default function PersonWallView({ loaderData }: Route.ComponentProps) {
  const rootData = useRootData();
  const { person, clusters } = loaderData;
  const location = useLocation();
  const [renameDialogOpen, setRenameDialogOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const navigate = useNavigate();
  const hideFetcher = useFetcher();

  const visibleClusters = clusters.filter((c) => !c.hidden);
  const hiddenClusters = clusters.filter((c) => c.hidden);
  const isSubmitting = hideFetcher.state !== "idle";

  const handleHideAll = () => {
    hideFetcher.submit({ hidden: "true" }, { method: "post", action: `/api/person/${person.id}/hide` });
  };

  const handleUnhideAll = () => {
    hideFetcher.submit({ hidden: "false" }, { method: "post", action: `/api/person/${person.id}/hide` });
  };

  // Convert clusters to wall tiles
  const tiles: WallTile[] = useMemo(
    () =>
      clusters
        .filter((cluster) => !cluster.hidden)
        .map((cluster) => {
          // Use direct face image URL (pre-extracted face crops)
          const imageUrl = cluster.detection_id ? `/api/face/${cluster.detection_id}` : null;

          return {
            id: cluster.id,
            imageUrls: imageUrl ? [imageUrl] : [],
            label: `Cluster #${cluster.id}`,
            navigateTo: `/cluster/${cluster.id}/wall`,
            metadata: {
              subtitle: `${cluster.face_count} photo${cluster.face_count !== 1 ? "s" : ""}`,
              count: cluster.face_count,
            },
          };
        }),
    [clusters],
  );

  const displayName = person.person_name || `Person ${person.id}`;

  const headerContent = (
    <Header
      homeTo="/wall"
      breadcrumbs={[
        { label: "People", to: "/people" },
        {
          label: (
            <span className="inline-flex items-center gap-2.5">
              {displayName}
              <button
                type="button"
                onClick={(e) => {
                  e.preventDefault();
                  setRenameDialogOpen(true);
                }}
                className="text-gray-400 hover:text-white"
              >
                <Pencil className="h-3.5 w-3.5" />
              </button>
            </span>
          ),
        },
      ]}
      user={rootData?.userAvatar}
      isAdmin={rootData?.user?.isAdmin}
      isImpersonating={rootData?.impersonation?.isImpersonating}
      viewAction={<WallViewSwitcher />}
    />
  );

  if (clusters.length === 0) {
    return (
      <div className="h-screen w-screen bg-gray-900 flex flex-col">
        <div className="p-4">{headerContent}</div>
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <Users className="h-16 w-16 text-gray-600 mx-auto mb-4" />
            <p className="text-gray-400 text-lg mb-4">No clusters found for this person.</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <>
      <PhotoWall
        key={location.key}
        tiles={tiles}
        sessionKey={`person-${person.id}-wall`}
        headerContent={headerContent}
      />
      <SecondaryControls variant="wall">
        {visibleClusters.length > 0 && (
          <button
            type="button"
            onClick={handleHideAll}
            disabled={isSubmitting}
            className="flex items-center gap-1.5 hover:text-white transition-colors disabled:opacity-50"
          >
            <EyeOff className="h-4 w-4" />
            <span>Hide All</span>
          </button>
        )}
        {hiddenClusters.length > 0 && (
          <>
            {visibleClusters.length > 0 && <ControlsDivider variant="wall" />}
            <button
              type="button"
              onClick={handleUnhideAll}
              disabled={isSubmitting}
              className="hover:text-white transition-colors disabled:opacity-50"
            >
              Unhide All ({hiddenClusters.length})
            </button>
          </>
        )}
        <ControlsDivider variant="wall" />
        <button
          type="button"
          onClick={() => setDeleteDialogOpen(true)}
          disabled={isSubmitting}
          className="flex items-center gap-1.5 hover:text-white transition-colors disabled:opacity-50"
        >
          <Trash2 className="h-4 w-4" />
          <span>Delete Person</span>
        </button>
      </SecondaryControls>
      <RenamePersonDialog
        open={renameDialogOpen}
        onOpenChange={setRenameDialogOpen}
        personId={person.id.toString()}
        currentFirstName={person.first_name || ""}
        currentLastName={person.last_name || ""}
        onSuccess={() => {
          window.location.reload();
        }}
      />
      <DeletePersonDialog
        open={deleteDialogOpen}
        onOpenChange={setDeleteDialogOpen}
        personId={person.id.toString()}
        personName={person.person_name || `Person ${person.id}`}
        clusterCount={clusters.length}
        onSuccess={() => navigate("/people")}
      />
    </>
  );
}
