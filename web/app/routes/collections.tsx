import { Check, FolderOpen, ImageIcon, User } from "lucide-react";
import { Link, useFetcher } from "react-router";
import { Header } from "~/components/header";
import { Card, CardContent } from "~/components/ui/card";
import { useRootData } from "~/hooks/use-root-data";
import { requireUser } from "~/lib/auth.server";
import { getUserCollections } from "~/lib/db.server";
import { getFaceCropStyle } from "~/lib/face-crop";
import type { Route } from "./+types/collections";

export function meta() {
  return [
    { title: "Storyteller - Collections" },
    { name: "description", content: "Manage your photo collections" },
  ];
}

export async function loader({ request }: Route.LoaderArgs) {
  const user = await requireUser(request);
  const collections = await getUserCollections(user.id);

  return {
    collections,
    currentCollectionId: user.default_collection_id,
  };
}

export default function CollectionsPage({ loaderData }: Route.ComponentProps) {
  const { collections, currentCollectionId } = loaderData;
  const rootData = useRootData();
  const fetcher = useFetcher();

  const handleSelectCollection = (collectionId: number) => {
    if (collectionId !== currentCollectionId) {
      fetcher.submit(
        { collectionId: collectionId.toString() },
        { method: "post", action: "/api/collections/switch" },
      );
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Header
        user={rootData?.userAvatar}
        isAdmin={rootData?.user?.isAdmin}
        breadcrumbs={[{ label: "Collections" }]}
      />
      <div className="h-16 bg-gray-900" />

      <main className="container mx-auto px-4 py-6">
        <div className="space-y-6">
          <div className="flex items-center space-x-3">
            <FolderOpen className="h-8 w-8 text-gray-700" />
            <h1 className="text-3xl font-bold text-gray-900">Collections</h1>
          </div>

          <p className="text-gray-600">
            Select which collection to view. Your current collection is highlighted.
          </p>

          {collections.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {collections.map((collection) => {
                const isActive = collection.id === currentCollectionId;
                const hasAvatar =
                  collection.avatar_photo_id &&
                  collection.avatar_bbox_x !== null &&
                  collection.avatar_med_width !== null;

                return (
                  <button
                    key={collection.id}
                    type="button"
                    onClick={() => handleSelectCollection(collection.id)}
                    className="text-left w-full"
                    disabled={fetcher.state !== "idle"}
                  >
                    <Card
                      className={`hover:shadow-lg transition-all h-full cursor-pointer ${
                        isActive
                          ? "ring-2 ring-blue-500 ring-offset-2 bg-blue-50"
                          : "hover:ring-2 hover:ring-gray-300"
                      }`}
                    >
                      <CardContent className="p-6">
                        <div className="flex items-start space-x-4">
                          {/* Avatar */}
                          <div className="flex-shrink-0">
                            {hasAvatar ? (
                              <div className="relative w-16 h-16 bg-gray-100 rounded-full overflow-hidden">
                                <img
                                  src={`/api/image/${collection.avatar_photo_id}`}
                                  alt={collection.person_name || collection.name}
                                  className="absolute max-w-none max-h-none"
                                  style={getFaceCropStyle(
                                    {
                                      bbox_x: collection.avatar_bbox_x!,
                                      bbox_y: collection.avatar_bbox_y!,
                                      bbox_width: collection.avatar_bbox_width!,
                                      bbox_height: collection.avatar_bbox_height!,
                                    },
                                    collection.avatar_med_width!,
                                    collection.avatar_med_height!,
                                    64,
                                  )}
                                />
                              </div>
                            ) : (
                              <div className="w-16 h-16 bg-gray-200 rounded-full flex items-center justify-center">
                                <FolderOpen className="h-8 w-8 text-gray-400" />
                              </div>
                            )}
                          </div>

                          {/* Collection Info */}
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center space-x-2">
                              <h3 className="text-lg font-semibold text-gray-900 truncate">
                                {collection.name}
                              </h3>
                              {isActive && (
                                <Check className="h-5 w-5 text-blue-500 flex-shrink-0" />
                              )}
                            </div>

                            <div className="mt-1 flex items-center text-sm text-gray-500">
                              <ImageIcon className="h-4 w-4 mr-1" />
                              {collection.photo_count.toLocaleString()} photo
                              {collection.photo_count !== 1 ? "s" : ""}
                            </div>

                            {collection.person_name && (
                              <div className="mt-1 flex items-center text-sm text-gray-500">
                                <User className="h-4 w-4 mr-1" />
                                Linked as {collection.person_name}
                              </div>
                            )}
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </button>
                );
              })}
            </div>
          ) : (
            <div className="text-center py-12">
              <FolderOpen className="h-16 w-16 text-gray-400 mx-auto mb-4" />
              <div className="text-gray-500 text-lg">No collections found.</div>
              <div className="text-gray-400 text-sm mt-2">
                You are not a member of any collections yet.
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
