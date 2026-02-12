import { Check, ChevronDown, ChevronRight, Loader2, User } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useFetcher, useRevalidator } from "react-router";
import { Header } from "~/components/header";
import { PersonSelectModal } from "~/components/person-select-modal";
import { Button } from "~/components/ui/button";
import { Card, CardContent } from "~/components/ui/card";
import { Input } from "~/components/ui/input";
import { useRootData } from "~/hooks/use-root-data";
import { requireUser } from "~/lib/auth.server";
import {
  getPersonsForCollection,
  getUserCollections,
  type PersonForSelection,
  type UserCollection,
} from "~/lib/db.server";
import type { Route } from "./+types/profile";

export function meta() {
  return [{ title: "Storyteller - Profile" }, { name: "description", content: "Manage your profile settings" }];
}

export async function loader({ request }: Route.LoaderArgs) {
  const user = await requireUser(request);
  const collections = await getUserCollections(user.id);

  // Get persons for each collection
  const collectionsWithPersons: Array<UserCollection & { persons: PersonForSelection[] }> = [];
  for (const collection of collections) {
    const persons = await getPersonsForCollection(collection.id);
    collectionsWithPersons.push({ ...collection, persons });
  }

  return {
    user: {
      id: user.id,
      firstName: user.first_name,
      lastName: user.last_name,
      defaultCollectionId: user.default_collection_id,
    },
    collections: collectionsWithPersons,
  };
}

export default function ProfilePage({ loaderData }: Route.ComponentProps) {
  const { user, collections } = loaderData;
  const rootData = useRootData();
  const revalidator = useRevalidator();

  // Account details state
  const [firstName, setFirstName] = useState(user.firstName);
  const [lastName, setLastName] = useState(user.lastName || "");
  const profileFetcher = useFetcher();
  const isProfileSubmitting = profileFetcher.state !== "idle";
  const hasProfileChanges = firstName !== user.firstName || lastName !== (user.lastName || "");

  // Default collection state
  const [selectedCollectionId, setSelectedCollectionId] = useState(user.defaultCollectionId);
  const [collectionDropdownOpen, setCollectionDropdownOpen] = useState(false);
  const collectionFetcher = useFetcher();
  const isCollectionSubmitting = collectionFetcher.state !== "idle";

  // Person select modal state
  const [personModalOpen, setPersonModalOpen] = useState(false);
  const [personModalCollection, setPersonModalCollection] = useState<(typeof collections)[0] | null>(null);

  // Handle profile save success
  const profileSuccessRef = useRef(false);
  useEffect(() => {
    if (profileFetcher.data?.success && !profileSuccessRef.current) {
      profileSuccessRef.current = true;
      revalidator.revalidate();
    }
    if (profileFetcher.state === "idle") {
      profileSuccessRef.current = false;
    }
  }, [profileFetcher.data, profileFetcher.state, revalidator]);

  // Handle collection change
  const handleCollectionChange = (collectionId: number) => {
    if (collectionId !== selectedCollectionId) {
      setSelectedCollectionId(collectionId);
      setCollectionDropdownOpen(false);
      collectionFetcher.submit(
        { collectionId: collectionId.toString() },
        { method: "post", action: "/api/user/set-default-collection" },
      );
    }
  };

  // Handle profile save
  const handleProfileSave = () => {
    profileFetcher.submit(
      { firstName: firstName.trim(), lastName: lastName.trim() },
      { method: "post", action: "/api/user/update-profile" },
    );
  };

  // Open person select modal
  const openPersonModal = (collection: (typeof collections)[0]) => {
    setPersonModalCollection(collection);
    setPersonModalOpen(true);
  };

  const selectedCollection = collections.find((c) => c.id === selectedCollectionId);

  return (
    <div className="min-h-screen bg-gray-50">
      <Header
        user={rootData?.userAvatar}
        isAdmin={rootData?.user?.isAdmin}
        isImpersonating={rootData?.impersonation?.isImpersonating}
        breadcrumbs={[{ label: "Profile" }]}
      />
      <div className="h-16 bg-gray-900" />

      <main className="container mx-auto px-4 py-6 max-w-2xl">
        <div className="space-y-6">
          {/* Page header */}
          <div className="flex items-center space-x-3">
            <User className="h-8 w-8 text-gray-700" />
            <h1 className="text-3xl font-bold text-gray-900">Profile</h1>
          </div>

          {/* Account Details Card */}
          <Card>
            <CardContent className="p-6">
              <h2 className="text-lg font-semibold mb-4">Account Details</h2>
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <label htmlFor="firstName" className="text-sm font-medium">
                      First Name
                    </label>
                    <Input
                      id="firstName"
                      value={firstName}
                      onChange={(e) => setFirstName(e.target.value)}
                      placeholder="First name"
                      autoComplete="off"
                      data-form-type="other"
                      data-1p-ignore
                      data-lpignore="true"
                    />
                  </div>
                  <div className="space-y-2">
                    <label htmlFor="lastName" className="text-sm font-medium">
                      Last Name
                    </label>
                    <Input
                      id="lastName"
                      value={lastName}
                      onChange={(e) => setLastName(e.target.value)}
                      placeholder="Last name (optional)"
                      autoComplete="off"
                      data-form-type="other"
                      data-1p-ignore
                      data-lpignore="true"
                    />
                  </div>
                </div>
                <div className="flex justify-end">
                  <Button
                    onClick={handleProfileSave}
                    disabled={!hasProfileChanges || !firstName.trim() || isProfileSubmitting}
                  >
                    {isProfileSubmitting ? (
                      <>
                        <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                        Saving...
                      </>
                    ) : (
                      "Save"
                    )}
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Default Collection Card - only show if multiple collections */}
          {collections.length > 1 && (
            <Card>
              <CardContent className="p-6">
                <h2 className="text-lg font-semibold mb-2">Default Collection</h2>
                <p className="text-sm text-gray-500 mb-4">Choose which collection loads when you sign in</p>
                <div className="relative">
                  <button
                    type="button"
                    onClick={() => setCollectionDropdownOpen(!collectionDropdownOpen)}
                    className="w-full flex items-center justify-between px-3 py-2 border rounded-md bg-white hover:bg-gray-50 transition-colors"
                    disabled={isCollectionSubmitting}
                  >
                    <span>{selectedCollection?.name || "Select collection"}</span>
                    {isCollectionSubmitting ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <ChevronDown className="h-4 w-4" />
                    )}
                  </button>
                  {collectionDropdownOpen && (
                    <div className="absolute z-10 w-full mt-1 bg-white border rounded-md shadow-lg">
                      {collections.map((collection) => (
                        <button
                          key={collection.id}
                          type="button"
                          onClick={() => handleCollectionChange(collection.id)}
                          className="w-full flex items-center justify-between px-3 py-2 hover:bg-gray-50 text-left"
                        >
                          <span>{collection.name}</span>
                          {collection.id === selectedCollectionId && <Check className="h-4 w-4 text-blue-500" />}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Collection Memberships Card */}
          <Card>
            <CardContent className="p-6">
              <h2 className="text-lg font-semibold mb-2">Your Collections</h2>
              <p className="text-sm text-gray-500 mb-4">Link yourself to a person in each collection for your avatar</p>
              <div className="divide-y border rounded-md">
                {collections.map((collection) => (
                  <div key={collection.id} className="flex items-center justify-between p-4">
                    <span className="font-medium">{collection.name}</span>
                    <button
                      type="button"
                      onClick={() => openPersonModal(collection)}
                      className="flex items-center gap-2 px-3 py-1.5 rounded-md border hover:bg-gray-50 transition-colors"
                    >
                      {collection.person_id && collection.avatar_detection_id ? (
                        <>
                          <div className="w-8 h-8 rounded-full overflow-hidden bg-gray-100">
                            <img
                              src={`/api/face/${collection.avatar_detection_id}`}
                              alt={collection.person_name || ""}
                              className="w-full h-full object-cover"
                            />
                          </div>
                          <span className="text-sm">{collection.person_name}</span>
                          <ChevronRight className="h-4 w-4 text-gray-400" />
                        </>
                      ) : (
                        <>
                          <div className="w-8 h-8 rounded-full border-2 border-dashed border-gray-300 flex items-center justify-center">
                            <User className="h-4 w-4 text-gray-400" />
                          </div>
                          <span className="text-sm text-gray-500">Select person</span>
                          <ChevronRight className="h-4 w-4 text-gray-400" />
                        </>
                      )}
                    </button>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </main>

      {/* Person Select Modal */}
      {personModalCollection && (
        <PersonSelectModal
          open={personModalOpen}
          onOpenChange={setPersonModalOpen}
          collectionId={personModalCollection.id}
          collectionName={personModalCollection.name}
          persons={personModalCollection.persons}
          currentPersonId={personModalCollection.person_id}
          currentUserId={user.id}
          onSuccess={() => revalidator.revalidate()}
        />
      )}
    </div>
  );
}
