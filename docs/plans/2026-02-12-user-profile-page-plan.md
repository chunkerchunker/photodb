# User Profile Page Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a user profile page with name editing, default collection selection, and person linking per collection.

**Architecture:** Card-based profile page at `/profile` route. Three API endpoints for profile updates, default collection, and person linking. Person selection uses a modal with search and single-select.

**Tech Stack:** React Router v7, TypeScript, PostgreSQL, TailwindCSS, shadcn/ui Dialog

---

## Task 1: Add Database Functions

**Files:**
- Modify: `web/app/lib/db.server.ts`

**Step 1: Add `updateUserProfile` function**

Add after line ~386 (after `updateUserPasswordHash`):

```typescript
/**
 * Update user's first and last name.
 */
export async function updateUserProfile(userId: number, firstName: string, lastName: string | null): Promise<void> {
  await pool.query(
    "UPDATE app_user SET first_name = $1, last_name = $2 WHERE id = $3",
    [firstName, lastName, userId]
  );
}
```

**Step 2: Add `setCollectionMemberPerson` function**

Add after `updateUserDefaultCollection` (around line ~2182):

```typescript
/**
 * Set the person linked to a user's collection membership.
 * Used for avatar display in that collection.
 */
export async function setCollectionMemberPerson(
  userId: number,
  collectionId: number,
  personId: number | null
): Promise<void> {
  const result = await pool.query(
    "UPDATE collection_member SET person_id = $1 WHERE user_id = $2 AND collection_id = $3 RETURNING user_id",
    [personId, userId, collectionId]
  );

  if (result.rows.length === 0) {
    throw new Error("User is not a member of this collection");
  }
}
```

**Step 3: Add `getPersonsForCollection` function**

Add after `setCollectionMemberPerson`:

```typescript
export type PersonForSelection = {
  id: number;
  first_name: string;
  last_name: string | null;
  person_name: string;
  detection_id: number | null;
  linked_user_id: number | null;
  linked_user_name: string | null;
};

/**
 * Get all persons in a collection with info about which user they're linked to.
 * Used for person selection modal.
 */
export async function getPersonsForCollection(collectionId: number): Promise<PersonForSelection[]> {
  const query = `
    SELECT
      per.id,
      per.first_name,
      per.last_name,
      TRIM(CONCAT(per.first_name, ' ', COALESCE(per.last_name, ''))) as person_name,
      COALESCE(
        per.representative_detection_id,
        (SELECT c.representative_detection_id
         FROM cluster c
         WHERE c.person_id = per.id
           AND c.representative_detection_id IS NOT NULL
           AND (c.hidden = false OR c.hidden IS NULL)
         ORDER BY c.face_count DESC
         LIMIT 1)
      ) as detection_id,
      cm.user_id as linked_user_id,
      CASE WHEN cm.user_id IS NOT NULL
        THEN TRIM(CONCAT(u.first_name, ' ', COALESCE(u.last_name, '')))
        ELSE NULL
      END as linked_user_name
    FROM person per
    LEFT JOIN collection_member cm ON cm.person_id = per.id AND cm.collection_id = per.collection_id
    LEFT JOIN app_user u ON cm.user_id = u.id
    WHERE per.collection_id = $1
      AND (per.hidden = false OR per.hidden IS NULL)
      AND EXISTS (
        SELECT 1 FROM cluster c
        WHERE c.person_id = per.id AND c.face_count > 0
      )
    ORDER BY per.first_name, per.last_name, per.id
  `;

  const result = await pool.query(query, [collectionId]);
  return result.rows;
}
```

**Step 4: Commit**

```bash
git add web/app/lib/db.server.ts
git commit -m "feat(db): add profile and person selection database functions"
```

---

## Task 2: Create Profile Update API Endpoint

**Files:**
- Create: `web/app/routes/api.user.update-profile.tsx`

**Step 1: Create the endpoint file**

```typescript
import { requireUser } from "~/lib/auth.server";
import { updateUserProfile } from "~/lib/db.server";
import type { Route } from "./+types/api.user.update-profile";

export async function action({ request }: Route.ActionArgs) {
  const user = await requireUser(request);
  const formData = await request.formData();

  const firstName = (formData.get("firstName") as string)?.trim();
  const lastName = (formData.get("lastName") as string)?.trim() || null;

  if (!firstName) {
    return Response.json({ success: false, message: "First name is required" }, { status: 400 });
  }

  try {
    await updateUserProfile(user.id, firstName, lastName);
    return Response.json({ success: true, message: "Profile updated" });
  } catch (error) {
    console.error("Failed to update profile:", error);
    return Response.json({ success: false, message: "Failed to update profile" }, { status: 500 });
  }
}
```

**Step 2: Add route to routes.ts**

In `web/app/routes.ts`, add after the admin routes (around line 12):

```typescript
  route("api/user/update-profile", "routes/api.user.update-profile.tsx"),
```

**Step 3: Commit**

```bash
git add web/app/routes/api.user.update-profile.tsx web/app/routes.ts
git commit -m "feat(api): add profile update endpoint"
```

---

## Task 3: Create Default Collection API Endpoint

**Files:**
- Create: `web/app/routes/api.user.set-default-collection.tsx`

**Step 1: Create the endpoint file**

```typescript
import { requireUser } from "~/lib/auth.server";
import { updateUserDefaultCollection } from "~/lib/db.server";
import type { Route } from "./+types/api.user.set-default-collection";

export async function action({ request }: Route.ActionArgs) {
  const user = await requireUser(request);
  const formData = await request.formData();

  const collectionId = parseInt(formData.get("collectionId") as string, 10);

  if (Number.isNaN(collectionId)) {
    return Response.json({ success: false, message: "Invalid collection ID" }, { status: 400 });
  }

  try {
    await updateUserDefaultCollection(user.id, collectionId);
    return Response.json({ success: true, message: "Default collection updated" });
  } catch (error) {
    console.error("Failed to update default collection:", error);
    return Response.json({ success: false, message: "Failed to update default collection" }, { status: 500 });
  }
}
```

**Step 2: Add route to routes.ts**

```typescript
  route("api/user/set-default-collection", "routes/api.user.set-default-collection.tsx"),
```

**Step 3: Commit**

```bash
git add web/app/routes/api.user.set-default-collection.tsx web/app/routes.ts
git commit -m "feat(api): add set default collection endpoint"
```

---

## Task 4: Create Set Member Person API Endpoint

**Files:**
- Create: `web/app/routes/api.collection.$id.set-member-person.tsx`

**Step 1: Create the endpoint file**

```typescript
import { requireUser } from "~/lib/auth.server";
import { setCollectionMemberPerson } from "~/lib/db.server";
import type { Route } from "./+types/api.collection.$id.set-member-person";

export async function action({ request, params }: Route.ActionArgs) {
  const user = await requireUser(request);
  const collectionId = parseInt(params.id, 10);

  if (Number.isNaN(collectionId)) {
    return Response.json({ success: false, message: "Invalid collection ID" }, { status: 400 });
  }

  const formData = await request.formData();
  const personIdStr = formData.get("personId") as string;
  const personId = personIdStr ? parseInt(personIdStr, 10) : null;

  if (personIdStr && Number.isNaN(personId)) {
    return Response.json({ success: false, message: "Invalid person ID" }, { status: 400 });
  }

  try {
    await setCollectionMemberPerson(user.id, collectionId, personId);
    return Response.json({ success: true, message: personId ? "Person linked" : "Person unlinked" });
  } catch (error) {
    console.error("Failed to set member person:", error);
    const message = error instanceof Error ? error.message : "Failed to set member person";
    return Response.json({ success: false, message }, { status: 500 });
  }
}
```

**Step 2: Add route to routes.ts**

```typescript
  route("api/collection/:id/set-member-person", "routes/api.collection.$id.set-member-person.tsx"),
```

**Step 3: Commit**

```bash
git add web/app/routes/api.collection.\$id.set-member-person.tsx web/app/routes.ts
git commit -m "feat(api): add set member person endpoint"
```

---

## Task 5: Create Person Selection Modal Component

**Files:**
- Create: `web/app/components/person-select-modal.tsx`

**Step 1: Create the modal component**

```typescript
import { Link2, Loader2, Search, User } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useFetcher } from "react-router";
import { Button } from "~/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "~/components/ui/dialog";
import { Input } from "~/components/ui/input";
import type { PersonForSelection } from "~/lib/db.server";

interface PersonSelectModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  collectionId: number;
  collectionName: string;
  persons: PersonForSelection[];
  currentPersonId: number | null;
  currentUserId: number;
  onSuccess?: () => void;
}

export function PersonSelectModal({
  open,
  onOpenChange,
  collectionId,
  collectionName,
  persons,
  currentPersonId,
  currentUserId,
  onSuccess,
}: PersonSelectModalProps) {
  const [search, setSearch] = useState("");
  const [selectedId, setSelectedId] = useState<number | null>(currentPersonId);
  const [showWarning, setShowWarning] = useState(false);
  const [warningPersonName, setWarningPersonName] = useState("");
  const [warningLinkedUser, setWarningLinkedUser] = useState("");
  const fetcher = useFetcher();

  const isSubmitting = fetcher.state !== "idle";

  // Filter persons by search
  const filteredPersons = useMemo(() => {
    if (!search.trim()) return persons;
    const lower = search.toLowerCase();
    return persons.filter((p) => p.person_name.toLowerCase().includes(lower));
  }, [persons, search]);

  // Reset state when modal opens
  useEffect(() => {
    if (open) {
      setSearch("");
      setSelectedId(currentPersonId);
      setShowWarning(false);
    }
  }, [open, currentPersonId]);

  // Handle successful submission
  useEffect(() => {
    if (fetcher.data?.success) {
      onSuccess?.();
      onOpenChange(false);
    }
  }, [fetcher.data, onSuccess, onOpenChange]);

  const handleSelect = (personId: number) => {
    setSelectedId(personId === selectedId ? null : personId);
  };

  const handleSetClick = () => {
    if (!selectedId) return;

    const person = persons.find((p) => p.id === selectedId);
    if (person?.linked_user_id && person.linked_user_id !== currentUserId) {
      // Show warning for already-linked person
      setWarningPersonName(person.person_name);
      setWarningLinkedUser(person.linked_user_name || "another user");
      setShowWarning(true);
    } else {
      submitSelection();
    }
  };

  const submitSelection = () => {
    fetcher.submit(
      { personId: selectedId?.toString() || "" },
      { method: "post", action: `/api/collection/${collectionId}/set-member-person` }
    );
  };

  const handleConfirmWarning = () => {
    setShowWarning(false);
    submitSelection();
  };

  if (showWarning) {
    return (
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent className="max-w-sm">
          <DialogHeader>
            <DialogTitle>Person Already Linked</DialogTitle>
            <DialogDescription>
              {warningPersonName} is linked to {warningLinkedUser}. Continue anyway?
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowWarning(false)}>
              Cancel
            </Button>
            <Button onClick={handleConfirmWarning} disabled={isSubmitting}>
              {isSubmitting ? (
                <>
                  <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                  Saving...
                </>
              ) : (
                "Continue"
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    );
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Select Person</DialogTitle>
          <DialogDescription>Choose who you are in {collectionName}</DialogDescription>
        </DialogHeader>

        {/* Search input */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
          <Input
            placeholder="Search by name..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-9"
            autoComplete="off"
            data-form-type="other"
            data-1p-ignore
            data-lpignore="true"
          />
        </div>

        {/* Person list */}
        <div className="max-h-[400px] overflow-y-auto border rounded-md">
          {filteredPersons.length === 0 ? (
            <div className="p-4 text-center text-gray-500">
              {search ? "No matching persons found" : "No persons in this collection"}
            </div>
          ) : (
            <div className="divide-y">
              {filteredPersons.map((person) => {
                const isSelected = person.id === selectedId;
                const isLinkedToOther = person.linked_user_id && person.linked_user_id !== currentUserId;

                return (
                  <button
                    key={person.id}
                    type="button"
                    onClick={() => handleSelect(person.id)}
                    className={`w-full flex items-center gap-3 p-3 text-left hover:bg-gray-50 transition-colors ${
                      isSelected ? "bg-blue-50" : ""
                    }`}
                  >
                    {/* Face thumbnail */}
                    {person.detection_id ? (
                      <div className="w-10 h-10 rounded-full overflow-hidden bg-gray-100 flex-shrink-0">
                        <img
                          src={`/api/face/${person.detection_id}`}
                          alt={person.person_name}
                          className="w-full h-full object-cover"
                        />
                      </div>
                    ) : (
                      <div className="w-10 h-10 rounded-full bg-gray-200 flex items-center justify-center flex-shrink-0">
                        <User className="h-5 w-5 text-gray-400" />
                      </div>
                    )}

                    {/* Name */}
                    <span className="flex-1 truncate font-medium">{person.person_name}</span>

                    {/* Linked indicator */}
                    {isLinkedToOther && (
                      <div className="flex-shrink-0" title={`Linked to ${person.linked_user_name}`}>
                        <Link2 className="h-4 w-4 text-amber-500" />
                      </div>
                    )}
                  </button>
                );
              })}
            </div>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleSetClick} disabled={!selectedId || isSubmitting}>
            {isSubmitting ? (
              <>
                <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                Saving...
              </>
            ) : (
              "Set"
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
```

**Step 2: Commit**

```bash
git add web/app/components/person-select-modal.tsx
git commit -m "feat(ui): add person selection modal component"
```

---

## Task 6: Create Profile Page

**Files:**
- Create: `web/app/routes/profile.tsx`

**Step 1: Create the profile page**

```typescript
import { Check, ChevronDown, Loader2, User } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useFetcher, useRevalidator } from "react-router";
import { Header } from "~/components/header";
import { PersonSelectModal } from "~/components/person-select-modal";
import { Button } from "~/components/ui/button";
import { Card, CardContent } from "~/components/ui/card";
import { Input } from "~/components/ui/input";
import { useRootData } from "~/hooks/use-root-data";
import { requireUser } from "~/lib/auth.server";
import { getPersonsForCollection, getUserCollections, type PersonForSelection, type UserCollection } from "~/lib/db.server";
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
        { method: "post", action: "/api/user/set-default-collection" }
      );
    }
  };

  // Handle profile save
  const handleProfileSave = () => {
    profileFetcher.submit(
      { firstName: firstName.trim(), lastName: lastName.trim() },
      { method: "post", action: "/api/user/update-profile" }
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
                  <Button onClick={handleProfileSave} disabled={!hasProfileChanges || !firstName.trim() || isProfileSubmitting}>
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
                        </>
                      ) : (
                        <>
                          <div className="w-8 h-8 rounded-full border-2 border-dashed border-gray-300 flex items-center justify-center">
                            <User className="h-4 w-4 text-gray-400" />
                          </div>
                          <span className="text-sm text-gray-500">Select person</span>
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
```

**Step 2: Add route to routes.ts**

Add after collections route (around line 7):

```typescript
  route("profile", "routes/profile.tsx"),
```

**Step 3: Commit**

```bash
git add web/app/routes/profile.tsx web/app/routes.ts
git commit -m "feat(ui): add profile page"
```

---

## Task 7: Add Profile Menu Item to Header

**Files:**
- Modify: `web/app/components/header.tsx`

**Step 1: Add Profile menu item**

In header.tsx, add a Profile menu item after the user name display (around line 104-106, after the closing `</>` of the user name block).

Add this import at the top with the other lucide imports:
```typescript
// Already have User imported, no change needed
```

Add the Profile menu item after `<DropdownMenuSeparator />` (line 104) and before the Collections item:

```typescript
                <DropdownMenuItem asChild>
                  <Link to="/profile" className="flex items-center cursor-pointer">
                    <User className="h-4 w-4 mr-2" />
                    Profile
                  </Link>
                </DropdownMenuItem>
```

The full section should look like:
```typescript
                {user && (
                  <>
                    <div className="px-2 py-1.5 text-sm font-medium text-gray-900">
                      {user.firstName} {user.lastName || ""}
                    </div>
                    <DropdownMenuSeparator />
                  </>
                )}
                <DropdownMenuItem asChild>
                  <Link to="/profile" className="flex items-center cursor-pointer">
                    <User className="h-4 w-4 mr-2" />
                    Profile
                  </Link>
                </DropdownMenuItem>
                <DropdownMenuItem asChild>
                  <Link to="/collections" className="flex items-center cursor-pointer">
```

**Step 2: Commit**

```bash
git add web/app/components/header.tsx
git commit -m "feat(ui): add Profile menu item to header dropdown"
```

---

## Task 8: Test the Implementation

**Step 1: Start the dev server**

```bash
cd web && npm run dev
```

**Step 2: Manual testing checklist**

1. Navigate to profile page from header menu
2. Update first/last name and save
3. If multiple collections, change default collection
4. Click person selector for a collection
5. Search for a person in the modal
6. Select a person and confirm
7. Try selecting a person already linked to another user (should show warning)
8. Verify avatar updates after selection

**Step 3: Run type check**

```bash
cd web && npm run typecheck
```

**Step 4: Run lint**

```bash
cd web && npm run check
```

**Step 5: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix: address lint/type issues in profile feature"
```
