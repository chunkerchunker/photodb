# User Profile Page Design

## Overview

Add a user profile page accessible from the header profile menu. The page allows users to:
- Update their first/last name
- Select a default collection (if they have multiple)
- Link themselves to a person in each collection they belong to

## Page Structure

**Route:** `/profile`
**File:** `routes/profile.tsx`

**Menu Addition:** Add "Profile" item to the header dropdown menu between the user name display and "Collections" link, with a `User` icon.

**Layout:** Card-based sections using the standard `<Layout>` wrapper with page header (User icon + "Profile" title).

## Card Sections

### 1. Account Details Card

- **Title:** "Account Details"
- **Fields:** First Name (required) | Last Name (optional) - side by side
- **Save button:** Right-aligned, disabled until changes made
- **Behavior:**
  - Pre-populated from current user data
  - Save enables when values differ from original
  - Submits via `useFetcher()` to `POST /api/user/update-profile`
  - Loading spinner during submission
  - Success toast: "Profile updated"

### 2. Default Collection Card

- **Title:** "Default Collection"
- **Description:** "Choose which collection loads when you sign in"
- **Control:** Dropdown showing all user's collections
- **Visibility:** Only shown if user has 2+ collections
- **Behavior:**
  - Current default pre-selected
  - Selection change saves immediately (no Save button)
  - Brief loading state on select while saving
  - Success toast: "Default collection updated"

### 3. Collection Memberships Card

- **Title:** "Your Collections"
- **Description:** "Link yourself to a person in each collection for your avatar"
- **Content:** List of collection rows

**Each Row:**
- Collection name (left)
- Person selector button (right):
  - If linked: face thumbnail (32px circle) + person name
  - If not linked: dashed outline circle with User icon + "Select person"
- Click opens person selection modal

## Person Selection Modal

**Structure:**
- Title: "Select Person"
- Subtitle: "Choose who you are in [Collection Name]"
- Search input at top: placeholder "Search by name..."
- Scrollable person list (max height ~400px)
- Footer: Cancel | Set buttons

**Person Row:**
- Face thumbnail (40px circle)
- Person name
- Link icon (right) if already linked to another user
  - Tooltip: "Linked to [username]"
- Hover highlight; selected row has distinct background (primary/10)

**Behavior:**
- Client-side search filtering as user types
- Single selection - click selects, deselects previous
- Cancel closes without changes
- Set disabled until selection made

**Warning Flow (for already-linked persons):**
1. On Set click: confirmation dialog "This person is linked to [username]. Continue anyway?"
2. Confirm proceeds; Cancel returns to modal

**Save:**
- `POST /api/collection/{id}/set-member-person`
- On success: close modal, refresh row, toast "Person linked"

## API Endpoints

### POST /api/user/update-profile

- **File:** `routes/api.user.update-profile.tsx`
- **Body:** `firstName`, `lastName`
- **Action:** Updates `app_user.first_name`, `app_user.last_name`
- **Auth:** Requires authenticated user
- **Response:** `{ success: boolean, message: string }`

### POST /api/user/set-default-collection

- **File:** `routes/api.user.set-default-collection.tsx`
- **Body:** `collectionId`
- **Action:** Updates `app_user.default_collection_id`
- **Validation:** User must be member of collection
- **Auth:** Requires authenticated user
- **Response:** `{ success: boolean, message: string }`

### POST /api/collection/:id/set-member-person

- **File:** `routes/api.collection.$id.set-member-person.tsx`
- **Body:** `personId` (or null to unlink)
- **Action:** Updates `collection_member.person_id`
- **Auth:** User must be member of that collection
- **Response:** `{ success: boolean, message: string }`

## Database Functions

Add to `lib/db.server.ts`:

```typescript
// Get all persons in a collection with linked user info
getPersonsForCollection(collectionId: number): Promise<{
  id: number;
  firstName: string;
  lastName: string | null;
  detectionId: number | null;
  linkedUserId: number | null;
  linkedUserName: string | null;
}[]>

// Update user profile
updateUserProfile(userId: number, firstName: string, lastName: string | null): Promise<void>

// Set default collection
setUserDefaultCollection(userId: number, collectionId: number): Promise<void>

// Set person link for collection membership
setCollectionMemberPerson(userId: number, collectionId: number, personId: number | null): Promise<void>
```

## Files to Create/Modify

| File | Action |
|------|--------|
| `routes/profile.tsx` | Create - main profile page |
| `routes/api.user.update-profile.tsx` | Create - profile update endpoint |
| `routes/api.user.set-default-collection.tsx` | Create - default collection endpoint |
| `routes/api.collection.$id.set-member-person.tsx` | Create - person link endpoint |
| `components/person-select-modal.tsx` | Create - reusable person selection modal |
| `components/header.tsx` | Modify - add Profile menu item |
| `lib/db.server.ts` | Modify - add database functions |
| `routes.ts` | Modify - add /profile route |
