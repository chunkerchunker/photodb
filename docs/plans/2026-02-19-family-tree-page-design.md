# Family Tree Page – Design

**Date**: 2026-02-19
**Status**: Approved
**Figma**: PhotoDB – Family Tree Page (in Drafts)

## Overview

A new page at `person/:id/family` that provides an interactive family tree visualization centered on a specific person. Users can browse existing relationships, add new ones via drag-and-drop, and create placeholder persons for unknown family members.

## Key Decisions

- **Rendering**: React Flow for the canvas (pan/zoom, custom nodes, drag-and-drop)
- **Entry point**: New "Family" tab on existing `person/:id` routes (alongside Grid/Wall)
- **Node display**: Face avatar thumbnail + name (compact)
- **Relationship editing**: Drag persons from a right sidebar panel onto drop zones in the tree; context via double-click to re-center

## Layout

Three zones within a full-viewport frame:

1. **Header** (56px): Breadcrumbs (`People / {name} / Family`) + Grid/Wall/Family view tabs
2. **Canvas** (flex): React Flow area, dark background (`bg-gray-900`), pan/zoom enabled
3. **Right sidebar** (280px): Light background (`bg-gray-50`), search input, scrollable person cards, "+ New Placeholder Person" button

### Canvas Elements

| Element | Description |
|---|---|
| **Person node** | 140x80px card: face avatar circle + name. Dark card with subtle border. |
| **Center node** | Same as person node but blue border + blue glow shadow. |
| **Placeholder node** | Dashed border, gray avatar, italic-style name (e.g. "Baby Kim"). Uses `is_placeholder=true`. |
| **Drop zone** | Dashed outline, `+` icon, label (e.g. "Add parent"). Accepts drag events. |
| **Partnership indicator** | Small circle with heart icon on the edge connecting partners. |
| **Connection lines** | Solid lines for real relationships, dashed for placeholder/drop-zone connections. |
| **Generation labels** | Left edge: "Grandparents", "Parents", "You + Siblings", "Children". |
| **Minimap** | Bottom-left: React Flow's built-in `<MiniMap>` component. |
| **Zoom controls** | Bottom-left above minimap: +/–/percentage. React Flow's `<Controls>`. |

### Right Sidebar

- **Search box**: Client-side filter on person name (same pattern as `person-select-modal.tsx`)
- **Person cards**: 48px height, face avatar + name + photo count. Draggable via React DnD or React Flow's drag API.
- **"+ New Placeholder Person" button**: Creates a placeholder person via API, then allows immediate drag onto a drop zone. Opens a small inline form for name/role before creation.

## Interaction Model

### Browsing
- Page loads centered on `person/:id` using `get_family_tree(id, max_generations=3)`
- Pan/zoom via mouse drag and scroll wheel (React Flow built-in)
- Minimap shows full tree extent with viewport indicator

### Re-centering
- Double-click any person node to re-center the tree on them
- Updates the URL to `person/:newId/family` via React Router navigation
- Tree re-fetches with the new center person

### Adding Relationships
1. User drags a person card from the sidebar onto a drop zone
2. Drop zone highlights blue on hover (valid target)
3. On drop: API call to create the relationship (e.g. `POST /api/person/:id/parent`)
4. Tree re-renders with the new node replacing the drop zone
5. New drop zones appear for the newly added person's missing relationships

### Creating Placeholders
1. Click "+ New Placeholder Person" in sidebar
2. Small inline form: name (optional), role hint (e.g. "father", "mother")
3. Creates placeholder person via `create_placeholder_parent()` or generic person creation
4. New placeholder appears in sidebar, ready to drag

### Removing Relationships
- Right-click a connection line or person node for context menu
- "Remove relationship" option (removes the link, not the person)
- "Delete person" option (for placeholders only)

## Data Flow

### Loader (`person.$id.family.tsx`)
```
loader({ params }) →
  getPersonById(collectionId, params.id)
  getFamilyTree(collectionId, params.id, maxGenerations=3)
  getPersonsForCollection(collectionId)  // for sidebar
```

### New `db.server.ts` Functions
```
getFamilyTree(collectionId, personId, maxGenerations)
  → calls get_family_tree(personId, maxGenerations, true) SQL function
  → returns: { id, first_name, last_name, relation, generation_offset, is_placeholder, detection_id }[]

getPersonParents(collectionId, personId)
getPersonPartnerships(collectionId, personId)
getPersonSiblings(collectionId, personId)
```

### New API Routes
```
POST /api/person/:id/parent        → { parentId, role: mother|father|parent }
POST /api/person/:id/child         → { childId, role }
POST /api/person/:id/partner       → { partnerId, type, startYear? }
POST /api/person/:id/sibling       → { siblingId }
DELETE /api/person/:id/parent/:parentId
DELETE /api/person/:id/partner/:partnerId
POST /api/person/placeholder       → { name?, gender?, description? }
```

All mutation routes call `refresh_genealogy_closures()` after the write.

## Tree Layout Algorithm

React Flow requires explicit `(x, y)` positions for each node. Layout is computed client-side from the `get_family_tree()` result:

1. Group nodes by `generation_offset` (0 = center person's generation)
2. Within each generation, order by: partners adjacent to their linked person, siblings by birth order, then remaining
3. Assign Y from generation: `y = generation_offset * (nodeHeight + verticalGap)`
4. Assign X within generation: centered, with `nodeWidth + horizontalGap` spacing
5. Insert drop zone nodes at expected empty positions (missing parents, open child/sibling slots)
6. Compute edges from parent-child and partnership relationships

The center person is placed at `(0, 0)` in React Flow coordinates; React Flow handles viewport centering.

## Tech Stack

- **React Flow** (`@xyflow/react`): Canvas, nodes, edges, minimap, controls
- **Custom nodes**: `PersonNode`, `PlaceholderNode`, `DropZoneNode` (React components)
- **Drag source**: HTML5 drag from sidebar, React Flow drop handler on canvas
- **Existing reuse**: Face avatar pattern (`/api/face/:detection_id`), shadcn Dialog/Input/Button, Lucide icons

## Out of Scope (for v1)

- Full-tree overview page at `/family` (can add later)
- Printing/exporting the tree as an image
- GEDCOM import/export
- Timeline view showing photos along the tree
- Auto-layout optimization (e.g. minimizing edge crossings)
