import { ArrowDownAZ, EyeOff, Grid, Loader2, Pencil, Search, User, Users, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useFetcher, useNavigate, useRevalidator } from "react-router";
import { CoverflowIcon } from "~/components/coverflow-icon";
import { Layout } from "~/components/layout";
import { Button } from "~/components/ui/button";
import { Card, CardContent } from "~/components/ui/card";
import { ContextMenu, ContextMenuContent, ContextMenuItem, ContextMenuTrigger } from "~/components/ui/context-menu";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "~/components/ui/dialog";
import { Input } from "~/components/ui/input";
import { dataWithViewMode } from "~/lib/cookies.server";
import { getPeople, getPeopleCount } from "~/lib/db.server";
import type { Route } from "./+types/people.grid";

export function meta() {
  return [
    { title: "PhotoDB - People" },
    {
      name: "description",
      content: "Browse identified people",
    },
  ];
}

const LIMIT = 24; // 4x6 grid

export async function loader({ request }: Route.LoaderArgs) {
  const url = new URL(request.url);
  const page = parseInt(url.searchParams.get("page") || "1", 10);
  const sortParam = url.searchParams.get("sort");
  const sort: "photos" | "name" = sortParam === "photos" ? "photos" : "name";
  const offset = (page - 1) * LIMIT;

  try {
    const people = await getPeople(LIMIT, offset, sort);
    const totalPeople = await getPeopleCount();
    const hasMore = offset + people.length < totalPeople;

    return dataWithViewMode(
      {
        people,
        totalPeople,
        hasMore,
        page,
        sort,
      },
      "grid",
    );
  } catch (error) {
    console.error("Failed to load people:", error);
    return dataWithViewMode(
      {
        people: [],
        totalPeople: 0,
        hasMore: false,
        page,
        sort,
      },
      "grid",
    );
  }
}

function getFaceCropStyle(
  bbox: {
    bbox_x: number;
    bbox_y: number;
    bbox_width: number;
    bbox_height: number;
  },
  imageWidth: number,
  imageHeight: number,
) {
  const containerSize = 128;
  const scaleX = containerSize / bbox.bbox_width;
  const scaleY = containerSize / bbox.bbox_height;

  const left = -bbox.bbox_x * scaleX;
  const top = -bbox.bbox_y * scaleY;
  const width = imageWidth * scaleX;
  const height = imageHeight * scaleY;

  return {
    transform: `translate(${left}px, ${top}px)`,
    transformOrigin: "0 0",
    width: `${width}px`,
    height: `${height}px`,
  };
}

type Person = Route.ComponentProps["loaderData"]["people"][number];

export default function PeopleView({ loaderData }: Route.ComponentProps) {
  const {
    people: initialPeople,
    totalPeople,
    hasMore: initialHasMore,
    page: initialPage,
    sort: initialSort,
  } = loaderData;

  const [people, setPeople] = useState<Person[]>(initialPeople);
  const [page, setPage] = useState(initialPage);
  const [hasMore, setHasMore] = useState(initialHasMore);
  const [sort, setSort] = useState(initialSort);
  const navigate = useNavigate();

  // Search state
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const searchInputRef = useRef<HTMLInputElement>(null);

  // Context menu state
  const [contextPerson, setContextPerson] = useState<Person | null>(null);
  const [renameDialogOpen, setRenameDialogOpen] = useState(false);
  const [editFirstName, setEditFirstName] = useState("");
  const [editLastName, setEditLastName] = useState("");
  const renameFetcher = useFetcher();
  const hideFetcher = useFetcher();
  const revalidator = useRevalidator();

  const fetcher = useFetcher<typeof loader>();
  const loadMoreRef = useRef<HTMLDivElement>(null);

  // Reset state when initial data changes
  useEffect(() => {
    setPeople(initialPeople);
    setPage(initialPage);
    setHasMore(initialHasMore);
    setSort(initialSort);
  }, [initialPeople, initialPage, initialHasMore, initialSort]);

  // Keyboard shortcut for search (Cmd+F / Ctrl+F)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "f") {
        e.preventDefault();
        setSearchOpen(true);
      }
      if (e.key === "Escape" && searchOpen) {
        setSearchOpen(false);
        setSearchQuery("");
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [searchOpen]);

  // Focus search input when opened
  useEffect(() => {
    if (searchOpen && searchInputRef.current) {
      searchInputRef.current.focus();
    }
  }, [searchOpen]);

  // Append new people when fetcher returns data
  useEffect(() => {
    if (fetcher.data && fetcher.data.people.length > 0) {
      setPeople((prev) => {
        const existingIds = new Set(prev.map((p) => p.id));
        const newPeople = fetcher.data!.people.filter((p) => !existingIds.has(p.id));
        return [...prev, ...newPeople];
      });
      setPage(fetcher.data.page);
      setHasMore(fetcher.data.hasMore);
      setSort(fetcher.data.sort);
    }
  }, [fetcher.data]);

  const loadMore = useCallback(() => {
    if (fetcher.state === "idle" && hasMore) {
      fetcher.load(`/people?page=${page + 1}&sort=${sort}`);
    }
  }, [fetcher, hasMore, page, sort]);

  // Intersection Observer for infinite scroll
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          loadMore();
        }
      },
      { rootMargin: "200px" },
    );

    const currentRef = loadMoreRef.current;
    if (currentRef) {
      observer.observe(currentRef);
    }

    return () => {
      if (currentRef) {
        observer.unobserve(currentRef);
      }
    };
  }, [loadMore]);

  const isLoading = fetcher.state === "loading";

  // Context menu handlers
  const handleRename = (person: Person) => {
    setContextPerson(person);
    const parts = person.person_name?.split(" ") || [];
    setEditFirstName(parts[0] || "");
    setEditLastName(parts.slice(1).join(" ") || "");
    setRenameDialogOpen(true);
  };

  const handleHide = (person: Person) => {
    hideFetcher.submit({ hidden: "true" }, { method: "post", action: `/api/person/${person.id}/hide` });
  };

  // Revalidate after hide completes
  useEffect(() => {
    if (hideFetcher.data?.success) {
      revalidator.revalidate();
    }
  }, [hideFetcher.data, revalidator]);

  const handleSaveRename = () => {
    if (contextPerson && editFirstName.trim()) {
      renameFetcher.submit(
        { firstName: editFirstName.trim(), lastName: editLastName.trim() },
        { method: "post", action: `/api/person/${contextPerson.id}/rename` },
      );
    }
  };

  // Update local state when rename completes
  useEffect(() => {
    if (renameFetcher.data?.success && contextPerson) {
      const newName = editLastName.trim() ? `${editFirstName.trim()} ${editLastName.trim()}` : editFirstName.trim();
      setPeople((prev) => prev.map((p) => (p.id === contextPerson.id ? { ...p, person_name: newName } : p)));
      setRenameDialogOpen(false);
      setContextPerson(null);
    }
  }, [renameFetcher.data, contextPerson, editFirstName, editLastName]);

  // Filter people based on search query
  const filteredPeople = searchQuery.trim()
    ? people.filter((person) => {
        const query = searchQuery.toLowerCase().trim();
        if (person.id.toString().includes(query)) return true;
        if (person.person_name?.toLowerCase().includes(query)) return true;
        return false;
      })
    : people;

  return (
    <Layout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <User className="h-8 w-8 text-gray-700" />
            <h1 className="text-3xl font-bold text-gray-900">People</h1>
          </div>
          <div className="flex items-center space-x-4">
            <div className="flex items-center rounded-lg border bg-gray-50 p-1" title="View mode">
              <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium bg-white text-gray-900 shadow-sm">
                <Grid className="h-4 w-4" />
              </div>
              <Link
                to={`/people/wall${sort !== "name" ? `?sort=${sort}` : ""}`}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium text-gray-500 hover:text-gray-700 transition-colors"
              >
                <CoverflowIcon className="size-4" />
              </Link>
            </div>
            <span className="text-gray-600">
              {totalPeople} {totalPeople === 1 ? "person" : "people"}
            </span>
            <div className="flex items-center rounded-lg border bg-gray-50 p-1" title="Sort order">
              <button
                type="button"
                onClick={() => navigate("/people/grid?sort=photos")}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                  sort === "photos" ? "bg-white text-gray-900 shadow-sm" : "text-gray-500 hover:text-gray-700"
                }`}
                title="Sort by most photos"
              >
                <Users className="h-4 w-4" />
              </button>
              <button
                type="button"
                onClick={() => navigate("/people/grid?sort=name")}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                  sort === "name" ? "bg-white text-gray-900 shadow-sm" : "text-gray-500 hover:text-gray-700"
                }`}
                title="Sort alphabetically"
              >
                <ArrowDownAZ className="h-4 w-4" />
              </button>
            </div>
          </div>
        </div>

        {/* Search Box */}
        {searchOpen && (
          <div className="relative w-full max-w-lg mx-auto mb-4">
            <div className="bg-white rounded-lg shadow-lg border">
              <div className="flex items-center px-4 py-3">
                <Search className="h-5 w-5 text-gray-400 mr-3" />
                <input
                  ref={searchInputRef}
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search by name..."
                  className="flex-1 outline-none text-lg placeholder:text-gray-400"
                  autoComplete="off"
                />
                {searchQuery && (
                  <button
                    type="button"
                    onClick={() => setSearchQuery("")}
                    className="p-1 hover:bg-gray-100 rounded mr-2"
                  >
                    <X className="h-4 w-4 text-gray-400" />
                  </button>
                )}
                <button
                  type="button"
                  onClick={() => {
                    setSearchOpen(false);
                    setSearchQuery("");
                  }}
                  className="text-xs text-gray-400 hover:text-gray-600"
                >
                  <kbd className="px-1.5 py-0.5 bg-gray-100 rounded">Esc</kbd>
                </button>
              </div>
              {searchQuery && (
                <div className="px-4 py-2 text-xs text-gray-500 border-t">
                  {filteredPeople.length} result{filteredPeople.length !== 1 ? "s" : ""}
                </div>
              )}
            </div>
          </div>
        )}

        {people.length > 0 ? (
          <>
            {searchQuery && filteredPeople.length === 0 ? (
              <div className="text-center py-12">
                <Search className="h-12 w-12 text-gray-300 mx-auto mb-4" />
                <div className="text-gray-500">No people match "{searchQuery}"</div>
                <button
                  type="button"
                  onClick={() => setSearchQuery("")}
                  className="mt-2 text-sm text-blue-600 hover:underline"
                >
                  Clear search
                </button>
              </div>
            ) : (
              <>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-4">
                  {filteredPeople.map((person) => (
                    <ContextMenu key={person.id}>
                      <ContextMenuTrigger asChild>
                        <div>
                          <Link to={`/person/${person.id}`}>
                            <Card className="hover:shadow-lg transition-all h-full">
                              <CardContent className="p-4">
                                <div className="text-center space-y-3">
                                  {person.photo_id &&
                                  person.bbox_x !== null &&
                                  person.normalized_width &&
                                  person.normalized_height ? (
                                    <div className="relative w-32 h-32 mx-auto bg-gray-100 rounded-lg border overflow-hidden">
                                      <img
                                        src={`/api/image/${person.photo_id}`}
                                        alt={person.person_name || `Person ${person.id}`}
                                        className="absolute max-w-none max-h-none"
                                        style={getFaceCropStyle(
                                          {
                                            bbox_x: person.bbox_x,
                                            bbox_y: person.bbox_y,
                                            bbox_width: person.bbox_width,
                                            bbox_height: person.bbox_height,
                                          },
                                          person.normalized_width,
                                          person.normalized_height,
                                        )}
                                        loading="lazy"
                                      />
                                    </div>
                                  ) : (
                                    <div className="w-full h-32 bg-gray-200 rounded-lg flex items-center justify-center">
                                      <User className="h-8 w-8 text-gray-400" />
                                    </div>
                                  )}

                                  <div className="space-y-1">
                                    <div className="font-semibold text-gray-900 truncate" title={person.person_name}>
                                      {person.person_name}
                                    </div>
                                    <div className="text-sm text-gray-600">
                                      {person.total_face_count} photo{person.total_face_count !== 1 ? "s" : ""}
                                      {person.cluster_count > 1 && (
                                        <span className="text-gray-400"> · {person.cluster_count} clusters</span>
                                      )}
                                    </div>
                                  </div>
                                </div>
                              </CardContent>
                            </Card>
                          </Link>
                        </div>
                      </ContextMenuTrigger>
                      <ContextMenuContent>
                        <ContextMenuItem onClick={() => handleRename(person)}>
                          <Pencil className="h-4 w-4 mr-2" />
                          Rename
                        </ContextMenuItem>
                        <ContextMenuItem onClick={() => handleHide(person)}>
                          <EyeOff className="h-4 w-4 mr-2" />
                          Hide
                        </ContextMenuItem>
                      </ContextMenuContent>
                    </ContextMenu>
                  ))}
                </div>

                {/* Infinite scroll trigger */}
                <div ref={loadMoreRef} className="flex justify-center py-8">
                  {isLoading && (
                    <div className="flex items-center space-x-2 text-gray-500">
                      <Loader2 className="h-5 w-5 animate-spin" />
                      <span>Loading more people...</span>
                    </div>
                  )}
                  {!hasMore && people.length > 0 && !searchQuery && (
                    <span className="text-gray-400 text-sm">All people loaded</span>
                  )}
                </div>
              </>
            )}
          </>
        ) : (
          <div className="text-center py-12">
            <User className="h-16 w-16 text-gray-400 mx-auto mb-4" />
            <div className="text-gray-500 text-lg">No identified people yet.</div>
            <div className="text-gray-400 text-sm mt-2">
              People appear here when face clusters are named or linked together.
            </div>
          </div>
        )}

        {/* Rename Dialog */}
        <Dialog open={renameDialogOpen} onOpenChange={setRenameDialogOpen}>
          <DialogContent className="max-w-sm">
            <DialogHeader>
              <DialogTitle>{contextPerson?.person_name ? "Rename Person" : "Set Name"}</DialogTitle>
              <DialogDescription>Enter the person's name.</DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <label htmlFor="firstName" className="text-sm font-medium">
                  First Name
                </label>
                <Input
                  id="firstName"
                  value={editFirstName}
                  onChange={(e) => setEditFirstName(e.target.value)}
                  placeholder="First name"
                  autoFocus
                />
              </div>
              <div className="space-y-2">
                <label htmlFor="lastName" className="text-sm font-medium">
                  Last Name
                </label>
                <Input
                  id="lastName"
                  value={editLastName}
                  onChange={(e) => setEditLastName(e.target.value)}
                  placeholder="Last name (optional)"
                />
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setRenameDialogOpen(false)}>
                Cancel
              </Button>
              <Button onClick={handleSaveRename} disabled={!editFirstName.trim()}>
                Save
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    </Layout>
  );
}
