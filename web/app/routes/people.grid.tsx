import { ArrowDownAZ, EyeOff, Grid, Loader2, Pencil, Search, User, Users } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { Link, useFetcher, useNavigate, useRevalidator } from "react-router";
import { CoverflowIcon } from "~/components/coverflow-icon";
import { Layout } from "~/components/layout";
import { RenamePersonDialog } from "~/components/rename-person-dialog";
import { SearchBox } from "~/components/search-box";
import { Card, CardContent } from "~/components/ui/card";
import { ContextMenu, ContextMenuContent, ContextMenuItem, ContextMenuTrigger } from "~/components/ui/context-menu";
import { ViewSwitcher } from "~/components/view-switcher";
import { useInfiniteScroll } from "~/hooks/use-infinite-scroll";
import { dataWithViewMode } from "~/lib/cookies.server";
import { getPeople, getPeopleCount } from "~/lib/db.server";
import { getFaceCropStyle } from "~/lib/face-crop";
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

  // Context menu state
  const [contextPerson, setContextPerson] = useState<Person | null>(null);
  const [renameDialogOpen, setRenameDialogOpen] = useState(false);
  const hideFetcher = useFetcher();
  const revalidator = useRevalidator();

  const fetcher = useFetcher<typeof loader>();

  // Reset state when initial data changes
  useEffect(() => {
    setPeople(initialPeople);
    setPage(initialPage);
    setHasMore(initialHasMore);
    setSort(initialSort);
  }, [initialPeople, initialPage, initialHasMore, initialSort]);

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

  const loadMoreRef = useInfiniteScroll({
    onLoadMore: loadMore,
    hasMore,
    isLoading: fetcher.state === "loading",
  });

  const isLoading = fetcher.state === "loading";

  // Context menu handlers
  const handleRename = (person: Person) => {
    setContextPerson(person);
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

  const handleRenameSuccess = (newFirstName: string, newLastName: string) => {
    if (contextPerson) {
      const newName = newLastName ? `${newFirstName} ${newLastName}` : newFirstName;
      setPeople((prev) => prev.map((p) => (p.id === contextPerson.id ? { ...p, person_name: newName } : p)));
      setContextPerson(null);
    }
  };

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
            <ViewSwitcher
              variant="light"
              modes={[
                { key: "grid", label: "Grid View", icon: <Grid className="h-4 w-4" />, isActive: true },
                {
                  key: "wall",
                  label: "3D Wall",
                  icon: <CoverflowIcon className="size-4" />,
                  to: `/people/wall${sort !== "name" ? `?sort=${sort}` : ""}`,
                  isActive: false,
                },
              ]}
            />
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

        <SearchBox
          open={searchOpen}
          onOpenChange={setSearchOpen}
          query={searchQuery}
          onQueryChange={setSearchQuery}
          placeholder="Search by name..."
          resultCount={searchQuery ? filteredPeople.length : undefined}
        />

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
        {contextPerson && (
          <RenamePersonDialog
            open={renameDialogOpen}
            onOpenChange={setRenameDialogOpen}
            personId={contextPerson.id.toString()}
            currentFirstName={contextPerson.person_name?.split(" ")[0] || ""}
            currentLastName={contextPerson.person_name?.split(" ").slice(1).join(" ") || ""}
            onSuccess={handleRenameSuccess}
          />
        )}
      </div>
    </Layout>
  );
}
