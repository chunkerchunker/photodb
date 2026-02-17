import { Eye, Loader2, User } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useFetcher } from "react-router";
import { Breadcrumb } from "~/components/breadcrumb";
import { Layout } from "~/components/layout";
import { Button } from "~/components/ui/button";
import { Card, CardContent } from "~/components/ui/card";
import { requireCollectionId } from "~/lib/auth.server";
import { getHiddenPeople, getHiddenPeopleCount, setPersonHidden } from "~/lib/db.server";
import type { Route } from "./+types/people.hidden";

export function meta() {
  return [
    { title: "Storyteller - Hidden People" },
    {
      name: "description",
      content: "View and manage hidden people",
    },
  ];
}

export async function action({ request }: Route.ActionArgs) {
  const { collectionId } = await requireCollectionId(request);
  const formData = await request.formData();
  const intent = formData.get("intent");

  if (intent === "unhide") {
    const personId = formData.get("personId") as string;
    if (personId) {
      const result = await setPersonHidden(collectionId, personId, false);
      return result;
    }
    return { success: false, message: "Invalid person ID" };
  }

  return { success: false, message: "Unknown action" };
}

const LIMIT = 24;

export async function loader({ request }: Route.LoaderArgs) {
  const { collectionId } = await requireCollectionId(request);
  const url = new URL(request.url);
  const page = parseInt(url.searchParams.get("page") || "1", 10);
  const offset = (page - 1) * LIMIT;

  try {
    const people = await getHiddenPeople(collectionId, LIMIT, offset);
    const totalPeople = await getHiddenPeopleCount(collectionId);
    const hasMore = offset + people.length < totalPeople;

    return {
      people,
      totalPeople,
      hasMore,
      page,
    };
  } catch (error) {
    console.error("Failed to load hidden people:", error);
    return {
      people: [],
      totalPeople: 0,
      hasMore: false,
      page,
    };
  }
}

type Person = Route.ComponentProps["loaderData"]["people"][number];

export default function HiddenPeopleView({ loaderData }: Route.ComponentProps) {
  const { people: initialPeople, totalPeople, hasMore: initialHasMore, page: initialPage } = loaderData;
  const fetcher = useFetcher();

  // Infinite scroll state
  const [people, setPeople] = useState<Person[]>(initialPeople);
  const [page, setPage] = useState(initialPage);
  const [hasMore, setHasMore] = useState(initialHasMore);
  const scrollFetcher = useFetcher<typeof loader>();
  const loadMoreRef = useRef<HTMLDivElement>(null);

  // Reset state when initial data changes (e.g., navigation or action)
  useEffect(() => {
    setPeople(initialPeople);
    setPage(initialPage);
    setHasMore(initialHasMore);
  }, [initialPeople, initialPage, initialHasMore]);

  // Append new people when scroll fetcher returns data
  useEffect(() => {
    if (scrollFetcher.data?.people && scrollFetcher.data.people.length > 0) {
      setPeople((prev) => {
        const existingIds = new Set(prev.map((p) => p.id));
        const newPeople = scrollFetcher.data?.people.filter((p) => !existingIds.has(p.id)) ?? [];
        return [...prev, ...newPeople];
      });
      setPage(scrollFetcher.data.page);
      setHasMore(scrollFetcher.data.hasMore);
    }
  }, [scrollFetcher.data]);

  const loadMore = useCallback(() => {
    if (scrollFetcher.state === "idle" && hasMore) {
      scrollFetcher.load(`/people/hidden?page=${page + 1}`);
    }
  }, [scrollFetcher, hasMore, page]);

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

  const isLoadingMore = scrollFetcher.state === "loading";

  const handleUnhide = (personId: string) => {
    fetcher.submit({ intent: "unhide", personId }, { method: "post" });
  };

  const breadcrumbItems = [{ label: "People", href: "/people" }, { label: "Hidden" }];

  return (
    <Layout>
      <div className="space-y-6">
        <Breadcrumb items={breadcrumbItems} />

        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <User className="h-8 w-8 text-gray-700" />
            <h1 className="text-3xl font-bold text-gray-900">Hidden People</h1>
          </div>
          <span className="text-gray-600">
            {totalPeople} hidden {totalPeople === 1 ? "person" : "people"}
          </span>
        </div>

        {people.length > 0 ? (
          <>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-4">
              {people.map((person) => (
                <Card key={person.id} className="h-full">
                  <CardContent className="p-4">
                    <div className="text-center space-y-3">
                      <Link to={`/person/${person.id}`}>
                        {person.detection_id ? (
                          <div className="w-32 h-32 mx-auto bg-gray-100 rounded-lg border overflow-hidden opacity-60 hover:opacity-100 transition-opacity">
                            <img
                              src={`/api/face/${person.detection_id}`}
                              alt={person.person_name || `Person ${person.id}`}
                              className="w-full h-full object-cover"
                              loading="lazy"
                            />
                          </div>
                        ) : (
                          <div className="w-full h-32 bg-gray-200 rounded-lg flex items-center justify-center opacity-60">
                            <User className="h-8 w-8 text-gray-400" />
                          </div>
                        )}
                      </Link>

                      <div className="space-y-1">
                        <div className="font-semibold text-gray-900 truncate" title={person.person_name}>
                          {person.person_name || `Person ${person.id}`}
                        </div>
                        <div className="text-sm text-gray-600">
                          {person.total_face_count} photo{person.total_face_count !== 1 ? "s" : ""}
                          {person.cluster_count > 1 && (
                            <span className="text-gray-400"> · {person.cluster_count} clusters</span>
                          )}
                        </div>
                      </div>

                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleUnhide(person.id.toString())}
                        disabled={fetcher.state === "submitting"}
                      >
                        <Eye className="h-4 w-4 mr-1" />
                        Unhide
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            {/* Infinite scroll trigger */}
            <div ref={loadMoreRef} className="flex justify-center py-8">
              {isLoadingMore && (
                <div className="flex items-center space-x-2 text-gray-500">
                  <Loader2 className="h-5 w-5 animate-spin" />
                  <span>Loading more people...</span>
                </div>
              )}
              {!hasMore && people.length > 0 && <span className="text-gray-400 text-sm">All hidden people loaded</span>}
            </div>
          </>
        ) : (
          <div className="text-center py-12">
            <User className="h-16 w-16 text-gray-400 mx-auto mb-4" />
            <div className="text-gray-500 text-lg">No hidden people.</div>
            <div className="text-gray-400 text-sm mt-2">
              <Link to="/people" className="text-blue-500 hover:underline">
                View all people
              </Link>
            </div>
          </div>
        )}
      </div>
    </Layout>
  );
}
