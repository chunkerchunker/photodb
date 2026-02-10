import { Shield, UserCog, Users } from "lucide-react";
import { useFetcher } from "react-router";
import { Header } from "~/components/header";
import { Button } from "~/components/ui/button";
import { Card, CardContent } from "~/components/ui/card";
import { useRootData } from "~/hooks/use-root-data";
import { requireAdmin } from "~/lib/auth.server";
import { getAllUsers } from "~/lib/db.server";
import type { Route } from "./+types/admin.users";

export function meta() {
  return [{ title: "Storyteller - Admin - Users" }, { name: "description", content: "Manage users" }];
}

export async function loader({ request }: Route.LoaderArgs) {
  await requireAdmin(request);
  const users = await getAllUsers();
  return { users };
}

export default function AdminUsersPage({ loaderData }: Route.ComponentProps) {
  const { users } = loaderData;
  const rootData = useRootData();
  const fetcher = useFetcher();

  const handleImpersonate = (userId: number) => {
    fetcher.submit({ userId: userId.toString() }, { method: "post", action: "/api/admin/impersonate" });
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Header
        user={rootData?.userAvatar}
        isAdmin={rootData?.user?.isAdmin}
        isImpersonating={rootData?.impersonation?.isImpersonating}
        breadcrumbs={[{ label: "Admin" }, { label: "Users" }]}
      />
      <div className="h-16 bg-gray-900" />

      <main className="container mx-auto px-4 py-6">
        <div className="space-y-6">
          <div className="flex items-center space-x-3">
            <Shield className="h-8 w-8 text-gray-700" />
            <h1 className="text-3xl font-bold text-gray-900">User Management</h1>
          </div>

          <p className="text-gray-600">View all users and impersonate them to troubleshoot issues.</p>

          {users.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {users.map((user) => {
                const isCurrentUser = user.id === rootData?.user?.id;
                const fullName = `${user.first_name} ${user.last_name}`.trim();

                return (
                  <Card key={user.id} className="hover:shadow-md transition-shadow">
                    <CardContent className="p-4">
                      <div className="flex items-start justify-between">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center space-x-2">
                            <h3 className="font-semibold text-gray-900 truncate">{fullName}</h3>
                            {user.is_admin && (
                              <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-purple-100 text-purple-800">
                                Admin
                              </span>
                            )}
                          </div>
                          <p className="text-sm text-gray-500 truncate">@{user.username}</p>
                          <div className="mt-2 flex items-center text-sm text-gray-500">
                            <Users className="h-4 w-4 mr-1" />
                            {user.collection_count} collection{user.collection_count !== 1 ? "s" : ""}
                          </div>
                        </div>

                        {!isCurrentUser && (
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handleImpersonate(user.id)}
                            disabled={fetcher.state !== "idle"}
                            className="ml-2 flex-shrink-0"
                          >
                            <UserCog className="h-4 w-4 mr-1" />
                            Impersonate
                          </Button>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          ) : (
            <div className="text-center py-12">
              <Users className="h-16 w-16 text-gray-400 mx-auto mb-4" />
              <div className="text-gray-500 text-lg">No users found.</div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
