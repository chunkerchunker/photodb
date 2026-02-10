import {
  isRouteErrorResponse,
  Links,
  Meta,
  Outlet,
  redirect,
  Scripts,
  ScrollRestoration,
  useLoaderData,
} from "react-router";

import type { Route } from "./+types/root";
import "./app.css";
import type { UserAvatarInfo } from "~/components/header";
import { getSessionInfo } from "~/lib/auth.server";
import { getCollectionMemberInfo } from "~/lib/db.server";

export const links: Route.LinksFunction = () => [
  { rel: "preconnect", href: "https://fonts.googleapis.com" },
  {
    rel: "preconnect",
    href: "https://fonts.gstatic.com",
    crossOrigin: "anonymous",
  },
  {
    rel: "stylesheet",
    href: "https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&display=swap",
  },
];

export type RootLoaderData = {
  user: {
    id: number;
    firstName: string;
    lastName: string | null;
    isAdmin: boolean;
  };
  userAvatar: UserAvatarInfo;
  impersonation: {
    isImpersonating: boolean;
    realAdminName: string | null;
  };
} | null;

export async function loader({ request }: Route.LoaderArgs): Promise<RootLoaderData> {
  const { pathname } = new URL(request.url);
  const publicPaths = new Set<string>(["/login", "/logout"]);

  if (publicPaths.has(pathname)) {
    return null;
  }

  const sessionInfo = await getSessionInfo(request);
  if (!sessionInfo) {
    throw redirect("/login");
  }

  const { realUser, effectiveUser, isImpersonating } = sessionInfo;

  // Get collection member info for avatar (for the effective user)
  let userAvatar: UserAvatarInfo = {
    firstName: effectiveUser.first_name,
    lastName: effectiveUser.last_name,
    avatarDetectionId: null,
  };

  if (effectiveUser.default_collection_id) {
    const memberInfo = await getCollectionMemberInfo(effectiveUser.id, effectiveUser.default_collection_id);
    if (memberInfo) {
      userAvatar = {
        firstName: effectiveUser.first_name,
        lastName: effectiveUser.last_name,
        avatarDetectionId: memberInfo.avatar_detection_id,
      };
    }
  }

  return {
    user: {
      id: effectiveUser.id,
      firstName: effectiveUser.first_name,
      lastName: effectiveUser.last_name,
      isAdmin: realUser.is_admin,
    },
    userAvatar,
    impersonation: {
      isImpersonating,
      realAdminName: isImpersonating ? `${realUser.first_name} ${realUser.last_name}`.trim() : null,
    },
  };
}

export function Layout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <Meta />
        <Links />
      </head>
      <body>
        {children}
        <ScrollRestoration />
        <Scripts />
      </body>
    </html>
  );
}

export default function App() {
  const data = useLoaderData<typeof loader>();
  return <Outlet context={data} />;
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  let message = "Oops!";
  let details = "An unexpected error occurred.";
  let stack: string | undefined;

  if (isRouteErrorResponse(error)) {
    message = error.status === 404 ? "404" : "Error";
    details = error.status === 404 ? "The requested page could not be found." : error.statusText || details;
  } else if (import.meta.env.DEV && error && error instanceof Error) {
    details = error.message;
    stack = error.stack;
  }

  return (
    <main className="pt-16 p-4 container mx-auto">
      <h1>{message}</h1>
      <p>{details}</p>
      {stack && (
        <pre className="w-full p-4 overflow-x-auto">
          <code>{stack}</code>
        </pre>
      )}
    </main>
  );
}
