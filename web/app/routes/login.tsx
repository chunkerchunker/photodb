import { Form, useActionData, useNavigation } from "react-router";
import type { Route } from "./+types/login";
import { handleLogin } from "~/lib/auth.server";

export async function action({ request }: Route.ActionArgs) {
  return await handleLogin(request);
}

export default function Login() {
  const actionData = useActionData<typeof action>() as { error?: string } | undefined;
  const navigation = useNavigation();
  const isSubmitting = navigation.state === "submitting";

  return (
    <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center px-4">
      <div className="w-full max-w-md bg-gray-800 border border-gray-700 rounded-xl p-8 shadow-lg">
        <h1 className="text-2xl font-semibold">Sign in</h1>
        <p className="text-sm text-gray-400 mt-2">Use your username and password.</p>

        <Form method="post" className="mt-6 space-y-4">
          <div>
            <label htmlFor="username" className="block text-sm font-medium text-gray-300">
              Username
            </label>
            <input
              id="username"
              name="username"
              type="text"
              autoComplete="username"
              className="mt-1 w-full rounded-md bg-gray-900 border border-gray-700 px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              required
            />
          </div>
          <div>
            <label htmlFor="password" className="block text-sm font-medium text-gray-300">
              Password
            </label>
            <input
              id="password"
              name="password"
              type="password"
              autoComplete="current-password"
              className="mt-1 w-full rounded-md bg-gray-900 border border-gray-700 px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              required
            />
          </div>

          {actionData?.error && (
            <div className="rounded-md bg-red-900/40 border border-red-800 text-red-200 px-3 py-2 text-sm">
              {actionData.error}
            </div>
          )}

          <button
            type="submit"
            className="w-full rounded-md bg-blue-600 hover:bg-blue-500 transition-colors px-4 py-2 font-medium"
            disabled={isSubmitting}
          >
            {isSubmitting ? "Signing in..." : "Sign in"}
          </button>
        </Form>
      </div>
    </div>
  );
}
