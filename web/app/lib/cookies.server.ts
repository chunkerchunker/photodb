import { data } from "react-router";

export function dataWithViewMode<T>(payload: T, mode: "grid" | "wall") {
  return data(payload, {
    headers: {
      "Set-Cookie": `viewMode=${mode}; Path=/; SameSite=Lax`,
    },
  });
}

export async function getViewModeCookie(request: Request): Promise<"grid" | "wall"> {
  const cookieHeader = request.headers.get("Cookie") || "";
  const match = cookieHeader.match(/viewMode=(grid|wall)/);
  return (match?.[1] as "grid" | "wall") || "wall";
}
