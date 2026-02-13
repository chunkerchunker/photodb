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

/**
 * Get the showWithoutImages preference from cookie.
 */
export function getShowWithoutImagesCookie(request: Request): boolean {
  const cookieHeader = request.headers.get("Cookie") || "";
  const match = cookieHeader.match(/showWithoutImages=(true|false)/);
  return match?.[1] === "true";
}

/**
 * Data response that sets the showWithoutImages cookie along with view mode.
 */
export function dataWithPreferences<T>(payload: T, mode: "grid" | "wall", showWithoutImages: boolean) {
  return data(payload, {
    headers: {
      "Set-Cookie": [
        `viewMode=${mode}; Path=/; SameSite=Lax`,
        `showWithoutImages=${showWithoutImages}; Path=/; SameSite=Lax`,
      ].join(", "),
    },
  });
}
