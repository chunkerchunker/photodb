import { data } from "react-router";

export function dataWithViewMode<T>(payload: T, mode: "grid" | "wall") {
  return data(payload, {
    headers: {
      "Set-Cookie": `viewMode=${mode}; Path=/; SameSite=Lax`,
    },
  });
}
