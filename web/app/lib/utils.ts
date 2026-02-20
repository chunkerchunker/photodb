import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Check if `query` is a subsequence of `text` (case-insensitive).
 * Characters must appear in order but can be non-contiguous.
 * e.g. "jhn" matches "John", "ae" matches "Jane".
 */
export function subsequenceMatch(text: string, query: string): boolean {
  const t = text.toLowerCase();
  const q = query.toLowerCase();
  let ti = 0;
  for (let qi = 0; qi < q.length; qi++) {
    const idx = t.indexOf(q[qi], ti);
    if (idx === -1) return false;
    ti = idx + 1;
  }
  return true;
}

/**
 * Convert a search string into a SQL LIKE pattern for subsequence matching.
 * e.g. "jhn" → "%j%h%n%"
 */
export function toSubsequenceLikePattern(query: string): string {
  return `%${query.split("").join("%")}%`;
}
