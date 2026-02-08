import { Search, X } from "lucide-react";
import { useEffect, useRef } from "react";

interface SearchBoxProps {
  /** Whether the search box is open */
  open: boolean;
  /** Callback when open state changes */
  onOpenChange: (open: boolean) => void;
  /** Current search query */
  query: string;
  /** Callback when query changes */
  onQueryChange: (query: string) => void;
  /** Placeholder text for the input */
  placeholder?: string;
  /** Number of results to display */
  resultCount?: number;
}

/**
 * Reusable search box component with keyboard shortcuts.
 * Listens for Cmd+F/Ctrl+F to open and Escape to close.
 */
export function SearchBox({
  open,
  onOpenChange,
  query,
  onQueryChange,
  placeholder = "Search...",
  resultCount,
}: SearchBoxProps): React.ReactNode {
  const inputRef = useRef<HTMLInputElement>(null);

  // Keyboard shortcut for search (Cmd+F / Ctrl+F)
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent): void {
      if ((e.metaKey || e.ctrlKey) && e.key === "f") {
        e.preventDefault();
        onOpenChange(true);
      }
      if (e.key === "Escape" && open) {
        onOpenChange(false);
        onQueryChange("");
      }
    }

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [open, onOpenChange, onQueryChange]);

  // Focus input when opened
  useEffect(() => {
    if (open && inputRef.current) {
      inputRef.current.focus();
    }
  }, [open]);

  if (!open) {
    return null;
  }

  function handleClose(): void {
    onOpenChange(false);
    onQueryChange("");
  }

  return (
    <div className="relative w-full max-w-lg mx-auto mb-4">
      <div className="bg-white rounded-lg shadow-lg border">
        <div className="flex items-center px-4 py-3">
          <Search className="h-5 w-5 text-gray-400 mr-3" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
            placeholder={placeholder}
            className="flex-1 outline-none text-lg placeholder:text-gray-400"
            autoComplete="off"
            aria-label="Search"
          />
          {query && (
            <button
              type="button"
              onClick={() => onQueryChange("")}
              className="p-1 hover:bg-gray-100 rounded mr-2"
              aria-label="Clear search"
            >
              <X className="h-4 w-4 text-gray-400" />
            </button>
          )}
          <button type="button" onClick={handleClose} className="text-xs text-gray-400 hover:text-gray-600">
            <kbd className="px-1.5 py-0.5 bg-gray-100 rounded">Esc</kbd>
          </button>
        </div>
        {query && resultCount !== undefined && (
          <div className="px-4 py-2 text-xs text-gray-500 border-t">
            {resultCount} result{resultCount !== 1 ? "s" : ""}
          </div>
        )}
      </div>
    </div>
  );
}
