import { ChevronDown, Loader2, Plus, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useFetcher } from "react-router";
import { Button } from "~/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "~/components/ui/collapsible";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "~/components/ui/dialog";
import { Input } from "~/components/ui/input";

interface RenamePersonDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  personId: string;
  currentFirstName?: string;
  currentLastName?: string;
  currentMiddleName?: string;
  currentMaidenName?: string;
  currentPreferredName?: string;
  currentSuffix?: string;
  currentAlternateNames?: string[];
  onSuccess?: (newFirstName: string, newLastName: string) => void;
  /** Title to show in the dialog header. Defaults to "Rename Person" or "Set Name" if no current name. */
  title?: string;
  /** Description to show in the dialog header. */
  description?: string;
  /** The API endpoint type - "person" or "cluster". Defaults to "person". */
  apiType?: "person" | "cluster";
}

const autocompleteOff = {
  autoComplete: "off",
  "data-form-type": "other",
  "data-1p-ignore": true,
  "data-lpignore": "true",
} as const;

/**
 * Reusable dialog for renaming a person or cluster.
 * Handles form state, API submission, and success callback internally.
 */
export function RenamePersonDialog({
  open,
  onOpenChange,
  personId,
  currentFirstName = "",
  currentLastName = "",
  currentMiddleName = "",
  currentMaidenName = "",
  currentPreferredName = "",
  currentSuffix = "",
  currentAlternateNames = [],
  onSuccess,
  title,
  description = "Enter the person's name.",
  apiType = "person",
}: RenamePersonDialogProps): React.ReactElement {
  const [firstName, setFirstName] = useState(currentFirstName);
  const [lastName, setLastName] = useState(currentLastName);
  const [middleName, setMiddleName] = useState(currentMiddleName);
  const [maidenName, setMaidenName] = useState(currentMaidenName);
  const [preferredName, setPreferredName] = useState(currentPreferredName);
  const [suffix, setSuffix] = useState(currentSuffix);
  const [alternateNames, setAlternateNames] = useState<string[]>(currentAlternateNames);
  const [newAltName, setNewAltName] = useState("");
  const [moreOpen, setMoreOpen] = useState(false);
  const submittedValuesRef = useRef<{ first: string; last: string } | null>(null);
  const renameFetcher = useFetcher();

  const isSubmitting = renameFetcher.state !== "idle";
  const hasCurrentName = currentFirstName || currentLastName;
  const dialogTitle = title ?? (hasCurrentName ? "Rename Person" : "Set Name");

  // Auto-open "More" section if any optional fields have values
  const hasOptionalFields =
    currentMiddleName || currentMaidenName || currentPreferredName || currentSuffix || currentAlternateNames.length > 0;

  // Reset form when dialog opens with new values
  useEffect(() => {
    if (open) {
      setFirstName(currentFirstName);
      setLastName(currentLastName);
      setMiddleName(currentMiddleName);
      setMaidenName(currentMaidenName);
      setPreferredName(currentPreferredName);
      setSuffix(currentSuffix);
      setAlternateNames(currentAlternateNames);
      setNewAltName("");
      setMoreOpen(!!hasOptionalFields);
    }
  }, [
    open,
    currentFirstName,
    currentLastName,
    currentMiddleName,
    currentMaidenName,
    currentPreferredName,
    currentSuffix,
    currentAlternateNames,
    hasOptionalFields,
  ]);

  // Handle successful submission
  useEffect(() => {
    if (renameFetcher.data?.success && submittedValuesRef.current) {
      const { first, last } = submittedValuesRef.current;
      submittedValuesRef.current = null;
      onSuccess?.(first, last);
      onOpenChange(false);
    }
  }, [renameFetcher.data, onSuccess, onOpenChange]);

  function handleSubmit(): void {
    const trimmedFirst = firstName.trim();
    const trimmedLast = lastName.trim();
    if (trimmedFirst) {
      submittedValuesRef.current = { first: trimmedFirst, last: trimmedLast };
      renameFetcher.submit(
        {
          firstName: trimmedFirst,
          lastName: trimmedLast,
          middleName: middleName.trim(),
          maidenName: maidenName.trim(),
          preferredName: preferredName.trim(),
          suffix: suffix.trim(),
          alternateNames: JSON.stringify(alternateNames),
        },
        { method: "post", action: `/api/${apiType}/${personId}/rename` },
      );
    }
  }

  function handleKeyDown(e: React.KeyboardEvent): void {
    if (e.key === "Enter") {
      handleSubmit();
    }
  }

  function addAlternateName(): void {
    const trimmed = newAltName.trim();
    if (trimmed && !alternateNames.includes(trimmed)) {
      setAlternateNames([...alternateNames, trimmed]);
      setNewAltName("");
    }
  }

  function removeAlternateName(index: number): void {
    setAlternateNames(alternateNames.filter((_, i) => i !== index));
  }

  function handleAltNameKeyDown(e: React.KeyboardEvent): void {
    if (e.key === "Enter") {
      e.preventDefault();
      addAlternateName();
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-sm">
        <DialogHeader>
          <DialogTitle>{dialogTitle}</DialogTitle>
          <DialogDescription>{description}</DialogDescription>
        </DialogHeader>
        <div className="space-y-4 py-4">
          <div className="space-y-2">
            <label htmlFor="renameFirstName" className="text-sm font-medium">
              First Name
            </label>
            <Input
              id="renameFirstName"
              value={firstName}
              onChange={(e) => setFirstName(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="First name"
              {...autocompleteOff}
              autoFocus
            />
          </div>
          <div className="space-y-2">
            <label htmlFor="renameLastName" className="text-sm font-medium">
              Last Name
            </label>
            <Input
              id="renameLastName"
              value={lastName}
              onChange={(e) => setLastName(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Last name (optional)"
              {...autocompleteOff}
            />
          </div>
          <Collapsible open={moreOpen} onOpenChange={setMoreOpen}>
            <CollapsibleTrigger asChild>
              <button
                type="button"
                className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground transition-colors"
              >
                <ChevronDown className={`h-4 w-4 transition-transform ${moreOpen ? "" : "-rotate-90"}`} />
                More name fields
              </button>
            </CollapsibleTrigger>
            <CollapsibleContent className="space-y-4 pt-3">
              <div className="space-y-2">
                <label htmlFor="renameMiddleName" className="text-sm font-medium">
                  Middle Name
                </label>
                <Input
                  id="renameMiddleName"
                  value={middleName}
                  onChange={(e) => setMiddleName(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Middle name (optional)"
                  {...autocompleteOff}
                />
              </div>
              <div className="space-y-2">
                <label htmlFor="renameSuffix" className="text-sm font-medium">
                  Suffix
                </label>
                <Input
                  id="renameSuffix"
                  value={suffix}
                  onChange={(e) => setSuffix(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Jr, Sr, III, etc. (optional)"
                  {...autocompleteOff}
                />
              </div>
              <div className="space-y-2">
                <label htmlFor="renameMaidenName" className="text-sm font-medium">
                  Maiden Name
                </label>
                <Input
                  id="renameMaidenName"
                  value={maidenName}
                  onChange={(e) => setMaidenName(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Maiden name (optional)"
                  {...autocompleteOff}
                />
              </div>
              <div className="space-y-2">
                <label htmlFor="renamePreferredName" className="text-sm font-medium">
                  Preferred Name
                </label>
                <Input
                  id="renamePreferredName"
                  value={preferredName}
                  onChange={(e) => setPreferredName(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Preferred name (optional)"
                  {...autocompleteOff}
                />
              </div>
              <div className="space-y-2">
                <label htmlFor="renameNewAltName" className="text-sm font-medium">
                  Alternate Names
                </label>
                <p className="text-xs text-muted-foreground">Names in other languages, nicknames, etc.</p>
                {alternateNames.length > 0 && (
                  <div className="flex flex-wrap gap-1.5">
                    {alternateNames.map((name, i) => (
                      <span
                        key={name}
                        className="inline-flex items-center gap-1 rounded-md bg-secondary px-2 py-1 text-sm"
                      >
                        {name}
                        <button
                          type="button"
                          onClick={() => removeAlternateName(i)}
                          className="text-muted-foreground hover:text-foreground transition-colors"
                        >
                          <X className="h-3 w-3" />
                        </button>
                      </span>
                    ))}
                  </div>
                )}
                <div className="flex gap-2">
                  <Input
                    id="renameNewAltName"
                    value={newAltName}
                    onChange={(e) => setNewAltName(e.target.value)}
                    onKeyDown={handleAltNameKeyDown}
                    placeholder="Add alternate name"
                    {...autocompleteOff}
                  />
                  <Button
                    type="button"
                    variant="outline"
                    size="icon"
                    onClick={addAlternateName}
                    disabled={!newAltName.trim()}
                    className="shrink-0"
                  >
                    <Plus className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </CollapsibleContent>
          </Collapsible>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleSubmit} disabled={!firstName.trim() || isSubmitting}>
            {isSubmitting ? (
              <>
                <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                Saving...
              </>
            ) : (
              "Save"
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
