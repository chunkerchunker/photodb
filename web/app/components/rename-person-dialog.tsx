import { Loader2 } from "lucide-react";
import { useEffect, useState } from "react";
import { useFetcher } from "react-router";
import { Button } from "~/components/ui/button";
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
  onSuccess?: (newFirstName: string, newLastName: string) => void;
  /** Title to show in the dialog header. Defaults to "Rename Person" or "Set Name" if no current name. */
  title?: string;
  /** Description to show in the dialog header. */
  description?: string;
  /** The API endpoint type - "person" or "cluster". Defaults to "person". */
  apiType?: "person" | "cluster";
}

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
  onSuccess,
  title,
  description = "Enter the person's name.",
  apiType = "person",
}: RenamePersonDialogProps): React.ReactElement {
  const [firstName, setFirstName] = useState(currentFirstName);
  const [lastName, setLastName] = useState(currentLastName);
  const renameFetcher = useFetcher();

  const isSubmitting = renameFetcher.state !== "idle";
  const hasCurrentName = currentFirstName || currentLastName;
  const dialogTitle = title ?? (hasCurrentName ? "Rename Person" : "Set Name");

  // Reset form when dialog opens with new values
  useEffect(() => {
    if (open) {
      setFirstName(currentFirstName);
      setLastName(currentLastName);
    }
  }, [open, currentFirstName, currentLastName]);

  // Handle successful submission
  useEffect(() => {
    if (renameFetcher.data?.success) {
      onSuccess?.(firstName.trim(), lastName.trim());
      onOpenChange(false);
    }
  }, [renameFetcher.data, firstName, lastName, onSuccess, onOpenChange]);

  function handleSubmit(): void {
    if (firstName.trim()) {
      renameFetcher.submit(
        { firstName: firstName.trim(), lastName: lastName.trim() },
        { method: "post", action: `/api/${apiType}/${personId}/rename` },
      );
    }
  }

  function handleKeyDown(e: React.KeyboardEvent): void {
    if (e.key === "Enter") {
      handleSubmit();
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
              placeholder="First name"
              autoComplete="off"
              data-form-type="other"
              data-1p-ignore
              data-lpignore="true"
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
              autoComplete="off"
              data-form-type="other"
              data-1p-ignore
              data-lpignore="true"
            />
          </div>
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
