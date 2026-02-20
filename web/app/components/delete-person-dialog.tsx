import { Loader2, Trash2 } from "lucide-react";
import { useEffect } from "react";
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

interface DeletePersonDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  personId: string;
  personName: string;
  clusterCount: number;
  onSuccess?: () => void;
}

export function DeletePersonDialog({
  open,
  onOpenChange,
  personId,
  personName,
  clusterCount,
  onSuccess,
}: DeletePersonDialogProps): React.ReactElement {
  const deleteFetcher = useFetcher();
  const isSubmitting = deleteFetcher.state !== "idle";

  useEffect(() => {
    if (deleteFetcher.data?.success) {
      onSuccess?.();
      onOpenChange(false);
    }
  }, [deleteFetcher.data, onSuccess, onOpenChange]);

  function handleDelete(): void {
    deleteFetcher.submit({}, { method: "post", action: `/api/person/${personId}/delete` });
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-sm">
        <DialogHeader>
          <DialogTitle>Remove All Clusters</DialogTitle>
          <DialogDescription>
            This will unlink all {clusterCount} cluster{clusterCount !== 1 ? "s" : ""} from{" "}
            <span className="font-medium text-gray-700">{personName}</span>. The clusters and their photos will not be
            deleted.
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={isSubmitting}>
            Cancel
          </Button>
          <Button variant="destructive" onClick={handleDelete} disabled={isSubmitting}>
            {isSubmitting ? (
              <>
                <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                Removing...
              </>
            ) : (
              <>
                <Trash2 className="h-4 w-4 mr-1" />
                Remove All
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
