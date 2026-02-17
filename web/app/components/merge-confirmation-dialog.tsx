import { AlertTriangle, Link2, Loader2 } from "lucide-react";
import { Button } from "~/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "~/components/ui/dialog";
import type { MergePreview } from "~/hooks/use-drag-to-merge";

interface MergeConfirmationDialogProps {
  open: boolean;
  sourceName: string;
  targetName: string;
  preview: MergePreview | null;
  isLoadingPreview: boolean;
  isSubmitting: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

export function MergeConfirmationDialog({
  open,
  sourceName,
  targetName,
  preview,
  isLoadingPreview,
  isSubmitting,
  onConfirm,
  onCancel,
}: MergeConfirmationDialogProps) {
  return (
    <Dialog open={open} onOpenChange={(o) => !o && onCancel()}>
      <DialogContent showCloseButton={false} className="max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Link2 className="h-5 w-5" />
            {preview?.willMergePersons ? "Merge People" : "Link as Same Person"}
          </DialogTitle>
          <DialogDescription>
            {preview?.willMergePersons ? (
              <>
                Merge <strong>{sourceName}</strong> into <strong>{targetName}</strong>?
              </>
            ) : (
              <>
                Link <strong>{sourceName}</strong> and <strong>{targetName}</strong> as the same person?
              </>
            )}
          </DialogDescription>
        </DialogHeader>

        {isLoadingPreview ? (
          <div className="flex items-center justify-center py-4 text-gray-500">
            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            Loading...
          </div>
        ) : preview?.willMergePersons ? (
          <div className="flex items-start gap-2 p-3 bg-amber-50 border border-amber-200 rounded-lg text-amber-800 text-sm">
            <AlertTriangle className="h-5 w-5 mt-0.5 flex-shrink-0" />
            <div>
              <strong>Person records will be merged.</strong>
              <br />"{preview.source?.personName}" will be merged into "{preview.target?.personName}
              ".
              {(preview.source?.personClusterCount ?? 0) > 1 && (
                <>
                  <br />
                  All {preview.source?.personClusterCount} clusters of "{preview.source?.personName}" will be
                  reassigned.
                </>
              )}
            </div>
          </div>
        ) : (
          <p className="text-sm text-gray-600">Both clusters will be preserved and assigned to the same identity.</p>
        )}

        <DialogFooter>
          <Button variant="outline" onClick={onCancel} disabled={isSubmitting}>
            Cancel
          </Button>
          <Button onClick={onConfirm} disabled={isSubmitting || isLoadingPreview}>
            {isSubmitting ? (
              <>
                <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                {preview?.willMergePersons ? "Merging..." : "Linking..."}
              </>
            ) : preview?.willMergePersons ? (
              "Merge"
            ) : (
              "Link"
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
