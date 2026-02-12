import { Link2, Loader2, Search, User } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
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
import type { PersonForSelection } from "~/lib/db.server";

interface PersonSelectModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  collectionId: number;
  collectionName: string;
  persons: PersonForSelection[];
  currentPersonId: number | null;
  currentUserId: number;
  onSuccess?: () => void;
}

export function PersonSelectModal({
  open,
  onOpenChange,
  collectionId,
  collectionName,
  persons,
  currentPersonId,
  currentUserId,
  onSuccess,
}: PersonSelectModalProps) {
  const [search, setSearch] = useState("");
  const [selectedId, setSelectedId] = useState<number | null>(currentPersonId);
  const [showWarning, setShowWarning] = useState(false);
  const [warningPersonName, setWarningPersonName] = useState("");
  const [warningLinkedUser, setWarningLinkedUser] = useState("");
  const fetcher = useFetcher();

  const isSubmitting = fetcher.state !== "idle";

  // Filter persons by search
  const filteredPersons = useMemo(() => {
    if (!search.trim()) return persons;
    const lower = search.toLowerCase();
    return persons.filter((p) => p.person_name.toLowerCase().includes(lower));
  }, [persons, search]);

  // Reset state when modal opens
  useEffect(() => {
    if (open) {
      setSearch("");
      setSelectedId(currentPersonId);
      setShowWarning(false);
    }
  }, [open, currentPersonId]);

  // Handle successful submission
  useEffect(() => {
    if (fetcher.data?.success) {
      onSuccess?.();
      onOpenChange(false);
    }
  }, [fetcher.data, onSuccess, onOpenChange]);

  const handleSelect = (personId: number) => {
    setSelectedId(personId === selectedId ? null : personId);
  };

  const handleSetClick = () => {
    if (!selectedId) return;

    const person = persons.find((p) => p.id === selectedId);
    if (person?.linked_user_id && person.linked_user_id !== currentUserId) {
      // Show warning for already-linked person
      setWarningPersonName(person.person_name);
      setWarningLinkedUser(person.linked_user_name || "another user");
      setShowWarning(true);
    } else {
      submitSelection();
    }
  };

  const submitSelection = () => {
    fetcher.submit(
      { personId: selectedId?.toString() || "" },
      { method: "post", action: `/api/collection/${collectionId}/set-member-person` },
    );
  };

  const handleConfirmWarning = () => {
    setShowWarning(false);
    submitSelection();
  };

  if (showWarning) {
    return (
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent className="max-w-sm">
          <DialogHeader>
            <DialogTitle>Person Already Linked</DialogTitle>
            <DialogDescription>
              {warningPersonName} is linked to {warningLinkedUser}. Continue anyway?
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowWarning(false)}>
              Cancel
            </Button>
            <Button onClick={handleConfirmWarning} disabled={isSubmitting}>
              {isSubmitting ? (
                <>
                  <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                  Saving...
                </>
              ) : (
                "Continue"
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    );
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Select Person</DialogTitle>
          <DialogDescription>Choose who you are in {collectionName}</DialogDescription>
        </DialogHeader>

        {/* Search input */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
          <Input
            placeholder="Search by name..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-9"
            autoComplete="off"
            data-form-type="other"
            data-1p-ignore
            data-lpignore="true"
          />
        </div>

        {/* Person list */}
        <div className="max-h-[400px] overflow-y-auto border rounded-md">
          {filteredPersons.length === 0 ? (
            <div className="p-4 text-center text-gray-500">
              {search ? "No matching persons found" : "No persons in this collection"}
            </div>
          ) : (
            <div className="divide-y">
              {filteredPersons.map((person) => {
                const isSelected = person.id === selectedId;
                const isLinkedToOther = person.linked_user_id && person.linked_user_id !== currentUserId;

                return (
                  <button
                    key={person.id}
                    type="button"
                    onClick={() => handleSelect(person.id)}
                    className={`w-full flex items-center gap-3 p-3 text-left hover:bg-gray-50 transition-colors ${
                      isSelected ? "bg-blue-50" : ""
                    }`}
                  >
                    {/* Face thumbnail */}
                    {person.detection_id ? (
                      <div className="w-10 h-10 rounded-full overflow-hidden bg-gray-100 flex-shrink-0">
                        <img
                          src={`/api/face/${person.detection_id}`}
                          alt={person.person_name}
                          className="w-full h-full object-cover"
                        />
                      </div>
                    ) : (
                      <div className="w-10 h-10 rounded-full bg-gray-200 flex items-center justify-center flex-shrink-0">
                        <User className="h-5 w-5 text-gray-400" />
                      </div>
                    )}

                    {/* Name */}
                    <span className="flex-1 truncate font-medium">{person.person_name}</span>

                    {/* Linked indicator */}
                    {isLinkedToOther && (
                      <div className="flex-shrink-0" title={`Linked to ${person.linked_user_name}`}>
                        <Link2 className="h-4 w-4 text-amber-500" />
                      </div>
                    )}
                  </button>
                );
              })}
            </div>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleSetClick} disabled={!selectedId || isSubmitting}>
            {isSubmitting ? (
              <>
                <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                Saving...
              </>
            ) : (
              "Set"
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
