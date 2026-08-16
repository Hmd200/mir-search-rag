export function ConfirmDialog({
  open,
  title,
  message,
  confirmLabel = "Delete",
  busy = false,
  onConfirm,
  onCancel,
}: {
  open: boolean;
  title: string;
  message: string;
  confirmLabel?: string;
  busy?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  if (!open) {
    return null;
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-end justify-center bg-ink/40 p-4 sm:items-center"
      role="presentation"
      onClick={onCancel}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="confirm-title"
        className="w-full max-w-md rounded-2xl border border-rule bg-card p-5 shadow-xl"
        onClick={(event) => event.stopPropagation()}
      >
        <h2 id="confirm-title" className="font-display text-xl text-ink">
          {title}
        </h2>
        <p className="mt-2 text-sm text-ink-soft">{message}</p>
        <div className="mt-5 flex justify-end gap-2">
          <button
            type="button"
            onClick={onCancel}
            disabled={busy}
            className="rounded-full border border-rule px-4 py-2 text-sm font-medium text-ink hover:bg-paper-2 disabled:opacity-60"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={onConfirm}
            disabled={busy}
            className="rounded-full bg-danger px-4 py-2 text-sm font-medium text-paper hover:bg-burgundy-dark disabled:opacity-60"
          >
            {busy ? "Working…" : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
