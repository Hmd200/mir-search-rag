export function ErrorBanner({
  message,
  onDismiss,
}: {
  message: string;
  onDismiss?: () => void;
}) {
  return (
    <div
      role="alert"
      className="flex items-start justify-between gap-3 rounded-xl border border-danger/30 bg-danger/10 px-4 py-3 text-sm text-danger"
    >
      <p>{message}</p>
      {onDismiss ? (
        <button
          type="button"
          onClick={onDismiss}
          className="shrink-0 rounded-md px-2 py-0.5 text-xs font-medium hover:bg-danger/10"
        >
          Dismiss
        </button>
      ) : null}
    </div>
  );
}
