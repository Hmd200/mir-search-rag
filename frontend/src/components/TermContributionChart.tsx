export function TermContributionChart({
  contributions,
}: {
  contributions: Record<string, number>;
}) {
  const entries = Object.entries(contributions).sort((left, right) => {
    return right[1] - left[1];
  });

  if (!entries.length) {
    return (
      <p className="text-sm text-ink-soft">No term contributions for this hit.</p>
    );
  }

  const max = Math.max(...entries.map(([, value]) => Math.abs(value)), Number.EPSILON);

  return (
    <ul className="space-y-2">
      {entries.map(([term, value]) => (
        <li key={term}>
          <div className="mb-1 flex items-baseline justify-between gap-3 text-xs">
            <span className="font-medium text-ink">{term}</span>
            <span className="font-mono text-ink-soft">{value.toFixed(4)}</span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-paper-2">
            <div
              className="h-full rounded-full bg-burgundy"
              style={{ width: `${Math.max(4, (Math.abs(value) / max) * 100)}%` }}
            />
          </div>
        </li>
      ))}
    </ul>
  );
}
