import type { ReactNode } from "react";

export function HighlightedSnippet({
  text,
  terms,
}: {
  text: string;
  terms?: readonly string[];
}) {
  const unique = [...new Set((terms ?? []).filter(Boolean))];
  if (!unique.length) {
    return <p className="whitespace-pre-wrap text-sm leading-relaxed text-ink">{text}</p>;
  }

  const escaped = unique.map((term) =>
    term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"),
  );
  const pattern = new RegExp(`(${escaped.join("|")})\\w*`, "gi");
  const parts: ReactNode[] = [];
  let lastIndex = 0;
  let match = pattern.exec(text);

  while (match) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }
    parts.push(
      <mark key={`${match.index}-${match[0]}`}>{match[0]}</mark>,
    );
    lastIndex = match.index + match[0].length;
    if (pattern.lastIndex === match.index) {
      pattern.lastIndex += 1;
    }
    match = pattern.exec(text);
  }

  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  return (
    <p className="whitespace-pre-wrap text-sm leading-relaxed text-ink">{parts}</p>
  );
}
