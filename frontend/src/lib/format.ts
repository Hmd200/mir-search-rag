import type { DocumentResponse } from "../api/client";

export function formatDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

export function fileTypeLabel(document: DocumentResponse): string {
  const mime = (document.mime_type ?? "").toLowerCase();
  const name = (document.original_filename ?? "").toLowerCase();

  if (mime.includes("pdf") || name.endsWith(".pdf")) {
    return "PDF";
  }
  if (
    mime.includes("wordprocessingml") ||
    mime.includes("msword") ||
    name.endsWith(".docx") ||
    name.endsWith(".doc")
  ) {
    return "DOCX";
  }
  if (document.source_type === "web") {
    return "WEB";
  }
  return "FILE";
}

export function formatScore(score: number): string {
  return score.toFixed(4);
}

export function formatLatency(elapsedMs: number): string {
  if (elapsedMs < 1000) {
    return `${elapsedMs.toFixed(1)} ms`;
  }
  return `${(elapsedMs / 1000).toFixed(2)} s`;
}
