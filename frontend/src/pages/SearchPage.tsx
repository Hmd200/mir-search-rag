import { useState } from "react";
import type { FormEvent } from "react";

import {
  formatApiError,
  searchBm25,
  searchKeyword,
  searchSemantic,
} from "../api/client";
import type {
  KeywordSearchResult,
  SemanticSearchResult,
} from "../api/client";
import { ErrorBanner } from "../components/ErrorBanner";
import { HighlightedSnippet } from "../components/HighlightedSnippet";
import { TermContributionChart } from "../components/TermContributionChart";
import { formatLatency, formatScore } from "../lib/format";

type Algorithm = "tfidf" | "bm25" | "semantic" | "rag";

type DisplayHit = {
  chunk_id: string;
  document_id: string;
  document_title: string;
  score: number;
  text: string;
  page_number: number | null;
  section_title: string | null;
  matched_terms?: string[];
  term_contributions?: Record<string, number>;
};

function toKeywordHits(results: KeywordSearchResult[]): DisplayHit[] {
  return results.map((result) => ({
    chunk_id: result.chunk_id,
    document_id: result.document_id,
    document_title: result.document_title,
    score: result.score,
    text: result.text,
    page_number: result.page_number,
    section_title: result.section_title,
    matched_terms: result.matched_terms,
    term_contributions: result.term_contributions,
  }));
}

function toSemanticHits(results: SemanticSearchResult[]): DisplayHit[] {
  return results.map((result) => ({
    chunk_id: result.chunk_id,
    document_id: result.document_id,
    document_title: result.document_title,
    score: result.score,
    text: result.text,
    page_number: result.page_number,
    section_title: result.section_title,
  }));
}

function ResultCard({ hit }: { hit: DisplayHit }) {
  const [open, setOpen] = useState(false);
  const hasContributions =
    hit.term_contributions && Object.keys(hit.term_contributions).length > 0;

  return (
    <article className="rounded-2xl border border-rule bg-card p-4 sm:p-5">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h2 className="font-display text-xl text-ink">{hit.document_title}</h2>
          <p className="text-sm text-ink-soft">
            Score {formatScore(hit.score)}
            {hit.page_number !== null ? ` · Page ${hit.page_number}` : ""}
            {hit.section_title ? ` · ${hit.section_title}` : ""}
          </p>
        </div>
      </div>
      <div className="mt-3">
        <HighlightedSnippet text={hit.text} terms={hit.matched_terms} />
      </div>
      {hasContributions ? (
        <div className="mt-4 border-t border-rule pt-3">
          <button
            type="button"
            onClick={() => setOpen((value) => !value)}
            className="text-sm font-medium text-burgundy hover:text-burgundy-dark"
            aria-expanded={open}
          >
            {open ? "Hide term contributions" : "Show term contributions"}
          </button>
          {open ? (
            <div className="mt-3">
              <TermContributionChart contributions={hit.term_contributions ?? {}} />
            </div>
          ) : null}
        </div>
      ) : null}
    </article>
  );
}

function ResultSkeleton() {
  return (
    <div className="animate-pulse rounded-2xl border border-rule bg-card p-5">
      <div className="h-5 w-1/3 rounded bg-paper-2" />
      <div className="mt-2 h-3 w-1/4 rounded bg-paper-2" />
      <div className="mt-4 space-y-2">
        <div className="h-3 w-full rounded bg-paper-2" />
        <div className="h-3 w-5/6 rounded bg-paper-2" />
        <div className="h-3 w-2/3 rounded bg-paper-2" />
      </div>
    </div>
  );
}

export function SearchPage() {
  const [query, setQuery] = useState("");
  const [algorithm, setAlgorithm] = useState<Algorithm>("tfidf");
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [topK, setTopK] = useState(10);
  const [k1, setK1] = useState(1.5);
  const [b, setB] = useState(0.75);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hits, setHits] = useState<DisplayHit[] | null>(null);
  const [elapsedMs, setElapsedMs] = useState<number | null>(null);
  const [resultCount, setResultCount] = useState<number | null>(null);
  const [submittedQuery, setSubmittedQuery] = useState<string | null>(null);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = query.trim();
    if (!trimmed) {
      setError("Enter a search query.");
      return;
    }
    if (algorithm === "rag") {
      setHits(null);
      setElapsedMs(null);
      setResultCount(null);
      setSubmittedQuery(trimmed);
      setError(null);
      return;
    }

    setLoading(true);
    setError(null);
    setSubmittedQuery(trimmed);

    try {
      if (algorithm === "tfidf") {
        const response = await searchKeyword({ q: trimmed, top_k: topK });
        setHits(toKeywordHits(response.results));
        setElapsedMs(response.elapsed_ms);
        setResultCount(response.result_count);
      } else if (algorithm === "bm25") {
        const response = await searchBm25({
          q: trimmed,
          top_k: topK,
          k1,
          b,
        });
        setHits(toKeywordHits(response.results));
        setElapsedMs(response.elapsed_ms);
        setResultCount(response.result_count);
      } else {
        const response = await searchSemantic({ q: trimmed, top_k: topK });
        setHits(toSemanticHits(response.results));
        setElapsedMs(response.elapsed_ms);
        setResultCount(response.result_count);
      }
    } catch (cause) {
      setHits(null);
      setElapsedMs(null);
      setResultCount(null);
      setError(formatApiError(cause));
    } finally {
      setLoading(false);
    }
  }

  const algorithms: { id: Algorithm; label: string }[] = [
    { id: "tfidf", label: "TF-IDF (VSM)" },
    { id: "bm25", label: "BM25" },
    { id: "semantic", label: "Semantic" },
    { id: "rag", label: "RAG" },
  ];

  return (
    <div className="space-y-8">
      <form onSubmit={(event) => void onSubmit(event)} className="space-y-4">
        <label className="block">
          <span className="sr-only">Search query</span>
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search the corpus"
            maxLength={500}
            className="w-full rounded-2xl border border-rule bg-card px-5 py-4 font-display text-2xl text-ink shadow-sm outline-none placeholder:text-ink-soft/70 focus:border-burgundy"
          />
        </label>

        <fieldset className="flex flex-wrap gap-2">
          <legend className="sr-only">Retrieval method</legend>
          {algorithms.map((option) => (
            <label
              key={option.id}
              className={[
                "cursor-pointer rounded-full border px-4 py-2 text-sm font-medium",
                algorithm === option.id
                  ? "border-burgundy bg-burgundy text-paper"
                  : "border-rule bg-card text-ink hover:border-burgundy/40",
              ].join(" ")}
            >
              <input
                type="radio"
                name="algorithm"
                value={option.id}
                checked={algorithm === option.id}
                onChange={() => setAlgorithm(option.id)}
                className="sr-only"
              />
              {option.label}
            </label>
          ))}
        </fieldset>

        <div>
          <button
            type="button"
            onClick={() => setAdvancedOpen((value) => !value)}
            className="text-sm font-medium text-burgundy"
            aria-expanded={advancedOpen}
          >
            {advancedOpen ? "Hide advanced settings" : "Advanced settings"}
          </button>
          {advancedOpen ? (
            <div className="mt-3 grid gap-3 rounded-2xl border border-rule bg-card p-4 sm:grid-cols-3">
              <label className="text-sm">
                <span className="mb-1 block text-ink-soft">top_k</span>
                <input
                  type="number"
                  min={1}
                  max={50}
                  value={topK}
                  onChange={(event) => setTopK(Number(event.target.value))}
                  className="w-full rounded-lg border border-rule bg-paper px-3 py-2"
                />
              </label>
              {algorithm === "bm25" ? (
                <>
                  <label className="text-sm">
                    <span className="mb-1 block text-ink-soft">k1</span>
                    <input
                      type="number"
                      min={0.01}
                      max={10}
                      step={0.1}
                      value={k1}
                      onChange={(event) => setK1(Number(event.target.value))}
                      className="w-full rounded-lg border border-rule bg-paper px-3 py-2"
                    />
                  </label>
                  <label className="text-sm">
                    <span className="mb-1 block text-ink-soft">b</span>
                    <input
                      type="number"
                      min={0}
                      max={1}
                      step={0.05}
                      value={b}
                      onChange={(event) => setB(Number(event.target.value))}
                      className="w-full rounded-lg border border-rule bg-paper px-3 py-2"
                    />
                  </label>
                </>
              ) : null}
            </div>
          ) : null}
        </div>

        <button
          type="submit"
          disabled={loading}
          className="rounded-full bg-burgundy px-6 py-2.5 text-sm font-semibold text-paper hover:bg-burgundy-dark disabled:opacity-60"
        >
          {loading ? "Searching…" : "Search"}
        </button>
      </form>

      {error ? (
        <ErrorBanner message={error} onDismiss={() => setError(null)} />
      ) : null}

      {algorithm === "rag" ? (
        <div className="rounded-2xl border border-rule bg-card px-6 py-12 text-center">
          <p className="font-display text-2xl">RAG generation is coming soon</p>
          <p className="mt-2 text-sm text-ink-soft">
            The API does not yet expose an answer-generation endpoint. Use
            TF-IDF, BM25, or semantic search for retrieval.
          </p>
        </div>
      ) : null}

      {loading ? (
        <div className="space-y-3">
          <ResultSkeleton />
          <ResultSkeleton />
          <ResultSkeleton />
        </div>
      ) : null}

      {!loading && algorithm !== "rag" && hits !== null && elapsedMs !== null ? (
        <p className="text-sm text-ink-soft">
          {resultCount ?? hits.length} result{(resultCount ?? hits.length) === 1 ? "" : "s"}{" "}
          for “{submittedQuery}” · {formatLatency(elapsedMs)}
        </p>
      ) : null}

      {!loading && algorithm !== "rag" && hits !== null && hits.length === 0 ? (
        <div className="rounded-2xl border border-rule bg-card px-6 py-12 text-center">
          <p className="font-display text-2xl">No matching chunks</p>
          <p className="mt-2 text-sm text-ink-soft">
            Try another query or a different retrieval method.
          </p>
        </div>
      ) : null}

      {!loading && hits === null && algorithm !== "rag" && !error ? (
        <div className="rounded-2xl border border-dashed border-rule px-6 py-12 text-center">
          <p className="font-display text-2xl">Search the indexed corpus</p>
          <p className="mt-2 text-sm text-ink-soft">
            Choose TF-IDF, BM25, or semantic search, then enter a query.
          </p>
        </div>
      ) : null}

      {!loading && hits && hits.length > 0 ? (
        <div className="space-y-3">
          {hits.map((hit) => (
            <ResultCard key={hit.chunk_id} hit={hit} />
          ))}
        </div>
      ) : null}
    </div>
  );
}
