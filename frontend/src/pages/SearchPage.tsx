import { useState } from "react";
import type { FormEvent, ReactNode } from "react";

import {
  ApiError,
  formatApiError,
  searchBm25,
  searchKeyword,
  searchRag,
  searchSemantic,
} from "../api/client";
import type {
  KeywordSearchResult,
  PrfExpansion,
  RagCitedChunk,
  RagResponse,
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

const CITATION_MARKER = /\[(\d+)\]/g;

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

function formatRagError(error: unknown): string {
  if (error instanceof ApiError && error.status === 503) {
    return "The language model is unavailable. Start Ollama and try again.";
  }
  return formatApiError(error);
}

function uniqueCitationNumbers(answer: string): number[] {
  const seen = new Set<number>();
  const ordered: number[] = [];
  const pattern = new RegExp(CITATION_MARKER.source, "g");
  let match = pattern.exec(answer);
  while (match) {
    const number = Number(match[1]);
    if (!seen.has(number)) {
      seen.add(number);
      ordered.push(number);
    }
    match = pattern.exec(answer);
  }
  return ordered;
}

function sourceElementId(citationNumber: number): string {
  return `rag-source-${citationNumber}`;
}

function contributionIntensityClass(normalized: number): string {
  if (normalized >= 0.75) {
    return "bg-burgundy/60";
  }
  if (normalized >= 0.5) {
    return "bg-burgundy/40";
  }
  if (normalized >= 0.25) {
    return "bg-burgundy/25";
  }
  return "bg-burgundy/10";
}

function lookupContribution(
  matched: string,
  contributions: Record<string, number>,
): { term: string; score: number } | null {
  const lowered = matched.toLowerCase();
  for (const [term, score] of Object.entries(contributions)) {
    if (lowered === term || lowered.startsWith(term)) {
      return { term, score };
    }
  }
  return null;
}

function GradedSnippet({
  text,
  terms,
  contributions,
}: {
  text: string;
  terms?: readonly string[];
  contributions: Record<string, number>;
}) {
  const unique = [...new Set((terms ?? []).filter(Boolean))];
  if (!unique.length) {
    return (
      <p className="whitespace-pre-wrap text-sm leading-relaxed text-ink">{text}</p>
    );
  }

  // Intensity is max-normalized inside this hit only. A global scale would
  // paint a weak result as strongly as the top hit just because one of its
  // terms is the strongest *in that weak document*.
  const maxContribution = Math.max(
    ...Object.values(contributions).map((value) => Math.abs(value)),
    Number.EPSILON,
  );

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
    const token = match[0];
    const found = lookupContribution(token, contributions);
    const normalized = found
      ? Math.abs(found.score) / maxContribution
      : 0;
    parts.push(
      <mark
        key={`${match.index}-${token}`}
        className={`rounded-sm ${contributionIntensityClass(normalized)}`}
        title={
          found ? `${found.term}: ${found.score.toFixed(2)}` : undefined
        }
      >
        {token}
      </mark>,
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
        {hasContributions ? (
          <GradedSnippet
            text={hit.text}
            terms={hit.matched_terms}
            contributions={hit.term_contributions ?? {}}
          />
        ) : (
          <HighlightedSnippet text={hit.text} terms={hit.matched_terms} />
        )}
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

function CitedAnswer({
  answer,
  onCite,
}: {
  answer: string;
  onCite: (citationNumber: number) => void;
}) {
  // Split the LLM answer on [n] markers so each marker can jump to the
  // matching source card (id rag-source-n) listed below the answer.
  const parts: ReactNode[] = [];
  const pattern = new RegExp(CITATION_MARKER.source, "g");
  let lastIndex = 0;
  let match = pattern.exec(answer);

  while (match) {
    if (match.index > lastIndex) {
      parts.push(answer.slice(lastIndex, match.index));
    }
    const citationNumber = Number(match[1]);
    parts.push(
      <button
        key={`${match.index}-${citationNumber}`}
        type="button"
        onClick={() => onCite(citationNumber)}
        className="mx-0.5 inline-flex rounded-md bg-burgundy/10 px-1 py-0.5 align-baseline text-sm font-semibold text-burgundy hover:bg-burgundy/20"
        aria-label={`Jump to source ${citationNumber}`}
      >
        [{citationNumber}]
      </button>,
    );
    lastIndex = match.index + match[0].length;
    match = pattern.exec(answer);
  }

  if (lastIndex < answer.length) {
    parts.push(answer.slice(lastIndex));
  }

  return (
    <p className="whitespace-pre-wrap text-base leading-relaxed text-ink">{parts}</p>
  );
}

function CitedChunkCard({
  citationNumber,
  chunk,
  highlighted,
}: {
  citationNumber: number;
  chunk: RagCitedChunk;
  highlighted: boolean;
}) {
  return (
    <article
      id={sourceElementId(citationNumber)}
      className={[
        "scroll-mt-24 rounded-2xl border bg-card p-4 sm:p-5",
        highlighted ? "border-burgundy ring-2 ring-burgundy/30" : "border-rule",
      ].join(" ")}
    >
      <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <p className="text-xs font-medium uppercase tracking-wide text-burgundy">
            [{citationNumber}]
          </p>
          <h2 className="font-display text-xl text-ink">{chunk.document_title}</h2>
          <p className="text-sm text-ink-soft">
            {chunk.page_number !== null ? `Page ${chunk.page_number}` : "Page unknown"}
            {" · "}
            Retrieval {formatScore(chunk.retrieval_score)}
            {chunk.rerank_score !== null
              ? ` · Rerank ${formatScore(chunk.rerank_score)}`
              : ""}
          </p>
        </div>
      </div>
      <p className="mt-3 whitespace-pre-wrap text-sm leading-relaxed text-ink">
        {chunk.text}
      </p>
    </article>
  );
}

function RagResults({
  result,
  highlightedCitation,
  onCite,
}: {
  result: RagResponse;
  highlightedCitation: number | null;
  onCite: (citationNumber: number) => void;
}) {
  // cited_chunks is first-appearance order of unique [n] markers, not sorted
  // by n. Pair each chunk with that marker so [2] still scrolls to source 2.
  const citationNumbers = uniqueCitationNumbers(result.answer);

  return (
    <div className="space-y-6">
      {result.abstained ? (
        <div className="rounded-2xl border border-rule bg-card px-6 py-12 text-center">
          <p className="font-display text-2xl">Not enough evidence in the corpus</p>
          <p className="mt-2 text-sm text-ink-soft">
            The retrieved chunks do not contain enough information to answer
            this question.
          </p>
        </div>
      ) : (
        <section className="rounded-2xl border border-rule bg-card p-5 sm:p-6">
          <h2 className="font-display text-2xl text-ink">Answer</h2>
          <div className="mt-3">
            <CitedAnswer answer={result.answer} onCite={onCite} />
          </div>
        </section>
      )}

      {!result.abstained && result.cited_chunks.length > 0 ? (
        <section className="space-y-3">
          <h3 className="font-display text-xl text-ink">Cited sources</h3>
          {result.cited_chunks.map((chunk, index) => {
            const citationNumber = citationNumbers[index] ?? index + 1;
            return (
              <CitedChunkCard
                key={chunk.chunk_id}
                citationNumber={citationNumber}
                chunk={chunk}
                highlighted={highlightedCitation === citationNumber}
              />
            );
          })}
        </section>
      ) : null}
    </div>
  );
}

export function SearchPage() {
  const [query, setQuery] = useState("");
  const [algorithm, setAlgorithm] = useState<Algorithm>("tfidf");
  const [usePrf, setUsePrf] = useState(false);
  const [useReranker, setUseReranker] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [topK, setTopK] = useState(10);
  const [k1, setK1] = useState(1.5);
  const [b, setB] = useState(0.75);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hits, setHits] = useState<DisplayHit[] | null>(null);
  const [expansion, setExpansion] = useState<PrfExpansion | null>(null);
  const [ragResult, setRagResult] = useState<RagResponse | null>(null);
  const [highlightedCitation, setHighlightedCitation] = useState<number | null>(
    null,
  );
  const [elapsedMs, setElapsedMs] = useState<number | null>(null);
  const [resultCount, setResultCount] = useState<number | null>(null);
  const [submittedQuery, setSubmittedQuery] = useState<string | null>(null);

  function resetResults() {
    setHits(null);
    setExpansion(null);
    setRagResult(null);
    setHighlightedCitation(null);
    setElapsedMs(null);
    setResultCount(null);
  }

  function jumpToCitation(citationNumber: number) {
    // Clicking [n] in the answer scrolls to #rag-source-n and highlights it.
    setHighlightedCitation(citationNumber);
    document.getElementById(sourceElementId(citationNumber))?.scrollIntoView({
      behavior: "smooth",
      block: "center",
    });
  }

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = query.trim();
    if (!trimmed) {
      setError("Enter a search query.");
      return;
    }

    setLoading(true);
    setError(null);
    setSubmittedQuery(trimmed);
    resetResults();

    try {
      if (algorithm === "tfidf") {
        const response = await searchKeyword({
          q: trimmed,
          top_k: topK,
          use_prf: usePrf,
        });
        setHits(toKeywordHits(response.results));
        setExpansion(response.expansion);
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
      } else if (algorithm === "semantic") {
        const response = await searchSemantic({ q: trimmed, top_k: topK });
        setHits(toSemanticHits(response.results));
        setElapsedMs(response.elapsed_ms);
        setResultCount(response.result_count);
      } else {
        const response = await searchRag({
          query: trimmed,
          top_k: topK,
          use_reranker: useReranker,
        });
        setRagResult(response);
        setElapsedMs(response.elapsed_ms);
        setResultCount(response.cited_chunks.length);
      }
    } catch (cause) {
      resetResults();
      setError(
        algorithm === "rag" ? formatRagError(cause) : formatApiError(cause),
      );
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

  const showPrfChips =
    algorithm === "tfidf" &&
    usePrf &&
    expansion !== null &&
    expansion.added_terms.length > 0;

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

        {algorithm === "tfidf" ? (
          <label className="flex cursor-pointer items-center gap-2 text-sm text-ink">
            <input
              type="checkbox"
              checked={usePrf}
              onChange={(event) => setUsePrf(event.target.checked)}
              className="size-4 accent-burgundy"
            />
            Pseudo-Relevance Feedback (Rocchio)
          </label>
        ) : null}

        {algorithm === "rag" ? (
          <label className="flex cursor-pointer items-center gap-2 text-sm text-ink">
            <input
              type="checkbox"
              checked={useReranker}
              onChange={(event) => setUseReranker(event.target.checked)}
              className="size-4 accent-burgundy"
            />
            Rerank results
          </label>
        ) : null}

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
          {loading
            ? algorithm === "rag"
              ? "Generating…"
              : "Searching…"
            : "Search"}
        </button>
      </form>

      {error ? (
        <ErrorBanner message={error} onDismiss={() => setError(null)} />
      ) : null}

      {loading ? (
        <div className="space-y-3">
          <ResultSkeleton />
          <ResultSkeleton />
          <ResultSkeleton />
        </div>
      ) : null}

      {!loading &&
      algorithm !== "rag" &&
      hits !== null &&
      elapsedMs !== null ? (
        <p className="text-sm text-ink-soft">
          {resultCount ?? hits.length} result
          {(resultCount ?? hits.length) === 1 ? "" : "s"} for “{submittedQuery}”
          {" · "}
          {formatLatency(elapsedMs)}
        </p>
      ) : null}

      {!loading &&
      algorithm === "rag" &&
      ragResult !== null &&
      elapsedMs !== null ? (
        <p className="text-sm text-ink-soft">
          Answer for “{submittedQuery}” · {formatLatency(elapsedMs)}
        </p>
      ) : null}

      {!loading && showPrfChips && expansion ? (
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-sm text-ink-soft">Expanded with:</span>
          {expansion.added_terms.map((item) => (
            <span
              key={item.term}
              className="rounded-full border border-rule bg-card px-3 py-1 text-sm text-ink"
              title={`weight ${item.weight.toFixed(4)}`}
            >
              {item.term}
            </span>
          ))}
        </div>
      ) : null}

      {!loading && algorithm !== "rag" && hits !== null && hits.length === 0 ? (
        <div className="rounded-2xl border border-rule bg-card px-6 py-12 text-center">
          <p className="font-display text-2xl">No matching chunks</p>
          <p className="mt-2 text-sm text-ink-soft">
            Try another query or a different retrieval method.
          </p>
        </div>
      ) : null}

      {!loading &&
      hits === null &&
      ragResult === null &&
      !error &&
      algorithm !== "rag" ? (
        <div className="rounded-2xl border border-dashed border-rule px-6 py-12 text-center">
          <p className="font-display text-2xl">Search the indexed corpus</p>
          <p className="mt-2 text-sm text-ink-soft">
            Choose TF-IDF, BM25, or semantic search, then enter a query.
          </p>
        </div>
      ) : null}

      {!loading &&
      ragResult === null &&
      !error &&
      algorithm === "rag" ? (
        <div className="rounded-2xl border border-dashed border-rule px-6 py-12 text-center">
          <p className="font-display text-2xl">Ask a question over the corpus</p>
          <p className="mt-2 text-sm text-ink-soft">
            RAG retrieves semantic chunks and generates a cited answer.
          </p>
        </div>
      ) : null}

      {!loading && algorithm !== "rag" && hits && hits.length > 0 ? (
        <div className="space-y-3">
          {(algorithm === "tfidf" || algorithm === "bm25") ? (
            <p className="text-sm text-ink-soft">
              Highlight intensity = term's contribution to the score
            </p>
          ) : null}
          {hits.map((hit) => (
            <ResultCard key={hit.chunk_id} hit={hit} />
          ))}
        </div>
      ) : null}

      {!loading && algorithm === "rag" && ragResult ? (
        <RagResults
          result={ragResult}
          highlightedCitation={highlightedCitation}
          onCite={jumpToCitation}
        />
      ) : null}
    </div>
  );
}
