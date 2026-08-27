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
  AbstentionReason,
  Bm25Mode,
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
type LlmProvider = "ollama" | "gemini";

const BM25_MODE_LABELS: Record<Bm25Mode, string> = {
  default: "Default",
  tunable: "Tunable",
  finetuned: "Calibrated",
};

const BM25_MODES: { id: Bm25Mode; label: string }[] = [
  { id: "default", label: `${BM25_MODE_LABELS.default} (k1=1.5, b=0.75)` },
  { id: "tunable", label: BM25_MODE_LABELS.tunable },
  { id: "finetuned", label: BM25_MODE_LABELS.finetuned },
];

type DisplayHit = {
  chunk_id: string;
  document_id: string;
  document_title: string;
  score: number;
  text: string;
  page_start: number | null;
  page_end: number | null;
  section_title: string | null;
  matched_terms?: string[];
  term_contributions?: Record<string, number>;
};

const CITATION_MARKER = /\[n?(\d+)\]/g;

function formatPageRange(
  pageStart: number | null,
  pageEnd: number | null,
): string | null {
  if (pageStart === null || pageEnd === null) {
    return null;
  }
  if (pageStart === pageEnd) {
    return `Page ${pageStart}`;
  }
  return `Pages ${pageStart}-${pageEnd}`;
}

function toKeywordHits(results: KeywordSearchResult[]): DisplayHit[] {
  return results.map((result) => ({
    chunk_id: result.chunk_id,
    document_id: result.document_id,
    document_title: result.document_title,
    score: result.score,
    text: result.text,
    page_start: result.page_start,
    page_end: result.page_end,
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
    page_start: result.page_start,
    page_end: result.page_end,
    section_title: result.section_title,
  }));
}

function formatRagError(error: unknown): string {
  if (error instanceof ApiError && error.status === 400) {
    return error.message;
  }
  if (error instanceof ApiError && error.status === 503) {
    return "The language model is unavailable. For Ollama, start the local server. For Gemini, check MIR_GEMINI_API_KEY.";
  }
  return formatApiError(error);
}

function abstentionMessage(reason: AbstentionReason | null): string {
  switch (reason) {
    case "no_context":
      return "No searchable context was found.";
    case "low_relevance":
      return "Retrieved passages were not relevant enough.";
    case "model_abstained":
      return "The generator could not find a supported answer.";
    case "citation_failure":
      return "The system could not produce an answer with valid citations.";
    case "grounding_failure":
      return "The answer could not be verified against the retrieved passages.";
    default:
      return "The retrieved chunks do not contain enough information to answer this question.";
  }
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

function resultDocumentElementId(documentId: string): string {
  return `search-doc-${documentId}`;
}

function shortDocumentTitle(title: string, maxLength = 18): string {
  const trimmed = title.trim() || "Untitled";
  if (trimmed.length <= maxLength) {
    return trimmed;
  }
  return `${trimmed.slice(0, Math.max(1, maxLength - 1))}…`;
}

type GraphNode = {
  documentId: string;
  title: string;
  x: number;
  y: number;
};

type GraphEdge = {
  from: string;
  to: string;
  weight: number;
};

function unionMatchedTerms(hits: DisplayHit[]): Set<string> {
  const terms = new Set<string>();
  for (const hit of hits) {
    for (const term of hit.matched_terms ?? []) {
      if (term) {
        terms.add(term);
      }
    }
  }
  return terms;
}

function sharedTermCount(left: Set<string>, right: Set<string>): number {
  let count = 0;
  for (const term of left) {
    if (right.has(term)) {
      count += 1;
    }
  }
  return count;
}

function documentScore(hits: DisplayHit[]): number {
  return Math.max(...hits.map((hit) => hit.score), 0);
}

function buildDocumentRelationGraph(
  hits: DisplayHit[],
  mode: "lexical" | "semantic",
): { nodes: GraphNode[]; edges: GraphEdge[] } {
  const byDocument = new Map<string, DisplayHit[]>();
  for (const hit of hits) {
    const group = byDocument.get(hit.document_id) ?? [];
    group.push(hit);
    byDocument.set(hit.document_id, group);
  }

  const documents = [...byDocument.entries()];
  const width = 480;
  const height = 280;
  const centerX = width / 2;
  const centerY = height / 2;
  const radius = documents.length <= 2 ? 80 : 96;
  const nodes: GraphNode[] = documents.map(([documentId, group], index) => {
    const angle =
      (2 * Math.PI * index) / documents.length - Math.PI / 2;
    return {
      documentId,
      title: group[0]?.document_title ?? documentId,
      x: centerX + radius * Math.cos(angle),
      y: centerY + radius * Math.sin(angle),
    };
  });

  const edges: GraphEdge[] = [];
  for (let i = 0; i < documents.length; i += 1) {
    for (let j = i + 1; j < documents.length; j += 1) {
      const [leftId, leftHits] = documents[i];
      const [rightId, rightHits] = documents[j];
      let weight = 0;
      if (mode === "lexical") {
        // TF-IDF/BM25: edge weight is |matched_terms ∩ matched_terms|
        // across all chunks of each document in this result set.
        weight = sharedTermCount(
          unionMatchedTerms(leftHits),
          unionMatchedTerms(rightHits),
        );
      } else {
        // Semantic responses have no matched_terms. Use score proximity:
        // 1 / (1 + |score_a - score_b|), with each document scored as its
        // best (max) chunk. Closer cosine scores => thicker edges. Title
        // overlap is unused because titles need not share tokens.
        const delta = Math.abs(
          documentScore(leftHits) - documentScore(rightHits),
        );
        weight = 1 / (1 + delta);
      }
      if (weight > 0) {
        edges.push({ from: leftId, to: rightId, weight });
      }
    }
  }

  return { nodes, edges };
}

function DocumentRelationGraph({
  hits,
  mode,
}: {
  hits: DisplayHit[];
  mode: "lexical" | "semantic";
}) {
  const { nodes, edges } = buildDocumentRelationGraph(hits, mode);
  if (nodes.length < 2) {
    return null;
  }

  const nodeById = new Map(nodes.map((node) => [node.documentId, node]));
  const maxWeight = Math.max(...edges.map((edge) => edge.weight), Number.EPSILON);

  function jumpToDocument(documentId: string) {
    document.getElementById(resultDocumentElementId(documentId))?.scrollIntoView({
      behavior: "smooth",
      block: "center",
    });
  }

  const caption =
    mode === "lexical"
      ? "Edges connect documents sharing matched terms; thicker = more overlap."
      : "Edges connect documents whose retrieval scores are close; thicker = closer scores.";

  return (
    <section className="rounded-2xl border border-rule bg-card p-4 sm:p-5">
      <h3 className="font-display text-xl text-ink">Document relations</h3>
      <svg
        viewBox="0 0 480 280"
        className="mt-3 h-auto w-full text-burgundy"
        role="img"
        aria-label="Document relation graph for the current search results"
      >
        {edges.map((edge) => {
          const from = nodeById.get(edge.from);
          const to = nodeById.get(edge.to);
          if (!from || !to) {
            return null;
          }
          const strength = edge.weight / maxWeight;
          return (
            <line
              key={`${edge.from}-${edge.to}`}
              x1={from.x}
              y1={from.y}
              x2={to.x}
              y2={to.y}
              className="stroke-burgundy"
              strokeWidth={1 + strength * 4}
              strokeOpacity={0.25 + strength * 0.7}
            />
          );
        })}
        {nodes.map((node) => (
          <g
            key={node.documentId}
            transform={`translate(${node.x} ${node.y})`}
            className="cursor-pointer"
            role="button"
            tabIndex={0}
            aria-label={`Jump to ${node.title}`}
            onClick={() => jumpToDocument(node.documentId)}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                jumpToDocument(node.documentId);
              }
            }}
          >
            <circle r={14} className="fill-burgundy" />
            <text
              y={32}
              textAnchor="middle"
              className="fill-ink font-display text-[11px]"
            >
              {shortDocumentTitle(node.title)}
            </text>
          </g>
        ))}
      </svg>
      <p className="mt-2 text-sm text-ink-soft">{caption}</p>
    </section>
  );
}

function heatmapCellClass(normalized: number): string {
  if (normalized <= 0) {
    return "bg-paper-2 text-ink-soft";
  }
  if (normalized >= 0.75) {
    return "bg-burgundy/70 text-paper";
  }
  if (normalized >= 0.5) {
    return "bg-burgundy/50 text-paper";
  }
  if (normalized >= 0.25) {
    return "bg-burgundy/30 text-ink";
  }
  return "bg-burgundy/15 text-ink";
}

function KeywordHeatmap({ hits }: { hits: DisplayHit[] }) {
  // TF-IDF/BM25 only: cells are summed term_contributions per document.
  const documents: {
    documentId: string;
    title: string;
    contributions: Record<string, number>;
  }[] = [];
  const totals = new Map<string, number>();

  for (const hit of hits) {
    let row = documents.find((item) => item.documentId === hit.document_id);
    if (!row) {
      row = {
        documentId: hit.document_id,
        title: hit.document_title,
        contributions: {},
      };
      documents.push(row);
    }
    for (const [term, value] of Object.entries(hit.term_contributions ?? {})) {
      row.contributions[term] = (row.contributions[term] ?? 0) + value;
      totals.set(term, (totals.get(term) ?? 0) + Math.abs(value));
    }
  }

  const terms = [...totals.entries()]
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .slice(0, 8)
    .map(([term]) => term);

  if (documents.length === 0 || terms.length === 0) {
    return null;
  }

  const maxCell = Math.max(
    ...documents.flatMap((row) =>
      terms.map((term) => Math.abs(row.contributions[term] ?? 0)),
    ),
    Number.EPSILON,
  );

  function jumpToDocument(documentId: string) {
    document.getElementById(resultDocumentElementId(documentId))?.scrollIntoView({
      behavior: "smooth",
      block: "center",
    });
  }

  return (
    <section className="rounded-2xl border border-rule bg-card p-4 sm:p-5">
      <h3 className="font-display text-xl text-ink">Keyword heatmap</h3>
      <div className="mt-3 overflow-x-auto">
        <table className="w-full min-w-[480px] border-separate border-spacing-1 text-left text-xs">
          <thead>
            <tr>
              <th className="px-2 py-1 font-medium text-ink-soft">Document</th>
              {terms.map((term) => (
                <th
                  key={term}
                  className="px-2 py-1 text-center font-medium text-ink-soft"
                >
                  {term}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {documents.map((row) => (
              <tr
                key={row.documentId}
                className="cursor-pointer"
                onClick={() => jumpToDocument(row.documentId)}
              >
                <th className="px-2 py-1 font-medium text-ink" scope="row">
                  <button
                    type="button"
                    className="text-left text-burgundy hover:text-burgundy-dark"
                    onClick={() => jumpToDocument(row.documentId)}
                  >
                    {shortDocumentTitle(row.title, 22)}
                  </button>
                </th>
                {terms.map((term) => {
                  const value = row.contributions[term] ?? 0;
                  const normalized = Math.abs(value) / maxCell;
                  return (
                    <td key={term} className="p-0">
                      <div
                        className={`rounded-md px-2 py-2 text-center font-mono ${heatmapCellClass(normalized)}`}
                        title={`${row.title} · ${term}: ${value.toFixed(3)}`}
                      >
                        {value === 0 ? "—" : value.toFixed(2)}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-sm text-ink-soft">
        Cell color is that term’s contribution to the document’s ranking
        score; darker burgundy is a stronger match.
      </p>
    </section>
  );
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
  const pageLabel = formatPageRange(hit.page_start, hit.page_end);

  return (
    <article className="rounded-2xl border border-rule bg-card p-4 sm:p-5">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h2 className="font-display text-xl text-ink">{hit.document_title}</h2>
          <p className="text-sm text-ink-soft">
            Score {formatScore(hit.score)}
            {pageLabel ? ` · ${pageLabel}` : ""}
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

function formatOptionalScore(value: number | null | undefined): string {
  return value == null ? "n/a" : formatScore(value);
}

function CitedChunkCard({
  citationNumber,
  chunk,
  highlighted,
  assignAnchor = true,
}: {
  citationNumber: number;
  chunk: RagCitedChunk;
  highlighted: boolean;
  assignAnchor?: boolean;
}) {
  return (
    <article
      id={assignAnchor ? sourceElementId(citationNumber) : undefined}
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
            {formatPageRange(chunk.page_start, chunk.page_end) ?? "Page unknown"}
            {" · "}
            Dense {formatOptionalScore(chunk.dense_score)}
            {" · "}
            BM25 {formatOptionalScore(chunk.bm25_score)}
            {" · "}
            RRF {formatOptionalScore(chunk.fusion_score)}
            {chunk.rerank_score !== null
              ? ` · Rerank ${formatScore(chunk.rerank_score)}`
              : ""}
          </p>
          {chunk.retrieval_sources.length > 0 ? (
            <p className="text-xs text-ink-soft">
              Sources: {chunk.retrieval_sources.join(", ")}
            </p>
          ) : null}
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
      {result.rewritten_query ? (
        <p className="text-sm text-ink-soft">
          Retrieved with:{" "}
          <span className="rounded-full border border-rule bg-card px-3 py-1 text-ink">
            {result.rewritten_query}
          </span>
        </p>
      ) : null}
      <p className="text-sm text-ink-soft">
        Generated with{" "}
        {result.llm_provider === "gemini" ? "Gemini" : "Ollama"}
      </p>
      {result.abstained ? (
        <div className="rounded-2xl border border-rule bg-card px-6 py-12 text-center">
          <p className="font-display text-2xl">Not enough evidence in the corpus</p>
          <p className="mt-2 text-sm text-ink-soft">
            {abstentionMessage(result.abstention_reason)}
          </p>
        </div>
      ) : (
        <section className="rounded-2xl border border-rule bg-card p-5 sm:p-6">
          <h2 className="font-display text-2xl text-ink">Answer</h2>
          {!result.cited_chunks.length ? (
            <p className="mt-2 rounded-lg border border-amber/40 bg-amber/10 px-3 py-2 text-sm text-ink">
              The model did not cite the retrieved passages. The list below is
              what was sent to the generator.
            </p>
          ) : null}
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
                key={`cited-${chunk.chunk_id}`}
                citationNumber={citationNumber}
                chunk={chunk}
                highlighted={false}
                assignAnchor={false}
              />
            );
          })}
        </section>
      ) : null}

      {result.context_chunks.length > 0 ? (
        <section className="space-y-3">
          <h3 className="font-display text-xl text-ink">Retrieved context</h3>
          {result.context_chunks.map((chunk, index) => {
            const citationNumber = chunk.prompt_index ?? index + 1;
            return (
              <CitedChunkCard
                key={`context-${chunk.chunk_id}`}
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

// Search UI: dispatches TF-IDF, BM25, semantic, and RAG; exposes PRF,
// BM25 modes (Default / Tunable / Calibrated; "finetuned" on the wire),
// cross-encoder reranker as toggles.
export function SearchPage() {
  const [query, setQuery] = useState("");
  const [algorithm, setAlgorithm] = useState<Algorithm>("tfidf");
  const [usePrf, setUsePrf] = useState(false);
  const [useReranker, setUseReranker] = useState(false);
  const [useQueryRewrite, setUseQueryRewrite] = useState(false);
  const [llmProvider, setLlmProvider] = useState<LlmProvider>("ollama");
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [topK, setTopK] = useState(10);
  const [ragTopK, setRagTopK] = useState(4);
  const [k1, setK1] = useState(1.5);
  const [b, setB] = useState(0.75);
  const [bm25Mode, setBm25Mode] = useState<Bm25Mode>("default");
  const [bm25Effective, setBm25Effective] = useState<{
    mode: Bm25Mode;
    k1: number;
    b: number;
  } | null>(null);
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
    setBm25Effective(null);
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
          bm25_mode: bm25Mode,
          ...(bm25Mode === "tunable" ? { k1, b } : {}),
        });
        setHits(toKeywordHits(response.results));
        setElapsedMs(response.elapsed_ms);
        setResultCount(response.result_count);
        if (response.bm25_mode !== null) {
          setBm25Effective({
            mode: response.bm25_mode,
            k1: response.k1,
            b: response.b,
          });
        }
      } else if (algorithm === "semantic") {
        const response = await searchSemantic({ q: trimmed, top_k: topK });
        setHits(toSemanticHits(response.results));
        setElapsedMs(response.elapsed_ms);
        setResultCount(response.result_count);
      } else {
        const response = await searchRag({
          query: trimmed,
          top_k: ragTopK,
          use_reranker: useReranker,
          use_query_rewrite: useQueryRewrite,
          llm_provider: llmProvider,
        });
        setRagResult(response);
        setElapsedMs(response.elapsed_ms);
        setResultCount(response.context_chunks.length);
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

        {algorithm === "bm25" ? (
          <fieldset className="flex flex-wrap gap-2">
            <legend className="mb-1 w-full text-sm text-ink-soft">
              BM25 parameters
            </legend>
            {BM25_MODES.map((option) => (
              <label
                key={option.id}
                className={[
                  "cursor-pointer rounded-full border px-4 py-2 text-sm font-medium",
                  bm25Mode === option.id
                    ? "border-burgundy bg-burgundy text-paper"
                    : "border-rule bg-card text-ink hover:border-burgundy/40",
                ].join(" ")}
              >
                <input
                  type="radio"
                  name="bm25-mode"
                  value={option.id}
                  checked={bm25Mode === option.id}
                  onChange={() => {
                    setBm25Mode(option.id);
                    setBm25Effective(null);
                  }}
                  className="sr-only"
                />
                {option.label}
              </label>
            ))}
          </fieldset>
        ) : null}

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
          <div className="space-y-3">
            <fieldset className="flex flex-wrap gap-2">
              <legend className="mb-1 w-full text-sm text-ink-soft">
                Generator
              </legend>
              {(
                [
                  { id: "ollama", label: "Ollama (local)" },
                  { id: "gemini", label: "Gemini (API)" },
                ] as const
              ).map((option) => (
                <label
                  key={option.id}
                  className={[
                    "cursor-pointer rounded-full border px-4 py-2 text-sm font-medium",
                    llmProvider === option.id
                      ? "border-burgundy bg-burgundy text-paper"
                      : "border-rule bg-card text-ink hover:border-burgundy/40",
                  ].join(" ")}
                >
                  <input
                    type="radio"
                    name="llm-provider"
                    value={option.id}
                    checked={llmProvider === option.id}
                    onChange={() => setLlmProvider(option.id)}
                    className="sr-only"
                  />
                  {option.label}
                </label>
              ))}
            </fieldset>
            <div className="flex flex-wrap gap-4">
            <label className="flex cursor-pointer items-center gap-2 text-sm text-ink">
              <input
                type="checkbox"
                checked={useReranker}
                onChange={(event) => setUseReranker(event.target.checked)}
                className="size-4 accent-burgundy"
              />
              Rerank results
            </label>
            <label className="flex cursor-pointer items-center gap-2 text-sm text-ink">
              <input
                type="checkbox"
                checked={useQueryRewrite}
                onChange={(event) => setUseQueryRewrite(event.target.checked)}
                className="size-4 accent-burgundy"
              />
              Rewrite query
            </label>
            </div>
          </div>
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
                  max={algorithm === "rag" ? 8 : 50}
                  value={algorithm === "rag" ? ragTopK : topK}
                  onChange={(event) => {
                    const value = Number(event.target.value);
                    if (algorithm === "rag") {
                      setRagTopK(value);
                    } else {
                      setTopK(value);
                    }
                  }}
                  className="w-full rounded-lg border border-rule bg-paper px-3 py-2"
                />
              </label>
              {algorithm === "bm25" && bm25Mode === "tunable" ? (
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
              {algorithm === "bm25" && bm25Mode !== "tunable" ? (
                <p className="text-sm text-ink-soft sm:col-span-2">
                  {bm25Mode === "default"
                    ? "Default mode always scores with k1=1.5 and b=0.75. Request k1/b are ignored."
                    : "Calibrated mode uses server-side k1 and b selected by an in-sample evaluation sweep (macro nDCG@4, then MRR). Several settings tied, so k1=1.5 and b=0.75 were retained. Request values are ignored."}
                </p>
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
          {algorithm === "bm25" && bm25Effective ? (
            <>
              {" · "}
              {BM25_MODE_LABELS[bm25Effective.mode]} · k1={bm25Effective.k1} · b={bm25Effective.b}
            </>
          ) : null}
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
          {hits.map((hit, index) => {
            const firstOfDocument =
              hits.findIndex((item) => item.document_id === hit.document_id) ===
              index;
            return (
              <div
                key={hit.chunk_id}
                id={
                  firstOfDocument
                    ? resultDocumentElementId(hit.document_id)
                    : undefined
                }
                className="scroll-mt-24"
              >
                <ResultCard hit={hit} />
              </div>
            );
          })}
          {algorithm === "tfidf" || algorithm === "bm25" ? (
            <KeywordHeatmap hits={hits} />
          ) : null}
          {new Set(hits.map((hit) => hit.document_id)).size >= 2 ? (
            <DocumentRelationGraph
              hits={hits}
              mode={algorithm === "semantic" ? "semantic" : "lexical"}
            />
          ) : null}
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
