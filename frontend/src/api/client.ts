import axios, { isAxiosError } from "axios";
import type { AxiosProgressEvent } from "axios";

export type SourceType = "upload" | "web";
export type DocumentStatus = "pending" | "processing" | "indexed" | "failed";

export type DocumentResponse = {
  id: string;
  title: string;
  original_filename: string | null;
  source_type: SourceType;
  source_url: string | null;
  mime_type: string | null;
  file_size_bytes: number | null;
  status: DocumentStatus;
  chunk_count: number;
  keyword_indexed: boolean;
  vector_indexed: boolean;
  extra_metadata: Record<string, unknown>;
  error_message: string | null;
  created_at: string;
  updated_at: string;
  indexed_at: string | null;
};

export type DocumentListResponse = {
  items: DocumentResponse[];
  total: number;
  offset: number;
  limit: number;
};

export type KeywordSearchResult = {
  chunk_id: string;
  document_id: string;
  document_title: string;
  score: number;
  text: string;
  page_number: number | null;
  section_title: string | null;
  matched_terms: string[];
  term_contributions: Record<string, number>;
};

export type KeywordSearchResponse = {
  query: string;
  mode: "tfidf";
  result_count: number;
  elapsed_ms: number;
  results: KeywordSearchResult[];
};

export type BM25SearchResponse = {
  query: string;
  mode: "bm25";
  k1: number;
  b: number;
  result_count: number;
  elapsed_ms: number;
  results: KeywordSearchResult[];
};

export type SemanticSearchResult = {
  chunk_id: string;
  document_id: string;
  document_title: string;
  score: number;
  distance: number;
  text: string;
  page_number: number | null;
  section_title: string | null;
};

export type SemanticSearchResponse = {
  query: string;
  mode: "semantic";
  result_count: number;
  elapsed_ms: number;
  results: SemanticSearchResult[];
};

export type KeywordIndexStatsResponse = {
  document_count: number;
  chunk_count: number;
  vocabulary_size: number;
  posting_count: number;
};

export type VectorStoreStatsResponse = {
  chunk_count: number;
};

export type HealthResponse = {
  status: "ok";
  app: string;
  version: string;
  environment: string;
};

export type ListDocumentsParams = {
  offset?: number;
  limit?: number;
  source_type?: SourceType;
  status?: DocumentStatus;
};

export type KeywordSearchParams = {
  q: string;
  top_k?: number;
  candidate_limit?: number;
};

export type BM25SearchParams = {
  q: string;
  top_k?: number;
  candidate_limit?: number;
  k1?: number;
  b?: number;
};

export type SemanticSearchParams = {
  q: string;
  top_k?: number;
};

export class ApiError extends Error {
  readonly status: number;
  readonly detail: unknown;

  constructor(status: number, detail: unknown, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

export function formatApiError(error: unknown): string {
  if (error instanceof ApiError) {
    return error.message;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "Something went wrong. Please try again.";
}

function formatDetail(detail: unknown): string {
  if (typeof detail === "string" && detail.trim()) {
    return detail;
  }
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => {
        if (item && typeof item === "object" && "msg" in item) {
          return String(item.msg);
        }
        return null;
      })
      .filter((message): message is string => Boolean(message));
    if (messages.length) {
      return messages.join("; ");
    }
  }
  if (detail && typeof detail === "object") {
    if ("message" in detail && typeof detail.message === "string") {
      return detail.message;
    }
  }
  return "The request failed.";
}

const client = axios.create({
  baseURL: "/api/v1",
  timeout: 180_000,
});

client.interceptors.response.use(
  (response) => response,
  (error: unknown) => {
    if (isAxiosError(error)) {
      if (!error.response) {
        throw new ApiError(
          0,
          error.message,
          "Cannot reach the API. Is the backend running?",
        );
      }
      const detail = (error.response.data as { detail?: unknown } | undefined)
        ?.detail;
      throw new ApiError(
        error.response.status,
        detail ?? error.response.data,
        formatDetail(detail ?? error.message),
      );
    }
    throw new ApiError(0, error, "The request failed.");
  },
);

export async function listDocuments(
  params: ListDocumentsParams = {},
): Promise<DocumentListResponse> {
  const { data } = await client.get<DocumentListResponse>("/documents", {
    params: {
      offset: params.offset ?? 0,
      limit: params.limit ?? 20,
      source_type: params.source_type,
      status: params.status,
    },
  });
  return data;
}

export async function listAllDocuments(): Promise<DocumentListResponse> {
  const pageSize = 100;
  const first = await listDocuments({ offset: 0, limit: pageSize });
  const items = [...first.items];

  while (items.length < first.total) {
    const next = await listDocuments({
      offset: items.length,
      limit: pageSize,
    });
    if (!next.items.length) {
      break;
    }
    items.push(...next.items);
  }

  return {
    items,
    total: first.total,
    offset: 0,
    limit: items.length,
  };
}

export async function getDocument(
  documentId: string,
): Promise<DocumentResponse> {
  const { data } = await client.get<DocumentResponse>(
    `/documents/${encodeURIComponent(documentId)}`,
  );
  return data;
}

export async function uploadDocument(
  file: File,
  onProgress?: (percent: number) => void,
): Promise<DocumentResponse> {
  const body = new FormData();
  body.append("file", file);
  const { data } = await client.post<DocumentResponse>("/documents", body, {
    onUploadProgress: (event: AxiosProgressEvent) => {
      if (!event.total) {
        return;
      }
      onProgress?.(Math.round((event.loaded / event.total) * 100));
    },
  });
  return data;
}

export async function deleteDocument(documentId: string): Promise<void> {
  await client.delete(`/documents/${encodeURIComponent(documentId)}`);
}

export async function searchKeyword(
  params: KeywordSearchParams,
): Promise<KeywordSearchResponse> {
  const { data } = await client.get<KeywordSearchResponse>("/search/keyword", {
    params,
  });
  return data;
}

export async function searchBm25(
  params: BM25SearchParams,
): Promise<BM25SearchResponse> {
  const { data } = await client.get<BM25SearchResponse>("/search/bm25", {
    params,
  });
  return data;
}

export async function searchSemantic(
  params: SemanticSearchParams,
): Promise<SemanticSearchResponse> {
  const { data } = await client.get<SemanticSearchResponse>(
    "/search/semantic",
    { params },
  );
  return data;
}

export async function getKeywordStats(): Promise<KeywordIndexStatsResponse> {
  const { data } = await client.get<KeywordIndexStatsResponse>(
    "/search/keyword/stats",
  );
  return data;
}

export async function getVectorStats(): Promise<VectorStoreStatsResponse> {
  const { data } = await client.get<VectorStoreStatsResponse>(
    "/search/semantic/stats",
  );
  return data;
}

export async function getHealth(): Promise<HealthResponse> {
  const { data } = await client.get<HealthResponse>("/health");
  return data;
}
