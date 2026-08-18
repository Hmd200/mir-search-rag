import { useCallback, useEffect, useRef, useState } from "react";
import type { DragEvent, ChangeEvent, FormEvent } from "react";

import {
  addDocumentFromUrl,
  deleteDocument,
  formatApiError,
  getKeywordStats,
  listAllDocuments,
  uploadDocument,
} from "../api/client";
import type { DocumentResponse, KeywordIndexStatsResponse } from "../api/client";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { ErrorBanner } from "../components/ErrorBanner";
import { fileTypeLabel, formatDate } from "../lib/format";

const ACCEPT = ".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document";

type UploadState =
  | { status: "idle" }
  | { status: "uploading"; fileName: string; percent: number }
  | { status: "indexing"; fileName: string }
  | { status: "success"; fileName: string }
  | { status: "error"; fileName: string; message: string };

function sourceTypeLabel(document: DocumentResponse): string {
  return document.source_type === "web" ? "Web" : "Upload";
}

function IndexBadge({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span
      className={[
        "inline-flex rounded-full px-2 py-0.5 text-xs font-medium",
        ok ? "bg-sage/15 text-sage" : "bg-paper-2 text-ink-soft",
      ].join(" ")}
    >
      {label}: {ok ? "indexed" : "pending"}
    </span>
  );
}

function StatusBadge({ status }: { status: DocumentResponse["status"] }) {
  const styles: Record<DocumentResponse["status"], string> = {
    indexed: "bg-sage/15 text-sage",
    pending: "bg-amber/15 text-amber",
    processing: "bg-amber/15 text-amber",
    failed: "bg-danger/10 text-danger",
  };
  return (
    <span className={`inline-flex rounded-full px-2 py-0.5 text-xs font-medium ${styles[status]}`}>
      {status}
    </span>
  );
}

export function AdminPage() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [documents, setDocuments] = useState<DocumentResponse[]>([]);
  const [total, setTotal] = useState(0);
  const [stats, setStats] = useState<KeywordIndexStatsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [upload, setUpload] = useState<UploadState>({ status: "idle" });
  const [pageUrl, setPageUrl] = useState("");
  const [pendingDelete, setPendingDelete] = useState<DocumentResponse | null>(null);
  const [deleting, setDeleting] = useState(false);

  const refresh = useCallback(async () => {
    const [list, keywordStats] = await Promise.all([
      listAllDocuments(),
      getKeywordStats(),
    ]);
    setDocuments(list.items);
    setTotal(list.total);
    setStats(keywordStats);
  }, []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    refresh()
      .catch((cause: unknown) => {
        if (!cancelled) {
          setError(formatApiError(cause));
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [refresh]);

  async function ingestFiles(files: FileList | File[]) {
    const selected = [...files].filter((file) => {
      const name = file.name.toLowerCase();
      return name.endsWith(".pdf") || name.endsWith(".docx");
    });
    if (!selected.length) {
      setUpload({
        status: "error",
        fileName: "",
        message: "Only PDF and DOCX files can be uploaded.",
      });
      return;
    }

    for (const file of selected) {
      try {
        setUpload({ status: "uploading", fileName: file.name, percent: 0 });
        const document = await uploadDocument(file, (percent) => {
          setUpload(
            percent >= 100
              ? { status: "indexing", fileName: file.name }
              : { status: "uploading", fileName: file.name, percent },
          );
        });
        setUpload({ status: "success", fileName: document.title || file.name });
        await refresh();
        setError(null);
      } catch (cause) {
        setUpload({
          status: "error",
          fileName: file.name,
          message: formatApiError(cause),
        });
        return;
      }
    }
  }

  async function ingestUrl(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = pageUrl.trim();
    if (!trimmed) {
      setUpload({
        status: "error",
        fileName: "",
        message: "Enter a URL to scrape.",
      });
      return;
    }

    try {
      setUpload({ status: "indexing", fileName: trimmed });
      const document = await addDocumentFromUrl(trimmed);
      setUpload({ status: "success", fileName: document.title || trimmed });
      setPageUrl("");
      await refresh();
      setError(null);
    } catch (cause) {
      setUpload({
        status: "error",
        fileName: trimmed,
        message: formatApiError(cause),
      });
    }
  }

  function onDrop(event: DragEvent<HTMLElement>) {
    event.preventDefault();
    setDragging(false);
    if (event.dataTransfer.files.length) {
      void ingestFiles(event.dataTransfer.files);
    }
  }

  function onSelect(event: ChangeEvent<HTMLInputElement>) {
    if (event.target.files?.length) {
      void ingestFiles(event.target.files);
      event.target.value = "";
    }
  }

  async function confirmDelete() {
    if (!pendingDelete) {
      return;
    }
    setDeleting(true);
    try {
      await deleteDocument(pendingDelete.id);
      setPendingDelete(null);
      await refresh();
      setError(null);
    } catch (cause) {
      setError(formatApiError(cause));
      setPendingDelete(null);
    } finally {
      setDeleting(false);
    }
  }

  const busy = upload.status === "uploading" || upload.status === "indexing";

  return (
    <div className="space-y-8">
      <div>
        <h1 className="font-display text-3xl text-ink">Admin dashboard</h1>
        <p className="mt-1 text-ink-soft">
          Upload PDF or DOCX files, or scrape a web page. Both keyword and vector
          indexes update together.
        </p>
      </div>

      {error ? (
        <ErrorBanner message={error} onDismiss={() => setError(null)} />
      ) : null}

      <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {[
          ["Documents", stats?.document_count],
          ["Chunks", stats?.chunk_count],
          ["Vocabulary", stats?.vocabulary_size],
          ["Postings", stats?.posting_count],
        ].map(([label, value]) => (
          <article
            key={String(label)}
            className="rounded-2xl border border-rule bg-card px-4 py-4"
          >
            <p className="text-xs font-medium uppercase tracking-wide text-ink-soft">
              {label}
            </p>
            <p className="mt-1 font-display text-2xl">
              {loading && stats === null ? "—" : (value ?? 0).toLocaleString()}
            </p>
          </article>
        ))}
      </section>

      <section>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          multiple
          className="sr-only"
          onChange={onSelect}
        />
        <button
          type="button"
          disabled={busy}
          onClick={() => inputRef.current?.click()}
          onDragEnter={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          className={[
            "w-full rounded-2xl border-2 border-dashed px-4 py-10 text-center transition",
            dragging
              ? "border-burgundy bg-burgundy/5"
              : "border-rule bg-card hover:border-burgundy/50",
            busy ? "cursor-wait opacity-80" : "cursor-pointer",
          ].join(" ")}
        >
          <p className="font-display text-lg">Drop PDF or DOCX files here</p>
          <p className="mt-1 text-sm text-ink-soft">or click to browse</p>
          {upload.status === "uploading" ? (
            <div className="mx-auto mt-4 max-w-md">
              <p className="text-sm text-ink">
                Uploading {upload.fileName} ({upload.percent}%)
              </p>
              <div className="mt-2 h-2 overflow-hidden rounded-full bg-paper-2">
                <div
                  className="h-full bg-burgundy transition-all"
                  style={{ width: `${upload.percent}%` }}
                />
              </div>
            </div>
          ) : null}
          {upload.status === "indexing" ? (
            <p className="mt-4 text-sm text-ink">
              Indexing {upload.fileName}…
            </p>
          ) : null}
          {upload.status === "success" ? (
            <p className="mt-4 text-sm text-sage">Indexed {upload.fileName}.</p>
          ) : null}
          {upload.status === "error" ? (
            <p className="mt-4 text-sm text-danger">{upload.message}</p>
          ) : null}
        </button>
      </section>

      <section className="rounded-2xl border border-rule bg-card p-4 sm:p-5">
        <h2 className="font-display text-xl text-ink">Add from URL</h2>
        <p className="mt-1 text-sm text-ink-soft">
          Scrape the main text of a public web page into the same corpus.
        </p>
        <form
          onSubmit={(event) => void ingestUrl(event)}
          className="mt-4 flex flex-col gap-3 sm:flex-row"
        >
          <label className="block min-w-0 flex-1">
            <span className="sr-only">Page URL</span>
            <input
              type="url"
              value={pageUrl}
              onChange={(event) => setPageUrl(event.target.value)}
              placeholder="https://example.com/article"
              disabled={busy}
              className="w-full rounded-lg border border-rule bg-paper px-3 py-2 text-sm outline-none focus:border-burgundy"
            />
          </label>
          <button
            type="submit"
            disabled={busy}
            className="rounded-full bg-burgundy px-5 py-2 text-sm font-semibold text-paper hover:bg-burgundy-dark disabled:opacity-60"
          >
            {upload.status === "indexing" ? "Indexing…" : "Add page"}
          </button>
        </form>
        {upload.status === "indexing" ? (
          <p className="mt-3 text-sm text-ink">Indexing {upload.fileName}…</p>
        ) : null}
        {upload.status === "success" ? (
          <p className="mt-3 text-sm text-sage">Indexed {upload.fileName}.</p>
        ) : null}
        {upload.status === "error" ? (
          <p className="mt-3 text-sm text-danger">{upload.message}</p>
        ) : null}
      </section>

      <section>
        <div className="mb-3 flex items-baseline justify-between gap-3">
          <h2 className="font-display text-2xl">Indexed documents</h2>
          <p className="text-sm text-ink-soft">{total} in corpus</p>
        </div>

        {loading ? (
          <div className="space-y-2">
            {[0, 1, 2].map((key) => (
              <div
                key={key}
                className="h-16 animate-pulse rounded-xl bg-paper-2"
              />
            ))}
          </div>
        ) : documents.length === 0 ? (
          <div className="rounded-2xl border border-rule bg-card px-6 py-12 text-center">
            <p className="font-display text-xl">The corpus is empty</p>
            <p className="mt-2 text-sm text-ink-soft">
              Upload a PDF or DOCX file to start indexing.
            </p>
          </div>
        ) : (
          <>
            <div className="hidden overflow-x-auto rounded-2xl border border-rule bg-card md:block">
              <table className="w-full min-w-[720px] text-left text-sm">
                <thead className="border-b border-rule bg-paper-2/60 text-xs uppercase tracking-wide text-ink-soft">
                  <tr>
                    <th className="px-4 py-3 font-medium">Title</th>
                    <th className="px-4 py-3 font-medium">Source</th>
                    <th className="px-4 py-3 font-medium">Type</th>
                    <th className="px-4 py-3 font-medium">Date added</th>
                    <th className="px-4 py-3 font-medium">Chunks</th>
                    <th className="px-4 py-3 font-medium">Indexes</th>
                    <th className="px-4 py-3 font-medium"> </th>
                  </tr>
                </thead>
                <tbody>
                  {documents.map((document) => (
                    <tr key={document.id} className="border-b border-rule last:border-0">
                      <td className="px-4 py-3">
                        <p className="font-medium text-ink">{document.title}</p>
                        {document.source_url ? (
                          <a
                            href={document.source_url}
                            target="_blank"
                            rel="noreferrer"
                            className="mt-1 block truncate text-xs text-burgundy hover:text-burgundy-dark"
                            title={document.source_url}
                          >
                            {document.source_url}
                          </a>
                        ) : (
                          <p className="text-xs text-ink-soft">
                            {document.original_filename ?? document.id}
                          </p>
                        )}
                        {document.error_message ? (
                          <p className="mt-1 text-xs text-danger">
                            {document.error_message}
                          </p>
                        ) : null}
                      </td>
                      <td className="px-4 py-3">{sourceTypeLabel(document)}</td>
                      <td className="px-4 py-3">{fileTypeLabel(document)}</td>
                      <td className="px-4 py-3 text-ink-soft">
                        {formatDate(document.created_at)}
                      </td>
                      <td className="px-4 py-3">{document.chunk_count}</td>
                      <td className="px-4 py-3">
                        <div className="flex flex-wrap gap-1">
                          <StatusBadge status={document.status} />
                          <IndexBadge ok={document.keyword_indexed} label="Keyword" />
                          <IndexBadge ok={document.vector_indexed} label="Vector" />
                        </div>
                      </td>
                      <td className="px-4 py-3 text-right">
                        <button
                          type="button"
                          onClick={() => setPendingDelete(document)}
                          className="rounded-full px-3 py-1 text-sm font-medium text-danger hover:bg-danger/10"
                        >
                          Delete
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <ul className="space-y-3 md:hidden">
              {documents.map((document) => (
                <li
                  key={document.id}
                  className="rounded-2xl border border-rule bg-card p-4"
                >
                  <p className="font-medium">{document.title}</p>
                  <p className="text-xs text-ink-soft">
                    {sourceTypeLabel(document)} · {fileTypeLabel(document)} ·{" "}
                    {formatDate(document.created_at)} · {document.chunk_count} chunks
                  </p>
                  {document.source_url ? (
                    <a
                      href={document.source_url}
                      target="_blank"
                      rel="noreferrer"
                      className="mt-1 block truncate text-xs text-burgundy"
                      title={document.source_url}
                    >
                      {document.source_url}
                    </a>
                  ) : null}
                  <div className="mt-2 flex flex-wrap gap-1">
                    <StatusBadge status={document.status} />
                    <IndexBadge ok={document.keyword_indexed} label="Keyword" />
                    <IndexBadge ok={document.vector_indexed} label="Vector" />
                  </div>
                  {document.error_message ? (
                    <p className="mt-2 text-xs text-danger">{document.error_message}</p>
                  ) : null}
                  <button
                    type="button"
                    onClick={() => setPendingDelete(document)}
                    className="mt-3 text-sm font-medium text-danger"
                  >
                    Delete
                  </button>
                </li>
              ))}
            </ul>
          </>
        )}
      </section>

      <ConfirmDialog
        open={pendingDelete !== null}
        title="Delete document?"
        message={
          pendingDelete
            ? `Remove “${pendingDelete.title}” from the corpus and both indexes.`
            : ""
        }
        busy={deleting}
        onCancel={() => setPendingDelete(null)}
        onConfirm={() => {
          void confirmDelete();
        }}
      />
    </div>
  );
}
