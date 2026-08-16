import { BrowserRouter, Route, Routes } from "react-router-dom";

import { ErrorBoundary } from "./components/ErrorBoundary";
import { Layout } from "./components/Layout";
import { AdminPage } from "./pages/AdminPage";
import { SearchPage } from "./pages/SearchPage";

export default function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/" element={<SearchPage />} />
            <Route path="/admin" element={<AdminPage />} />
            <Route
              path="*"
              element={
                <div className="rounded-2xl border border-rule bg-card px-6 py-12 text-center">
                  <p className="font-display text-2xl">Page not found</p>
                  <p className="mt-2 text-sm text-ink-soft">
                    Use Search or Admin in the header.
                  </p>
                </div>
              }
            />
          </Route>
        </Routes>
      </BrowserRouter>
    </ErrorBoundary>
  );
}
