import { NavLink, Outlet } from "react-router-dom";

const navClass = ({ isActive }: { isActive: boolean }) =>
  [
    "rounded-full px-3 py-1.5 text-sm font-medium transition",
    isActive
      ? "bg-burgundy text-paper"
      : "text-ink-soft hover:bg-paper-2 hover:text-ink",
  ].join(" ");

export function Layout() {
  return (
    <div className="min-h-svh">
      <header className="border-b border-rule bg-card/80 backdrop-blur">
        <div className="mx-auto flex max-w-6xl flex-col gap-3 px-4 py-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="font-display text-xl font-medium tracking-tight text-ink">
              MIR Search
            </p>
            <p className="text-sm text-ink-soft">
              Dual-engine document retrieval
            </p>
          </div>
          <nav className="flex gap-2" aria-label="Primary">
            <NavLink to="/" className={navClass} end>
              Search
            </NavLink>
            <NavLink to="/admin" className={navClass}>
              Admin
            </NavLink>
          </nav>
        </div>
      </header>
      <main className="mx-auto w-full max-w-6xl px-4 py-8">
        <Outlet />
      </main>
    </div>
  );
}
