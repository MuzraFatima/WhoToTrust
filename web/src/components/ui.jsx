import { ROUTE_NOTES } from "../lib/format.js";

export function Stat({ value, caption, title }) {
  return (
    <div className="card stat">
      {title && <div className="eyebrow">{title}</div>}
      <div className="value">{value}</div>
      <div className="caption">{caption}</div>
    </div>
  );
}

export function RouteChip({ route }) {
  return (
    <span className="chip" title={ROUTE_NOTES[route]}>
      <span className="dot" style={{ background: `var(--${route})` }} />
      {route}
    </span>
  );
}

export function PassChip({ pass }) {
  return pass ? (
    <span className="chip pass">pass</span>
  ) : (
    <span className="chip miss">miss</span>
  );
}

export function Notice({ children, kind }) {
  return (
    <div className={`notice ${kind ?? ""}`} role={kind === "error" ? "alert" : undefined}>
      {children}
    </div>
  );
}

export function ErrorNotice({ error }) {
  return <Notice kind="error">Could not load this: {String(error?.message ?? error)}</Notice>;
}

export function EmptyRuns() {
  return (
    <div className="card empty">
      <h2>No results yet</h2>
      <p className="muted">
        Run the pipeline once, then reload this page. It takes about 45 minutes on a laptop CPU the
        first time, and nothing leaves your machine.
      </p>
      <pre>
        python scripts/make_mini.py --seed-cache{"\n"}python scripts/run_mini.py
      </pre>
    </div>
  );
}

export function Legend({ items }) {
  return (
    <div className="legend">
      {items.map((it) => (
        <span key={it.key}>
          <i style={{ background: it.colour }} />
          {it.label}
        </span>
      ))}
    </div>
  );
}

export function Pager({ page, pages, total, pageSize, onPage, onPageSize }) {
  return (
    <div className="pager">
      <span className="muted small">
        {total.toLocaleString()} items, page {page} of {pages}
      </span>
      <span style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
        <label className="small muted" htmlFor="page-size">
          Per page
        </label>
        <select
          id="page-size"
          value={pageSize}
          onChange={(e) => onPageSize(Number(e.target.value))}
        >
          {[10, 25, 50, 100].map((n) => (
            <option key={n}>{n}</option>
          ))}
        </select>
        <button className="btn" disabled={page <= 1} onClick={() => onPage(page - 1)}>
          Previous
        </button>
        <button className="btn" disabled={page >= pages} onClick={() => onPage(page + 1)}>
          Next
        </button>
      </span>
    </div>
  );
}
