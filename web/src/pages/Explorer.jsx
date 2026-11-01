import { useEffect, useMemo, useState } from "react";
import { getJSON, imageUrl } from "../lib/api.js";
import { useAsync, useDebounced } from "../lib/hooks.js";
import { DATASETS, ERROR_LABELS, ROUTE_NOTES, num, voteSummary } from "../lib/format.js";
import { Scatter } from "../components/Scatter.jsx";
import { CardSkeleton, Skeleton } from "../components/Skeleton.jsx";
import { EmptyRuns, ErrorNotice, Pager, RouteChip } from "../components/ui.jsx";

const ROUTES = ["clear", "ambiguous", "unreliable"];

/** Keep only the points the route and error filters allow (the table filters on the server). */
function filterPoints(points, routesOn, onlyErrors) {
  const keep = [];
  points.route.forEach((r, i) => {
    if (!routesOn.includes(points.routes[r])) return;
    if (onlyErrors && points.true_errors && !(points.true_errors[i] > 0)) return;
    keep.push(i);
  });
  const pick = (a) => keep.map((i) => a[i]);
  return {
    routes: points.routes,
    item_id: pick(points.item_id),
    entropy: pick(points.entropy),
    max_p_error: pick(points.max_p_error),
    route: pick(points.route),
  };
}

function ItemDetail({ run, item }) {
  const { data, error, loading } = useAsync(() => getJSON("/api/item", { run, item }), [run, item]);
  if (loading) return <CardSkeleton lines={4} height={140} />;
  if (error) return <ErrorNotice error={error} />;
  return (
    <section className="card" aria-label="Selected item">
      <div className="detail">
        <div>
          <div style={{ display: "flex", gap: "0.6rem", alignItems: "center", flexWrap: "wrap" }}>
            <h2 style={{ margin: 0 }}>Item {data.item_id}</h2>
            <RouteChip route={data.route} />
          </div>
          <p className="muted small" style={{ marginTop: 4 }}>
            {ROUTE_NOTES[data.route]} Votes: {voteSummary(data.labels, data.classes)}. Entropy{" "}
            {num(data.entropy)}.
          </p>
          {data.has_image && (
            <img
              className="pixel"
              src={imageUrl(run, item)}
              alt={`The picture for item ${data.item_id}`}
              loading="lazy"
              width="128"
              height="128"
            />
          )}
          {data.text && (
            <>
              <blockquote className="quote">
                <strong>Premise.</strong> {data.text.premise}
              </blockquote>
              <blockquote className="quote">
                <strong>Hypothesis.</strong> {data.text.hypothesis}
              </blockquote>
            </>
          )}
        </div>
        <div className="table-wrap">
          <table aria-label="Votes on this item">
            <thead>
              <tr>
                <th scope="col">Annotator</th>
                <th scope="col">Voted</th>
                <th scope="col">p(error)</th>
                <th scope="col">Why</th>
              </tr>
            </thead>
            <tbody>
              {data.votes.map((v, i) => (
                <tr key={`${v.annotator_id}-${i}`} className={v.is_error ? "vote-error" : undefined}>
                  <td>
                    {v.annotator_id}
                    {v.is_error && (
                      <div className="small" style={{ color: "var(--bad)" }}>
                        injected {v.error_type} error
                      </div>
                    )}
                  </td>
                  <td>{v.label_name}</td>
                  <td>
                    <div className="bar" title={`p(error) ${num(v.p_error, 2)}`}>
                      <span style={{ width: `${Math.round(v.p_error * 100)}%` }} />
                    </div>
                    <span className="small">{num(v.p_error, 2)}</span>
                  </td>
                  <td className="wrap">
                    <ul className="reasons small">
                      {v.reasons.map((r) => (
                        <li key={r}>{r}</li>
                      ))}
                    </ul>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

export default function Explorer() {
  const runsState = useAsync(() => getJSON("/api/runs"), []);
  const [dataset, setDataset] = useState("cifar10h");
  const [errorType, setErrorType] = useState("spammer");
  const [seed, setSeed] = useState(0);
  const [routesOn, setRoutesOn] = useState(ROUTES);
  const [onlyErrors, setOnlyErrors] = useState(false);
  const [query, setQuery] = useState("");
  const q = useDebounced(query, 300);
  const [pageSize, setPageSize] = useState(10);
  // The page and the selected item are only valid for the run and filters they were chosen under,
  // so they carry that key. A different key reads as "page 1, nothing chosen" with no stale
  // request in between (a request for the old item against the new run would be a 404).
  const [pageAt, setPageAt] = useState({ key: "", page: 1 });
  const [pick, setPick] = useState({ run: null, item: null });

  const runs = useMemo(
    () => (runsState.data ?? []).filter((r) => r.method === "labeltriage"),
    [runsState.data],
  );
  const datasets = useMemo(() => [...new Set(runs.map((r) => r.dataset))], [runs]);
  const errorTypes = useMemo(
    () => [...new Set(runs.filter((r) => r.dataset === dataset).map((r) => r.error_type))],
    [runs, dataset],
  );
  const seeds = useMemo(
    () =>
      runs
        .filter((r) => r.dataset === dataset && r.error_type === errorType)
        .map((r) => r.seed)
        .sort((a, b) => a - b),
    [runs, dataset, errorType],
  );

  // Keep the three selectors pointing at a run that exists.
  useEffect(() => {
    if (!runs.length) return;
    if (!datasets.includes(dataset)) setDataset(datasets[0]);
    else if (!errorTypes.includes(errorType)) setErrorType(errorTypes[0]);
    else if (!seeds.includes(seed)) setSeed(seeds[0]);
  }, [runs, datasets, errorTypes, seeds, dataset, errorType, seed]);

  const run = runs.find(
    (r) => r.dataset === dataset && r.error_type === errorType && r.seed === seed,
  )?.id;

  const filterKey = JSON.stringify([run, routesOn, onlyErrors, q, pageSize]);
  const page = pageAt.key === filterKey ? pageAt.page : 1;
  const setPage = (p) => setPageAt({ key: filterKey, page: p });
  const selected = pick.run === run ? pick.item : null;
  const setSelected = (item) => setPick({ run, item });

  const points = useAsync(() => (run ? getJSON("/api/points", { run }) : Promise.resolve(null)), [run]);
  const list = useAsync(
    () =>
      run
        ? getJSON("/api/items", {
            run,
            route: routesOn.length === ROUTES.length ? undefined : routesOn,
            only_errors: onlyErrors,
            q,
            page,
            page_size: pageSize,
          })
        : Promise.resolve(null),
    [run, routesOn, onlyErrors, q, page, pageSize],
  );

  // Open the most suspicious item of a page by default, so the detail panel is never empty.
  const firstItem = list.data?.items?.[0]?.item_id ?? null;
  useEffect(() => {
    if (selected === null && firstItem !== null && !list.loading) {
      setPick({ run, item: firstItem });
    }
  }, [selected, firstItem, list.loading, run]);

  const shown = useMemo(
    () => (points.data ? filterPoints(points.data, routesOn, onlyErrors) : null),
    [points.data, routesOn, onlyErrors],
  );

  if (runsState.loading) return <CardSkeleton height={260} />;
  if (runsState.error) return <ErrorNotice error={runsState.error} />;
  if (!runs.length) return <EmptyRuns />;

  const toggleRoute = (r) =>
    setRoutesOn((on) => (on.includes(r) ? on.filter((x) => x !== r) : [...on, r]));

  return (
    <div className="stack">
      <section className="card">
        <div className="controls">
          <div className="field">
            Dataset
            <div className="segmented" role="group" aria-label="Dataset">
              {datasets.map((d) => (
                <button key={d} aria-pressed={dataset === d} onClick={() => setDataset(d)}>
                  {DATASETS[d] ?? d}
                </button>
              ))}
            </div>
          </div>
          <label className="field">
            Error type
            <select value={errorType} onChange={(e) => setErrorType(e.target.value)}>
              {errorTypes.map((t) => (
                <option key={t} value={t}>
                  {ERROR_LABELS[t] ?? t}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            Seed
            <select value={seed} onChange={(e) => setSeed(Number(e.target.value))}>
              {seeds.map((s) => (
                <option key={s}>{s}</option>
              ))}
            </select>
          </label>
          <div className="field">
            Routes
            <span style={{ display: "flex", gap: "0.4rem" }}>
              {ROUTES.map((r) => (
                <button
                  key={r}
                  className="route-toggle"
                  aria-pressed={routesOn.includes(r)}
                  title={ROUTE_NOTES[r]}
                  onClick={() => toggleRoute(r)}
                >
                  <span
                    aria-hidden="true"
                    style={{
                      display: "inline-block",
                      width: 9,
                      height: 9,
                      borderRadius: "50%",
                      background: `var(--${r})`,
                    }}
                  />
                  {r}
                </button>
              ))}
            </span>
          </div>
          <label className="toggle">
            <input type="checkbox" checked={onlyErrors} onChange={(e) => setOnlyErrors(e.target.checked)} />
            Only items with an injected error
          </label>
          <label className="field">
            Find an item
            <input
              type="search"
              placeholder="item id"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </label>
        </div>
      </section>

      <div className="split">
        <section className="card">
          <h2>Where each item sits</h2>
          <p className="muted small">
            Click a dot to open that item. Higher means LabelTriage suspects one of its votes; further
            right means its votes split more.
          </p>
          {points.loading && <Skeleton height={380} />}
          {points.error && <ErrorNotice error={points.error} />}
          {shown && (shown.item_id.length ? (
            <Scatter points={shown} selected={selected} onSelect={setSelected} />
          ) : (
            <p className="muted">No items match these filters.</p>
          ))}
        </section>

        <section className="card">
          <h2>Most suspicious first</h2>
          {list.loading && !list.data && <Skeleton height={300} />}
          {list.error && <ErrorNotice error={list.error} />}
          {list.data && (
            <>
              <div className="table-wrap">
                <table aria-label="Items">
                  <thead>
                    <tr>
                      <th scope="col">Item</th>
                      <th scope="col">Route</th>
                      <th scope="col">Votes</th>
                      <th scope="col" className="num">Max p(error)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {list.data.items.map((it) => (
                      <tr
                        key={it.item_id}
                        className={`clickable ${it.item_id === selected ? "selected" : ""}`}
                        onClick={() => setSelected(it.item_id)}
                        aria-selected={it.item_id === selected}
                      >
                        <td>
                          <button
                            className="btn"
                            style={{ padding: "0.1rem 0.5rem" }}
                            onClick={(e) => {
                              e.stopPropagation();
                              setSelected(it.item_id);
                            }}
                          >
                            {it.item_id}
                          </button>
                        </td>
                        <td>
                          <RouteChip route={it.route} />
                        </td>
                        <td className="wrap" style={{ minWidth: 150 }}>
                          {voteSummary(it.labels, list.data.classes)}
                        </td>
                        <td className="num">{num(it.max_p_error, 2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {list.data.total === 0 && <p className="muted">No items match these filters.</p>}
              <Pager
                page={list.data.page}
                pages={list.data.pages}
                total={list.data.total}
                pageSize={pageSize}
                onPage={setPage}
                onPageSize={setPageSize}
              />
            </>
          )}
        </section>
      </div>

      {run && selected && <ItemDetail run={run} item={selected} />}
    </div>
  );
}
