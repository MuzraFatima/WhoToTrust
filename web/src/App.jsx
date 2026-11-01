import { Suspense, lazy } from "react";
import { useHashRoute } from "./lib/hooks.js";
import { CardSkeleton } from "./components/Skeleton.jsx";
import Overview from "./pages/Overview.jsx";

// Each page after the first is its own chunk, fetched when the page is opened.
const Results = lazy(() => import("./pages/Results.jsx"));
const Explorer = lazy(() => import("./pages/Explorer.jsx"));
const About = lazy(() => import("./pages/About.jsx"));

const PAGES = [
  { key: "overview", label: "Overview", Page: Overview },
  { key: "results", label: "Results", Page: Results },
  { key: "explorer", label: "Explorer", Page: Explorer },
  { key: "about", label: "About", Page: About },
];

export default function App() {
  const route = useHashRoute(
    PAGES.map((p) => p.key),
    "overview",
  );
  const { Page } = PAGES.find((p) => p.key === route);
  return (
    <>
      <header className="header">
        <div className="header-inner">
          <a className="brand" href="#/overview">
            <span className="brand-dots" aria-hidden="true">
              <i style={{ background: "var(--clear)" }} />
              <i style={{ background: "var(--ambiguous)" }} />
              <i style={{ background: "var(--unreliable)" }} />
            </span>
            WhoToTrust
          </a>
          <nav className="nav" aria-label="Pages">
            {PAGES.map((p) => (
              <a key={p.key} href={`#/${p.key}`} aria-current={p.key === route ? "page" : undefined}>
                {p.label}
              </a>
            ))}
          </nav>
        </div>
      </header>
      <main className="page" id="main">
        <Suspense fallback={<CardSkeleton height={220} />}>
          <Page />
        </Suspense>
      </main>
      <footer className="footer">
        WhoToTrust, laptop edition. Results are from validation data only. MIT licensed.
      </footer>
    </>
  );
}
