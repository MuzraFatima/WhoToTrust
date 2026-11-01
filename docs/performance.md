# Performance checklist

This project runs on one laptop and is not deployed, so items that only make sense for a hosted
service are marked **skipped**, with the reason. Everything else was done and measured.

| Item | Status | What was done |
|---|---|---|
| Cache API responses | done | Every `/api/*` answer carries an `ETag`; a repeat request gets a `304` with no body. The browser side keeps answers in memory and shares requests in flight (`web/src/lib/api.js`). Built assets are served with `Cache-Control: immutable`. |
| Load balancer | skipped | One local process. Nothing to balance. |
| Index the database | skipped | There is no database. Runs are read from Parquet files, and each file is read once per version. |
| Compress images | done | The README screenshots were palette-compressed, 870 KB to 345 KB. Item pictures are 32 px CIFAR images served as 128 px optimised PNGs of about 3.5 KB, cached for a day. |
| Loading skeletons | done | Every section shows a shimmer block while its data loads (`Skeleton.jsx`); `prefers-reduced-motion` turns the animation off. |
| Cache expensive queries | done | Parquet reads, the per-run item tables and the results summary are cached per file version, so they recompute only when a run changes (`src/labeltriage/api.py`). |
| N+1 database queries | not applicable | No database. The item list is one request per page, and the detail panel is one request per click. |
| Debounce input handlers | done | The item search waits 300 ms after the last keystroke (`useDebounced`, unit tested). |
| Split code into chunks | done | Each page after the first is its own chunk: the first load is 154 KB of JavaScript (50 KB gzipped); Results is 8 KB, Explorer 10 KB, About 2 KB. |
| Add CDN | skipped | Not deployed; assets come from the local server. |
| Server-side caching | done | Same caches as "expensive queries", plus a background warm-up of the CIFAR images at server start so the first picture is not slow. |
| Paginate large lists | done | The item list is paged on the server (10 to 100 per page, at most 100 enforced). One run has 800 to 1,000 items; a page is 4 KB. |
| Lighthouse audit | done | See below. |
| Compress API payloads | done | gzip: the scatter data for 800 items goes from 26.6 KB to 6.9 KB, the results summary from 10.9 KB to 2.5 KB, a page of items from 4.1 KB to 0.6 KB. |
| Unnecessary re-renders | done | The scatter plot is split so that hovering updates only the tooltip, never the 1,000 dots (`Dots` is memoised, handlers are stable). The charts are memoised and their inputs are `useMemo`ed. |
| Minify JS and CSS | done | Vite production build (esbuild minification, content-hashed file names). |
| Add lazy loading | done | Pages load on demand (`React.lazy`), item pictures use `loading="lazy"`. |
| Defer non-critical scripts | done | One module script, which browsers defer by default; nothing else blocks rendering. |
| Unused dependencies | done | `depcheck` reports none for the web app. Streamlit and Plotly were removed together with the old app. Python dependencies were checked against imports. |
| Database connection pooling | skipped | No database. |

## Lighthouse

Run with Google Chrome (headless) against the production build, using Lighthouse's default
mobile throttling.

| Page | Performance | Accessibility | Best practices | SEO | LCP | Total blocking time | Layout shift |
|---|---|---|---|---|---|---|---|
| Overview | 100 | 100 | 100 | 100 | 1.4 s | 10 ms | 0 |
| Results | 99 | 100 | 100 | 100 | 1.7 s | 10 ms | 0 |
| Explorer | 99 | 100 | 100 | 100 | 1.9 s | 80 ms | 0 |

The first audit found three things, all fixed: heat-map cells with too little text contrast
(accent mix capped at 40%), a layout shift of 0.12 from the footer moving when content arrived
(the page now reserves its height), and table headers without `scope="col"`.

One note remains. Explorer draws one SVG circle per item, which is 1,261 DOM elements, over
Lighthouse's 800-element advisory line. It scores well anyway, and the table below the plot is
the keyboard and screen-reader route to every item.

## Re-running the checks

```bash
cd web
npm ci
npm run build
npm run lint      # ESLint
npm test          # 24 unit tests
npm run e2e       # 7 end-to-end tests in real Google Chrome, against the real server
```

`npm run e2e` needs Google Chrome installed (it drives that browser, nothing is downloaded) and
the mini results from `python scripts/run_mini.py`.
