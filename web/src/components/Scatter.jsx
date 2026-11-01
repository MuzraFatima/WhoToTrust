import { memo, useCallback, useMemo, useState } from "react";
import { ROUTE_NOTES } from "../lib/format.js";

const W = 760;
const H = 420;
const M = { top: 16, right: 16, bottom: 46, left: 52 };
const TICKS = [0, 0.25, 0.5, 0.75, 1];

// Spread equal entropies sideways a little (the same way every render) so dots do not hide.
const jitter = (i) => ((Math.sin(i * 12.9898) * 43758.5453) % 1) * 0.018;

/** The dots only. Memoised, so hovering (which changes the tooltip) never redraws them. */
const Dots = memo(function Dots({ xy, route, colours, onPick, onHover }) {
  const index = (e) => (e.target.dataset.i === undefined ? null : Number(e.target.dataset.i));
  return (
    <g
      onClick={(e) => index(e) !== null && onPick(index(e))}
      onMouseMove={(e) => onHover(index(e), e)}
      onMouseLeave={() => onHover(null)}
    >
      {xy.map(([cx, cy], i) => (
        <circle
          key={i}
          data-i={i}
          cx={cx}
          cy={cy}
          r="3.6"
          fill={colours[route[i]]}
          fillOpacity="0.72"
          style={{ cursor: "pointer" }}
        />
      ))}
    </g>
  );
});

function ScatterImpl({ points, selected, onSelect }) {
  const [hover, setHover] = useState(null);
  const { routes, entropy, max_p_error: score, route, item_id: ids } = points;

  const { xy, xMax, colours } = useMemo(() => {
    const xMax = Math.max(0.1, ...entropy) * 1.04;
    const x = (v) => M.left + ((v + 0.01) / (xMax + 0.02)) * (W - M.left - M.right);
    const y = (v) => M.top + (1 - v) * (H - M.top - M.bottom);
    return {
      xMax,
      xy: entropy.map((e, i) => [x(e + jitter(i)), y(score[i])]),
      colours: routes.map((r) => `var(--${r})`),
    };
  }, [entropy, score, routes]);

  const onPick = useCallback((i) => onSelect(ids[i]), [ids, onSelect]);
  const onHover = useCallback((i, e) => {
    if (i === null) return setHover(null);
    const box = e.currentTarget.ownerSVGElement.getBoundingClientRect();
    setHover({ i, left: e.clientX - box.left, top: e.clientY - box.top });
  }, []);

  const picked = ids.indexOf(selected);
  return (
    <div className="scatter-wrap">
      <svg
        className="chart"
        viewBox={`0 0 ${W} ${H}`}
        width="100%"
        role="img"
        aria-label="Each dot is an item. Vote entropy runs left to right, its most suspicious vote bottom to top."
      >
        {TICKS.map((t) => {
          const yy = M.top + (1 - t) * (H - M.top - M.bottom);
          return (
            <g key={t}>
              <line className="grid-line" x1={M.left} x2={W - M.right} y1={yy} y2={yy} />
              <text x={M.left - 8} y={yy + 4} textAnchor="end">
                {t.toFixed(2)}
              </text>
            </g>
          );
        })}
        {TICKS.map((t) => (
          <text
            key={t}
            x={M.left + t * (W - M.left - M.right)}
            y={H - 26}
            textAnchor="middle"
          >
            {(t * xMax).toFixed(2)}
          </text>
        ))}
        <text x={(W + M.left) / 2} y={H - 6} textAnchor="middle">
          vote entropy (nats): further right means the votes split more
        </text>
        <text transform={`translate(13 ${H / 2}) rotate(-90)`} textAnchor="middle">
          highest p(error) among its votes
        </text>
        <Dots xy={xy} route={route} colours={colours} onPick={onPick} onHover={onHover} />
        {picked >= 0 && (
          <circle
            cx={xy[picked][0]}
            cy={xy[picked][1]}
            r="8"
            fill="none"
            stroke="var(--text)"
            strokeWidth="2.2"
            pointerEvents="none"
          />
        )}
      </svg>
      {hover && (
        <div className="tooltip" style={{ left: hover.left, top: hover.top }}>
          item {ids[hover.i]}: {routes[route[hover.i]]}, p(error) {score[hover.i].toFixed(2)}
        </div>
      )}
      <div className="legend" style={{ marginTop: 4 }}>
        {routes.map((r) => (
          <span key={r} title={ROUTE_NOTES[r]}>
            <i style={{ background: `var(--${r})`, borderRadius: "50%" }} />
            {r}
          </span>
        ))}
      </div>
    </div>
  );
}

export const Scatter = memo(ScatterImpl);
