import { memo } from "react";

const W = 720;
const H = 300;
const M = { top: 14, right: 12, bottom: 44, left: 44 };

/** Bars for each group (an error type) and series (a method), drawn as plain SVG. */
function GroupedBarsImpl({ groups, series, min = 0, max = 1, ticks = 5, title }) {
  const innerW = W - M.left - M.right;
  const innerH = H - M.top - M.bottom;
  const y = (v) => M.top + innerH * (1 - (v - min) / (max - min));
  const groupW = innerW / groups.length;
  const barW = Math.min(34, (groupW * 0.78) / series.length);
  const tickValues = Array.from({ length: ticks + 1 }, (_, i) => min + ((max - min) * i) / ticks);

  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={title} width="100%">
      <title>{title}</title>
      {tickValues.map((t) => (
        <g key={t}>
          <line className="grid-line" x1={M.left} x2={W - M.right} y1={y(t)} y2={y(t)} />
          <text x={M.left - 8} y={y(t) + 4} textAnchor="end">
            {t.toFixed(2)}
          </text>
        </g>
      ))}
      {groups.map((g, gi) => {
        const x0 = M.left + gi * groupW + (groupW - barW * series.length) / 2;
        return (
          <g key={g.label}>
            {series.map((s, si) => {
              const v = g.values[s.key];
              if (v == null) return null;
              const top = y(Math.min(max, Math.max(min, v)));
              return (
                <rect
                  key={s.key}
                  x={x0 + si * barW}
                  y={top}
                  width={barW - 3}
                  height={Math.max(0, y(min) - top)}
                  rx="3"
                  fill={s.colour}
                >
                  <title>{`${g.label}, ${s.label}: ${v.toFixed(3)}`}</title>
                </rect>
              );
            })}
            <text x={M.left + gi * groupW + groupW / 2} y={H - 18} textAnchor="middle">
              {g.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export const GroupedBars = memo(GroupedBarsImpl);
