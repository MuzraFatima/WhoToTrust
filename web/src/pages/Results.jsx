import { useMemo, useState } from "react";
import { getJSON } from "../lib/api.js";
import { useAsync } from "../lib/hooks.js";
import {
  DATASETS,
  ERROR_LABELS,
  ERROR_TYPES,
  METHODS,
  METHOD_COLOURS,
  METHOD_LABELS,
  num,
  signed,
} from "../lib/format.js";
import { compareByType, findMisses, gains } from "../lib/summary.js";
import { CardSkeleton } from "../components/Skeleton.jsx";
import { GroupedBars } from "../components/GroupedBars.jsx";
import { EmptyRuns, ErrorNotice, Legend, Notice, PassChip } from "../components/ui.jsx";

const GROUPS = ["g1", "g2", "g3", "g4", "g5", "all"];
const GROUP_NAMES = {
  g1: "G1 votes",
  g2: "G2 annotators",
  g3: "G3 plausibility",
  g4: "G4 neighbours",
  g5: "G5 dynamics",
  all: "All together",
};
const VERDICT_TEXT = {
  ahead: "ahead",
  level: "level (within seed noise)",
  behind: "behind",
};

const heat = (v) =>
  v == null
    ? undefined
    : { background: `color-mix(in srgb, var(--accent) ${Math.round(Math.max(0, v - 0.5) * 80)}%, transparent)` };

function Section({ title, note, children }) {
  return (
    <section className="card">
      <h2>{title}</h2>
      {note && <p className="muted">{note}</p>}
      {children}
    </section>
  );
}

function Comparison({ compare, dataset }) {
  const byType = useMemo(() => compareByType(compare, dataset), [compare, dataset]);
  const rows = useMemo(() => gains(compare, dataset), [compare, dataset]);
  const types = ERROR_TYPES.filter((t) => byType[t]);
  const values = types.flatMap((t) => METHODS.map((m) => byType[t][m]?.val_acc).filter((v) => v != null));
  if (!values.length) return <p className="muted">No comparison runs for this dataset.</p>;
  const min = Math.max(0, Math.floor((Math.min(...values) - 0.03) * 20) / 20);
  const groups = types.map((t) => ({
    label: ERROR_LABELS[t],
    values: Object.fromEntries(METHODS.map((m) => [m, byType[t][m]?.val_acc])),
  }));
  const series = METHODS.map((m) => ({ key: m, label: METHOD_LABELS[m], colour: METHOD_COLOURS[m] }));
  return (
    <>
      <GroupedBars
        groups={groups}
        series={series}
        min={min}
        max={1}
        title={`Validation accuracy by error type and training method, ${DATASETS[dataset]}`}
      />
      <Legend items={series} />
      <p className="muted small">
        Validation accuracy, mean of 3 seeds. The vertical axis starts at {num(min, 2)} so small gaps
        are readable; read the table for the exact differences.
      </p>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th scope="col">Error type</th>
              <th scope="col" className="num">LabelTriage</th>
              <th scope="col">Best simple rule</th>
              <th scope="col" className="num">Gap</th>
              <th scope="col">Verdict</th>
              <th scope="col" className="num">KL to human votes</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const m = byType[r.type];
              return (
                <tr key={r.type}>
                  <td>{ERROR_LABELS[r.type]}</td>
                  <td className="num">
                    {num(m.labeltriage.val_acc)} ± {num(m.labeltriage.val_acc_std, 3)}
                  </td>
                  <td>
                    {METHOD_LABELS[r.against]} {num(m[r.against].val_acc)}
                  </td>
                  <td className="num">{signed(r.gain)}</td>
                  <td>{VERDICT_TEXT[r.verdict]}</td>
                  <td className="num">
                    {num(m.labeltriage.val_kl)} vs {num(Math.min(...["majority_vote", "soft_labels"].map((k) => m[k]?.val_kl ?? Infinity)))}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </>
  );
}

function Detection({ triage, features, dataset }) {
  const t = triage.filter((r) => r.dataset === dataset);
  const f = features.filter((r) => r.dataset === dataset);
  return (
    <>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th scope="col">Error type</th>
              <th scope="col" className="num">LabelTriage AUROC</th>
              <th scope="col" className="num">Minority share only</th>
              <th scope="col" className="num">Calibration error</th>
              <th scope="col">AUROC gate (≥ 0.80)</th>
              <th scope="col">Calibration gate</th>
            </tr>
          </thead>
          <tbody>
            {t.map((r) => (
              <tr key={r.error_type}>
                <td>{ERROR_LABELS[r.error_type] ?? r.error_type}</td>
                <td className="num">{num(r.auroc)}</td>
                <td className="num">{num(r.minority_share)}</td>
                <td className="num">{num(r.calibration_error)}</td>
                <td>
                  <PassChip pass={r.auroc_pass} />
                </td>
                <td>
                  <PassChip pass={r.calibration_pass} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <h3 style={{ marginTop: "1.2rem" }}>Each kind of evidence on its own (AUROC)</h3>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th scope="col">Error type</th>
              {GROUPS.map((g) => (
                <th scope="col" className="num" key={g}>
                  {GROUP_NAMES[g]}
                </th>
              ))}
              <th scope="col" className="num">Shuffled labels</th>
            </tr>
          </thead>
          <tbody>
            {f.map((r) => (
              <tr key={r.error_type}>
                <td>{ERROR_LABELS[r.error_type] ?? r.error_type}</td>
                {GROUPS.map((g) => (
                  <td className="num" key={g} style={heat(r[g])}>
                    {num(r[g])}
                  </td>
                ))}
                <td className="num">{num(r.leakage_shuffled)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="muted small">
        Shuffled labels should sit near 0.5; they do, so the features are not leaking the answer. A
        group that stays under 0.60 on its own misses the Phase 3 bar.
      </p>
    </>
  );
}

export default function Results() {
  const [dataset, setDataset] = useState("cifar10h");
  const { data, error, loading } = useAsync(() => getJSON("/api/results"), []);
  if (loading)
    return (
      <div className="stack">
        <CardSkeleton height={260} />
        <CardSkeleton />
      </div>
    );
  if (error) return <ErrorNotice error={error} />;
  if (!data.n_runs) return <EmptyRuns />;
  const s = data.sections;
  const transfer = s.transfer?.[0];
  const misses = findMisses(s);
  return (
    <div className="stack">
      <div className="controls">
        <div className="segmented" role="group" aria-label="Dataset">
          {Object.entries(DATASETS).map(([key, name]) => (
            <button key={key} aria-pressed={dataset === key} onClick={() => setDataset(key)}>
              {name}
            </button>
          ))}
        </div>
      </div>

      <Section
        title="Does triage help the classifier?"
        note="Four training recipes on the same noisy votes: majority vote, soft labels, LabelTriage, and an oracle that knows which votes are wrong (the ceiling)."
      >
        <Comparison compare={s.compare} dataset={dataset} />
      </Section>

      <Section
        title="Can it spot the errors?"
        note="How well each vote's error score separates injected mistakes from honest votes, on held-out validation votes."
      >
        <Detection triage={s.triage ?? []} features={s.features ?? []} dataset={dataset} />
      </Section>

      <div className="grid cols-2">
        <Section
          title="Does it transfer to real errors?"
          note="Trained on injected errors only, scored on VariErr, where real annotators flagged real mistakes (dev half)."
        >
          {transfer ? (
            <>
              <p style={{ fontSize: "1.6rem", fontWeight: 750, margin: "0.2rem 0" }}>
                {num(transfer.auroc)} <span className="muted small">AUROC</span>
              </p>
              <p>
                Just counting minority votes scores {num(transfer.minority_share)}. The gap is{" "}
                {signed(transfer.auroc - transfer.minority_share)}, 95% interval{" "}
                [{signed(transfer.gain_lo)}, {signed(transfer.gain_hi)}]: it includes zero, so this is
                level with vote counting, not better.
              </p>
            </>
          ) : (
            <p className="muted">No transfer run yet.</p>
          )}
        </Section>
        <Section title="Starting point" note="Majority vote against soft labels, with nothing else added.">
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th scope="col">Dataset</th>
                  <th scope="col">Method</th>
                  <th scope="col" className="num">Val accuracy</th>
                  <th scope="col" className="num">KL</th>
                </tr>
              </thead>
              <tbody>
                {(s.reproduce ?? []).map((r) => (
                  <tr key={`${r.dataset}-${r.method}`}>
                    <td>{DATASETS[r.dataset] ?? r.dataset}</td>
                    <td>{METHOD_LABELS[r.method] ?? r.method}</td>
                    <td className="num">{num(r.val_acc)}</td>
                    <td className="num">{num(r.val_kl)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      </div>

      {misses.length > 0 && (
        <Notice>
          <strong>Gates this run misses, shown rather than hidden:</strong>
          <ul style={{ margin: "0.3rem 0 0", paddingLeft: "1.1rem" }}>
            {misses.map((m) => (
              <li key={m}>{m}</li>
            ))}
          </ul>
        </Notice>
      )}
    </div>
  );
}
