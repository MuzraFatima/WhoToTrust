import { useMemo } from "react";
import { getJSON } from "../lib/api.js";
import { useAsync } from "../lib/hooks.js";
import { num, pct, signed } from "../lib/format.js";
import { summarise } from "../lib/summary.js";
import { CardSkeleton } from "../components/Skeleton.jsx";
import { EmptyRuns, ErrorNotice, Notice, Stat } from "../components/ui.jsx";

const STEPS = [
  {
    title: "1. AmbiNoise",
    body: "Take real many-annotator votes, sample a few per item, and inject labelled mistakes on top. Now every vote is known to be an error or a genuine opinion.",
  },
  {
    title: "2. LabelTriage",
    body: "Score each vote's chance of being an error from five kinds of evidence, then route each item to a hard, soft or down-weighted training target.",
  },
  {
    title: "3. Harm analysis",
    body: "Retrain with each kind of disagreement removed or kept, to see which one actually hurts accuracy and which one hurts calibration.",
  },
];

const points = (x) => (x == null ? "–" : `${signed(x * 100, 1)} pts`);

function Findings({ results }) {
  const s = useMemo(() => summarise(results), [results]);
  const { images, text, transfer } = s;
  return (
    <div className="grid cols-4">
      <Stat
        title="Pipeline"
        value={`${s.runs} runs`}
        caption={`all finished, ${s.minutes.toFixed(0)} min of training on a laptop CPU`}
      />
      <Stat
        title="Images"
        value={points(images.mean)}
        caption={`LabelTriage vs the better simple rule, averaged over 4 error types (ahead in ${images.ahead}, behind in ${images.behind})`}
      />
      <Stat
        title="Text"
        value={points(text.mean)}
        caption={`same comparison on ChaosNLI (ahead in ${text.ahead}, behind in ${text.behind}; the rest are level)`}
      />
      <Stat
        title="Real errors (VariErr)"
        value={num(transfer?.auroc)}
        caption={`AUROC, against ${num(transfer?.minority_share)} for just counting minority votes. Level, not better.`}
      />
    </div>
  );
}

export default function Overview() {
  const { data, error, loading } = useAsync(() => getJSON("/api/results"), []);
  return (
    <div className="stack">
      <section className="hero">
        <div className="eyebrow">When labels disagree</div>
        <h1>Is that disagreement a mistake, or a real difference of opinion?</h1>
        <p>
          When several people label the same item and disagree, most pipelines either delete the
          minority votes or keep them all. LabelTriage decides, vote by vote, which is which, and
          trains on each differently.
        </p>
        <p style={{ display: "flex", gap: "0.6rem", flexWrap: "wrap" }}>
          <a className="btn" href="#/results">
            See the results
          </a>
          <a className="btn" href="#/explorer">
            Explore every vote
          </a>
        </p>
      </section>

      {loading && (
        <div className="grid cols-4">
          {[0, 1, 2, 3].map((i) => (
            <CardSkeleton key={i} lines={2} height={50} />
          ))}
        </div>
      )}
      {error && <ErrorNotice error={error} />}
      {data && data.n_runs === 0 && <EmptyRuns />}
      {data && data.n_runs > 0 && <Findings results={data} />}

      <section>
        <h2>How it works</h2>
        <div className="grid cols-3">
          {STEPS.map((step) => (
            <div className="card" key={step.title}>
              <h3>{step.title}</h3>
              <p className="muted">{step.body}</p>
            </div>
          ))}
        </div>
      </section>

      <Notice>
        <strong>Read these numbers as direction, not as the final study.</strong> This is the laptop
        edition: a fixed subset of each dataset ({pct(1500 / 10000, 0)} of the images, about 46% of
        the text items), a small classifier on frozen embeddings, scored on validation data only.
        The text encoder was already trained on inference tasks, which lifts every text number
        alike.
      </Notice>
    </div>
  );
}
