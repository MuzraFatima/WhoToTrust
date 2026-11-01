import { ROUTE_NOTES } from "../lib/format.js";
import { RouteChip } from "../components/ui.jsx";

const FEATURES = [
  ["G1 Votes", "How the item's votes split, and how far this vote sits from the crowd."],
  ["G2 Annotators", "How reliable this annotator looks across every item they labelled."],
  ["G3 Plausibility", "Whether a frozen model finds this label believable for the item."],
  ["G4 Neighbours", "What labels similar items received."],
  ["G5 Dynamics", "Whether a small probe model ever managed to learn this label."],
];

export default function About() {
  return (
    <div className="stack">
      <section className="hero" style={{ paddingBottom: 0 }}>
        <div className="eyebrow">About</div>
        <h1>One problem, two camps, one question</h1>
        <p>
          Noisy-label methods treat disagreement as damage and delete it. Label-variation methods
          treat it as signal and keep it. Both are right sometimes. LabelTriage asks which one a
          given vote is.
        </p>
      </section>

      <section>
        <h2>The three routes</h2>
        <div className="grid cols-3">
          {["clear", "ambiguous", "unreliable"].map((r) => (
            <div className="card" key={r}>
              <RouteChip route={r} />
              <p style={{ marginTop: "0.6rem" }}>{ROUTE_NOTES[r]}</p>
            </div>
          ))}
        </div>
      </section>

      <section>
        <h2>Five kinds of evidence</h2>
        <div className="grid cols-3">
          {FEATURES.map(([name, body]) => (
            <div className="card" key={name}>
              <h3>{name}</h3>
              <p className="muted">{body}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="card">
        <h2>What this edition is, and is not</h2>
        <ul>
          <li>
            It runs end to end on a laptop CPU, using only the data in this folder. Every number on
            these pages comes from real runs you can repeat.
          </li>
          <li>
            It uses a fixed subset of each dataset and a small classifier on frozen embeddings. The
            blueprint&apos;s full-scale, pre-registered study (fine-tuned ResNet and DeBERTa, GPU) was
            not run, so these results are indicative, not confirmatory.
          </li>
          <li>
            Scores are on validation data only; the sealed test splits were never read.
          </li>
          <li>
            Misses stay visible: the lazy-error gate for text, and the VariErr transfer, which is
            level with simply counting minority votes.
          </li>
        </ul>
      </section>
    </div>
  );
}
