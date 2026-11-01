import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App.jsx";
import { clearApiCache } from "./lib/api.js";

const results = {
  n_runs: 4,
  train_seconds_total: 600,
  sections: {
    compare: [
      { dataset: "cifar10h", error_type: "lazy", method: "majority_vote", val_acc: 0.9, val_acc_std: 0.01, val_kl: 0.3 },
      { dataset: "cifar10h", error_type: "lazy", method: "soft_labels", val_acc: 0.8, val_acc_std: 0.01, val_kl: 0.4 },
      { dataset: "cifar10h", error_type: "lazy", method: "labeltriage", val_acc: 0.95, val_acc_std: 0.01, val_kl: 0.2 },
    ],
    triage: [{ dataset: "cifar10h", error_type: "lazy", auroc: 0.99, minority_share: 0.9, calibration_error: 0.02, auroc_pass: true, calibration_pass: true }],
    features: [],
    reproduce: [],
    transfer: [{ auroc: 0.83, minority_share: 0.86, gain_lo: -0.06, gain_hi: 0.01 }],
  },
};

function serve(routes) {
  vi.stubGlobal(
    "fetch",
    vi.fn((url) => {
      const key = Object.keys(routes).find((k) => url.startsWith(k));
      const body = key ? routes[key] : { error: "unexpected " + url };
      return Promise.resolve({ ok: Boolean(key), status: key ? 200 : 404, statusText: "", json: () => Promise.resolve(body) });
    }),
  );
}

beforeEach(() => {
  clearApiCache();
  window.location.hash = "";
});
afterEach(() => vi.unstubAllGlobals());

describe("App", () => {
  it("shows the headline findings on the overview", async () => {
    serve({ "/api/results": results });
    render(<App />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(/mistake, or a real difference/i);
    expect(await screen.findByText("4 runs")).toBeInTheDocument();
    expect(screen.getByText("+5.0 pts")).toBeInTheDocument();
    expect(screen.getByText("0.830")).toBeInTheDocument();
  });

  it("explains how to produce results when there are none", async () => {
    serve({ "/api/results": { n_runs: 0, sections: {} } });
    render(<App />);
    expect(await screen.findByText("No results yet")).toBeInTheDocument();
    expect(screen.getByText(/run_mini.py/)).toBeInTheDocument();
  });

  it("shows a readable message when the server is down", async () => {
    serve({});
    render(<App />);
    expect(await screen.findByRole("alert")).toHaveTextContent(/Could not load this/);
  });

  it("opens the results page from the nav and loads its chunk lazily", async () => {
    serve({ "/api/results": results });
    render(<App />);
    await userEvent.click(screen.getByRole("link", { name: "Results" }));
    expect(await screen.findByText("Does triage help the classifier?")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Results" })).toHaveAttribute("aria-current", "page");
    await waitFor(() => expect(screen.getAllByText("ahead").length).toBeGreaterThan(0));
  });

  it("shows the about page", async () => {
    window.location.hash = "#/about";
    render(<App />);
    expect(await screen.findByText("The three routes")).toBeInTheDocument();
  });
});
