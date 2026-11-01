"""Turn Phase 6 / 7 run summaries into verdicts on the pre-registered hypotheses H1–H4.

Inputs are the per-run tables from :func:`labeltriage.eval.summary.collect`. Thresholds come from
``conf/prereg/hypotheses.yaml``. Every verdict reports its numbers, so a reader can check it.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from labeltriage.eval.stats import holm, paired_wilcoxon

SETTING = re.compile(r"ambinoise_(?P<base>\w+?)_k(?P<k>\d+)_r(?P<rate>[\d.]+)_(?P<etype>\w+)$")
GLOBAL_RULES = ("majority_vote", "soft_labels")


def with_settings(runs: pd.DataFrame) -> pd.DataFrame:
    parsed = runs["data"].str.extract(SETTING)
    return runs.assign(
        base=parsed["base"],
        k=pd.to_numeric(parsed["k"]),
        rate=pd.to_numeric(parsed["rate"]),
        etype=parsed["etype"],
    )


def _seed_bootstrap(gains: np.ndarray, n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(gains, size=(n_boot, len(gains)), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def h2(runs: pd.DataFrame, hyp: dict, metric: str = "clean_test/acc") -> dict:
    """LabelTriage vs the better global rule, per setting (A6: chosen on test, per setting)."""
    r = with_settings(runs)
    r = r[(r["base"] == "10h") & (r["k"] == hyp["H2"]["main_setting"]["k"])]
    rows, main_gains = [], []
    for (etype, rate), g in r.groupby(["etype", "rate"]):
        means = g.groupby("method")[metric].mean()
        if "labeltriage" not in means or not set(GLOBAL_RULES) <= set(means.index):
            continue
        best = means[list(GLOBAL_RULES)].idxmax()
        paired = g.pivot_table(index="seed", columns="method", values=metric)[
            ["labeltriage", best]
        ].dropna()
        gains = (paired["labeltriage"] - paired[best]).to_numpy()
        lo, hi = _seed_bootstrap(gains)
        rows.append(
            {
                "etype": etype,
                "rate": rate,
                "best_global": best,
                "gain": gains.mean(),
                "lo": lo,
                "hi": hi,
            }
        )
        if rate == hyp["H2"]["main_setting"]["rate"]:
            main_gains.extend(gains)
    table = pd.DataFrame(rows)
    if table.empty:
        return {"table": table, "verdict": "no data"}

    main = np.array(main_gains)
    main_lo, main_hi = _seed_bootstrap(main)
    p = paired_wilcoxon(table["gain"].to_numpy(), np.zeros(len(table)))
    need = hyp["H2"]["gain_over_best_global_min_points"] / 100
    floor = hyp["H2"]["never_worse_than_points"] / 100
    falsified = main_lo <= 0 <= main_hi
    holds = (main.mean() >= need) and (table["gain"].min() >= floor) and not falsified
    return {
        "table": table,
        "main_gain": float(main.mean()),
        "main_ci": (main_lo, main_hi),
        "worst_setting_gain": float(table["gain"].min()),
        "p": p,
        "verdict": "holds" if holds else ("falsified" if falsified else "not supported"),
    }


def h4(runs: pd.DataFrame, hyp: dict, metric: str = "human_test/kl") -> dict:
    """KL to human votes: LabelTriage close to soft labels and well below majority vote."""
    r = with_settings(runs)
    r = r[r["base"] == "10h"]
    piv = r.pivot_table(
        index=["etype", "rate", "k", "seed"], columns="method", values=metric
    ).dropna(subset=["labeltriage", "soft_labels", "majority_vote"])
    if piv.empty:
        return {"verdict": "no data"}
    lt, sl, mv = (piv[m].mean() for m in ("labeltriage", "soft_labels", "majority_vote"))
    rel_soft = (lt - sl) / sl
    rel_mv = (mv - lt) / mv
    p = paired_wilcoxon(piv["majority_vote"].to_numpy(), piv["labeltriage"].to_numpy())
    holds = (
        rel_soft <= hyp["H4"]["within_soft_label_relative"]
        and rel_mv >= hyp["H4"]["below_majority_vote_relative"]
        and p < 0.05
    )
    return {
        "kl": {"labeltriage": lt, "soft_labels": sl, "majority_vote": mv},
        "relative_to_soft": float(rel_soft),
        "reduction_vs_majority": float(rel_mv),
        "p": p,
        "verdict": "holds" if holds else ("falsified" if p >= 0.05 else "not supported"),
    }


def h1(runs: pd.DataFrame, transfer: dict | None, hyp: dict) -> dict:
    """Error detection on AmbiNoise (val AUROC) and transfer to the VariErr test half."""
    lt = runs[runs["method"] == "labeltriage"].dropna(subset=["triage/val_auroc"])
    if lt.empty:
        return {"verdict": "no data"}
    auc = lt["triage/val_auroc"].to_numpy()
    gain = auc - lt["triage/val_entropy_only"].to_numpy()
    out = {
        "ambinoise_auroc": float(auc.mean()),
        "gain_over_entropy": float(gain.mean()),
        "p": paired_wilcoxon(auc, lt["triage/val_entropy_only"].to_numpy()),
    }
    h = hyp["H1"]
    ok = (
        out["ambinoise_auroc"] >= h["ambinoise_min"]
        and out["gain_over_entropy"] >= h["gain_over_entropy_min"]
    )
    falsified = out["gain_over_entropy"] < h["falsified_if"]["gain_over_entropy_below"]
    if transfer is not None:
        out["varierr_auroc"] = transfer["transfer/test_auroc"]
        out["varierr_gain_over_minority_ci"] = (
            transfer["transfer/test_gain_over_minority_lo"],
            transfer["transfer/test_gain_over_minority_hi"],
        )
        ok = ok and out["varierr_auroc"] >= h["varierr_min"]
        ok = ok and out["varierr_gain_over_minority_ci"][0] > h["gain_over_minority_share_ci_above"]
        falsified = falsified or out["varierr_auroc"] < h["falsified_if"]["varierr_below"]
    out["verdict"] = "holds" if ok else ("falsified" if falsified else "not supported")
    return out


def h3(harm: pd.DataFrame, hyp: dict) -> dict:
    """Harm counterfactuals (Phase 7): what removing errors or collapsing ambiguity changes."""
    need = {"neither", "errors_removed", "ambiguity_collapsed"}
    if harm.empty or not need <= set(harm["method"]):
        return {"verdict": "no data"}
    piv = harm.pivot_table(
        index=["data", "seed"], columns="method", values=["clean_test/acc", "clean_test/ece"]
    )
    acc, ece = piv["clean_test/acc"], piv["clean_test/ece"]
    d_acc_err = (acc["errors_removed"] - acc["neither"]).to_numpy()
    rel_ece_amb = ((ece["ambiguity_collapsed"] - ece["neither"]) / ece["neither"]).to_numpy()
    p = paired_wilcoxon(ece["ambiguity_collapsed"].to_numpy(), ece["neither"].to_numpy())
    holds = d_acc_err.mean() > 0 and rel_ece_amb.mean() >= 0.20 and p < 0.05
    falsified = abs(rel_ece_amb.mean()) < 0.05
    return {
        "acc_gain_removing_errors": float(d_acc_err.mean()),
        "ece_change_collapsing_ambiguity": float(rel_ece_amb.mean()),
        "p": p,
        "verdict": "holds" if holds else ("falsified" if falsified else "not supported"),
    }


def holm_family(results: dict[str, dict]) -> dict[str, bool]:
    ps = {k: v["p"] for k, v in results.items() if "p" in v and not np.isnan(v["p"])}
    return holm(ps) if ps else {}
