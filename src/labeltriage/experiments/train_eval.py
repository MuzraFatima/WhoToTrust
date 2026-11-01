"""Train one classifier on one method's targets and evaluate it on the configured sets.

This is the unit every grid is made of: a (dataset, method, seed) triple in, a metrics dict out.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from labeltriage import targets
from labeltriage.data import bundles
from labeltriage.eval import metrics
from labeltriage.tasks import task
from labeltriage.utils.seed import seed_everything

Predictor = Callable[[Any], np.ndarray]


def load_bundle(cfg: DictConfig) -> bundles.Bundle:
    """The base dataset's bundle, with an AmbiNoise draw applied when ``data.noise`` is set."""
    bundle = bundles.load(cfg.data.get("base", cfg.data.name), cfg.paths.data_dir)
    noise = cfg.data.get("noise")
    if noise is None:
        return bundle

    from labeltriage.ambinoise.injector import NoiseSpec

    spec = NoiseSpec(
        k=noise.k,
        rate=noise.rate,
        error_type=noise.error_type,
        num_annotators=noise.num_annotators,
        concentration=noise.concentration,
        native_cutoff=noise.native_cutoff,
        seed=noise.seed,
    )
    ranking = None
    if spec.error_type == "plausible_wrong":
        ranking = pd.read_parquet(noise.ranking_path)
        ranking.columns = ranking.columns.astype(int)
    return bundles.ambinoise(bundle, spec, ranking)


def _triage_targets(cfg: DictConfig, bundle: bundles.Bundle):
    """LabelTriage: features, out-of-fold p(error) per train vote, then the router."""
    from labeltriage.experiments.features import build_or_load, feature_columns
    from labeltriage.triage.pipeline import score

    m = cfg.method
    table = build_or_load(cfg, bundle.votes)
    train = (table["split"] == "train").to_numpy()

    if "is_error" not in table:
        # Real noise without error labels (CIFAR-10N): train triage on AmbiNoise, apply here.
        from labeltriage.experiments.triage import source_model

        groups = tuple(m.transfer_groups)
        cols = [c for c in feature_columns(table) if c.split("_")[0] in groups]
        p_error = source_model(cfg, cols).predict(table[train])
        metrics = {"triage/mean_p_error": float(p_error.mean())}
    else:
        groups = tuple(m.get("groups", ("g1", "g2", "g3", "g4", "g5")))
        cols = [c for c in feature_columns(table) if c.split("_")[0] in groups]
        res = score(table, cols, kind=m.get("model", "lgbm"), seed=cfg.seed)
        p_error, metrics = res.p_error[train], res.metrics

    ts = targets.route(table[train], bundle.num_classes, p_error, m.nu, m.h0, m.get("hard_drop"))
    return ts, metrics


def _probe_probs(cfg: DictConfig, votes: pd.DataFrame, k: int, rng) -> np.ndarray:
    """Out-of-fold class probabilities for Confident Learning and CROWDLAB."""
    from labeltriage.baselines import oof_pred_probs
    from labeltriage.experiments.features import feature_inputs

    mv = targets.majority(votes, k, rng)
    emb = feature_inputs(cfg.paths.cache_dir, cfg.data.get("base", cfg.data.name)).embeddings
    return oof_pred_probs(mv.item_ids, emb, mv.targets.argmax(axis=1), k, seed=cfg.seed)


def _baseline(name: str):
    def build(cfg, votes, k, rng):
        from labeltriage import baselines

        if name == "dawid_skene":
            return baselines.dawid_skene_targets(votes, k)
        if name == "entropy_weight":
            return baselines.entropy_weight_targets(votes, k, rng)
        probs = _probe_probs(cfg, votes, k, np.random.default_rng(cfg.seed))
        if name == "confident_learning":
            return baselines.confident_learning_targets(votes, k, probs, rng)
        return baselines.crowdlab_targets(votes, k, probs)

    return build


BASELINES = {
    n: _baseline(n) for n in ("dawid_skene", "entropy_weight", "confident_learning", "crowdlab")
}


def make_targets(
    cfg: DictConfig, bundle: bundles.Bundle, rng: np.random.Generator
) -> tuple[targets.TargetSet, dict[str, float]]:
    """Targets for the train items, plus any metrics the method reports about itself."""
    kind, votes, k = cfg.method.target, bundle.train_votes, bundle.num_classes
    extra: dict[str, float] = {}
    if kind == "clean":
        # The sampled human votes before any injected corruption (AmbiNoise only).
        ts = targets.soft(votes.assign(label=votes["original_label"]), k)
    elif kind == "oracle":
        e = votes["is_error"].to_numpy(dtype=float)
        ts = targets.route(votes, k, e, nu=cfg.method.nu, h0=cfg.method.h0)
    elif kind == "triage":
        ts, extra = _triage_targets(cfg, bundle)
    elif kind in BASELINES:
        ts = BASELINES[kind](cfg, votes, k, rng)
    else:
        ts = targets.build(kind, votes, k, rng)

    if ts.routes is not None and "is_error" in votes:
        from labeltriage.triage.pipeline import ambiguity_recall

        routes = pd.Series(ts.routes, index=ts.item_ids)
        extra["route/ambiguity_recall"] = ambiguity_recall(votes, routes, k, cfg.method.h0)
    return ts.align(bundle.train_item_ids), extra


def coteaching_setting(cfg: DictConfig, bundle: bundles.Bundle) -> tuple[float, int] | None:
    """(tau, ramp epochs) for Co-teaching, or None for ordinary training.

    Tau, the share of each batch to drop, is estimated from the votes as the share disagreeing
    with their item's majority (capped at 0.5), since the true noise rate is unknown in practice.
    """
    if cfg.method.get("trainer") != "coteaching":
        return None
    votes = bundle.train_votes
    majority = votes.groupby("item_id")["label"].agg(lambda s: s.value_counts().idxmax())
    disagree = (votes["label"].to_numpy() != votes["item_id"].map(majority).to_numpy()).mean()
    ramp = max(1, round(cfg.method.get("ramp_frac", 0.2) * cfg.data.train.epochs))
    return float(min(0.5, disagree)), int(ramp)


def _train_image(cfg, bundle, ts) -> tuple[Predictor, list[float], str]:
    from labeltriage.train import image, loop

    t = cfg.data.train
    icfg = image.ImageConfig(
        loop=loop.LoopConfig(
            epochs=t.epochs,
            batch_size=t.batch_size,
            lr=t.lr,
            weight_decay=t.weight_decay,
            amp=t.get("amp", True),
        ),
        width=t.get("width", 64),
        augment=t.get("augment", True),
    )
    device = loop.device_default()
    x = bundle.train_inputs[np.arange(len(bundle.train_inputs))]
    coteach = coteaching_setting(cfg, bundle)
    model, history = image.fit(x, ts, bundle.num_classes, icfg, cfg.seed, device, coteach=coteach)

    def predict(inputs):
        return image.predict(model, inputs, icfg.eval_batch_size, device)

    return predict, history, str(device)


def _train_text(cfg, bundle, ts) -> tuple[Predictor, list[float], str]:
    import torch
    from transformers import AutoTokenizer

    from labeltriage.train import loop, models, text

    t = cfg.data.train
    tcfg = text.TextConfig(
        model_name=t.model_name,
        max_length=t.max_length,
        loop=loop.LoopConfig(
            epochs=t.epochs,
            batch_size=t.batch_size,
            optimizer="adamw",
            lr=t.lr,
            weight_decay=t.weight_decay,
            warmup_frac=t.warmup_frac,
            amp=t.get("amp", True),
        ),
    )
    device = loop.device_default()
    tok = AutoTokenizer.from_pretrained(tcfg.model_name)
    enc = text.tokenize(*bundle.train_inputs, tok, tcfg.max_length)
    model = models.text_classifier(tcfg.model_name, bundle.num_classes)
    coteach = coteaching_setting(cfg, bundle)
    peer = None
    if coteach is not None:
        torch.manual_seed(cfg.seed + 7919)
        peer = models.text_classifier(tcfg.model_name, bundle.num_classes)
    model, history = text.fit(model, enc, ts, tcfg, cfg.seed, device, peer=peer, coteach=coteach)

    def predict(inputs):
        return text.predict(
            model, text.tokenize(*inputs, tok, tcfg.max_length), tcfg.eval_batch_size, device
        )

    return predict, history, str(device)


TRAINERS = {"image": _train_image, "text": _train_text}


def evaluate(predict: Predictor, bundle: bundles.Bundle, names, stage: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in names:
        es = bundle.eval_set(name, stage)
        probs = predict(es.inputs)
        if name == "cifar10_c":
            out["cifar10_c/robust_acc"] = metrics.accuracy(probs, es.labels)
            continue
        out |= metrics.classification_report(probs, es.labels, es.human, prefix=f"{name}/")
    return out


def eval_names(cfg: DictConfig) -> list[str]:
    names = list(cfg.data.evals.dev)
    if cfg.stage == "final":
        names += list(cfg.data.evals.final)
    return names


@task("train_eval")
def train_eval(cfg: DictConfig) -> dict[str, Any]:
    rng = seed_everything(cfg.seed)
    bundle = load_bundle(cfg)
    ts, method_metrics = make_targets(cfg, bundle, rng)

    start = time.perf_counter()
    predict, history, device = TRAINERS[bundle.domain](cfg, bundle, ts)
    train_seconds = time.perf_counter() - start

    routes = {}
    if ts.routes is not None:
        names, counts = np.unique(ts.routes, return_counts=True)
        routes = {f"route/{n}": float(c / len(ts)) for n, c in zip(names, counts, strict=True)}

    return {
        "metrics": evaluate(predict, bundle, eval_names(cfg), cfg.stage) | routes | method_metrics,
        "history": history,
        "train_seconds": train_seconds,
        "device": device,
        "n_train": len(ts),
    }
