"""Train one classifier on one method's targets and evaluate it on the configured sets.

This is the unit every grid is made of: a (dataset, method, seed) triple in, a metrics dict out.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
from omegaconf import DictConfig

from labeltriage import targets
from labeltriage.data import bundles
from labeltriage.eval import metrics
from labeltriage.tasks import task
from labeltriage.utils.seed import seed_everything

Predictor = Callable[[Any], np.ndarray]


def make_targets(cfg: DictConfig, bundle: bundles.Bundle, rng: np.random.Generator):
    ts = targets.build(cfg.method.target, bundle.train_votes, bundle.num_classes, rng)
    return ts.align(bundle.train_item_ids)


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
    model, history = image.fit(x, ts, bundle.num_classes, icfg, cfg.seed, device)

    def predict(inputs):
        return image.predict(model, inputs, icfg.eval_batch_size, device)

    return predict, history, str(device)


def _train_text(cfg, bundle, ts) -> tuple[Predictor, list[float], str]:
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
    model, history = text.fit(model, enc, ts, tcfg, cfg.seed, device)

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
    bundle = bundles.load(cfg.data.name, cfg.paths.data_dir)
    ts = make_targets(cfg, bundle, rng)

    start = time.perf_counter()
    predict, history, device = TRAINERS[bundle.domain](cfg, bundle, ts)
    train_seconds = time.perf_counter() - start

    return {
        "metrics": evaluate(predict, bundle, eval_names(cfg), cfg.stage),
        "history": history,
        "train_seconds": train_seconds,
        "device": device,
        "n_train": len(ts),
    }
