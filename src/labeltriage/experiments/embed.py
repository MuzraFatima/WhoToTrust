"""Frozen-model outputs that feature groups G3–G5 read, cached once per base dataset.

Images (upsampled from 32x32 by each model's own processor, bicubic, amendment A6):
- CLIP ViT-B/32 zero-shot class probabilities -> G3 plausibility
- DINOv2 ViT-S/14 CLS embeddings -> G4 neighbours and the G5 probe

NLI:
- ``roberta-large-mnli`` class probabilities -> G3 plausibility (a different model from the
  DeBERTa-v3-small injection model, amendment A1)
- a sentence encoder over "premise [SEP] hypothesis" -> G4 and G5

Everything goes through ``transformers`` with no remote code.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from labeltriage.data import images, prepare, sources
from labeltriage.tasks import task

NLI_ORDER = ("entailment", "neutral", "contradiction")


def cache_paths(cache_dir: str | Path, base: str) -> dict[str, Path]:
    root = Path(cache_dir)
    return {
        "embeddings": root / "embeddings" / f"{base}.parquet",
        "plausibility": root / "plausibility" / f"{base}.parquet",
    }


def batched(fn: Callable[[Any], np.ndarray], inputs: Sequence, batch_size: int) -> np.ndarray:
    return np.concatenate(
        [fn(inputs[i : i + batch_size]) for i in range(0, len(inputs), batch_size)]
    )


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clip_zero_shot(classes: Sequence[str], model_name: str, batch_size: int):
    import torch
    from transformers import CLIPModel, CLIPProcessor

    dev = _device()
    model = CLIPModel.from_pretrained(model_name).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(model_name)
    with torch.no_grad():
        text = proc(text=[f"a photo of a {c}." for c in classes], return_tensors="pt", padding=True)
        t = model.get_text_features(**{k: v.to(dev) for k, v in text.items()})
        t = t / t.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def run(batch: np.ndarray) -> np.ndarray:
        px = proc(images=list(batch), return_tensors="pt")["pixel_values"].to(dev)
        f = model.get_image_features(pixel_values=px)
        f = f / f.norm(dim=-1, keepdim=True)
        return (model.logit_scale.exp() * f @ t.T).softmax(-1).cpu().numpy()

    return lambda x: batched(run, x, batch_size)


def dinov2_embed(model_name: str, batch_size: int):
    import torch
    from transformers import AutoImageProcessor, AutoModel

    dev = _device()
    model = AutoModel.from_pretrained(model_name).to(dev).eval()
    proc = AutoImageProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def run(batch: np.ndarray) -> np.ndarray:
        px = proc(images=list(batch), return_tensors="pt")["pixel_values"].to(dev)
        return model(pixel_values=px).pooler_output.float().cpu().numpy()

    return lambda x: batched(run, x, batch_size)


def mnli_probs(model_name: str, batch_size: int):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = _device()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(dev).eval()
    names = {i: n.lower() for i, n in model.config.id2label.items()}
    order = [next(i for i, n in names.items() if n.startswith(c[:4])) for c in NLI_ORDER]

    @torch.no_grad()
    def run(pairs) -> np.ndarray:
        prem, hyp = zip(*pairs, strict=True)
        enc = tok(
            list(prem),
            list(hyp),
            truncation=True,
            max_length=256,
            padding=True,
            return_tensors="pt",
        )
        logits = model(**{k: v.to(dev) for k, v in enc.items()}).logits.float()
        return logits.softmax(-1)[:, order].cpu().numpy()

    return lambda x: batched(run, x, batch_size)


def sentence_embed(model_name: str, batch_size: int):
    import torch
    from transformers import AutoModel, AutoTokenizer

    dev = _device()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(dev).eval()

    @torch.no_grad()
    def run(pairs) -> np.ndarray:
        prem, hyp = zip(*pairs, strict=True)
        enc = tok(
            list(prem),
            list(hyp),
            truncation=True,
            max_length=256,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(dev) for k, v in enc.items()}
        hidden = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        return ((hidden * mask).sum(1) / mask.sum(1)).float().cpu().numpy()

    return lambda x: batched(run, x, batch_size)


def item_inputs(
    base: str, data_dir: str | Path, subset: str | None = None
) -> tuple[pd.Index, Any, str, tuple[str, ...]]:
    """(item ids, inputs aligned with them, domain, class names) for every item of ``base``
    (or of its ``subset``)."""
    _, items, meta = prepare.load_prepared(base, data_dir, subset)
    classes = tuple(meta["classes"])
    if base in ("cifar10h", "cifar10n"):
        x, _ = images.cifar10(data_dir, train=(base == "cifar10n"))
        return items.index, x[items["image_index"].to_numpy()], "image", classes
    pairs = list(zip(items["premise"], items["hypothesis"], strict=True))
    return items.index, pairs, "text", classes


def save(ids: pd.Index, array: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(array, index=pd.Index(ids, name="item_id"))
    frame.columns = frame.columns.astype(str)
    frame.to_parquet(path)
    return path


def load(path: str | Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame.columns = frame.columns.astype(int)
    return frame


@task("embed")
def embed(cfg: DictConfig) -> dict[str, Any]:
    base = cfg.data.get("base", cfg.data.name)
    m = cfg.experiment.models
    ids, inputs, domain, classes = item_inputs(base, cfg.paths.data_dir, cfg.data.get("subset"))
    if domain == "image":
        plaus = clip_zero_shot(classes, m.clip, m.batch_size)(inputs)
        emb = dinov2_embed(m.dinov2, m.batch_size)(inputs)
    else:
        if classes != sources.NLI_CLASSES:
            raise ValueError(f"unexpected NLI class order {classes}")
        plaus = mnli_probs(m.mnli, m.batch_size)(inputs)
        emb = sentence_embed(m.sentence, m.batch_size)(inputs)

    paths = cache_paths(cfg.paths.cache_dir, base)
    save(ids, plaus, paths["plausibility"])
    save(ids, emb, paths["embeddings"])
    return {
        "metrics": {"plausibility_top1_mean": float(plaus.max(axis=1).mean())},
        "n_items": len(ids),
        "embedding_dim": int(emb.shape[1]),
        "paths": {k: str(v) for k, v in paths.items()},
    }
