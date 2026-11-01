"""JSON API behind the web app, and the server that also hosts the built React app.

    python -m labeltriage.api                # http://127.0.0.1:8000
    python -m labeltriage.api --open         # and open it in the default browser

It reads what the runs already saved: ``scored_votes.parquet`` and ``result.json`` under the
runs folder (``outputs_mini/grid`` if present, else ``outputs/grid``), and the prepared datasets
under ``data/processed`` for class names, NLI text and CIFAR images. Nothing is written.

Responses are gzip-compressed and carry an ETag, so a repeat request is a cheap 304. Parquet
files are read once per file version, and the heavier summaries are cached until a run changes.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import io
import json
import os
import re
import threading
import webbrowser
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from labeltriage import explorer
from labeltriage.eval.summary import aggregate, collect

SCORED = "scored_votes.parquet"
ROUTES = tuple(explorer.ROUTE_COLOURS)
NLI_CLASSES = ("entailment", "neutral", "contradiction")
BASES = {"10h": "cifar10h", "nli": "chaosnli"}
DATA_NAME = re.compile(r"ambinoise_(10h|nli)_k(\d+)_r([0-9.]+)_(.+)$")
PAGE_SIZES = (10, 25, 50, 100)
NOT_BUILT = """<!doctype html><meta charset="utf-8"><title>WhoToTrust</title>
<body style="font:16px system-ui;max-width:42rem;margin:4rem auto;padding:0 1rem">
<h1>The web app is not built yet</h1>
<p>The API is running (see <a href="/api/docs">/api/docs</a>). To build the app, once:</p>
<pre>cd web
npm ci
npm run build</pre>
<p>Then reload this page.</p></body>"""
WEB_DIST = Path(__file__).resolve().parents[2] / "web" / "dist"


def default_root() -> Path:
    """The folder holding finished runs: mini results if there are any, else the full ones."""
    env = os.environ.get("LT_API_ROOT")
    if env:
        return Path(env)
    for candidate in ("outputs_mini/grid", "outputs/grid", "outputs"):
        if Path(candidate).is_dir() and any(Path(candidate).rglob(SCORED)):
            return Path(candidate)
    return Path("outputs_mini/grid")


# --- reading files, once per file version -----------------------------------------------------


def _stamp(path: Path) -> int:
    return path.stat().st_mtime_ns


@functools.lru_cache(maxsize=256)
def _read_parquet(path: str, stamp: int) -> pd.DataFrame:
    return pd.read_parquet(path)


def read_parquet(path: Path) -> pd.DataFrame:
    return _read_parquet(str(path), _stamp(path))


@functools.lru_cache(maxsize=16)
def _read_json(path: str, stamp: int) -> dict:
    return json.loads(Path(path).read_text(encoding="utf8"))


def clean(value: Any) -> Any:
    """Make numpy / pandas values JSON-safe: NaN and NA become null."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [clean(v) for v in value]
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.bool_ | bool):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return None if not np.isfinite(value) else float(value)
    return value


def records(df: pd.DataFrame, columns: dict[str, str]) -> list[dict]:
    """Rows of ``df`` as dicts, keeping ``columns`` (source name -> API name) that exist."""
    keep = {src: dst for src, dst in columns.items() if src in df}
    return [
        clean({dst: row[src] for src, dst in keep.items()})
        for _, row in df.sort_values(list(keep)[:2]).iterrows()
    ]


# --- what a run is ------------------------------------------------------------------------------


class Runs:
    """The scored runs under one folder, safe to address by relative path."""

    def __init__(self, root: Path, data_dir: Path):
        self.root, self.data_dir = Path(root).resolve(), Path(data_dir)

    def resolve(self, run_id: str) -> Path:
        root = self.root
        path = (root / run_id).resolve()
        if not path.is_relative_to(root) or not (path / SCORED).is_file():
            raise HTTPException(404, f"no scored run {run_id!r}")
        return path

    def describe(self, path: Path) -> dict:
        cfg = {}
        result = path / "result.json"
        if result.is_file():
            cfg = _read_json(str(result), _stamp(result)).get("config", {})
        data = cfg.get("data", {})
        name = data.get("name") or path.parent.parent.name
        m = DATA_NAME.match(name)
        base = data.get("base") or (BASES[m.group(1)] if m else None)
        noise = data.get("noise") or {}
        return {
            "id": path.relative_to(self.root).as_posix(),
            "dataset": base,
            "error_type": noise.get("error_type") or (m.group(4) if m else None),
            "k": noise.get("k") or (int(m.group(2)) if m else None),
            "rate": noise.get("rate") or (float(m.group(3)) if m else None),
            "seed": cfg.get("seed"),
            "method": cfg.get("method", {}).get("name"),
            "experiment": cfg.get("experiment", {}).get("name"),
            "stage": cfg.get("stage"),
        }

    def listing(self) -> list[dict]:
        if not self.root.is_dir():
            return []
        out = []
        for scored in sorted(self.root.rglob(SCORED)):
            info = self.describe(scored.parent)
            df = read_parquet(scored)
            out.append({**info, "n_votes": len(df), "n_items": int(df["item_id"].nunique())})
        return out

    def items(self, path: Path) -> pd.DataFrame:
        return _item_table(str(path / SCORED), _stamp(path / SCORED))


@functools.lru_cache(maxsize=16)
def _item_table(path: str, stamp: int) -> pd.DataFrame:
    return explorer.item_table(_read_parquet(path, stamp))


@functools.lru_cache(maxsize=8)
def _base_info(data_dir: str, base: str) -> dict:
    """Class names, plus the items table (text, image index) when the dataset is prepared."""
    d = Path(data_dir) / "processed" / base
    classes = list(NLI_CLASSES) if base == "chaosnli" else []
    items = None
    try:
        classes = json.loads((d / "meta.json").read_text())["classes"]
        items = pd.read_parquet(d / "items.parquet")
    except (OSError, ValueError, KeyError):
        pass
    return {"classes": classes, "items": items}


def _cifar_present(data_dir: str) -> bool:
    return (Path(data_dir) / "raw" / "cifar10" / "cifar-10-batches-py" / "test_batch").is_file()


@functools.lru_cache(maxsize=1)
def _cifar_test_images(data_dir: str) -> np.ndarray | None:
    """CIFAR-10's 10,000 test images if they are already on disk. Never downloads anything.

    Loading goes through torchvision, which takes a while to import, so the answer to "does this
    item have a picture" uses ``_cifar_present`` and only a picture request pays for this.
    """
    if not _cifar_present(data_dir):
        return None
    try:
        from labeltriage.data import images

        return images.cifar10(data_dir, train=False)[0]
    except Exception:  # torchvision missing or files unreadable: no pictures, the app still works
        return None


# --- results across the pipeline ---------------------------------------------------------------


def _signature(root: Path) -> tuple[int, int]:
    done = list(root.rglob("DONE")) if root.is_dir() else []
    return len(done), max((_stamp(p) for p in done), default=0)


def _split_data(df: pd.DataFrame) -> pd.DataFrame:
    parts = df["data"].astype(str).str.extract(DATA_NAME)
    out = df.copy()
    out["base"] = parts[0].map(BASES).fillna(df["data"])
    out["error_type"] = parts[3]
    return out


@functools.lru_cache(maxsize=4)
def _results(root: str, signature: tuple[int, int]) -> dict:
    runs = collect(root)
    if runs.empty:
        return {"n_runs": 0, "sections": {}}
    runs = _split_data(runs)
    by = {name: runs[runs["experiment"] == name] for name in runs["experiment"].dropna().unique()}

    def section(name: str, columns: dict[str, str]) -> list[dict]:
        df = by.get(name)
        return [] if df is None or df.empty else records(df, columns)

    compare = by.get("pilot")
    compare_rows: list[dict] = []
    if compare is not None and not compare.empty:
        agg = _split_data(aggregate(compare))
        compare_rows = records(
            agg,
            {
                "base": "dataset",
                "error_type": "error_type",
                "method": "method",
                "n_seeds": "n_seeds",
                "val/acc_mean": "val_acc",
                "val/acc_std": "val_acc_std",
                "val/kl_mean": "val_kl",
                "clean_dev/acc_mean": "clean_dev_acc",
            },
        )
    transfer = by.get("transfer")
    return clean(
        {
            "n_runs": len(runs),
            "train_seconds_total": float(runs["train_seconds"].fillna(0).sum()),
            "sections": {
                "reproduce": section(
                    "reproduce",
                    {
                        "data": "dataset",
                        "method": "method",
                        "val/acc": "val_acc",
                        "val/kl": "val_kl",
                        "clean_dev/acc": "clean_dev_acc",
                        "train_seconds": "train_seconds",
                    },
                ),
                "injection": section(
                    "injection", {"data": "dataset", "heldout_acc_vs_majority": "heldout_acc"}
                ),
                "features": section(
                    "features",
                    {
                        "base": "dataset",
                        "error_type": "error_type",
                        **{f"auroc/{g}": g for g in ("g1", "g2", "g3", "g4", "g5", "all")},
                        "auroc/minority_share_only": "minority_share",
                        "auroc/leakage_shuffled": "leakage_shuffled",
                    },
                ),
                "triage": section(
                    "triage",
                    {
                        "base": "dataset",
                        "error_type": "error_type",
                        "triage/val_auroc": "auroc",
                        "triage/val_minority_share_only": "minority_share",
                        "triage/val_calibration_error": "calibration_error",
                        "gate/auroc_pass": "auroc_pass",
                        "gate/calibration_pass": "calibration_pass",
                    },
                ),
                "compare": compare_rows,
                "transfer": (
                    []
                    if transfer is None or transfer.empty
                    else records(
                        transfer,
                        {
                            "transfer/dev_auroc": "auroc",
                            "transfer/dev_minority_share_only": "minority_share",
                            "transfer/dev_gain_over_minority_lo": "gain_lo",
                            "transfer/dev_gain_over_minority_hi": "gain_hi",
                        },
                    )
                ),
            },
        }
    )


# --- the app ----------------------------------------------------------------------------------


class _Static(StaticFiles):
    """Built assets are content-hashed, so browsers may keep them for a year."""

    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        if path.replace("\\", "/").startswith("assets/") and resp.status_code == 200:
            resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return resp


def create_app(
    root: str | Path | None = None,
    data_dir: str | Path | None = None,
    web_dir: str | Path | None = WEB_DIST,
) -> FastAPI:
    runs = Runs(
        Path(root) if root else default_root(),
        Path(data_dir or os.environ.get("LT_DATA_DIR", "data")),
    )
    app = FastAPI(title="WhoToTrust API", docs_url="/api/docs", openapi_url="/api/openapi.json")

    @app.middleware("http")
    async def etag(request: Request, call_next):
        """Revalidate cheaply: identical JSON answers a repeat request with 304."""
        response = await call_next(request)
        if request.method != "GET" or not request.url.path.startswith("/api/"):
            return response
        if response.status_code != 200 or "etag" in response.headers:
            return response
        body = b"".join([chunk async for chunk in response.body_iterator])
        tag = '"' + hashlib.sha1(body).hexdigest()[:20] + '"'
        headers = {k: v for k, v in response.headers.items() if k.lower() != "content-length"}
        headers["ETag"] = tag
        headers.setdefault("Cache-Control", "no-cache")
        if request.headers.get("if-none-match") == tag:
            return Response(status_code=304, headers=headers)
        return Response(body, status_code=200, headers=headers)

    # Added after the ETag layer, so it wraps it: the tag hashes the plain body, then gzip packs it.
    app.add_middleware(GZipMiddleware, minimum_size=500, compresslevel=5)

    def run_path(run: str) -> Path:
        return runs.resolve(run)

    @app.get("/api/health")
    def health():
        return {"ok": True, "root": str(runs.root), "runs": len(runs.listing())}

    @app.get("/api/runs")
    def list_runs():
        return runs.listing()

    @app.get("/api/results")
    def results():
        return _results(str(runs.root), _signature(runs.root))

    @app.get("/api/points")
    def points(run: str):
        """Every item of a run as parallel arrays (compact for the scatter plot)."""
        items = runs.items(run_path(run))
        out = {
            "routes": list(ROUTES),
            "item_id": items["item_id"].tolist(),
            "entropy": items["entropy"].round(4).tolist(),
            "max_p_error": items["max_p_error"].round(4).tolist(),
            "route": items["route"]
            .map({r: i for i, r in enumerate(ROUTES)})
            .fillna(0)
            .astype(int)
            .tolist(),
            "true_errors": items["true_errors"].tolist() if "true_errors" in items else None,
        }
        return clean(out)

    @app.get("/api/items")
    def list_items(
        run: str,
        route: Annotated[list[str] | None, Query()] = None,
        only_errors: bool = False,
        q: Annotated[str, Query(max_length=80)] = "",
        sort: Annotated[str, Query(pattern="^(max_p_error|entropy|item_id)$")] = "max_p_error",
        page: Annotated[int, Query(ge=1)] = 1,
        page_size: Annotated[int, Query(ge=1, le=100)] = 25,
    ):
        """A page of the run's items, most suspicious first, with the filters the app offers."""
        path = run_path(run)
        view = runs.items(path)
        classes = _base_info(str(runs.data_dir), runs.describe(path)["dataset"] or "")["classes"]
        if route:
            view = view[view["route"].isin(route)]
        if only_errors and "true_errors" in view:
            view = view[view["true_errors"] > 0]
        if q:
            view = view[view["item_id"].str.contains(q, case=False, regex=False)]
        view = view.sort_values(sort, ascending=sort == "item_id", kind="stable")
        total = len(view)
        shown = view.iloc[(page - 1) * page_size : page * page_size]
        cols = ["item_id", "route", "labels", "entropy", "max_p_error", "n_votes", "true_errors"]
        return clean(
            {
                "total": total,
                "page": page,
                "page_size": page_size,
                "pages": max(1, -(-total // page_size)),
                "classes": classes,
                "items": shown[[c for c in cols if c in shown]].to_dict("records"),
            }
        )

    def item_rows(path: Path, item: str) -> pd.DataFrame:
        scored = read_parquet(path / SCORED)
        if item not in set(runs.items(path)["item_id"]):
            raise HTTPException(404, f"no item {item!r} in this run")
        return scored

    @app.get("/api/item")
    def item_detail(run: str, item: str):
        path = run_path(run)
        scored = item_rows(path, item)
        info = runs.describe(path)
        base = _base_info(str(runs.data_dir), info["dataset"] or "")
        classes = base["classes"]
        summary = runs.items(path).set_index("item_id").loc[item]
        votes = explorer.vote_table(scored, item)
        votes["label_name"] = [
            classes[int(v)] if classes and int(v) < len(classes) else str(v) for v in votes["label"]
        ]
        votes["reasons"] = [s.split("; ") for s in votes.pop("why")]
        meta = base["items"]
        text = None
        has_image = False
        if meta is not None and item in meta.index:
            row = meta.loc[item]
            if "premise" in meta:
                text = {"premise": str(row["premise"]), "hypothesis": str(row["hypothesis"])}
            elif "image_index" in meta:
                has_image = _cifar_present(str(runs.data_dir))
        return clean(
            {
                "item_id": item,
                "route": summary["route"],
                "entropy": summary["entropy"],
                "max_p_error": summary["max_p_error"],
                "n_votes": int(summary["n_votes"]),
                "labels": summary["labels"],
                "classes": classes,
                "text": text,
                "has_image": has_image,
                "votes": votes.to_dict("records"),
            }
        )

    @app.get("/api/image")
    def image(run: str, item: str):
        path = run_path(run)
        item_rows(path, item)
        base = _base_info(str(runs.data_dir), runs.describe(path)["dataset"] or "")
        meta = base["items"]
        x = _cifar_test_images(str(runs.data_dir))
        if meta is None or x is None or "image_index" not in meta or item not in meta.index:
            raise HTTPException(404, "no picture for this item")
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(x[int(meta.loc[item, "image_index"])]).resize(
            (128, 128), Image.NEAREST
        ).save(buf, "PNG", optimize=True)
        return Response(
            buf.getvalue(),
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException):
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    if web_dir and Path(web_dir).is_dir():
        app.mount("/", _Static(directory=str(web_dir), html=True), name="web")
    else:

        @app.get("/", include_in_schema=False)
        def not_built():
            return HTMLResponse(NOT_BUILT, status_code=503)

    app.state.runs = runs
    return app


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--root", help="folder of finished runs (default: outputs_mini/grid)")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--open", action="store_true", help="open the app in the default browser")
    args = parser.parse_args(argv)
    app = create_app(args.root, args.data_dir)
    data_dir = str(app.state.runs.data_dir)
    if _cifar_present(data_dir):
        # Importing torchvision and reading the images takes a while: do it now, in the
        # background, so the first picture a visitor asks for is already waiting.
        threading.Thread(target=_cifar_test_images, args=(data_dir,), daemon=True).start()
    if args.open:
        webbrowser.open(f"http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
