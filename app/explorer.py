"""Disagreement Explorer: see which disagreements LabelTriage called errors, and why.

    pip install -e ".[demo]"
    streamlit run app/explorer.py

Reads the ``scored_votes.parquet`` a LabelTriage run saves next to its ``result.json``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from labeltriage import explorer

st.set_page_config(page_title="Disagreement Explorer", layout="wide")


@st.cache_data
def load_scored(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data
def load_cifar10h_images(data_dir: str) -> tuple[np.ndarray, pd.Series] | None:
    """CIFAR-10 test images and each CIFAR-10H item's image index, if available locally."""
    try:
        from labeltriage.data import images, prepare

        _, items, _ = prepare.load_prepared("cifar10h", data_dir)
        x, _ = images.cifar10(data_dir, train=False)
        return x, items["image_index"]
    except Exception:  # the explorer works without images
        return None


def newest_scored(root: str) -> list[str]:
    paths = sorted(Path(root).rglob("scored_votes.parquet"), key=lambda p: p.stat().st_mtime)
    return [str(p) for p in reversed(paths)]


st.title("Disagreement Explorer")
st.caption(
    "Each dot is an item. Further right means its votes split more; higher means LabelTriage "
    "suspects one of its votes is an error. Colour shows the training target it was routed to."
)

with st.sidebar:
    root = st.text_input("Run outputs folder", "outputs")
    options = newest_scored(root)
    path = st.selectbox("Scored votes", options) if options else st.text_input("Scored votes file")
    data_dir = st.text_input("Data folder (for images)", "data")
    if not path:
        st.info("Run LabelTriage first; it saves scored_votes.parquet next to its result.")
        st.stop()

scored = load_scored(path)
items = explorer.item_table(scored)

with st.sidebar:
    routes = st.multiselect("Routes", list(explorer.ROUTE_COLOURS), list(explorer.ROUTE_COLOURS))
    only_errors = "true_errors" in items and st.checkbox("Only items with a known error")

view = items[items["route"].isin(routes)]
if only_errors:
    view = view[view["true_errors"] > 0]

cols = st.columns(3)
cols[0].metric("Items", f"{len(view):,}")
for col, route in zip(cols[1:], ("ambiguous", "unreliable"), strict=True):
    col.metric(f"Routed {route}", f"{(items['route'] == route).mean():.1%}")

rng = np.random.default_rng(0)
plot = view.assign(entropy_jitter=view["entropy"] + rng.normal(0, 0.01, len(view)))
fig = px.scatter(
    plot,
    x="entropy_jitter",
    y="max_p_error",
    color="route",
    color_discrete_map=explorer.ROUTE_COLOURS,
    hover_data=["item_id", "labels", "n_votes"]
    + (["true_errors"] if "true_errors" in plot else []),
    labels={
        "entropy_jitter": "vote entropy (nats)",
        "max_p_error": "highest p(error) among its votes",
    },
    custom_data=["item_id"],
    opacity=0.7,
)
fig.update_traces(marker={"size": 6})
event = st.plotly_chart(fig, width="stretch", on_select="rerun", key="scatter")

picked = None
if event and event.selection and event.selection.points:
    picked = event.selection.points[0]["customdata"][0]
ranked = view.sort_values("max_p_error", ascending=False)["item_id"].tolist()
if not ranked:
    st.stop()
item_id = st.selectbox(
    "Item (click a dot, or pick from the most suspicious first)",
    ranked,
    index=ranked.index(picked) if picked in ranked else 0,
)

row = items.set_index("item_id").loc[item_id]
left, right = st.columns([1, 3])
with left:
    imgs = load_cifar10h_images(data_dir)
    if imgs is not None and item_id in imgs[1].index:
        st.image(imgs[0][int(imgs[1][item_id])], width=160, caption=f"item {item_id}")
    st.markdown(f"**Route:** {row['route']}  \n**Votes:** {row['labels']}")
with right:
    st.dataframe(explorer.vote_table(scored, item_id), width="stretch", hide_index=True)
