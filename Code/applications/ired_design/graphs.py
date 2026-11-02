#!/usr/bin/env python3
"""Render model and consensus lineage graphs with campaign measurements marked."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from applications.ired_design.run_search import ARMS
from crossfit.utils import REPO_ROOT, load_config


def _draw(frame: pd.DataFrame, path: Path, title: str, *, minimum_support: int = 1) -> None:
    selected = frame.loc[frame["selected_for_next_round"]].copy()
    edges = selected.groupby(["parent_variant_id", "variant_id"]).size().rename("support").reset_index()
    edges = edges.loc[edges["support"] >= minimum_support]
    retained = set(edges["parent_variant_id"]) | set(edges["variant_id"])
    nodes = selected.loc[selected["variant_id"].isin(retained)].drop_duplicates("variant_id")
    graph = nx.DiGraph()
    for row in nodes.itertuples():
        graph.add_node(
            row.variant_id, round=int(row.round), measured=bool(row.in_curated_dataset)
        )
    for row in edges.itertuples():
        graph.add_edge(row.parent_variant_id, row.variant_id, support=int(row.support))
        if row.parent_variant_id not in graph:
            graph.add_node(row.parent_variant_id, round=0, measured=True)
    if not graph:
        return
    layers = {}
    for node, values in graph.nodes(data=True):
        layers.setdefault(values["round"], []).append(node)
    positions = {}
    for round_number, values in sorted(layers.items()):
        for index, node in enumerate(sorted(values)):
            positions[node] = (round_number, index - (len(values) - 1) / 2)
    figure, axis = plt.subplots(figsize=(14, 10))
    colors = [graph.nodes[node]["round"] for node in graph]
    widths = [0.35 + 0.25 * graph.edges[edge]["support"] for edge in graph.edges]
    nx.draw_networkx_edges(graph, positions, width=widths, alpha=0.25, arrows=False, ax=axis)
    nx.draw_networkx_nodes(
        graph, positions, node_color=colors, cmap="viridis", node_size=16,
        edgecolors=["black" if graph.nodes[node].get("measured") else "none" for node in graph],
        linewidths=0.8, ax=axis,
    )
    axis.set_title(title)
    axis.set_xlabel("in-silico round")
    axis.set_ylabel("layout index (only horizontal position is meaningful)")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> int:
    config = load_config(REPO_ROOT / "Code/config/ired_design.yaml")
    report = REPO_ROOT / config["paths"]["reports"]
    root = report / "full_lineage"
    graph_dir = report / "trajectory_graphs"
    all_selected = []
    for arm in ARMS:
        frame = pd.concat([pd.read_parquet(path) for path in sorted((root / arm).glob("*.parquet"))])
        _draw(frame, graph_dir / f"{arm}.png", arm)
        all_selected.append(frame.loc[frame["selected_for_next_round"]])
    ensemble = pd.concat(all_selected, ignore_index=True)
    _draw(ensemble, graph_dir / "ensemble.png", "ensemble-supported trajectory", minimum_support=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
