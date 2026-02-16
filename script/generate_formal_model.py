#!/usr/bin/env python3
"""Generate a one-page conceptual flow PDF for the AS-level network model."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
MPL_CACHE_DIR = ROOT_DIR / ".mpl-cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _add_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    facecolor: str = "#f8fafc",
    title_size: float = 11.0,
    body_size: float = 9.5,
) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.4,
        edgecolor="#334155",
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(x + 0.012, y + h - 0.038, title, fontsize=title_size, fontweight="bold", va="top")
    ax.text(
        x + 0.012,
        y + h - 0.072,
        body,
        fontsize=body_size,
        va="top",
        linespacing=1.35,
        wrap=True,
    )


def _add_arrow(
    ax,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    color: str = "#1e3a8a",
    lw: float = 1.5,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start_xy,
            end_xy,
            arrowstyle="-|>",
            mutation_scale=12,
            lw=lw,
            color=color,
        )
    )


def _draw_conceptual_flow(ax, k_paths: int) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.02,
        0.965,
        "Network Model Conceptual Flow (AS Graph and Routing Decision)",
        fontsize=15,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.02,
        0.925,
        "Directed topology, fixed candidate path set, and shared decision input across algorithms.",
        fontsize=10.5,
        color="#334155",
        va="top",
    )

    _add_box(
        ax,
        0.03,
        0.16,
        0.37,
        0.68,
        "Directed AS Graph  $G=(V,E)$",
        "$V$: Autonomous Systems (ASes)\n$E\\subseteq V\\times V$: inter-domain routing links\nEach edge $(u,v)$ represents an inter-AS forwarding relation.",
        facecolor="#eef2ff",
    )

    gax = ax.inset_axes([0.075, 0.24, 0.28, 0.48])
    gax.axis("off")
    graph = nx.DiGraph()
    graph.add_edges_from(
        [
            ("AS_u", "AS_1"),
            ("AS_u", "AS_2"),
            ("AS_1", "AS_3"),
            ("AS_2", "AS_3"),
            ("AS_2", "AS_v"),
            ("AS_3", "AS_v"),
        ]
    )
    pos = {
        "AS_u": (0.05, 0.5),
        "AS_1": (0.35, 0.8),
        "AS_2": (0.35, 0.2),
        "AS_3": (0.68, 0.5),
        "AS_v": (0.95, 0.5),
    }
    nx.draw_networkx_nodes(
        graph,
        pos=pos,
        node_size=1500,
        node_color="#c7d2fe",
        edgecolors="#312e81",
        linewidths=1.2,
        ax=gax,
    )
    nx.draw_networkx_edges(
        graph,
        pos=pos,
        width=1.6,
        edge_color="#334155",
        arrows=True,
        arrowstyle="-|>",
        arrowsize=14,
        ax=gax,
    )
    nx.draw_networkx_labels(graph, pos=pos, font_size=8.5, font_weight="bold", ax=gax)

    _add_box(
        ax,
        0.46,
        0.76,
        0.22,
        0.1,
        "Demand Input",
        "Pair $(s,d)\\in\\mathcal{S}$",
        facecolor="#ecfeff",
    )
    _add_box(
        ax,
        0.71,
        0.7,
        0.26,
        0.16,
        "Candidate Path Generation",
        "Run latency-weighted\n$k$-shortest path on $G$\n($k=%d$)." % k_paths,
        facecolor="#fff7ed",
    )
    _add_box(
        ax,
        0.71,
        0.49,
        0.26,
        0.15,
        "Fixed Candidate Set",
        "$\\mathcal{P}_{sd}=\\{P_{sd}^{(1)},...,P_{sd}^{(k)}\\}$\nConstant for the full simulation.",
        facecolor="#fef9c3",
    )
    _add_box(
        ax,
        0.71,
        0.28,
        0.26,
        0.16,
        "Routing Decision Flow",
        "Algorithms select from the same\n$\\mathcal{P}_{sd}$ only.\nOutput path: $p_t(s,d)\\in\\mathcal{P}_{sd}$.",
        facecolor="#f0fdf4",
    )
    _add_box(
        ax,
        0.46,
        0.1,
        0.51,
        0.12,
        "Control Constraint",
        "No algorithm may add or remove candidate paths.\nObserved differences arise from path-evaluation logic, not path availability.",
        facecolor="#f8fafc",
        title_size=10.5,
        body_size=9.25,
    )

    _add_arrow(ax, (0.4, 0.5), (0.71, 0.77))
    _add_arrow(ax, (0.68, 0.81), (0.71, 0.78))
    _add_arrow(ax, (0.84, 0.7), (0.84, 0.64))
    _add_arrow(ax, (0.84, 0.49), (0.84, 0.44))
    _add_arrow(ax, (0.84, 0.28), (0.84, 0.22))
    _add_arrow(ax, (0.61, 0.22), (0.61, 0.13), color="#475569", lw=1.2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a one-page conceptual flow PDF for the AS-level network model."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of latency-weighted candidate paths per source-destination pair.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("script/network_model_conceptual_flow.pdf"),
        help="Output PDF file path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(args.output) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))
        ax = fig.add_subplot(111)
        _draw_conceptual_flow(ax, k_paths=args.k)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Wrote conceptual flow PDF to {args.output}")


if __name__ == "__main__":
    main()
