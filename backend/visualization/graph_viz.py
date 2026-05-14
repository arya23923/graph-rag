"""
GraphViz — renders NetworkX graphs to Matplotlib figures and Pyvis HTML.
Used by the Streamlit frontend pages.
"""
import io
import base64
import logging
from typing import Dict, List, Optional, Tuple
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from backend.visualization.network_builder import NetworkBuilder, TYPE_COLORS, REL_COLORS

logger = logging.getLogger(__name__)


class GraphViz:
    def __init__(self):
        self.builder = NetworkBuilder()

    # ── Matplotlib ─────────────────────────────────────────────────────────

    def render_matplotlib(
        self,
        nodes: List[Dict],
        edges: List[Dict],
        title: str = "Tech Knowledge Graph",
        figsize: Tuple[int, int] = (14, 10),
    ) -> plt.Figure:
        G = self.builder.build(nodes, edges)
        fig, ax = plt.subplots(figsize=figsize, facecolor="#0E1117")
        ax.set_facecolor("#0E1117")

        if G.number_of_nodes() == 0:
            ax.text(0.5, 0.5, "No graph data available", color="white",
                    ha="center", va="center", fontsize=14, transform=ax.transAxes)
            return fig

        # Layout
        try:
            pos = nx.spring_layout(G, k=2.5, seed=42, iterations=60)
        except Exception:
            pos = nx.circular_layout(G)

        # Draw edges
        edge_colors = [G.edges[u, v].get("color", "#95A5A6") for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors,
                               alpha=0.7, arrows=True, arrowsize=20,
                               width=1.5, connectionstyle="arc3,rad=0.1")

        # Draw nodes
        node_colors = [G.nodes[n].get("color", "#4A90D9") for n in G.nodes()]
        node_sizes  = [G.nodes[n].get("size", 15) * 80 for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                               node_size=node_sizes, alpha=0.9)

        # Labels
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8,
                                font_color="white", font_weight="bold")

        # Edge labels (relationship type)
        edge_labels = {(u, v): G.edges[u, v].get("relationship", "")[:12]
                       for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax,
                                     font_size=6, font_color="#CCCCCC",
                                     bbox=dict(alpha=0))

        # Legend
        legend_patches = [
            mpatches.Patch(color=col, label=lbl)
            for lbl, col in TYPE_COLORS.items()
            if any(G.nodes[n].get("type") == lbl for n in G.nodes())
        ]
        if legend_patches:
            ax.legend(handles=legend_patches, loc="upper left",
                      fontsize=7, facecolor="#1E1E2E", labelcolor="white")

        ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=12)
        ax.axis("off")
        plt.tight_layout()
        return fig

    def fig_to_base64(self, fig: plt.Figure) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                    facecolor="#0E1117")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    # ── Pyvis HTML ─────────────────────────────────────────────────────────

    def render_pyvis_html(
        self,
        nodes: List[Dict],
        edges: List[Dict],
        height: str = "600px",
    ) -> str:
        """Return self-contained HTML string for st.components.html()."""
        try:
            from pyvis.network import Network
            net = Network(height=height, width="100%", bgcolor="#0E1117",
                          font_color="white", directed=True)
            net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=150)

            G = self.builder.build(nodes, edges)
            for node in G.nodes():
                nd = G.nodes[node]
                net.add_node(
                    node, label=node,
                    color=nd.get("color", "#4A90D9"),
                    size=nd.get("size", 15),
                    title=f"Type: {nd.get('type','?')}\nConfidence: {nd.get('confidence',0):.2f}",
                )
            for u, v in G.edges():
                ed = G.edges[u, v]
                net.add_edge(u, v, label=ed.get("relationship",""), color=ed.get("color","#95A5A6"))

            return net.generate_html()
        except ImportError:
            return "<p style='color:white'>Pyvis not installed. Falling back to Matplotlib.</p>"

    # ── convenience ────────────────────────────────────────────────────────

    def render_from_neo4j(self, related: List[Dict], title: str = "Entity Graph") -> plt.Figure:
        G = self.builder.build_from_neo4j_result(related)
        nodes = [{"name": n, **G.nodes[n]} for n in G.nodes()]
        edges = [{"source": u, "target": v, **G.edges[u, v]} for u, v in G.edges()]
        return self.render_matplotlib(nodes, edges, title=title)
