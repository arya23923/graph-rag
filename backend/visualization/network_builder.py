"""
NetworkBuilder — constructs a NetworkX DiGraph from entity/relationship data.
Tech-domain node colours and edge labels.
"""
import networkx as nx
from typing import List, Dict, Optional

# Node colour by entity type
TYPE_COLORS = {
    "TECH":    "#4A90D9",
    "ORG":     "#E8A838",
    "PERSON":  "#50C878",
    "PRODUCT": "#9B59B6",
    "GPE":     "#E74C3C",
    "UNKNOWN": "#95A5A6",
}

# Edge colour by relationship type
REL_COLORS = {
    "INTEGRATES_WITH": "#2ECC71",
    "RUNS_ON":         "#3498DB",
    "ORCHESTRATES":    "#E67E22",
    "BUILT_WITH":      "#9B59B6",
    "USES":            "#1ABC9C",
    "CONNECTS_TO":     "#F39C12",
    "SUPPORTS":        "#27AE60",
    "WORKS_WITH":      "#2980B9",
    "DEPLOYS_ON":      "#E74C3C",
    "CO_OCCURS_WITH":  "#BDC3C7",
}


class NetworkBuilder:
    def build(self, nodes: List[Dict], edges: List[Dict]) -> nx.DiGraph:
        G = nx.DiGraph()

        for node in nodes:
            name = node["name"]
            ntype = node.get("type", "UNKNOWN")
            G.add_node(
                name,
                type=ntype,
                confidence=node.get("confidence", 0.5),
                color=TYPE_COLORS.get(ntype, TYPE_COLORS["UNKNOWN"]),
                size=max(10, int(node.get("confidence", 0.5) * 30)),
            )

        for edge in edges:
            rel = edge.get("relationship", "RELATED_TO")
            G.add_edge(
                edge["source"],
                edge["target"],
                relationship=rel,
                color=REL_COLORS.get(rel, "#95A5A6"),
                weight=1,
            )

        return G

    def build_from_neo4j_result(self, related: List[Dict]) -> nx.DiGraph:
        """Build graph directly from get_related_entities() output."""
        nodes_map: Dict[str, Dict] = {}
        edges: List[Dict] = []
        for r in related:
            for name in (r["source"], r["target"]):
                if name not in nodes_map:
                    nodes_map[name] = {"name": name, "type": r.get("target_type","TECH"), "confidence": 0.85}
            edges.append({"source": r["source"], "target": r["target"], "relationship": r["relationship"]})
        return self.build(list(nodes_map.values()), edges)

    def get_graph_metrics(self, G: nx.DiGraph) -> Dict:
        if G.number_of_nodes() == 0:
            return {}
        return {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "density": round(nx.density(G), 4),
            "most_connected": sorted(G.degree(), key=lambda x: x[1], reverse=True)[:5],
            "is_weakly_connected": nx.is_weakly_connected(G) if G.number_of_nodes() > 1 else True,
        }

    def to_dict(self, G: nx.DiGraph) -> Dict:
        return {
            "nodes": [{"id": n, **G.nodes[n]} for n in G.nodes],
            "edges": [{"source": u, "target": v, **G.edges[u, v]} for u, v in G.edges],
        }
