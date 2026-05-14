"""
Load sample tech documents into the knowledge graph.
Run: python scripts/load_sample_data.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.graph_builder import GraphBuilder

SAMPLE_TEXTS = [
    ("cloud_tech.txt", Path("data/sample_docs/cloud_tech.txt").read_text()),
    ("ai_stack", (
        "LangChain integrates with OpenAI and uses Neo4j for graph-based retrieval. "
        "Python supports LangChain and FastAPI. NetworkX works with Python for graph analysis. "
        "Hugging Face works with LangChain to serve open-source models."
    )),
    ("devops", (
        "Kubernetes orchestrates Docker containers and runs on AWS and Azure. "
        "Terraform integrates with Kubernetes and Ansible for full infrastructure automation. "
        "Jenkins deploys using Docker and integrates with Terraform for CI/CD pipelines."
    )),
]


def main():
    builder = GraphBuilder()
    for name, text in SAMPLE_TEXTS:
        print(f"\nBuilding graph from: {name}")
        result = builder.build_from_text(text, source_doc=name)
        if result["success"]:
            print(f"  ✅ {result['entities_count']} entities, {result['relationships_count']} relationships")
        else:
            print(f"  ❌ {result['error']}")

    stats = builder.get_stats()
    print(f"\nGraph stats: {stats}")


if __name__ == "__main__":
    main()
