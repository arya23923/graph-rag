"""
Initialise Neo4j schema constraints and load sample data.
Run: python scripts/init_neo4j.py
Requires Neo4j running at NEO4J_URI (see .env or docker-compose.yml)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.database.neo4j_client import Neo4jClient


def main():
    client = Neo4jClient()
    if not client.connected:
        print("❌ Neo4j not connected. Start it with: docker-compose up -d")
        print("   Then re-run this script.")
        return

    print("✅ Neo4j connected")
    stats = client.get_graph_stats()
    print(f"   Current: {stats['entities']} entities, {stats['relationships']} relationships")

    if stats["entities"] > 0:
        ans = input("Graph already has data. Clear and reload? [y/N] ")
        if ans.lower() == "y":
            client.clear_graph()
            print("   Graph cleared.")

    print("\nLoading sample data...")
    import subprocess
    subprocess.run([sys.executable, "scripts/load_sample_data.py"])


if __name__ == "__main__":
    main()
