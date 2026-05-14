"""
Neo4j Client — full graph operations for Tech Domain Graph RAG
"""
import logging
from typing import List, Dict
from backend.config import config

logger = logging.getLogger(__name__)


class Neo4jClient:
    def __init__(self):
        self._driver = None
        self.connected = False
        self._connect()

    def _connect(self):
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
            )
            self._driver.verify_connectivity()
            self.connected = True
            logger.info("Neo4j connected at %s", config.NEO4J_URI)
            self._create_constraints()
        except Exception as exc:
            logger.warning("Neo4j unavailable (%s). Running in mock mode.", exc)
            self.connected = False

    def _create_constraints(self):
        constraints = [
            "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
        ]
        with self._driver.session() as session:
            for cql in constraints:
                try:
                    session.run(cql)
                except Exception:
                    pass

    def close(self):
        if self._driver:
            self._driver.close()

    # ── write ──────────────────────────────────────────────────────────────

    def create_knowledge_graph(self, entities: List[Dict], relationships: List[Dict]) -> Dict:
        if not self.connected:
            return self._mock_result(entities, relationships)

        entities_created = 0
        relationships_created = 0
        with self._driver.session() as session:
            for ent in entities:
                r = session.run(
                    "MERGE (e:Entity {name:$name}) "
                    "ON CREATE SET e.type=$type, e.source=$source, e.confidence=$confidence, e.created_at=timestamp() "
                    "ON MATCH SET  e.source=$source RETURN e",
                    name=ent["name"], type=ent.get("type","UNKNOWN"),
                    source=ent.get("source","unknown"), confidence=ent.get("confidence",0.5),
                )
                if r.single(): entities_created += 1

            for rel in relationships:
                rel_type = rel.get("relationship","RELATED_TO").replace(" ","_").upper()
                try:
                    r = session.run(
                        f"MATCH (src:Entity {{name:$src}}),(tgt:Entity {{name:$tgt}}) "
                        f"MERGE (src)-[r:{rel_type}]->(tgt) "
                        "ON CREATE SET r.sentence=$sentence, r.source_doc=$source_doc, r.confidence=$confidence RETURN r",
                        src=rel["source"], tgt=rel["target"],
                        sentence=rel.get("sentence",""), source_doc=rel.get("source_doc","unknown"),
                        confidence=rel.get("confidence",0.5),
                    )
                    if r.single(): relationships_created += 1
                except Exception as e:
                    logger.debug("Rel merge failed: %s", e)

        return {"status":"success","entities_created":entities_created,"relationships_created":relationships_created}

    # ── read ───────────────────────────────────────────────────────────────

    def get_related_entities(self, entity_name: str, depth: int = 2) -> List[Dict]:
        if not self.connected:
            return self._mock_related(entity_name)
        with self._driver.session() as session:
            result = session.run(
                f"MATCH path=(start:Entity {{name:$name}})-[*1..{depth}]-(related:Entity) "
                "WITH path,related,[r IN relationships(path)|type(r)] AS rel_types,length(path) AS hop "
                "RETURN nodes(path)[-2].name AS source, rel_types[-1] AS relationship, "
                "related.name AS target, related.type AS target_type, hop ORDER BY hop,target LIMIT 100",
                name=entity_name,
            )
            return [{"source":r["source"],"relationship":r["relationship"],"target":r["target"],
                     "target_type":r["target_type"],"depth":r["hop"]} for r in result]

    def get_entity_subgraph(self, entity_name: str, depth: int = 2) -> Dict:
        if not self.connected:
            return self._mock_subgraph(entity_name)
        nodes, edges = {}, []
        with self._driver.session() as session:
            nr = session.run(
                f"MATCH path=(start:Entity {{name:$name}})-[*0..{depth}]-(n:Entity) "
                "UNWIND nodes(path) AS node WITH DISTINCT node "
                "RETURN node.name AS name, node.type AS type, node.confidence AS confidence",
                name=entity_name,
            )
            for rec in nr:
                nodes[rec["name"]] = {"name":rec["name"],"type":rec["type"] or "UNKNOWN","confidence":rec["confidence"] or 0.5}

            er = session.run(
                f"MATCH (a:Entity {{name:$name}})-[r*1..{depth}]-(b:Entity) "
                "WITH a,b,r UNWIND r AS rel "
                "RETURN startNode(rel).name AS source, type(rel) AS relationship, endNode(rel).name AS target",
                name=entity_name,
            )
            seen = set()
            for rec in er:
                key = (rec["source"],rec["relationship"],rec["target"])
                if key not in seen:
                    seen.add(key)
                    edges.append({"source":rec["source"],"relationship":rec["relationship"],"target":rec["target"]})

        return {"nodes":list(nodes.values()),"edges":edges}

    def search_entities(self, query: str, limit: int = 10) -> List[Dict]:
        if not self.connected:
            return []
        with self._driver.session() as session:
            result = session.run(
                "MATCH (e:Entity) WHERE toLower(e.name) CONTAINS toLower($query) "
                "RETURN e.name AS name, e.type AS type, e.confidence AS confidence "
                "ORDER BY e.confidence DESC LIMIT $limit",
                query=query, limit=limit,
            )
            return [dict(r) for r in result]

    def find_path(self, source: str, target: str, max_depth: int = 4) -> List[Dict]:
        if not self.connected:
            return []
        with self._driver.session() as session:
            result = session.run(
                f"MATCH (src:Entity {{name:$src}}),(tgt:Entity {{name:$tgt}}) "
                f"MATCH path=shortestPath((src)-[*..{max_depth}]-(tgt)) "
                "RETURN [n IN nodes(path)|n.name] AS node_names,"
                "[r IN relationships(path)|type(r)] AS rel_types, length(path) AS path_length",
                src=source, tgt=target,
            )
            return [{"nodes":r["node_names"],"relationships":r["rel_types"],"length":r["path_length"]} for r in result]

    def get_all_entities(self, limit: int = 200) -> List[Dict]:
        if not self.connected:
            return self._mock_all_entities()
        with self._driver.session() as session:
            result = session.run(
                "MATCH (e:Entity) RETURN e.name AS name, e.type AS type, e.confidence AS confidence LIMIT $limit",
                limit=limit,
            )
            return [dict(r) for r in result]

    def get_graph_stats(self) -> Dict:
        if not self.connected:
            return {"entities":0,"relationships":0,"connected":False}
        with self._driver.session() as session:
            nodes = session.run("MATCH (e:Entity) RETURN count(e) AS cnt").single()["cnt"]
            rels  = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()["cnt"]
            return {"entities":nodes,"relationships":rels,"connected":True}

    def clear_graph(self) -> Dict:
        if not self.connected:
            return {"status":"mock - nothing to clear"}
        with self._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        return {"status":"cleared"}

    # ── mock helpers ───────────────────────────────────────────────────────

    def _mock_result(self, entities, relationships):
        return {"status":"success (mock)","entities_created":len(entities),"relationships_created":len(relationships)}

    MOCK_GRAPH = {
        "Kubernetes":  [("Kubernetes","ORCHESTRATES","Docker"),("Kubernetes","RUNS_ON","AWS"),("Kubernetes","INTEGRATES_WITH","Terraform")],
        "Docker":      [("Docker","INTEGRATES_WITH","Kubernetes"),("Docker","BUILT_WITH","Python"),("Docker","RUNS_ON","AWS")],
        "FastAPI":     [("FastAPI","BUILT_WITH","Python"),("FastAPI","CONNECTS_TO","Neo4j"),("FastAPI","USES","Pydantic")],
        "Neo4j":       [("Neo4j","INTEGRATES_WITH","LangChain"),("Neo4j","SUPPORTS","Python"),("Neo4j","CONNECTS_TO","FastAPI")],
        "LangChain":   [("LangChain","INTEGRATES_WITH","OpenAI"),("LangChain","USES","Neo4j"),("LangChain","SUPPORTS","Python")],
        "AWS":         [("AWS","SUPPORTS","Kubernetes"),("AWS","INTEGRATES_WITH","Terraform"),("AWS","RUNS_ON","Docker")],
        "Python":      [("Python","SUPPORTS","FastAPI"),("Python","SUPPORTS","LangChain"),("Python","USES","NetworkX")],
        "Kafka":       [("Kafka","INTEGRATES_WITH","Spark"),("Kafka","RUNS_ON","AWS"),("Kafka","CONNECTS_TO","Elasticsearch")],
        "Spark":       [("Spark","INTEGRATES_WITH","Kafka"),("Spark","RUNS_ON","AWS"),("Spark","USES","Python")],
    }

    def _mock_related(self, entity_name: str) -> List[Dict]:
        triples = self.MOCK_GRAPH.get(entity_name, [(entity_name,"CO_OCCURS_WITH","Python")])
        return [{"source":s,"relationship":r,"target":t,"target_type":"TECH","depth":1} for s,r,t in triples]

    def _mock_subgraph(self, entity_name: str) -> Dict:
        triples = self.MOCK_GRAPH.get(entity_name, [(entity_name,"CO_OCCURS_WITH","Python")])
        node_names = {entity_name}
        edges = []
        for s,r,t in triples:
            node_names.add(s); node_names.add(t)
            edges.append({"source":s,"relationship":r,"target":t})
        nodes = [{"name":n,"type":"TECH","confidence":0.9} for n in node_names]
        return {"nodes":nodes,"edges":edges}

    def _mock_all_entities(self) -> List[Dict]:
        names = {"Kubernetes","Docker","FastAPI","Neo4j","LangChain","AWS","Python","Kafka","Spark",
                 "Redis","MongoDB","Terraform","Ansible","Jenkins","NetworkX","OpenAI","React"}
        return [{"name":n,"type":"TECH","confidence":0.9} for n in names]
