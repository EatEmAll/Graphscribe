#!/usr/bin/env python3
"""
Neo4j Graph Consolidation Quality Review

This script connects to Neo4j and performs a comprehensive review of graph consolidation quality,
including node/relationship counts, duplicate detection, and consolidation metrics.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any

from notebooklm_graph_pipe.paths import CONSOLIDATION_REPORT_DIR

try:
    from neo4j import GraphDatabase
except ImportError:
    print("Neo4j driver not available. Please install: pip install neo4j")
    exit(1)

# Default connection settings
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password123")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")


class GraphConsolidationReviewer:
    def __init__(self, uri: str, user: str, password: str, database: str):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None

    def connect(self) -> bool:
        """Establish connection to Neo4j."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )
            # Test connection
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            return True
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            return False

    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()

    def get_basic_counts(self) -> Dict[str, Any]:
        """Get basic node and relationship counts."""
        queries = {
            "total_nodes": "MATCH (n) RETURN count(n) as count",
            "entity_nodes": "MATCH (n:__Entity__) RETURN count(n) as count",
            "chunk_nodes": "MATCH (n:__Chunk__) RETURN count(n) as count",
            "document_nodes": "MATCH (n:__Document__) RETURN count(n) as count",
            "community_nodes": "MATCH (n:__Community__) RETURN count(n) as count",
            "total_relationships": "MATCH ()-[r]->() RETURN count(r) as count",
            "entity_entity_rels": "MATCH (n:__Entity__)-[r]-(m:__Entity__) RETURN count(r) as count",
            "chunk_entity_rels": "MATCH (c:__Chunk__)-[r]-(e:__Entity__) RETURN count(r) as count",
        }

        results = {}
        with self.driver.session(database=self.database) as session:
            for key, query in queries.items():
                try:
                    result = session.run(query)
                    results[key] = result.single()["count"]
                except Exception as e:
                    results[key] = f"Error: {e}"
        return results

    def get_label_distribution(self) -> Dict[str, int]:
        """Get distribution of labels across entity nodes."""
        query = """
        MATCH (n:__Entity__)
        UNWIND labels(n) AS label
        WHERE label <> '__Entity__'
        RETURN label, count(*) as count
        ORDER BY count DESC
        """

        results = {}
        with self.driver.session(database=self.database) as session:
            try:
                result = session.run(query)
                for record in result:
                    results[record["label"]] = record["count"]
            except Exception as e:
                results["error"] = str(e)
        return results

    def detect_duplicate_nodes(self) -> Dict[str, Any]:
        """Detect potential duplicate nodes based on name similarity."""
        query = """
        MATCH (n:__Entity__)
        WHERE n.id IS NOT NULL
        WITH toLower(trim(n.id)) as normalized_name, collect({eid: elementId(n), name: n.id, labels: labels(n)}) as nodes
        WHERE size(nodes) > 1
        RETURN normalized_name, size(nodes) as duplicate_count, nodes
        ORDER BY duplicate_count DESC
        LIMIT 20
        """

        duplicates = []
        with self.driver.session(database=self.database) as session:
            try:
                result = session.run(query)
                for record in result:
                    duplicates.append(
                        {
                            "normalized_name": record["normalized_name"],
                            "duplicate_count": record["duplicate_count"],
                            "nodes": record["nodes"],
                        }
                    )
            except Exception as e:
                return {"error": str(e)}

        return {
            "duplicate_groups": duplicates,
            "total_groups": len(duplicates),
            "total_duplicate_nodes": sum(d["duplicate_count"] for d in duplicates),
        }

    def get_consolidation_metrics(self) -> Dict[str, Any]:
        """Get metrics related to consolidation quality."""
        queries = {
            "concept_only_nodes": """
                MATCH (n:__Entity__)
                WHERE NOT n:__Entity__ IN labels(n) AND size(labels(n)) = 1
                RETURN count(n) as count
            """,
            "nodes_with_multiple_labels": """
                MATCH (n:__Entity__)
                WHERE size([l IN labels(n) WHERE l <> '__Entity__']) > 1
                RETURN count(n) as count
            """,
            "orphaned_entities": """
                MATCH (n:__Entity__)
                WHERE NOT (n)--()
                RETURN count(n) as count
            """,
            "high_degree_entities": """
                MATCH (n:__Entity__)
                WITH n, size([(n)--() | 1]) as degree
                WHERE degree > 10
                RETURN count(n) as count
            """,
            "taxonomy_relationships": """
                MATCH (n:__Entity__)-[r:SUBCLASS_OF|INSTANCE_OF|TYPE_OF|IS_A]->(m:__Entity__)
                RETURN count(r) as count
            """,
            "entities_without_taxonomy": """
                MATCH (n:__Entity__)
                WHERE NOT (n)-[:SUBCLASS_OF|INSTANCE_OF|TYPE_OF|IS_A]-()
                RETURN count(n) as count
            """,
        }

        results = {}
        with self.driver.session(database=self.database) as session:
            for key, query in queries.items():
                try:
                    result = session.run(query)
                    results[key] = result.single()["count"]
                except Exception as e:
                    results[key] = f"Error: {e}"
        return results

    def get_relationship_patterns(self) -> Dict[str, Any]:
        """Analyze relationship patterns for consolidation quality."""
        query = """
        MATCH ()-[r]->()
        RETURN type(r) as rel_type, count(*) as count
        ORDER BY count DESC
        LIMIT 20
        """

        patterns = {}
        with self.driver.session(database=self.database) as session:
            try:
                result = session.run(query)
                for record in result:
                    patterns[record["rel_type"]] = record["count"]
            except Exception as e:
                patterns["error"] = str(e)
        return patterns

    def get_entity_quality_issues(self) -> Dict[str, Any]:
        """Identify potential quality issues with entities."""
        queries = {
            "empty_names": """
                MATCH (n:__Entity__)
                WHERE n.id IS NULL OR trim(n.id) = ''
                RETURN count(n) as count
            """,
            "very_long_names": """
                MATCH (n:__Entity__)
                WHERE n.id IS NOT NULL AND size(n.id) > 100
                RETURN count(n) as count
            """,
            "single_character_names": """
                MATCH (n:__Entity__)
                WHERE n.id IS NOT NULL AND size(trim(n.id)) = 1
                RETURN count(n) as count
            """,
            "duplicate_names_exact": """
                MATCH (n:__Entity__)
                WHERE n.id IS NOT NULL
                WITH n.id as name, count(*) as cnt
                WHERE cnt > 1
                RETURN count(name) as count
            """,
        }

        results = {}
        with self.driver.session(database=self.database) as session:
            for key, query in queries.items():
                try:
                    result = session.run(query)
                    results[key] = result.single()["count"]
                except Exception as e:
                    results[key] = f"Error: {e}"
        return results

    def generate_consolidation_report(self) -> Dict[str, Any]:
        """Generate a comprehensive consolidation quality report."""
        print("Generating Neo4j Graph Consolidation Quality Report...")
        print(f"Connection: {self.uri}")
        print(f"Database: {self.database}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print("-" * 50)

        report = {
            "metadata": {
                "connection_uri": self.uri,
                "database": self.database,
                "timestamp": datetime.now().isoformat(),
                "review_version": "1.0",
            },
            "basic_counts": self.get_basic_counts(),
            "label_distribution": self.get_label_distribution(),
            "duplicate_analysis": self.detect_duplicate_nodes(),
            "consolidation_metrics": self.get_consolidation_metrics(),
            "relationship_patterns": self.get_relationship_patterns(),
            "quality_issues": self.get_entity_quality_issues(),
        }

        return report


def main():
    """Main function to run the consolidation review."""
    reviewer = GraphConsolidationReviewer(
        NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
    )

    if not reviewer.connect():
        print("Failed to connect to Neo4j. Please check:")
        print("1. Neo4j is running")
        print("2. Connection settings are correct")
        print("3. Network connectivity")
        return

    try:
        report = reviewer.generate_consolidation_report()

        # Save report to file
        report_file = CONSOLIDATION_REPORT_DIR / f"graph_consolidation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n" + "=" * 60)
        print("CONSOLIDATION QUALITY REPORT SUMMARY")
        print("=" * 60)

        # Basic counts
        counts = report["basic_counts"]
        print(f"Total Nodes: {counts.get('total_nodes', 'N/A')}")
        print(f"Entity Nodes: {counts.get('entity_nodes', 'N/A')}")
        print(f"Chunk Nodes: {counts.get('chunk_nodes', 'N/A')}")
        print(f"Document Nodes: {counts.get('document_nodes', 'N/A')}")
        print(f"Total Relationships: {counts.get('total_relationships', 'N/A')}")

        # Consolidation metrics
        metrics = report["consolidation_metrics"]
        print(f"\nConsolidation Metrics:")
        print(f"  Concept-only nodes: {metrics.get('concept_only_nodes', 'N/A')}")
        print(
            f"  Multi-label entities: {metrics.get('nodes_with_multiple_labels', 'N/A')}"
        )
        print(f"  Orphaned entities: {metrics.get('orphaned_entities', 'N/A')}")
        print(
            f"  Entities without taxonomy: {metrics.get('entities_without_taxonomy', 'N/A')}"
        )

        # Duplicate analysis
        dup_analysis = report["duplicate_analysis"]
        print(f"\nDuplicate Analysis:")
        print(f"  Duplicate groups: {dup_analysis.get('total_groups', 'N/A')}")
        print(
            f"  Total duplicate nodes: {dup_analysis.get('total_duplicate_nodes', 'N/A')}"
        )

        # Quality issues
        issues = report["quality_issues"]
        print(f"\nQuality Issues:")
        print(f"  Empty names: {issues.get('empty_names', 'N/A')}")
        print(f"  Very long names: {issues.get('very_long_names', 'N/A')}")
        print(f"  Duplicate exact names: {issues.get('duplicate_names_exact', 'N/A')}")

        # Top labels
        labels = report["label_distribution"]
        print(f"\nTop 10 Labels:")
        top_labels = list(labels.items())[:10]
        for label, count in top_labels:
            print(f"  {label}: {count}")

        print(f"\nDetailed report saved to: {report_file}")

    finally:
        reviewer.close()


if __name__ == "__main__":
    main()
