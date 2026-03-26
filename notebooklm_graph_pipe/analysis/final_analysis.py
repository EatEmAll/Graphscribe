#!/usr/bin/env python3
"""
Neo4j Graph Consolidation Quality Review - Final Analysis
"""

from neo4j import GraphDatabase
import json
import os
from datetime import datetime

from notebooklm_graph_pipe.paths import CONSOLIDATION_REPORT_DIR


def analyze_consolidation_quality():
    """Perform comprehensive consolidation quality analysis."""

    try:
        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        user = os.environ.get("NEO4J_USERNAME", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD", "password123")
        database = os.environ.get("NEO4J_DATABASE", "neo4j")

        driver = GraphDatabase.driver(uri, auth=(user, password))

        with driver.session(database=database) as session:
            print("Neo4j Graph Consolidation Quality Review")
            print("=" * 50)
            print(f"Analysis Date: {datetime.now().isoformat()}")
            print(f"Database: {database}")
            print()

            # 1. Basic Graph Statistics
            print("1. BASIC GRAPH STATISTICS")
            print("-" * 30)

            total_nodes = session.run("MATCH (n) RETURN count(n) as count").single()[
                "count"
            ]
            total_rels = session.run(
                "MATCH ()-[r]->() RETURN count(r) as count"
            ).single()["count"]

            print(f"Total Nodes: {total_nodes}")
            print(f"Total Relationships: {total_rels}")

            # 2. Entity Analysis
            print("\n2. ENTITY ANALYSIS")
            print("-" * 30)

            entity_stats = session.run("""
                MATCH (e:__Entity__)
                RETURN count(e) as entity_count,
                       avg(size([(e)--() | 1])) as avg_degree,
                       count {MATCH (e) WHERE NOT (e)--()} as orphaned_entities,
                       count {MATCH (e) WHERE size(labels(e)) = 1} as single_label_entities
            """).single()

            print(f"Entity Nodes: {entity_stats['entity_count']}")
            print(f"Average Degree: {entity_stats['avg_degree']:.2f}")
            print(f"Orphaned Entities: {entity_stats['orphaned_entities']}")
            print(f"Single-label Entities: {entity_stats['single_label_entities']}")

            # 3. Label Distribution
            print("\n3. LABEL DISTRIBUTION")
            print("-" * 30)

            label_result = session.run("""
                MATCH (n:__Entity__)
                UNWIND labels(n) as label
                WHERE label <> '__Entity__'
                RETURN label, count(*) as count
                ORDER BY count DESC
            """)

            labels = []
            for record in label_result:
                labels.append({"label": record["label"], "count": record["count"]})
                print(f"  {record['label']}: {record['count']}")

            # 4. Duplicate Detection
            print("\n4. DUPLICATE ANALYSIS")
            print("-" * 30)

            duplicate_analysis = session.run("""
                MATCH (n:__Entity__)
                WHERE n.id IS NOT NULL
                WITH toLower(trim(n.id)) as normalized_name, 
                     count(*) as duplicate_count,
                     collect({eid: elementId(n), name: n.id, labels: labels(n)}) as nodes
                WHERE duplicate_count > 1
                RETURN normalized_name, duplicate_count, nodes
                ORDER BY duplicate_count DESC
                LIMIT 10
            """)

            duplicates = []
            for record in duplicate_analysis:
                duplicates.append(
                    {
                        "normalized_name": record["normalized_name"],
                        "duplicate_count": record["duplicate_count"],
                        "nodes": record["nodes"],
                    }
                )
                print(
                    f"  {record['normalized_name']}: {record['duplicate_count']} duplicates"
                )

            # 5. Relationship Patterns
            print("\n5. RELATIONSHIP PATTERNS")
            print("-" * 30)

            rel_patterns = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as rel_type, count(*) as count
                ORDER BY count DESC
                LIMIT 15
            """)

            relationships = []
            for record in rel_patterns:
                relationships.append(
                    {"type": record["rel_type"], "count": record["count"]}
                )
                print(f"  {record['rel_type']}: {record['count']}")

            # 6. Taxonomy Structure
            print("\n6. TAXONOMY STRUCTURE")
            print("-" * 30)

            taxonomy_result = session.run("""
                MATCH (n:__Entity__)-[r:SUBCLASS_OF|INSTANCE_OF|TYPE_OF|IS_A]->(m:__Entity__)
                RETURN type(r) as rel_type, count(*) as count
            """)

            taxonomy_rels = []
            for record in taxonomy_result:
                taxonomy_rels.append(
                    {"type": record["rel_type"], "count": record["count"]}
                )
                print(f"  {record['rel_type']}: {record['count']}")

            if not taxonomy_rels:
                print("  No taxonomy relationships found")

            # 7. Quality Metrics
            print("\n7. CONSOLIDATION QUALITY METRICS")
            print("-" * 30)

            # Calculate consolidation ratios
            entity_count = entity_stats["entity_count"]
            duplicate_groups = len(duplicates)
            total_duplicates = sum(d["duplicate_count"] for d in duplicates)

            consolidation_ratio = (
                (total_duplicates - duplicate_groups) / entity_count
                if entity_count > 0
                else 0
            )
            taxonomy_ratio = (
                sum(t["count"] for t in taxonomy_rels) / entity_count
                if entity_count > 0
                else 0
            )

            print(f"Consolidation Potential: {consolidation_ratio:.2%}")
            print(f"Taxonomy Coverage: {taxonomy_ratio:.2%}")
            print(f"Duplicate Groups: {duplicate_groups}")
            print(f"Total Duplicate Nodes: {total_duplicates}")

            # 8. Recommendations
            print("\n8. RECOMMENDATIONS")
            print("-" * 30)

            recommendations = []

            if duplicate_groups > 0:
                recommendations.append("Run Tier 1 consolidation for lemmatized merges")
                recommendations.append(
                    "Consider Tier 3 semantic merging for similar entities"
                )

            if len(taxonomy_rels) == 0:
                recommendations.append(
                    "Run taxonomy cleanup to establish hierarchical structure"
                )

            if entity_stats["orphaned_entities"] > 0:
                recommendations.append(
                    "Review orphaned entities for connection opportunities"
                )

            if entity_stats["single_label_entities"] > entity_count * 0.8:
                recommendations.append(
                    "Consider Tier 2 relabeling for better categorization"
                )

            if not recommendations:
                recommendations.append("Graph appears well-consolidated")
                recommendations.append("Monitor for ongoing optimization opportunities")

            for rec in recommendations:
                print(f"  • {rec}")

            # 9. Generate Report
            report = {
                "analysis_date": datetime.now().isoformat(),
                "database": database,
                "basic_stats": {
                    "total_nodes": total_nodes,
                    "total_relationships": total_rels,
                    "entity_nodes": entity_count,
                },
                "label_distribution": labels,
                "duplicates": duplicates,
                "relationships": relationships,
                "taxonomy": taxonomy_rels,
                "quality_metrics": {
                    "consolidation_potential": consolidation_ratio,
                    "taxonomy_coverage": taxonomy_ratio,
                    "duplicate_groups": duplicate_groups,
                    "orphaned_entities": entity_stats["orphaned_entities"],
                },
                "recommendations": recommendations,
            }

            # Save report
            report_file = CONSOLIDATION_REPORT_DIR / f"consolidation_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report_file.parent.mkdir(parents=True, exist_ok=True)
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            print(f"\n9. REPORT GENERATION")
            print("-" * 30)
            print(f"Detailed report saved to: {report_file}")

            return report

    except Exception as e:
        print(f"Error analyzing consolidation quality: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure Neo4j is running")
        print("2. Check connection settings")
        print("3. Verify database permissions")
        return None


if __name__ == "__main__":
    analyze_consolidation_quality()
