#!/usr/bin/env python3
"""
Check Neo4j Graph State
"""

from neo4j import GraphDatabase
import os


def check_neo4j_state():
    try:
        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        user = os.environ.get("NEO4J_USERNAME", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD", "password123")
        database = os.environ.get("NEO4J_DATABASE", "neo4j")

        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session(database=database) as session:
            print("Neo4j Graph State Analysis")
            print("=" * 40)

            # Get all node counts by label
            result = session.run("""
                MATCH (n) 
                RETURN labels(n) as labels, count(*) as count
                ORDER BY count DESC
            """)

            nodes_by_label = {}
            total_nodes = 0
            for record in result:
                labels = record["labels"]
                count = record["count"]
                total_nodes += count
                for label in labels:
                    if label not in nodes_by_label:
                        nodes_by_label[label] = 0
                    nodes_by_label[label] += count

            print(f"Total Nodes: {total_nodes}")
            print("\nNode Distribution by Label:")
            for label, count in sorted(
                nodes_by_label.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"  {label}: {count}")

            # Get relationship counts
            rel_result = session.run("""
                MATCH ()-[r]->() 
                RETURN type(r) as type, count(*) as count 
                ORDER BY count DESC 
                LIMIT 15
            """)

            print("\nTop Relationship Types:")
            total_rels = 0
            for record in rel_result:
                rel_type = record["type"]
                count = record["count"]
                total_rels += count
                print(f"  {rel_type}: {count}")

            print(f"\nTotal Relationships: {total_rels}")

            # Check for consolidation-specific metrics
            entity_result = session.run("""
                MATCH (n:__Entity__)
                RETURN count(n) as entity_count,
                       avg(size([(n)--() | 1])) as avg_degree,
                       count {MATCH (n) WHERE NOT (n)--()} as orphaned_nodes
            """)

            entity_data = entity_result.single()
            print(f"\nEntity Analysis:")
            print(f"  Total Entity Nodes: {entity_data['entity_count']}")
            print(f"  Average Degree: {entity_data['avg_degree']:.2f}")
            print(f"  Orphaned Entities: {entity_data['orphaned_nodes']}")

            # Check for duplicate detection potential
            dup_result = session.run("""
                MATCH (n:__Entity__)
                WHERE n.id IS NOT NULL
                WITH toLower(trim(n.id)) as normalized, count(*) as cnt
                WHERE cnt > 1
                RETURN count(*) as duplicate_groups, sum(cnt) as total_duplicates
            """)

            dup_data = dup_result.single()
            print(f"\nDuplicate Analysis:")
            print(f"  Duplicate Groups: {dup_data['duplicate_groups']}")
            print(f"  Total Duplicate Nodes: {dup_data['total_duplicates']}")

            # Check label distribution for entities
            label_result = session.run("""
                MATCH (n:__Entity__)
                UNWIND labels(n) as label
                WHERE label <> '__Entity__'
                RETURN label, count(*) as count
                ORDER BY count DESC
                LIMIT 10
            """)

            print(f"\nTop Entity Labels:")
            for record in label_result:
                print(f"  {record['label']}: {record['count']}")

            # Check consolidation metadata
            meta_result = session.run("""
                MATCH (n)
                WHERE n.consolidated_on IS NOT NULL OR n.merged_from IS NOT NULL
                RETURN count(*) as consolidated_nodes
            """)

            meta_data = meta_result.single()
            print(f"\nConsolidation Metadata:")
            print(f"  Consolidated Nodes: {meta_data['consolidated_nodes']}")

        driver.close()

    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure Neo4j is running: neo4j status")
        print("2. Check connection settings:")
        print("   URI:", os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
        print("   User:", os.environ.get("NEO4J_USERNAME", "neo4j"))
        print("   Database:", os.environ.get("NEO4J_DATABASE", "neo4j"))
        print("3. Check Neo4j logs for errors")
        print("4. Verify network connectivity to Neo4j port 7687")


if __name__ == "__main__":
    check_neo4j_state()
