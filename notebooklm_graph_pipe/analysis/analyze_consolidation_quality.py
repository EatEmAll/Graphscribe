#!/usr/bin/env python3
"""
Neo4j Graph Consolidation Quality Review - Enhanced Version

This script provides a comprehensive review of graph consolidation quality using Neo4j.
It includes both live analysis (when Neo4j is running) and static analysis based on
consolidation logs and patterns.
"""

import json
import os
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from notebooklm_graph_pipe.paths import CONSOLIDATION_REPORT_DIR, REPO_ROOT


class ConsolidationQualityAnalyzer:
    def __init__(self, repo_root: str = None):
        self.repo_root = Path(repo_root).resolve() if repo_root else REPO_ROOT
        self.runs_dir = self.repo_root / "runs"
        self.consolidation_logs = []

    def find_consolidation_runs(self) -> List[Path]:
        """Find recent consolidation run directories."""
        if not self.runs_dir.exists():
            return []

        run_dirs = [d for d in self.runs_dir.iterdir() if d.is_dir()]
        # Sort by modification time, most recent first
        run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return run_dirs[:5]  # Last 5 runs

    def analyze_consolidation_logs(self, run_dir: Path) -> Dict[str, Any]:
        """Analyze consolidation logs from a specific run."""
        analysis = {
            "run_dir": str(run_dir),
            "timestamp": datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(),
            "tier1_analysis": {},
            "tier2_analysis": {},
            "tier3_analysis": {},
            "taxonomy_analysis": {},
            "overall_metrics": {},
        }

        # Analyze Tier 1 logs
        tier1_log = run_dir / "tier1.log"
        if tier1_log.exists():
            with open(tier1_log, "r") as f:
                content = f.read()
                analysis["tier1_analysis"] = {
                    "log_exists": True,
                    "total_entities": self._extract_count(
                        content, r"Total entities: (\d+)"
                    ),
                    "merge_candidates": self._extract_count(
                        content, r"Merge candidate groups: (\d+)"
                    ),
                    "merged_nodes": self._extract_count(
                        content, r"Merged (\d+) duplicate nodes"
                    ),
                    "would_merge": self._extract_count(
                        content, r"Would merge (\d+) nodes"
                    ),
                    "dry_run": "DRY RUN" in content,
                }

        # Analyze Tier 2 logs and summaries
        tier2_summary = run_dir / "tier2_summary.json"
        if tier2_summary.exists():
            try:
                with open(tier2_summary, "r") as f:
                    tier2_data = json.load(f)
                    analysis["tier2_analysis"] = {
                        "summary_exists": True,
                        "processed_nodes": tier2_data.get("processed_nodes", 0),
                        "relabelled_nodes": tier2_data.get("relabelled_nodes", 0),
                        "labels_added": tier2_data.get("labels_added", []),
                        "decisions_made": tier2_data.get("decisions_made", 0),
                        "errors": tier2_data.get("errors", []),
                    }
            except Exception as e:
                analysis["tier2_analysis"] = {"error": str(e)}

        # Analyze Tier 3 logs and summaries
        tier3_summary = run_dir / "tier3_summary.json"
        if tier3_summary.exists():
            try:
                with open(tier3_summary, "r") as f:
                    tier3_data = json.load(f)
                    analysis["tier3_analysis"] = {
                        "summary_exists": True,
                        "candidates_evaluated": tier3_data.get(
                            "candidates_evaluated", 0
                        ),
                        "merges_performed": tier3_data.get("merges_performed", 0),
                        "alias_acceptance_rate": tier3_data.get(
                            "alias_acceptance_rate", 0
                        ),
                        "judged_pairs": tier3_data.get("judged_pairs", 0),
                        "threshold_used": tier3_data.get("threshold_used", 0),
                    }
            except Exception as e:
                analysis["tier3_analysis"] = {"error": str(e)}

        # Analyze Taxonomy logs
        taxonomy_summary = run_dir / "taxonomy_summary.json"
        if taxonomy_summary.exists():
            try:
                with open(taxonomy_summary, "r") as f:
                    tax_data = json.load(f)
                    analysis["taxonomy_analysis"] = {
                        "summary_exists": True,
                        "nodes_processed": tax_data.get("nodes_processed", 0),
                        "taxonomy_relations_added": tax_data.get(
                            "taxonomy_relations_added", 0
                        ),
                        "concepts_without_taxonomy": tax_data.get(
                            "concepts_without_taxonomy", 0
                        ),
                        "concept_only_ratio": tax_data.get("concept_only_ratio", 0),
                    }
            except Exception as e:
                analysis["taxonomy_analysis"] = {"error": str(e)}

        # Analyze overall state
        state_file = run_dir / "state.json"
        if state_file.exists():
            try:
                with open(state_file, "r") as f:
                    state_data = json.load(f)
                    analysis["overall_metrics"] = {
                        "iterations": len(state_data.get("iterations", [])),
                        "final_concept_ratio": state_data.get("iterations", [{}])[-1]
                        .get("kpis", {})
                        .get("concept_only_ratio"),
                        "final_duplicate_rate": state_data.get("iterations", [{}])[-1]
                        .get("kpis", {})
                        .get("duplicate_candidate_rate"),
                        "stop_reason": state_data.get("stop_reason"),
                    }
            except Exception as e:
                analysis["overall_metrics"] = {"error": str(e)}

        return analysis

    def _extract_count(self, content: str, pattern: str) -> Optional[int]:
        """Extract count from log content using regex."""
        import re

        match = re.search(pattern, content)
        return int(match.group(1)) if match else None

    def analyze_consolidation_patterns(self) -> Dict[str, Any]:
        """Analyze consolidation patterns across all runs."""
        runs = self.find_consolidation_runs()
        if not runs:
            return {
                "status": "no_runs_found",
                "message": "No consolidation runs found. Run consolidation first.",
                "suggested_commands": [
                    "python scripts/consolidation/consolidate_self_improving.py",
                    "python scripts/consolidation/consolidate_tier1_lemmatize.py --dry-run",
                    "python scripts/consolidation/consolidate_tier2_relabel.py --dry-run",
                ],
            }

        all_analyses = []
        for run_dir in runs:
            analysis = self.analyze_consolidation_logs(run_dir)
            all_analyses.append(analysis)

        # Aggregate insights
        total_merges = sum(
            [
                a.get("tier1_analysis", {}).get("merged_nodes", 0)
                or a.get("tier1_analysis", {}).get("would_merge", 0)
                for a in all_analyses
            ]
        )

        total_relabels = sum(
            [
                a.get("tier2_analysis", {}).get("relabelled_nodes", 0)
                for a in all_analyses
            ]
        )

        total_semantic_merges = sum(
            [
                a.get("tier3_analysis", {}).get("merges_performed", 0)
                for a in all_analyses
            ]
        )

        taxonomy_relations = sum(
            [
                a.get("taxonomy_analysis", {}).get("taxonomy_relations_added", 0)
                for a in all_analyses
            ]
        )

        return {
            "status": "analysis_complete",
            "total_runs_analyzed": len(runs),
            "runs": all_analyses,
            "aggregated_metrics": {
                "total_tier1_merges": total_merges,
                "total_tier2_relabels": total_relabels,
                "total_tier3_merges": total_semantic_merges,
                "total_taxonomy_relations": taxonomy_relations,
            },
            "consolidation_quality_indicators": {
                "tier1_effectiveness": "Good"
                if total_merges > 0
                else "No merges performed",
                "tier2_coverage": "Good"
                if total_relabels > 0
                else "No relabels performed",
                "tier3_precision": "Good"
                if total_semantic_merges > 0
                else "No semantic merges",
                "taxonomy_structure": "Good"
                if taxonomy_relations > 0
                else "No taxonomy relations",
            },
        }

    def check_neo4j_connection(self) -> Dict[str, Any]:
        """Check if Neo4j is running and accessible."""
        try:
            from neo4j import GraphDatabase

            # Try to connect with default settings
            uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
            user = os.environ.get("NEO4J_USERNAME", "neo4j")
            password = os.environ.get("NEO4J_PASSWORD", "password123")
            database = os.environ.get("NEO4J_DATABASE", "neo4j")

            driver = GraphDatabase.driver(uri, auth=(user, password))
            with driver.session(database=database) as session:
                result = session.run("RETURN 1 as test")
                test_result = result.single()["test"]

            driver.close()

            return {
                "status": "connected",
                "uri": uri,
                "database": database,
                "test_result": test_result,
                "message": "Neo4j is running and accessible",
            }
        except Exception as e:
            return {
                "status": "not_connected",
                "error": str(e),
                "message": "Neo4j is not running or not accessible",
                "suggestions": [
                    "Start Neo4j: neo4j start",
                    "Check Neo4j service status",
                    "Verify connection settings in environment variables",
                    "Check Neo4j logs for errors",
                ],
            }

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive consolidation quality report."""
        print("Neo4j Graph Consolidation Quality Review")
        print("=" * 50)

        # Check Neo4j connection
        neo4j_status = self.check_neo4j_connection()
        print(f"Neo4j Status: {neo4j_status['message']}")

        # Analyze consolidation patterns
        consolidation_analysis = self.analyze_consolidation_patterns()

        # Generate final report
        report = {
            "timestamp": datetime.now().isoformat(),
            "neo4j_connection": neo4j_status,
            "consolidation_analysis": consolidation_analysis,
            "recommendations": self._generate_recommendations(consolidation_analysis),
            "next_steps": self._get_next_steps(consolidation_analysis),
        }

        # Save report
        report_file = CONSOLIDATION_REPORT_DIR / f"consolidation_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return report

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        if analysis.get("status") == "no_runs_found":
            recommendations.extend(
                [
                    "Run initial consolidation to establish baseline",
                    "Start with Tier 1 lemmatization for basic deduplication",
                    "Use dry-run mode first to preview changes",
                ]
            )
        else:
            # Analyze consolidation effectiveness
            metrics = analysis.get("aggregated_metrics", {})

            if metrics.get("total_tier1_merges", 0) == 0:
                recommendations.append(
                    "Consider running Tier 1 consolidation for basic lemmatized merges"
                )

            if metrics.get("total_tier2_relabels", 0) < 10:
                recommendations.append(
                    "Tier 2 relabeling seems limited - review label catalog and parameters"
                )

            if metrics.get("total_tier3_merges", 0) == 0:
                recommendations.append(
                    "No semantic merges detected - consider adjusting Tier 3 threshold"
                )

            if metrics.get("total_taxonomy_relations", 0) < 5:
                recommendations.append(
                    "Limited taxonomy structure - run taxonomy cleanup for better organization"
                )

        if not recommendations:
            recommendations.append(
                "Consolidation appears to be working well - monitor for ongoing optimization"
            )

        return recommendations

    def _get_next_steps(self, analysis: Dict[str, Any]) -> Dict[str, List[str]]:
        """Provide next steps for consolidation improvement."""
        steps = {"immediate": [], "short_term": [], "long_term": []}

        # Immediate steps
        if analysis.get("status") == "no_runs_found":
            steps["immediate"] = [
                "python scripts/consolidation/consolidate_tier1_lemmatize.py --dry-run",
                "Review dry-run output",
                "python scripts/consolidation/consolidate_tier1_lemmatize.py",
            ]
        else:
            steps["immediate"] = [
                "Review latest consolidation logs",
                "Check Neo4j connection and service status",
                "Verify consolidation parameters are appropriate",
            ]

        # Short-term steps
        steps["short_term"] = [
            "Run full consolidation pipeline: python scripts/consolidation/consolidate_self_improving.py",
            "Monitor consolidation metrics during execution",
            "Review generated taxonomy decisions",
            "Validate consolidation results in Neo4j browser",
        ]

        # Long-term steps
        steps["long_term"] = [
            "Establish monitoring for consolidation quality metrics",
            "Set up automated consolidation runs with appropriate thresholds",
            "Create validation scripts for ongoing quality checks",
            "Document consolidation patterns and lessons learned",
        ]

        return steps


def main():
    """Main function to run the consolidation quality review."""
    analyzer = ConsolidationQualityAnalyzer()
    report = analyzer.generate_comprehensive_report()

    # Print summary
    print("\nCONSOLIDATION QUALITY SUMMARY")
    print("=" * 40)

    neo4j_status = report["neo4j_connection"]
    print(f"Neo4j Connection: {neo4j_status['status']}")

    consolidation = report["consolidation_analysis"]
    if consolidation.get("status") == "analysis_complete":
        print(f"Total Runs Analyzed: {consolidation['total_runs_analyzed']}")
        metrics = consolidation["aggregated_metrics"]
        print(f"Tier 1 Merges: {metrics.get('total_tier1_merges', 0)}")
        print(f"Tier 2 Relabels: {metrics.get('total_tier2_relabels', 0)}")
        print(f"Tier 3 Semantic Merges: {metrics.get('total_tier3_merges', 0)}")
        print(f"Taxonomy Relations: {metrics.get('total_taxonomy_relations', 0)}")

    print("\nRECOMMENDATIONS:")
    for rec in report["recommendations"]:
        print(f"  â€¢ {rec}")

    print(f"\nReport saved to: {report_file}")

    return report


if __name__ == "__main__":
    main()
