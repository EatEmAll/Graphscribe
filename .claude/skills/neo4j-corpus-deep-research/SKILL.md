---
name: neo4j-corpus-deep-research
description: Iterative source-grounded research across the local corpus retrieval MCP and its matching Neo4j graph.
---

# Neo4j Corpus Deep Research

Treat `corpus_answer` as the high-context, cited reader and `graph_neighbors` or the Neo4j MCP as the topology explorer. Use only active local corpus revisions.

## Workflow

1. Select the requested corpus with `corpus_list` and state its key.
2. Ask `corpus_answer` for a direct cited answer and 5-10 concrete entities, aliases, related subtopics, ambiguities, and overly generic seeds.
3. Keep specific, source-supported seeds; drop generic hubs.
4. Expand promising seeds by one graph hop. Use two hops only when the first hop remains relevant and bounded.
5. Convert graph discoveries into 1-3 focused follow-up questions and revalidate them with `corpus_answer`.
6. Score each branch from 0-2 on relevance, novelty, graph support, and source explainability. Keep branches scoring at least 5/8 with nonzero relevance and explainability.
7. Stop after two low-yield rounds, when no branch reaches 5/8, when results become generic hubs, or after three iterations.

## Evidence rules

- A graph relationship is a discovery lead, not sufficient evidence by itself.
- Claims in the final answer require citations returned by `corpus_answer` or `corpus_search`.
- Distinguish graph-discovered hypotheses from source-supported conclusions.
- Never write to Neo4j unless the user explicitly asks.

## Output

Return: final answer, iteration log, extracted corpus concepts and aliases, Neo4j seeds, accepted branches, rejected branches, stop reason, and self-critique.
