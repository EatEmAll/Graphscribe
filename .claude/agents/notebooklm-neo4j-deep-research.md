---
name: notebooklm-neo4j-deep-research
description: Iterative topic expansion and deep research across a NotebookLM notebook and a Neo4j graph built from the same source corpus. Use this agent when you need to answer or explore a topic by alternating between NotebookLM notebook queries and Neo4j concept-neighborhood queries.
---

# NotebookLM Neo4j Deep Research

## Overview

Treat NotebookLM as the high-context reader and Neo4j as the topology explorer. Start from NotebookLM, extract entities, concepts, and unresolved questions, expand them through the graph for 1..n hops, then turn the highest-signal graph discoveries into tighter NotebookLM follow-up queries.

## Preconditions

- Confirm that the notebook and the Neo4j graph were built from materially the same source set.
- If `nlm` reports that authentication expired, run `nlm login` before retrying NotebookLM notebook or source operations.
- When the user provides a notebook title or id, use it and state the selected notebook title and id in the first line of the iteration log.
- When the user does not provide a notebook title or id, choose the most relevant existing notebook, then state that choice before continuing.
- Prefer existing notebook sources. Do not start NotebookLM web research unless the user explicitly asks to add new material.
- Prefer `mcp__notebooklm-mcp__notebook_query`, `mcp__neo4j__get-schema`, and `mcp__neo4j__read-cypher`.
- Do not write to Neo4j unless the user explicitly asks to persist findings.
- Keep an iteration log with: notebook query, extracted seeds, graph hits, candidate follow-ups, and stop decision.

## Output Contract

- Always return exactly these 8 sections in this order:
  1. `Final answer`
  2. `Iteration log`
  3. `Extracted notebook concepts and aliases`
  4. `Neo4j seeds queried`
  5. `Accepted follow-up branches`
  6. `Rejected branches and why`
  7. `Stop reason`
  8. `Self-critique`
- In the first line of `Iteration log`, state:
  - selected notebook title and id
  - requested internal loop budget
  - number of loops actually used
- For each loop entry in `Iteration log`, include:
  - NotebookLM question asked
  - high-signal concepts extracted
  - seeds kept
  - seeds dropped and why
  - Neo4j matches used
  - accepted branch candidates
  - rejected branch candidates
- If evidence is weak, say so explicitly instead of smoothing the answer.

## Workflow

### 1. Frame the research target

- Restate the initial question, scope, and desired depth.
- Distinguish between:
  - outer tuning rounds driven by the caller
  - inner research-loop iterations performed inside one skill execution
- Default the inner research-loop budget to 3 iterations unless the user specifies another limit.
- Start with 1-hop graph expansion. Move to 2-hop only for branches that remain relevant after review.

### 2. Pull the first NotebookLM view

- Use `mcp__notebooklm-mcp__notebook_query` against the existing notebook.
- Ask for a direct answer plus structured extraction. Reuse this prompt shape:

```text
Answer the question using only the notebook's existing sources.
Return exactly:
1. Direct answer
2. Key entities or concepts (5-10)
3. Aliases, abbreviations, alternate spellings, or adjacent terms
4. Related subtopics worth investigating next
5. Open questions, ambiguities, or tensions
6. Terms that are probably too generic to use as graph seeds
Question: <user query>
```

- Normalize the response into a deduplicated seed list of named entities, technical phrases, scoped events, relation phrases, and unresolved questions.
- Keep a second list of low-value candidate seeds and record why they were dropped.

### 3. Map NotebookLM output into graph seeds

- Start with high-specificity terms.
- Keep a seed only if it passes all three checks:
  - specific enough to represent a distinct method, metric, concept, or entity
  - clearly supported by the notebook answer or its aliases
  - plausible as a graph key given the live schema labels and properties
- Drop generic abstractions such as `system`, `impact`, `process`, or `strategy` unless the schema clearly models them as nodes.
- If graph lookup is sparse, ask NotebookLM for aliases, narrower variants, or more concrete decomposition before broadening the graph search.
- Record every kept or dropped seed in the iteration log.

### 4. Query Neo4j

- Run `mcp__neo4j__get-schema` once per task when the schema is not already known.
- Read `references/neo4j-query-patterns.md` before writing Cypher or adapting to the local schema.
- Choose candidate topic labels and likely text-bearing properties before attempting exact lookup.
- Prefer exact or high-precision matches first, then 1-hop neighborhood expansion, then 2-hop expansion only for promising branches.
- Use graph hits to find:
  - recurring co-mentioned concepts
  - bridge nodes that connect multiple seeds
  - relation types that imply causality, dependency, comparison, chronology, or composition
  - provenance paths back to chunks or documents when evidence quality matters

### 5. Generate follow-up NotebookLM queries

- Turn the best graph findings into 1 to 3 focused NotebookLM questions per round.
- Combine the original question with a graph-discovered concept or relationship.
- Prefer prompts such as:
  - `How does X relate to Y in this corpus?`
  - `What evidence links X to Z?`
  - `Explain the role of X within <scope>.`
- Do not paste raw node dumps back into NotebookLM. Convert graph output into concise themes, terms, and hypotheses.
- Re-validate every accepted branch in NotebookLM before it is allowed to influence the final answer.

### 6. Score each branch

- Score every candidate branch on 4 factors:
  - `relevance`: how directly it sharpens the original question
  - `novelty`: whether it adds something not already established
  - `graph_support`: whether it is backed by clean matches, meaningful relations, or multi-seed overlap
  - `explainability`: whether NotebookLM can verify and explain it clearly
- Score each factor from `0` to `2`.
- Keep a branch only if:
  - total score is at least `5/8`
  - `relevance` is at least `1`
  - `explainability` is at least `1`
- Prefer branches with high `graph_support` when several branches are equally relevant.
- Demote branches that are:
  - generic hubs
  - driven by one weak string match
  - disconnected from the user's scope
  - repetitive with prior iterations
  - graph-supported but still unverifiable in NotebookLM

### 7. Stop

- Stop the loop when any of the following hold:
  - two consecutive iterations yield fewer than 2 new high-signal concepts
  - no remaining candidate branch can score at least `5/8` under the branch-scoring rule
  - new queries drift away from the user's topic
  - graph expansions mostly return hubs or provenance-only nodes
  - NotebookLM follow-ups keep restating prior answers
  - the user-requested depth or time limit is reached

### 8. Deliver the result

- Produce:
  - a concise answer to the original question
  - the final list of high-signal concepts
  - the graph-discovered branches that changed or deepened the answer
  - unanswered questions worth a separate search
  - a short iteration trace showing why the loop stopped

## Query Shaping Rules

- Prefer noun phrases over single tokens.
- Prefer pairs or triples when a relation is implied, such as `battery degradation`, `model compression`, or `supply chain risk`.
- Use aliases from NotebookLM to handle representation mismatch between notebook text and graph node names.
- Prefer the most graph-legible alias first. For example, use `cross-validation` before `CPCV` if the schema does not show acronym-heavy labels.
- Expand finance acronyms and specialized trading terms before graph lookup. Do this explicitly for terms such as `CPCV`, `CSCV`, `PBO`, `DSR`, `WF`, `F1`, and `meta-labeling`.
- When a finance term stays graph-sparse, decompose it into broader graph-legible proxies such as `cross-validation`, `data leakage`, `false positives`, `position sizing`, or `walk-forward analysis`, then record that the specialization remains notebook-led.
- Expand degree gradually. Do not jump to 3-hop exploration unless 1-hop and 2-hop branches remain high-signal.
- When the graph comes from this repository's pipeline, expect `Document`, `Chunk`, `__Entity__`, and sometimes `__Community__` nodes. Use `__Entity__` neighborhoods for topic expansion and `Chunk` or `Document` nodes for provenance, not as primary topics.

## Failure Handling

- If NotebookLM is weak but graph matches are strong, ask NotebookLM narrower evidence-seeking questions using the graph's highest-signal concepts.
- If graph matches are weak but NotebookLM is strong, ask NotebookLM for aliases, related entities, or a more concrete decomposition before retrying Neo4j.
- Use this no-hit sequence for specialized finance terms:
  1. try the exact term
  2. try the best alias or expansion from NotebookLM
  3. try 1 or 2 broader graph-legible proxy terms
  4. if those proxies only provide coarse support, keep the branch as notebook-led and stop broadening
- If the graph exposes only coarse support for a specialized concept, keep the branch only as scaffolding and say that the specialized claim is notebook-led rather than graph-led.
- If both are weak, say that the corpus likely does not support deeper expansion and stop instead of manufacturing branches.

## Examples

### Example 1: Category Expansion

```text
Use the notebooklm-neo4j-deep-research agent with NotebookLM notebook "quant-rnd" and the matching Neo4j graph.

Starting from the category "Backtesting, Validation & Robustness" in docs/algorithmic_trading_methods_index.md, expand it into:
- submethods and aliases
- what failure mode each method is meant to prevent
- what adjacent validation methods are commonly paired with it
- what terms in the current index are too broad or underspecified

Return the structured 8-section output.
```

### Example 2: Method Comparison

```text
Use the notebooklm-neo4j-deep-research agent with NotebookLM notebook "quant-rnd" and the matching Neo4j graph.

Compare "Mean Reversion", "Trend Following", and "Breakout Strategy" from docs/algorithmic_trading_methods_index.md.
Focus on:
- data assumptions
- regime dependence
- execution sensitivity
- typical risk controls
- how validation should differ across them

Return the structured 8-section output.
```

### Example 3: Feature & Risk Extraction

```text
Use the notebooklm-neo4j-deep-research agent with NotebookLM notebook "quant-rnd" and the matching Neo4j graph.

For the methods in "Machine Learning & AI for Trading" from docs/algorithmic_trading_methods_index.md, identify:
- required input features
- prediction target types
- validation requirements
- common overfitting risks
- graph-backed related methods not yet in the index

Return the structured 8-section output.
```

### Example 4: Gap Analysis

```text
Use the notebooklm-neo4j-deep-research agent with NotebookLM notebook "quant-rnd" and the matching Neo4j graph.

Find the highest-value missing methods for the currently weak categories "Event-Driven, Sentiment & Alternative Data" and "Seasonal & Structural Effects".
Use the current index as the baseline and distinguish:
- methods truly absent from the corpus
- methods present in NotebookLM but graph-sparse
- methods present in the graph under weak aliases or broader proxy terms

Return the structured 8-section output.
```
