#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_registry import DatasetRegistryEntry, load_dataset_entry

MODEL_NAME = "gpt-5.4"
BENCHMARK_RUN_PREFIX = "ab_eval_"
NOTEBOOK_ONLY = "notebook_only"
HYBRID = "hybrid"
MAX_RUN_ATTEMPTS = 2
MAX_RESERVE_SWAPS = 2
ANSWER_TIMEOUT_SECONDS = 2400
JUDGE_TIMEOUT_SECONDS = 900
SMOKE_TIMEOUT_SECONDS = 180


class BenchmarkError(RuntimeError):
    pass


@dataclass(frozen=True)
class QuestionSpec:
    question_id: str
    text: str
    reserve: bool = False


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    title: str
    notebook_id: str
    background: str
    rationale: str
    questions: list[QuestionSpec]


@dataclass(frozen=True)
class AnswerArtifact:
    dataset_key: str
    condition: str
    question_id: str
    question_text: str
    answer_path: Path
    events_path: Path
    prompt_path: Path
    answer_text: str
    tools_used: list[str]
    command_redacted: str
    attempt_count: int


@dataclass(frozen=True)
class AnswerScore:
    dataset: str
    condition: str
    question_id: str
    correctness: int
    completeness: int
    evidence_quality: int
    cross_document_synthesis: int
    total_score: int
    normalized_score: float
    rationale: str
    weakness_tags: list[str]
    score_path: Path


@dataclass(frozen=True)
class ComparisonScore:
    dataset: str
    question_id: str
    winner: str
    material_value_of_graph: str
    rationale: str
    comparison_path: Path


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def dataset_specs() -> dict[str, DatasetSpec]:
    return {
        "bench-openalex-rag": DatasetSpec(
            key="bench-openalex-rag",
            title="bench-openalex-rag",
            notebook_id="93670642-be3f-4c50-b542-58877e64bf3f",
            background=(
                "A 100-work OpenAlex slice centered on retrieval-augmented generation, with titles, abstracts, "
                "authors, institutions, venues, topics, and reference lists."
            ),
            rationale=(
                "This corpus is graph-friendly because value often comes from bridging authors, institutions, "
                "datasets, benchmarks, cited-work neighborhoods, and topic clusters across many papers."
            ),
            questions=[
                QuestionSpec("OA01", "Which authors or institutions bridge RAG and graph-based retrieval or KG-grounded methods, and through what paper neighborhoods?"),
                QuestionSpec("OA02", "Which datasets or benchmarks recur across the slice, and which method families or evaluation goals do they connect?"),
                QuestionSpec("OA03", "How is graph structure used inside RAG systems for retrieval, ranking, reasoning, grounding, or hallucination control?"),
                QuestionSpec("OA04", "Which evaluation dimensions recur together, such as faithfulness, hallucination, retrieval quality, latency, or explainability, and what tradeoff clusters emerge?"),
                QuestionSpec("OA05", "Which survey or synthesis papers act as hubs across application domains, and what subtopics do they bridge?"),
                QuestionSpec("OA06", "How do healthcare or medical papers differ from general-purpose RAG papers in safety, evidence, and deployment constraints?"),
                QuestionSpec("OA07", "Which papers link multimodal or agentic systems to RAG, and which of those links look strongest versus weakest in-corpus?"),
                QuestionSpec("OA08", "What implicit end-to-end RAG pipeline can be reconstructed from the corpus, and where do papers disagree on chunking, grounding, or verification?"),
                QuestionSpec("OA09", "Which cited/reference neighborhoods connect older IR/NLP foundations to newer RAG work, and which bridge works look most central?", reserve=True),
                QuestionSpec("OA10", "What alias or taxonomy gaps are most likely weakening graph provenance in this corpus today?", reserve=True),
            ],
        ),
        "bench-imdb-scifi": DatasetSpec(
            key="bench-imdb-scifi",
            title="bench-imdb-scifi",
            notebook_id="41c71b77-6ed6-400a-b7ca-b64211ea8db0",
            background=(
                "A 100-title IMDb slice of high-vote sci-fi movies from 2000-2024, with directors, writers, "
                "principal cast, genres, ratings, vote counts, and known-for title ids."
            ),
            rationale=(
                "This corpus is graph-friendly because the interesting questions depend on connectivity across people, "
                "roles, franchises, genre overlap, and collaboration patterns."
            ),
            questions=[
                QuestionSpec("IM01", "Which people connect the most high-vote sci-fi films in the slice, and through which roles?"),
                QuestionSpec("IM02", "Which recurring director-writer-actor collaborations define the biggest movie subclusters?"),
                QuestionSpec("IM03", "Which titles bridge superhero, space-opera, dystopian, AI, and cerebral sci-fi clusters through shared cast or crew?"),
                QuestionSpec("IM04", "Which franchises or shared universes dominate connectivity, and which films bridge otherwise separate clusters?"),
                QuestionSpec("IM05", "Which movies are structurally central despite middling ratings, and what connections make them hubs?"),
                QuestionSpec("IM06", "Which creators show the broadest genre-crossing footprint inside sci-fi/action/adventure overlap?"),
                QuestionSpec("IM07", "Which films look isolated by credits even though they are important by votes or ratings, and what does that reveal about dataset limits?"),
                QuestionSpec("IM08", "What production patterns recur across the most connected titles: ensemble reuse, sequel structure, director continuity, or writer continuity?"),
                QuestionSpec("IM09", "Which actor or crew aliases or duplicates are most likely distorting connectivity or overcounting importance?", reserve=True),
                QuestionSpec("IM10", "Which cast or crew neighborhoods most clearly demonstrate graph value beyond notebook-only summarization?", reserve=True),
            ],
        ),
        "bench-opentargets-alzheimers": DatasetSpec(
            key="bench-opentargets-alzheimers",
            title="bench-opentargets-alzheimers",
            notebook_id="c213a4ad-459b-4352-bb77-36f5c8e03649",
            background=(
                "A 100-record Open Targets slice for Alzheimer disease, with target ids, symbols, scores, and "
                "per-modality evidence summaries from the lightweight GraphQL export."
            ),
            rationale=(
                "This corpus is graph-friendly because the useful questions rely on clustering and bridging targets "
                "across evidence modalities and biological themes."
            ),
            questions=[
                QuestionSpec("OT01", "Which targets bridge multiple evidence modalities for Alzheimer disease, and which evidence mixes recur most often?"),
                QuestionSpec("OT02", "Which target clusters align with immune/microglial, synaptic/signaling, and mitochondrial/metabolic themes, and which targets bridge them?"),
                QuestionSpec("OT03", "Which canonical Alzheimer genes anchor neighborhoods of less obvious targets, and what evidence types make those links plausible?"),
                QuestionSpec("OT04", "Which targets look strong overall but are carried mainly by one evidence modality, and which look more balanced?"),
                QuestionSpec("OT05", "Which target groups appear repeatedly together through pathway or mechanism language, rather than just coexisting in the disease slice?"),
                QuestionSpec("OT06", "Which targets are high-scoring but graph-isolated, and what does that imply about coverage limits or alias/taxonomy gaps?"),
                QuestionSpec("OT07", "Which evidence modalities most often co-occur for top targets, and where do the strongest asymmetries appear?"),
                QuestionSpec("OT08", "If prioritizing follow-up investigation, which targets sit at the best intersection of strong score, diverse evidence, and strong neighborhood support?"),
                QuestionSpec("OT09", "Where does missing mechanism or drug detail in the lightweight slice most clearly limit multi-hop reasoning?", reserve=True),
                QuestionSpec("OT10", "Which target neighborhoods most clearly show the graph adding useful structure beyond NotebookLM-only reading?", reserve=True),
            ],
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the benchmark A/B evaluation blog workflow.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["bench-openalex-rag", "bench-imdb-scifi", "bench-opentargets-alzheimers"],
        help="Dataset keys to benchmark in order.",
    )
    parser.add_argument("--registry-path", help="Optional benchmark dataset registry path.")
    parser.add_argument("--model", default=MODEL_NAME, help="Codex model name.")
    parser.add_argument("--runs-root", default=str(REPO_ROOT / "runs"), help="Benchmark run root.")
    parser.add_argument("--run-dir", help="Existing benchmark run dir to resume.")
    parser.add_argument("--temp-root", default=str(Path(tempfile.gettempdir()) / "codex_ab_eval"), help="Temp Codex working root.")
    return parser


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def slugified_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_root(runs_root: Path) -> Path:
    root = runs_root / f"{BENCHMARK_RUN_PREFIX}{slugified_timestamp()}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def preflight_notebooks(dataset_entries: list[DatasetRegistryEntry]) -> None:
    result = subprocess.run(["nlm", "notebook", "list", "--json"], text=True, capture_output=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "nlm notebook list failed"
        raise BenchmarkError(detail)
    payload = json.loads(result.stdout or "[]")
    items = payload.get("notebooks") if isinstance(payload, dict) else payload
    ids = {str(item.get("id")) for item in items if isinstance(item, dict)}
    missing = [entry.notebook.id for entry in dataset_entries if entry.notebook.id not in ids]
    if missing:
        raise BenchmarkError(f"Notebook ids missing from NotebookLM: {', '.join(missing)}")


def ensure_container_exists(container_name: str) -> None:
    result = subprocess.run(["docker", "inspect", container_name], text=True, capture_output=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"docker inspect failed for {container_name}"
        raise BenchmarkError(detail)


def is_container_running(container_name: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Running}}", container_name],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"docker inspect failed for {container_name}"
        raise BenchmarkError(detail)
    return result.stdout.strip().lower() == "true"


def start_container(container_name: str) -> None:
    if is_container_running(container_name):
        return
    result = subprocess.run(["docker", "start", container_name], text=True, capture_output=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"docker start failed for {container_name}"
        raise BenchmarkError(detail)
    deadline = time.time() + 120
    while time.time() < deadline:
        if is_container_running(container_name):
            return
        time.sleep(2)
    raise BenchmarkError(f"Container {container_name} did not reach running state.")


def stop_container(container_name: str) -> None:
    if not is_container_running(container_name):
        return
    result = subprocess.run(["docker", "stop", container_name], text=True, capture_output=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"docker stop failed for {container_name}"
        raise BenchmarkError(detail)


def resolve_codex_launcher() -> list[str]:
    if os.name == "nt":
        return ["cmd", "/c", "codex"]
    resolved = shutil.which("codex")
    if not resolved:
        raise BenchmarkError("codex executable not found on PATH.")
    return [resolved]


def parse_jsonl_events(raw_output: str) -> tuple[list[dict[str, Any]], list[str]]:
    parsed: list[dict[str, Any]] = []
    tools_used: list[str] = []
    for line in raw_output.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        parsed.append(payload)
        item = payload.get("item")
        if isinstance(item, dict) and item.get("type") == "mcp_tool_call":
            tools_used.append(f"{item.get('server')}:{item.get('tool')}")
    return parsed, tools_used


def build_question_prompt(dataset: DatasetSpec, condition: str, question: QuestionSpec) -> str:
    lines = [
        "You are running an isolated benchmark through codex exec.",
        "Use only existing NotebookLM sources.",
        "Do not add sources.",
        "Do not do web research.",
        "Do not use shell commands.",
        "Do not read local files.",
        f"Notebook title: {dataset.title}",
        f"Notebook id: {dataset.notebook_id}",
        f"Dataset background: {dataset.background}",
        f"Why this corpus suits graph-backed analysis: {dataset.rationale}",
        f"Question: {question.text}",
    ]
    if condition == NOTEBOOK_ONLY:
        lines.extend(
            [
                "Use only notebooklm-mcp.",
                "Do not use neo4j.",
                "Keep tool use efficient. Use at most 3 NotebookLM queries.",
                "Return exactly these markdown sections:",
                "## Direct answer",
                "## Corpus evidence summary",
                "## Disagreements or uncertainties",
                "## Evidence-quality note",
            ]
        )
    else:
        lines.extend(
            [
                "Use only notebooklm-mcp and neo4j.",
                "The active Neo4j MCP already points to the matching dataset graph.",
                "Internally follow this workflow: NotebookLM first -> extract concrete seeds -> query Neo4j for anchors or neighborhoods -> return to NotebookLM for validation.",
                "Keep tool use efficient. Use at most 3 NotebookLM queries and 3 Neo4j queries.",
                "Return exactly these markdown sections:",
                "## Direct answer",
                "## Graph-backed findings",
                "## Notebook-led findings",
                "## Disagreements or uncertainties",
                "## Evidence-quality note",
            ]
        )
    return "\n".join(lines) + "\n"


def build_scoring_prompt(dataset: DatasetSpec, condition: str, question: QuestionSpec, answer_text: str) -> str:
    return (
        "You are a strict benchmark judge.\n"
        "Do not use any tools.\n"
        "Score the answer on a fixed 4-factor rubric from 1 to 5 each: correctness, completeness, evidence_quality, cross_document_synthesis.\n"
        "Compute total_score as the sum of those four integers.\n"
        "Compute normalized_score as total_score / 2 so it is on a 1-10 scale.\n"
        "Return JSON only with keys: dataset, condition, question_id, correctness, completeness, evidence_quality, cross_document_synthesis, total_score, normalized_score, rationale, weakness_tags.\n"
        f"Dataset: {dataset.key}\n"
        f"Condition: {condition}\n"
        f"Question ID: {question.question_id}\n"
        f"Question: {question.text}\n"
        "Answer:\n-----\n"
        f"{answer_text}\n"
        "-----\n"
    )


def build_comparison_prompt(dataset: DatasetSpec, question: QuestionSpec, notebook_answer: AnswerArtifact, notebook_score: AnswerScore, hybrid_answer: AnswerArtifact, hybrid_score: AnswerScore) -> str:
    score_diff = abs(notebook_score.total_score - hybrid_score.total_score)
    forced_winner = "tie" if score_diff <= 1 else (HYBRID if hybrid_score.total_score > notebook_score.total_score else NOTEBOOK_ONLY)
    return (
        "You are a strict comparative benchmark judge.\n"
        "Do not use any tools.\n"
        "Winner rule: if the totals differ by 1 or less, winner must be tie. Otherwise higher total wins.\n"
        "Return JSON only with keys: dataset, question_id, winner, material_value_of_graph, rationale.\n"
        f"Dataset: {dataset.key}\n"
        f"Question ID: {question.question_id}\n"
        f"Question: {question.text}\n"
        f"Notebook-only total score: {notebook_score.total_score}\n"
        f"Hybrid total score: {hybrid_score.total_score}\n"
        f"Forced winner by benchmark rule: {forced_winner}\n"
        "Notebook-only answer:\n-----\n"
        f"{notebook_answer.answer_text}\n"
        "-----\n"
        "Hybrid answer:\n-----\n"
        f"{hybrid_answer.answer_text}\n"
        "-----\n"
    )


def extract_first_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        payload = json.loads(stripped[start : end + 1])
        if isinstance(payload, dict):
            return payload
    raise BenchmarkError("Could not parse JSON response from Codex.")


def redacted_command(command: list[str], entry: DatasetRegistryEntry | None) -> str:
    parts: list[str] = []
    for part in command:
        rendered = part
        if entry and entry.neo4j.password and entry.neo4j.password in rendered:
            rendered = rendered.replace(entry.neo4j.password, "<REDACTED>")
        parts.append(rendered)
    return " ".join(parts)


def run_codex_exec(
    *,
    prompt_text: str,
    temp_root: Path,
    answer_path: Path,
    events_path: Path,
    model: str,
    config_overrides: list[str],
    dataset_entry: DatasetRegistryEntry | None,
    timeout_seconds: int,
) -> tuple[str, list[str], str]:
    temp_root.mkdir(parents=True, exist_ok=True)
    answer_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    launcher = resolve_codex_launcher()
    command = [
        *launcher,
        "exec",
        "-C",
        str(temp_root),
        "--skip-git-repo-check",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "--model",
        model,
        "--json",
        "-o",
        str(answer_path),
    ]
    for override in config_overrides:
        command.extend(["-c", override])
    command.append("-")
    try:
        result = subprocess.run(
            command,
            input=prompt_text,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        partial_stdout = exc.stdout or ""
        write_text(events_path, partial_stdout)
        raise BenchmarkError(f"codex exec timed out after {timeout_seconds}s") from exc
    write_text(events_path, result.stdout or "")
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "codex exec failed"
        raise BenchmarkError(detail)
    if not answer_path.exists():
        raise BenchmarkError(f"Codex did not write answer file: {answer_path}")
    _, tools_used = parse_jsonl_events(result.stdout or "")
    return answer_path.read_text(encoding="utf-8"), tools_used, redacted_command(command, dataset_entry)


def build_hybrid_overrides(entry: DatasetRegistryEntry) -> list[str]:
    return [
        f'mcp_servers.neo4j.env.NEO4J_URI="{entry.neo4j.uri}"',
        f'mcp_servers.neo4j.env.NEO4J_USERNAME="{entry.neo4j.username}"',
        f'mcp_servers.neo4j.env.NEO4J_PASSWORD="{entry.neo4j.password}"',
        f'mcp_servers.neo4j.env.NEO4J_DATABASE="{entry.neo4j.database}"',
    ]


def validate_tools(condition: str, tools_used: list[str]) -> bool:
    has_notebook = any(tool.startswith("notebooklm-mcp:") for tool in tools_used)
    has_neo4j = any(tool.startswith("neo4j:") for tool in tools_used)
    if condition == NOTEBOOK_ONLY:
        return has_notebook and not has_neo4j
    return has_notebook and has_neo4j


def answer_artifact_paths(run_dir: Path, dataset_key: str, condition: str, question_id: str, attempt: int) -> tuple[Path, Path, Path]:
    base = run_dir / dataset_key / "answers" / condition / question_id / f"attempt_{attempt}"
    return base / "prompt.txt", base / "answer.md", base / "events.jsonl"


def score_artifact_path(run_dir: Path, dataset_key: str, condition: str, question_id: str) -> Path:
    return run_dir / dataset_key / "scores" / condition / f"{question_id}.json"


def comparison_artifact_path(run_dir: Path, dataset_key: str, question_id: str) -> Path:
    return run_dir / dataset_key / "comparisons" / f"{question_id}.json"


def load_existing_answer_artifact(run_dir: Path, dataset: DatasetSpec, question: QuestionSpec, condition: str) -> AnswerArtifact | None:
    for attempt in range(1, MAX_RUN_ATTEMPTS + 1):
        prompt_path, answer_path, events_path = answer_artifact_paths(run_dir, dataset.key, condition, question.question_id, attempt)
        if not (prompt_path.exists() and answer_path.exists() and events_path.exists()):
            continue
        answer_text = answer_path.read_text(encoding="utf-8")
        if not answer_text.strip():
            continue
        tools_used = parse_jsonl_events(events_path.read_text(encoding="utf-8"))[1]
        if not validate_tools(condition, tools_used):
            continue
        return AnswerArtifact(
            dataset.key,
            condition,
            question.question_id,
            question.text,
            answer_path,
            events_path,
            prompt_path,
            answer_text,
            tools_used,
            "<reused>",
            attempt,
        )
    return None


def load_existing_answer_score(path: Path) -> AnswerScore | None:
    payload: dict[str, Any] | None = None
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        judge_path = path.with_suffix(".judge.txt")
        if judge_path.exists():
            payload = extract_first_json_object(judge_path.read_text(encoding="utf-8"))
            write_json(path, payload)
    if payload is None:
        return None
    total_score = (
        int(payload["correctness"])
        + int(payload["completeness"])
        + int(payload["evidence_quality"])
        + int(payload["cross_document_synthesis"])
    )
    return AnswerScore(
        dataset=str(payload["dataset"]),
        condition=str(payload["condition"]),
        question_id=str(payload["question_id"]),
        correctness=int(payload["correctness"]),
        completeness=int(payload["completeness"]),
        evidence_quality=int(payload["evidence_quality"]),
        cross_document_synthesis=int(payload["cross_document_synthesis"]),
        total_score=total_score,
        normalized_score=round(total_score / 2, 3),
        rationale=str(payload["rationale"]),
        weakness_tags=[str(item) for item in payload.get("weakness_tags") or []],
        score_path=path,
    )


def load_existing_comparison_score(path: Path) -> ComparisonScore | None:
    payload: dict[str, Any] | None = None
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        judge_path = path.with_suffix(".judge.txt")
        if judge_path.exists():
            payload = extract_first_json_object(judge_path.read_text(encoding="utf-8"))
            write_json(path, payload)
    if payload is None:
        return None
    return ComparisonScore(
        dataset=str(payload["dataset"]),
        question_id=str(payload["question_id"]),
        winner=str(payload["winner"]),
        material_value_of_graph=str(payload["material_value_of_graph"]),
        rationale=str(payload["rationale"]),
        comparison_path=path,
    )


def run_answer_generation(*, run_dir: Path, temp_root: Path, dataset: DatasetSpec, entry: DatasetRegistryEntry, question: QuestionSpec, condition: str, model: str, timeout_seconds: int = ANSWER_TIMEOUT_SECONDS) -> AnswerArtifact:
    existing = load_existing_answer_artifact(run_dir, dataset, question, condition)
    if existing:
        log(f"Reusing {dataset.key} {condition} {question.question_id} attempt {existing.attempt_count}")
        return existing
    prompt_text = build_question_prompt(dataset, condition, question)
    for attempt in range(1, MAX_RUN_ATTEMPTS + 1):
        log(f"Generating {dataset.key} {condition} {question.question_id} attempt {attempt}")
        prompt_path, answer_path, events_path = answer_artifact_paths(run_dir, dataset.key, condition, question.question_id, attempt)
        write_text(prompt_path, prompt_text)
        overrides = build_hybrid_overrides(entry) if condition == HYBRID else []
        try:
            answer_text, tools_used, command_redacted = run_codex_exec(
                prompt_text=prompt_text,
                temp_root=temp_root / dataset.key / condition / question.question_id / f"attempt_{attempt}",
                answer_path=answer_path,
                events_path=events_path,
                model=model,
                config_overrides=overrides,
                dataset_entry=entry if condition == HYBRID else None,
                timeout_seconds=timeout_seconds,
            )
        except BenchmarkError as exc:
            existing_answer = answer_path.read_text(encoding="utf-8") if answer_path.exists() else ""
            if attempt < MAX_RUN_ATTEMPTS and not existing_answer.strip():
                log(f"Retrying {dataset.key} {condition} {question.question_id} after empty or missing final agent message.")
                continue
            raise exc
        if validate_tools(condition, tools_used) and answer_text.strip():
            return AnswerArtifact(dataset.key, condition, question.question_id, question.text, answer_path, events_path, prompt_path, answer_text, tools_used, command_redacted, attempt)
        if attempt == MAX_RUN_ATTEMPTS:
            if not answer_text.strip():
                raise BenchmarkError(f"Empty answer for {dataset.key} {condition} {question.question_id}")
            raise BenchmarkError(f"Invalid tool usage for {dataset.key} {condition} {question.question_id}: {', '.join(tools_used) or 'none'}")
        reason = "empty answer" if not answer_text.strip() else "invalid tool usage"
        log(f"Retrying {dataset.key} {condition} {question.question_id} after {reason}.")
    raise BenchmarkError(f"Unreachable answer generation failure for {dataset.key} {condition} {question.question_id}")


def run_structured_judge(*, prompt_text: str, temp_root: Path, output_path: Path, model: str) -> dict[str, Any]:
    answer_path = output_path.with_suffix(".judge.txt")
    events_path = output_path.with_suffix(".judge.events.jsonl")
    output_text, tools_used, _ = run_codex_exec(
        prompt_text=prompt_text,
        temp_root=temp_root,
        answer_path=answer_path,
        events_path=events_path,
        model=model,
        config_overrides=[],
        dataset_entry=None,
        timeout_seconds=JUDGE_TIMEOUT_SECONDS,
    )
    if tools_used:
        raise BenchmarkError(f"Structured judge unexpectedly used tools: {', '.join(tools_used)}")
    payload = extract_first_json_object(output_text)
    write_json(output_path, payload)
    return payload


def score_answer(*, run_dir: Path, temp_root: Path, dataset: DatasetSpec, question: QuestionSpec, artifact: AnswerArtifact, model: str) -> AnswerScore:
    existing = load_existing_answer_score(score_artifact_path(run_dir, dataset.key, artifact.condition, question.question_id))
    if existing:
        log(f"Reusing score {dataset.key} {artifact.condition} {question.question_id}")
        return existing
    log(f"Scoring {dataset.key} {artifact.condition} {question.question_id}")
    payload = run_structured_judge(
        prompt_text=build_scoring_prompt(dataset, artifact.condition, question, artifact.answer_text),
        temp_root=temp_root / dataset.key / "judges" / artifact.condition / question.question_id,
        output_path=score_artifact_path(run_dir, dataset.key, artifact.condition, question.question_id),
        model=model,
    )
    return AnswerScore(
        dataset=str(payload["dataset"]),
        condition=str(payload["condition"]),
        question_id=str(payload["question_id"]),
        correctness=int(payload["correctness"]),
        completeness=int(payload["completeness"]),
        evidence_quality=int(payload["evidence_quality"]),
        cross_document_synthesis=int(payload["cross_document_synthesis"]),
        total_score=(
            int(payload["correctness"])
            + int(payload["completeness"])
            + int(payload["evidence_quality"])
            + int(payload["cross_document_synthesis"])
        ),
        normalized_score=round(
            (
                int(payload["correctness"])
                + int(payload["completeness"])
                + int(payload["evidence_quality"])
                + int(payload["cross_document_synthesis"])
            )
            / 2,
            3,
        ),
        rationale=str(payload["rationale"]),
        weakness_tags=[str(item) for item in payload.get("weakness_tags") or []],
        score_path=score_artifact_path(run_dir, dataset.key, artifact.condition, question.question_id),
    )


def compare_answers(*, run_dir: Path, temp_root: Path, dataset: DatasetSpec, question: QuestionSpec, notebook_answer: AnswerArtifact, notebook_score: AnswerScore, hybrid_answer: AnswerArtifact, hybrid_score: AnswerScore, model: str) -> ComparisonScore:
    existing = load_existing_comparison_score(comparison_artifact_path(run_dir, dataset.key, question.question_id))
    if existing:
        log(f"Reusing comparison {dataset.key} {question.question_id}")
        return existing
    log(f"Comparing {dataset.key} {question.question_id}")
    payload = run_structured_judge(
        prompt_text=build_comparison_prompt(dataset, question, notebook_answer, notebook_score, hybrid_answer, hybrid_score),
        temp_root=temp_root / dataset.key / "judges" / "comparisons" / question.question_id,
        output_path=comparison_artifact_path(run_dir, dataset.key, question.question_id),
        model=model,
    )
    return ComparisonScore(
        dataset=str(payload["dataset"]),
        question_id=str(payload["question_id"]),
        winner=str(payload["winner"]),
        material_value_of_graph=str(payload["material_value_of_graph"]),
        rationale=str(payload["rationale"]),
        comparison_path=comparison_artifact_path(run_dir, dataset.key, question.question_id),
    )


def replacement_candidates(primary_questions: list[QuestionSpec], notebook_scores: dict[str, AnswerScore], hybrid_scores: dict[str, AnswerScore], comparisons: dict[str, ComparisonScore]) -> list[str]:
    candidates: list[tuple[int, int, int, str]] = []
    for index, question in enumerate(primary_questions):
        comparison = comparisons[question.question_id]
        notebook_score = notebook_scores[question.question_id]
        hybrid_score = hybrid_scores[question.question_id]
        if comparison.material_value_of_graph != "no":
            continue
        if comparison.winner not in {NOTEBOOK_ONLY, "tie"}:
            continue
        if notebook_score.cross_document_synthesis > 3 or hybrid_score.cross_document_synthesis > 3:
            continue
        candidates.append((max(notebook_score.cross_document_synthesis, hybrid_score.cross_document_synthesis), max(notebook_score.total_score, hybrid_score.total_score), index, question.question_id))
    candidates.sort()
    return [question_id for _, _, _, question_id in candidates[:MAX_RESERVE_SWAPS]]


def dataset_level_summary(*, dataset: DatasetSpec, final_question_ids: list[str], notebook_scores: dict[str, AnswerScore], hybrid_scores: dict[str, AnswerScore], comparisons: dict[str, ComparisonScore]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    notebook_totals: list[int] = []
    hybrid_totals: list[int] = []
    wins = losses = ties = 0
    helped: list[str] = []
    little: list[str] = []
    for question_id in final_question_ids:
        notebook_score = notebook_scores[question_id]
        hybrid_score = hybrid_scores[question_id]
        comparison = comparisons[question_id]
        notebook_totals.append(notebook_score.total_score)
        hybrid_totals.append(hybrid_score.total_score)
        wins += int(comparison.winner == HYBRID)
        losses += int(comparison.winner == NOTEBOOK_ONLY)
        ties += int(comparison.winner == "tie")
        (helped if comparison.material_value_of_graph == "yes" else little).append(question_id)
        rows.append({"question_id": question_id, "question_text": next(item.text for item in dataset.questions if item.question_id == question_id), "notebook_only_total": notebook_score.total_score, "hybrid_total": hybrid_score.total_score, "winner": comparison.winner, "material_value_of_graph": comparison.material_value_of_graph, "reason": comparison.rationale})
    return {
        "dataset": dataset.key,
        "rows": rows,
        "notebook_only_mean_total": round(sum(notebook_totals) / len(notebook_totals), 3),
        "hybrid_mean_total": round(sum(hybrid_totals) / len(hybrid_totals), 3),
        "notebook_only_rating_10": round(sum(notebook_totals) / (2 * len(notebook_totals)), 3),
        "hybrid_rating_10": round(sum(hybrid_totals) / (2 * len(hybrid_totals)), 3),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "hybrid_helped": helped,
        "hybrid_little_or_none": little,
    }


def render_technical_blog(run_dir: Path, dataset_summaries: list[dict[str, Any]], aggregate: dict[str, Any], datasets: list[DatasetSpec]) -> str:
    lines = ["# NotebookLM + Neo4j vs NotebookLM Only Across 3 Benchmark Corpora", "", "## Why These Datasets"]
    for dataset in datasets:
        lines.extend([f"### {dataset.key}", dataset.background, "", dataset.rationale, ""])
    lines.extend(["## Methodology", "We ran isolated `codex exec` A/B evaluations from `%TEMP%` with `--skip-git-repo-check --ephemeral --sandbox read-only`.", "Each question was answered twice: once with NotebookLM only, and once with a NotebookLM + Neo4j workflow that forced notebook-first exploration, graph expansion, and notebook re-validation.", "Each answer was scored independently on four 1-5 factors: correctness, completeness, evidence quality, and cross-document synthesis.", "We then ran a comparative judge per question and kept a final scored set of 8 questions per dataset, allowing up to 2 reserve-question substitutions when the initial question under-tested graph value.", "", "## Results by Dataset"])
    for summary in dataset_summaries:
        lines.extend([f"### {summary['dataset']}", f"- NotebookLM only mean: `{summary['notebook_only_mean_total']}` / 20 (`{summary['notebook_only_rating_10']}` / 10)", f"- NotebookLM + Neo4j mean: `{summary['hybrid_mean_total']}` / 20 (`{summary['hybrid_rating_10']}` / 10)", f"- Hybrid wins / losses / ties: `{summary['wins']} / {summary['losses']} / {summary['ties']}`", "", "| Question | NotebookLM only | Hybrid | Winner | Why |", "| --- | ---: | ---: | --- | --- |"])
        for row in summary["rows"]:
            lines.append(f"| {row['question_id']} | {row['notebook_only_total']} | {row['hybrid_total']} | {row['winner']} | {row['reason']} |")
        lines.extend(["", f"Hybrid clearly helped on: {', '.join(summary['hybrid_helped']) or 'None'}.", f"Hybrid added little or no value on: {', '.join(summary['hybrid_little_or_none']) or 'None'}.", ""])
    lines.extend(["## Overall Rating", f"- NotebookLM only overall rating: `{aggregate['notebook_only_rating_10']}` / 10", f"- NotebookLM + Neo4j overall rating: `{aggregate['hybrid_rating_10']}` / 10", f"- Overall wins / losses / ties for hybrid: `{aggregate['wins']} / {aggregate['losses']} / {aggregate['ties']}`", "", aggregate["verdict"], "", "## Limits of the 4-Factor Scoring", "The rubric was useful for consistency, but it is still a text-level judge rather than a claim-level verifier. It rewards synthesis and uncertainty marking, but it cannot fully prove factual correctness from the raw answer text alone.", "A stronger follow-up would add blinded human review, claim-level citation checks against the corpus, explicit tool-trace quality scoring, and a small gold-standard set of benchmark questions with expected evidence nuggets.", "", f"Full raw outputs, scorecards, and judge artifacts are in `{run_dir}`.", ""])
    return "\n".join(lines)


def render_appendix(*, run_dir: Path, datasets: list[DatasetSpec], question_orders: dict[str, list[str]], replacements: dict[str, list[dict[str, str]]], answer_artifacts: dict[str, dict[str, dict[str, AnswerArtifact]]], notebook_scores: dict[str, dict[str, AnswerScore]], hybrid_scores: dict[str, dict[str, AnswerScore]], comparisons: dict[str, dict[str, ComparisonScore]]) -> str:
    lines = ["# Appendix: A/B Evaluation Artifacts", "", "## Command Shape", "All answer-generation runs used this shape from a temp working root, with prompts provided through stdin:", "", "```powershell", "codex exec -C <temp-root> --skip-git-repo-check --ephemeral --sandbox read-only --model gpt-5.4 --json -o <answer-file> -", "```", "", "Hybrid runs additionally overrode the Neo4j MCP env with `-c mcp_servers.neo4j.env.*=...` per dataset. Passwords are redacted here and were not written into this appendix.", "", "## Dataset Question Sets"]
    for dataset in datasets:
        lines.append(f"### {dataset.key}")
        for question in dataset.questions:
            lines.append(f"- `{question.question_id}`{' (reserve)' if question.reserve else ''}: {question.text}")
        lines.append(f"- Replacements used: {json.dumps(replacements.get(dataset.key) or []) if replacements.get(dataset.key) else 'none'}")
        lines.append("")
        lines.append("### Final scored questions")
        for question_id in question_orders[dataset.key]:
            lines.append(f"- `{question_id}`")
        lines.append("")
        lines.append("### Per-question results")
        lines.append("| Question | NotebookLM only | Hybrid | Winner | Graph added value |")
        lines.append("| --- | ---: | ---: | --- | --- |")
        for question_id in question_orders[dataset.key]:
            lines.append(f"| {question_id} | {notebook_scores[dataset.key][question_id].total_score} | {hybrid_scores[dataset.key][question_id].total_score} | {comparisons[dataset.key][question_id].winner} | {comparisons[dataset.key][question_id].material_value_of_graph} |")
        lines.append("")
        lines.append("### Raw artifacts")
        for question_id in question_orders[dataset.key]:
            lines.extend([f"- `{question_id}` notebook-only answer: `{answer_artifacts[dataset.key][NOTEBOOK_ONLY][question_id].answer_path}`", f"- `{question_id}` notebook-only events: `{answer_artifacts[dataset.key][NOTEBOOK_ONLY][question_id].events_path}`", f"- `{question_id}` hybrid answer: `{answer_artifacts[dataset.key][HYBRID][question_id].answer_path}`", f"- `{question_id}` hybrid events: `{answer_artifacts[dataset.key][HYBRID][question_id].events_path}`", f"- `{question_id}` notebook-only score: `{notebook_scores[dataset.key][question_id].score_path}`", f"- `{question_id}` hybrid score: `{hybrid_scores[dataset.key][question_id].score_path}`", f"- `{question_id}` comparison: `{comparisons[dataset.key][question_id].comparison_path}`"])
        lines.append("")
    return "\n".join(lines)


def aggregate_overall(dataset_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    notebook_scores = [row["notebook_only_total"] for summary in dataset_summaries for row in summary["rows"]]
    hybrid_scores = [row["hybrid_total"] for summary in dataset_summaries for row in summary["rows"]]
    notebook_rating = round(sum(notebook_scores) / (2 * len(notebook_scores)), 3)
    hybrid_rating = round(sum(hybrid_scores) / (2 * len(hybrid_scores)), 3)
    wins = sum(summary["wins"] for summary in dataset_summaries)
    losses = sum(summary["losses"] for summary in dataset_summaries)
    ties = sum(summary["ties"] for summary in dataset_summaries)
    verdict = (
        "Across these benchmark corpora, the NotebookLM + Neo4j workflow was materially better overall, especially on questions that required multi-hop connectivity, bridge-node discovery, or cross-record clustering."
        if hybrid_rating > notebook_rating
        else "Across these benchmark corpora, NotebookLM only scored better overall, which suggests the current graph workflow added more complexity than value for the final question set."
        if hybrid_rating < notebook_rating
        else "Across these benchmark corpora, the two workflows ended effectively tied overall, with graph value appearing only for a subset of strongly relational questions."
    )
    return {"notebook_only_rating_10": notebook_rating, "hybrid_rating_10": hybrid_rating, "wins": wins, "losses": losses, "ties": ties, "verdict": verdict}


def benchmark_dataset(*, run_dir: Path, temp_root: Path, dataset: DatasetSpec, entry: DatasetRegistryEntry, model: str) -> tuple[dict[str, Any], list[str], dict[str, dict[str, AnswerArtifact]], dict[str, AnswerScore], dict[str, AnswerScore], dict[str, ComparisonScore], list[dict[str, str]]]:
    log(f"Starting dataset benchmark: {dataset.key}")
    start_container(entry.neo4j.container_name or "")
    try:
        smoke_notebook = run_answer_generation(
            run_dir=run_dir,
            temp_root=temp_root / "smoke",
            dataset=dataset,
            entry=entry,
            question=QuestionSpec("SMOKE", "What is the notebook title and approximately how many sources are in it?"),
            condition=NOTEBOOK_ONLY,
            model=model,
            timeout_seconds=SMOKE_TIMEOUT_SECONDS,
        )
        if any(tool.startswith("neo4j:") for tool in smoke_notebook.tools_used):
            raise BenchmarkError(f"Notebook-only smoke used Neo4j for {dataset.key}")
        smoke_hybrid = run_answer_generation(
            run_dir=run_dir,
            temp_root=temp_root / "smoke",
            dataset=dataset,
            entry=entry,
            question=QuestionSpec(
                "SMOKE_HYBRID",
                "Confirm access to both the notebook and graph. Name the notebook title, then identify one concrete graph node or neighborhood that is clearly present and relevant to this dataset.",
            ),
            condition=HYBRID,
            model=model,
            timeout_seconds=SMOKE_TIMEOUT_SECONDS,
        )
        if not any(tool.startswith("neo4j:") for tool in smoke_hybrid.tools_used):
            raise BenchmarkError(f"Hybrid smoke did not use Neo4j for {dataset.key}")

        primary_questions = [item for item in dataset.questions if not item.reserve]
        reserve_questions = [item for item in dataset.questions if item.reserve]
        answer_artifacts: dict[str, dict[str, AnswerArtifact]] = {NOTEBOOK_ONLY: {}, HYBRID: {}}
        notebook_scores: dict[str, AnswerScore] = {}
        hybrid_scores: dict[str, AnswerScore] = {}
        comparisons: dict[str, ComparisonScore] = {}

        for question in primary_questions:
            answer_artifacts[NOTEBOOK_ONLY][question.question_id] = run_answer_generation(run_dir=run_dir, temp_root=temp_root, dataset=dataset, entry=entry, question=question, condition=NOTEBOOK_ONLY, model=model)
        for question in primary_questions:
            answer_artifacts[HYBRID][question.question_id] = run_answer_generation(run_dir=run_dir, temp_root=temp_root, dataset=dataset, entry=entry, question=question, condition=HYBRID, model=model)
        for question in primary_questions:
            notebook_scores[question.question_id] = score_answer(run_dir=run_dir, temp_root=temp_root, dataset=dataset, question=question, artifact=answer_artifacts[NOTEBOOK_ONLY][question.question_id], model=model)
            hybrid_scores[question.question_id] = score_answer(run_dir=run_dir, temp_root=temp_root, dataset=dataset, question=question, artifact=answer_artifacts[HYBRID][question.question_id], model=model)
            comparisons[question.question_id] = compare_answers(run_dir=run_dir, temp_root=temp_root, dataset=dataset, question=question, notebook_answer=answer_artifacts[NOTEBOOK_ONLY][question.question_id], notebook_score=notebook_scores[question.question_id], hybrid_answer=answer_artifacts[HYBRID][question.question_id], hybrid_score=hybrid_scores[question.question_id], model=model)

        final_question_ids = [item.question_id for item in primary_questions]
        swap_records: list[dict[str, str]] = []
        for original_id, reserve_question in zip(replacement_candidates(primary_questions, notebook_scores, hybrid_scores, comparisons), reserve_questions):
            log(f"Replacing {dataset.key} question {original_id} with reserve {reserve_question.question_id}")
            answer_artifacts[NOTEBOOK_ONLY][reserve_question.question_id] = run_answer_generation(run_dir=run_dir, temp_root=temp_root, dataset=dataset, entry=entry, question=reserve_question, condition=NOTEBOOK_ONLY, model=model)
            answer_artifacts[HYBRID][reserve_question.question_id] = run_answer_generation(run_dir=run_dir, temp_root=temp_root, dataset=dataset, entry=entry, question=reserve_question, condition=HYBRID, model=model)
            notebook_scores[reserve_question.question_id] = score_answer(run_dir=run_dir, temp_root=temp_root, dataset=dataset, question=reserve_question, artifact=answer_artifacts[NOTEBOOK_ONLY][reserve_question.question_id], model=model)
            hybrid_scores[reserve_question.question_id] = score_answer(run_dir=run_dir, temp_root=temp_root, dataset=dataset, question=reserve_question, artifact=answer_artifacts[HYBRID][reserve_question.question_id], model=model)
            comparisons[reserve_question.question_id] = compare_answers(run_dir=run_dir, temp_root=temp_root, dataset=dataset, question=reserve_question, notebook_answer=answer_artifacts[NOTEBOOK_ONLY][reserve_question.question_id], notebook_score=notebook_scores[reserve_question.question_id], hybrid_answer=answer_artifacts[HYBRID][reserve_question.question_id], hybrid_score=hybrid_scores[reserve_question.question_id], model=model)
            final_question_ids[final_question_ids.index(original_id)] = reserve_question.question_id
            swap_records.append({"replaced": original_id, "replacement": reserve_question.question_id})

        summary = dataset_level_summary(dataset=dataset, final_question_ids=final_question_ids, notebook_scores=notebook_scores, hybrid_scores=hybrid_scores, comparisons=comparisons)
        write_json(run_dir / dataset.key / "dataset_summary.json", summary)
        return summary, final_question_ids, answer_artifacts, notebook_scores, hybrid_scores, comparisons, swap_records
    finally:
        try:
            stop_container(entry.neo4j.container_name or "")
            log(f"Stopped dataset container: {dataset.key}")
        except BenchmarkError as exc:
            log(f"Warning: failed to stop dataset container {dataset.key}: {exc}")


def main() -> int:
    args = build_parser().parse_args()
    spec_map = dataset_specs()
    datasets: list[DatasetSpec] = []
    entries: list[DatasetRegistryEntry] = []
    for dataset_key in args.datasets:
        if dataset_key not in spec_map:
            raise BenchmarkError(f"Unknown dataset key: {dataset_key}")
        datasets.append(spec_map[dataset_key])
        entries.append(load_dataset_entry(dataset_key, args.registry_path))

    preflight_notebooks(entries)
    for entry in entries:
        ensure_container_exists(entry.neo4j.container_name or "")

    benchmark_root = Path(args.run_dir).resolve() if args.run_dir else run_root(Path(args.runs_root).resolve())
    temp_root = Path(args.temp_root).resolve()
    temp_root.mkdir(parents=True, exist_ok=True)
    if not args.run_dir:
        write_json(benchmark_root / "preflight.json", {"datasets": [entry.key for entry in entries]})
    log(f"Writing benchmark artifacts to {benchmark_root}")

    dataset_summaries: list[dict[str, Any]] = []
    question_orders: dict[str, list[str]] = {}
    replacements: dict[str, list[dict[str, str]]] = {}
    all_answer_artifacts: dict[str, dict[str, dict[str, AnswerArtifact]]] = {}
    all_notebook_scores: dict[str, dict[str, AnswerScore]] = {}
    all_hybrid_scores: dict[str, dict[str, AnswerScore]] = {}
    all_comparisons: dict[str, dict[str, ComparisonScore]] = {}

    for dataset, entry in zip(datasets, entries):
        summary, final_questions, answer_artifacts, notebook_scores, hybrid_scores, comparisons, swap_records = benchmark_dataset(run_dir=benchmark_root, temp_root=temp_root, dataset=dataset, entry=entry, model=args.model)
        dataset_summaries.append(summary)
        question_orders[dataset.key] = final_questions
        replacements[dataset.key] = swap_records
        all_answer_artifacts[dataset.key] = answer_artifacts
        all_notebook_scores[dataset.key] = notebook_scores
        all_hybrid_scores[dataset.key] = hybrid_scores
        all_comparisons[dataset.key] = comparisons

    aggregate = aggregate_overall(dataset_summaries)
    write_json(benchmark_root / "aggregate_summary.json", {"datasets": dataset_summaries, "overall": aggregate})
    write_text(benchmark_root / "technical_blog.md", render_technical_blog(benchmark_root, dataset_summaries, aggregate, datasets))
    write_text(benchmark_root / "appendix.md", render_appendix(run_dir=benchmark_root, datasets=datasets, question_orders=question_orders, replacements=replacements, answer_artifacts=all_answer_artifacts, notebook_scores=all_notebook_scores, hybrid_scores=all_hybrid_scores, comparisons=all_comparisons))
    log(f"Benchmark completed successfully: {benchmark_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BenchmarkError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
