#!/usr/bin/env python3
"""Command-line tool for computing message metrics from SWE trajectories."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[38;5;203m"
GREEN = "\033[38;5;42m"
BLUE = "\033[38;5;39m"
GOLD = "\033[38;5;221m"
CYAN = "\033[38;5;45m"
SLATE = "\033[38;5;244m"
WHITE = "\033[38;5;15m"

ROLE_COLORS = {
    "system": RED,
    "user": GOLD,
    "assistant": BLUE,
    "tool": GREEN,
    "unknown": SLATE
}
TRACK_CHARS = {
    "system": "■",
    "user": "■",
    "assistant": "■",
    "tool": "■",
    "unknown": "■",
}
KNOWN_ROLES = ("system", "user", "assistant", "tool")


@dataclass(frozen=True)
class Metrics:
    system: int
    user: int
    assistant: int
    tool: int
    total: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class AnalysisResult:
    source: str
    metrics: Metrics
    collection_id: str | None = None
    collection_name: str | None = None
    run_id: str | None = None
    trajectory_roles: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "collection_id": self.collection_id,
            "collection_name": self.collection_name,
            "run_id": self.run_id,
            "metrics": self.metrics.to_dict(),
        }


@dataclass(frozen=True)
class RemoteTarget:
    collection_id: str
    collection_name: str
    run_id: str


@dataclass(frozen=True)
class RemoteSelection:
    mode: str
    targets: list[RemoteTarget]

    @property
    def is_batch(self) -> bool:
        return self.mode in {"all_runs", "all_collections"}


@dataclass(frozen=True)
class AverageMetrics:
    system: float
    user: float
    assistant: float
    tool: float
    total: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class CollectionAggregate:
    collection_id: str
    collection_name: str
    short_name: str
    run_count: int
    average_metrics: AverageMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "collection_id": self.collection_id,
            "collection_name": self.collection_name,
            "short_name": self.short_name,
            "run_count": self.run_count,
            "average_metrics": self.average_metrics.to_dict(),
        }


@dataclass(frozen=True)
class BatchAggregate:
    scope: str
    run_count: int
    overall_average: AverageMetrics
    collections: list[CollectionAggregate]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "run_count": self.run_count,
            "overall_average": self.overall_average.to_dict(),
            "collections": [collection.to_dict() for collection in self.collections],
        }


class UserCancelled(RuntimeError):
    """Raised when the user aborts the interactive selector."""


def try_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def load_local(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_docent_client(api_key: str | None = None) -> Any:
    try_load_dotenv()

    try:
        from docent import Docent
    except ImportError as exc:
        raise RuntimeError(
            "Docent support requires the 'docent' package to be installed."
        ) from exc

    resolved_api_key = api_key or os.getenv("DOCENT_API_KEY")
    if not resolved_api_key:
        raise RuntimeError(
            "Docent support requires DOCENT_API_KEY or --api-key."
        )

    return Docent(api_key=resolved_api_key)


def fetch_docent_run(collection_id: str, run_id: str, api_key: str | None = None) -> Any:
    client = build_docent_client(api_key=api_key)
    return client.get_agent_run(collection_id, run_id)


def list_docent_collections(api_key: str | None = None) -> list[Any]:
    client = build_docent_client(api_key=api_key)
    return list(client.list_collections())


def list_docent_run_ids(collection_id: str, api_key: str | None = None) -> list[str]:
    client = build_docent_client(api_key=api_key)
    return list(client.list_agent_run_ids(collection_id))





def extract_mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def attr_or_key(value: Any, name: str, default: Any = None) -> Any:
    mapping = extract_mapping(value)
    if mapping is not None:
        return mapping.get(name, default)
    return getattr(value, name, default)


def normalize_message(message: Any) -> dict[str, Any]:
    mapping = extract_mapping(message)
    if mapping is not None:
        return dict(mapping)

    normalized: dict[str, Any] = {}
    for field in ("id", "role", "content", "metadata", "tool_call_id", "function"):
        value = getattr(message, field, None)
        if value is not None:
            normalized[field] = value
    return normalized


def flatten_transcripts(transcripts: Iterable[Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for transcript in transcripts:
        transcript_messages = attr_or_key(transcript, "messages", [])
        messages.extend(normalize_message(message) for message in transcript_messages)
    return messages


def extract_messages(payload: Any) -> list[dict[str, Any]]:
    """Handle local JSON payloads and Docent SDK objects."""
    if payload is None:
        raise ValueError("The trajectory payload is empty.")

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        if not payload:
            raise ValueError("The trajectory payload does not contain any runs.")
        messages: list[dict[str, Any]] = []
        for item in payload:
            try:
                messages.extend(extract_messages(item))
            except ValueError:
                continue
        if messages:
            return messages
        raise ValueError("Unsupported trajectory format: could not find messages.")

    direct_messages = attr_or_key(payload, "messages")
    if direct_messages is not None:
        return [normalize_message(message) for message in direct_messages]

    agent_run = attr_or_key(payload, "agent_run")
    if agent_run is not None:
        nested_messages = attr_or_key(agent_run, "messages")
        if nested_messages is not None:
            return [normalize_message(message) for message in nested_messages]

    transcripts = attr_or_key(payload, "transcripts")
    if transcripts is not None:
        messages = flatten_transcripts(transcripts)
        if messages:
            return messages

    raise ValueError("Unsupported trajectory format: could not find messages.")


def normalize_role(message: Mapping[str, Any]) -> str:
    return str(message.get("role", "unknown")).strip().lower()


def compute_metrics(messages: Sequence[Mapping[str, Any]]) -> Metrics:
    counts = Counter(normalize_role(message) for message in messages)
    return Metrics(
        system=counts.get("system", 0),
        user=counts.get("user", 0),
        assistant=counts.get("assistant", 0),
        tool=counts.get("tool", 0),
        total=len(messages),
    )


def extract_trajectory_roles(messages: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(normalize_role(message) for message in messages)


def analyze_payload(payload: Any, source_label: str) -> AnalysisResult:
    messages = extract_messages(payload)
    return AnalysisResult(
        source=source_label,
        metrics=compute_metrics(messages),
        trajectory_roles=extract_trajectory_roles(messages),
    )


def analyze_remote_run(
    client: Any,
    collection_id: str,
    run_id: str,
    collection_name: str | None = None,
) -> AnalysisResult:
    payload = client.get_agent_run(collection_id, run_id)
    messages = extract_messages(payload)
    return AnalysisResult(
        source=f"Docent run: {collection_id}/{run_id}",
        collection_id=collection_id,
        collection_name=collection_name,
        run_id=run_id,
        metrics=compute_metrics(messages),
        trajectory_roles=extract_trajectory_roles(messages),
    )


def percent(count: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{(count / total) * 100:4.1f}%"


def colorize(text: str, color: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{color}{text}{RESET}"


def format_metric_value(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.1f}"


def draw_bar(count: int, total: int, role: str, enabled: bool, width: int = 20) -> str:
    if total <= 0:
        filled = 0
    else:
        filled = max(1, round((count / total) * width)) if count else 0
    bar = TRACK_CHARS[role] * filled
    muted = "·" * (width - filled)
    return colorize(bar, ROLE_COLORS[role], enabled) + colorize(muted, SLATE, enabled)


def draw_trajectory_track(
    roles: Sequence[str],
    enabled: bool,
    *,
    width: int = 80,
) -> list[str]:
    if not roles:
        return [colorize("No messages", SLATE, enabled)]

    rendered_boxes = [
        colorize(TRACK_CHARS.get(role, TRACK_CHARS["unknown"]), ROLE_COLORS.get(role, SLATE), enabled)
        for role in roles
    ]
    return [
        "".join(rendered_boxes[index:index + width])
        for index in range(0, len(rendered_boxes), width)
    ]


def print_report(
    metrics: Metrics,
    source_label: str,
    color: bool,
    *,
    trajectory_roles: Sequence[str] = (),
) -> None:
    title = colorize("Trajectory Message Analysis", BOLD + GOLD, color)
    subtitle = colorize(source_label, DIM, color)
    track_label = colorize("Trajectory", BOLD, color)

    print()
    print(title)
    print(subtitle)
    trajectory_lines = draw_trajectory_track(trajectory_roles, color)
    print(f"{track_label}  {trajectory_lines[0]}")
    for line in trajectory_lines[1:]:
        print(f"{' ' * 12}{line}")
    print(colorize("─" * 58, SLATE, color))

    rows = [
        ("System", metrics.system, "system"),
        ("User", metrics.user, "user"),
        ("Assistant", metrics.assistant, "assistant"),
        ("Tool", metrics.tool, "tool"),
    ]
    for label, count, role in rows:
        role_label = colorize(f"{label:<10}", ROLE_COLORS[role] + BOLD, color)
        print(
            f"{role_label} {count:>5}  "
            f"{percent(count, metrics.total):>6}  "
            f"{draw_bar(count, metrics.total, role, color)}"
        )

    print(colorize("─" * 58, SLATE, color))
    total_label = colorize("Total messages:", BOLD, color)
    print(f"{total_label:<10}   {metrics.total:>5}")


def print_average_report(
    metrics: AverageMetrics,
    source_label: str,
    color: bool,
    *,
    title_text: str,
    run_count: int,
) -> None:
    title = colorize(title_text, BOLD + GOLD, color)
    subtitle = colorize(f"{source_label}", DIM, color)

    print()
    print(title)
    print(subtitle)
    print(f"Averaged across {run_count} runs")
    print(colorize("─" * 58, SLATE, color))

    rows = [
        ("System", metrics.system, "system"),
        ("User", metrics.user, "user"),
        ("Assistant", metrics.assistant, "assistant"),
        ("Tool", metrics.tool, "tool"),
    ]
    for label, count, role in rows:
        role_label = colorize(f"{label:<10}", ROLE_COLORS[role] + BOLD, color)
        print(
            f"{role_label} {format_metric_value(count):>5}  "
            f"{percent(count, metrics.total):>6}  "
            f"{draw_bar(count, metrics.total, role, color)}"
        )

    print(colorize("─" * 58, SLATE, color))
    total_label = colorize("Average messages in a trajectory:", BOLD, color)
    print(f"{total_label:<10}  {format_metric_value(metrics.total):>5}")


def print_batch_header(count: int, color: bool) -> None:
    print()
    print(colorize("Remote Trajectory Analysis", BOLD + GOLD, color))
    print(colorize(f"Selected trajectories: {count}", DIM, color))


def print_batch_report(results: Sequence[AnalysisResult], color: bool) -> None:
    print_batch_header(len(results), color=color)
    for index, result in enumerate(results, start=1):
        if len(results) > 1:
            print(colorize(f"\n[{index}/{len(results)}]", DIM, color))
        print_report(
            result.metrics,
            result.source,
            color=color,
            trajectory_roles=result.trajectory_roles,
        )


def print_progress(current: int, total: int, color: bool, label: str = "Processing trajectories") -> None:
    if total <= 0:
        return

    width = 28
    ratio = current / total
    filled = round(ratio * width)
    bar = "#" * filled + "-" * (width - filled)
    progress_text = f"\r{label} {current:>4}/{total:<4} [{bar}] {ratio * 100:5.1f}%"
    sys.stdout.write(colorize(progress_text, CYAN if current < total else GREEN, color))
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")


def shorten_collection_name(name: str, collection_id: str, max_width: int = 10) -> str:
    base = (name or "").strip() or collection_id
    if len(base) <= max_width:
        return base
    return f"{base[: max_width - 3]}..."


def compute_average_metrics(metrics_values: Sequence[Metrics]) -> AverageMetrics:
    if not metrics_values:
        raise ValueError("Cannot compute averages without any trajectories.")

    count = len(metrics_values)
    return AverageMetrics(
        system=sum(metrics.system for metrics in metrics_values) / count,
        user=sum(metrics.user for metrics in metrics_values) / count,
        assistant=sum(metrics.assistant for metrics in metrics_values) / count,
        tool=sum(metrics.tool for metrics in metrics_values) / count,
        total=sum(metrics.total for metrics in metrics_values) / count,
    )


def aggregate_batch_results(results: Sequence[AnalysisResult], scope: str) -> BatchAggregate:
    if not results:
        raise ValueError("Cannot aggregate an empty result set.")

    grouped: dict[str, list[AnalysisResult]] = {}
    order: list[str] = []
    for result in results:
        collection_id = result.collection_id or "unknown-collection"
        if collection_id not in grouped:
            grouped[collection_id] = []
            order.append(collection_id)
        grouped[collection_id].append(result)

    collections: list[CollectionAggregate] = []
    for collection_id in order:
        collection_results = grouped[collection_id]
        collection_name = collection_results[0].collection_name or collection_id
        collections.append(
            CollectionAggregate(
                collection_id=collection_id,
                collection_name=collection_name,
                short_name=shorten_collection_name(collection_name, collection_id),
                run_count=len(collection_results),
                average_metrics=compute_average_metrics(
                    [result.metrics for result in collection_results]
                ),
            )
        )

    return BatchAggregate(
        scope=scope,
        run_count=len(results),
        overall_average=compute_average_metrics([result.metrics for result in results]),
        collections=collections,
    )


def sanitize_filename_part(value: str) -> str:
    sanitized = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    return sanitized.strip("_") or "export"


def ensure_out_dir(base_dir: Path | None = None) -> Path:
    out_dir = (base_dir or Path.cwd()) / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def build_all_runs_export_payload(
    selection: RemoteSelection,
    aggregate: BatchAggregate,
    results: Sequence[AnalysisResult],
) -> dict[str, Any]:
    collection = aggregate.collections[0]
    return {
        "scope": "all_runs",
        "collection": {
            "collection_id": collection.collection_id,
            "collection_name": collection.collection_name,
        },
        "run_count": aggregate.run_count,
        "average_metrics": aggregate.overall_average.to_dict(),
        "trajectory_metrics": [result.to_dict() for result in results],
    }


def build_all_collections_runs_payload(
    aggregate: BatchAggregate,
    results: Sequence[AnalysisResult],
) -> dict[str, Any]:
    grouped: dict[str, list[AnalysisResult]] = {}
    for result in results:
        collection_id = result.collection_id or "unknown-collection"
        grouped.setdefault(collection_id, []).append(result)

    collections_payload: list[dict[str, Any]] = []
    for collection in aggregate.collections:
        collection_results = grouped.get(collection.collection_id, [])
        collections_payload.append(
            {
                "collection_id": collection.collection_id,
                "collection_name": collection.collection_name,
                "short_name": collection.short_name,
                "run_count": collection.run_count,
                "aggregate_metrics": collection.average_metrics.to_dict(),
                "trajectory_metrics": [result.to_dict() for result in collection_results],
            }
        )

    return {
        "scope": "all_collections",
        "run_count": aggregate.run_count,
        "overall_average": aggregate.overall_average.to_dict(),
        "collections": collections_payload,
    }


def write_json_file(path: Path, payload: Mapping[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def export_batch_results(
    selection: RemoteSelection,
    aggregate: BatchAggregate,
    results: Sequence[AnalysisResult],
    *,
    base_dir: Path | None = None,
) -> list[Path]:
    out_dir = ensure_out_dir(base_dir)
    if selection.mode == "all_runs":
        collection_id = selection.targets[0].collection_id
        export_path = out_dir / f"all_runs_{sanitize_filename_part(collection_id)}_metrics.json"
        write_json_file(
            export_path,
            build_all_runs_export_payload(selection, aggregate, results),
        )
        return [export_path]

    runs_path = out_dir / "all_collections_runs_data.json"
    write_json_file(
        runs_path,
        build_all_collections_runs_payload(aggregate, results),
    )
    return [runs_path]


def load_cached_all_runs_metrics(
    collection_id: str,
    base_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Try to load cached metrics for all_runs mode."""
    out_dir = ensure_out_dir(base_dir)
    cache_path = out_dir / f"all_runs_{sanitize_filename_part(collection_id)}_metrics.json"
    
    if not cache_path.exists():
        return None
    
    try:
        return load_local(str(cache_path))
    except (OSError, json.JSONDecodeError):
        return None


def load_cached_all_collections_metrics(
    base_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Try to load cached metrics for all_collections mode."""
    out_dir = ensure_out_dir(base_dir)
    runs_path = out_dir / "all_collections_runs_data.json"

    if not runs_path.exists():
        return None

    try:
        return load_local(str(runs_path))
    except (OSError, json.JSONDecodeError):
        return None


def reconstruct_results_from_cached(
    cached_data: dict[str, Any],
    is_all_collections: bool = False,
) -> list[AnalysisResult]:
    """Reconstruct AnalysisResult objects from cached metrics data."""
    results: list[AnalysisResult] = []
    
    if is_all_collections:
        # For all_collections, iterate through collections and their trajectory_metrics
        for collection_data in cached_data.get("collections", []):
            for trajectory_data in collection_data.get("trajectory_metrics", []):
                result = AnalysisResult(
                    source=trajectory_data.get("source", ""),
                    collection_id=trajectory_data.get("collection_id"),
                    collection_name=trajectory_data.get("collection_name"),
                    run_id=trajectory_data.get("run_id"),
                    metrics=Metrics(
                        system=trajectory_data["metrics"]["system"],
                        user=trajectory_data["metrics"]["user"],
                        assistant=trajectory_data["metrics"]["assistant"],
                        tool=trajectory_data["metrics"]["tool"],
                        total=trajectory_data["metrics"]["total"],
                    ),
                )
                results.append(result)
    else:
        # For all_runs, trajectory_metrics is at the top level
        for trajectory_data in cached_data.get("trajectory_metrics", []):
            result = AnalysisResult(
                source=trajectory_data.get("source", ""),
                collection_id=trajectory_data.get("collection_id"),
                collection_name=trajectory_data.get("collection_name"),
                run_id=trajectory_data.get("run_id"),
                metrics=Metrics(
                    system=trajectory_data["metrics"]["system"],
                    user=trajectory_data["metrics"]["user"],
                    assistant=trajectory_data["metrics"]["assistant"],
                    tool=trajectory_data["metrics"]["tool"],
                    total=trajectory_data["metrics"]["total"],
                ),
            )
            results.append(result)
    
    return results


def try_load_cached_metrics(
    selection: RemoteSelection,
    base_dir: Path | None = None,
) -> list[AnalysisResult] | None:
    """Try to load cached metrics for the given selection.
    
    Returns: list of AnalysisResult if cache found and valid, None otherwise.
    """
    if selection.mode == "single":
        # Single runs are not cached
        return None
    
    if selection.mode == "all_runs":
        collection_id = selection.targets[0].collection_id
        cached_data = load_cached_all_runs_metrics(collection_id, base_dir)
        if cached_data and cached_data.get("scope") == "all_runs":
            # Verify collection_id matches
            collection_info = cached_data.get("collection", {})
            if collection_info.get("collection_id") == collection_id:
                return reconstruct_results_from_cached(cached_data, is_all_collections=False)
        return None
    
    if selection.mode == "all_collections":
        runs_data = load_cached_all_collections_metrics(base_dir)
        if runs_data and runs_data.get("scope") == "all_collections":
            return reconstruct_results_from_cached(runs_data, is_all_collections=True)
        return None
    
    return None


def print_export_paths(paths: Sequence[Path], color: bool) -> None:
    if not paths:
        return

    print()
    print(colorize("Saved JSON exports", BOLD + GOLD, color))
    for path in paths:
        print(colorize(f"- {path}", DIM, color))


def print_collection_comparison_table(collections: Sequence[CollectionAggregate], color: bool) -> None:
    if not collections:
        return

    print()
    print(colorize("Collection Comparison", BOLD + GOLD, color))
    print(colorize("Percent share of messages within the average run", DIM, color))

    label_width = 12
    column_width = max(
        12,
        max(len(collection.collection_name) for collection in collections)+2,
    )

    header = f"{'Metric':<{label_width}}" + "".join(
        f"{collection.collection_name:>{column_width}}" for collection in collections
    )
    print(colorize(header, BOLD, color))
    print(colorize("-" * len(header), SLATE, color))

    rows = [
        ("System %", lambda collection: percent(collection.average_metrics.system, collection.average_metrics.total)),
        ("User %", lambda collection: percent(collection.average_metrics.user, collection.average_metrics.total)),
        ("Assist %", lambda collection: percent(collection.average_metrics.assistant, collection.average_metrics.total)),
        ("Tool %", lambda collection: percent(collection.average_metrics.tool, collection.average_metrics.total)),
        ("Avg msgs", lambda collection: format_metric_value(collection.average_metrics.total)),
    ]
    for label, formatter in rows:
        row = f"{label:<{label_width}}" + "".join(
            f"{formatter(collection):>{column_width}}" for collection in collections
        )
        print(row)


def print_aggregate_report(aggregate: BatchAggregate, color: bool) -> None:
    if aggregate.scope == "all_runs":
        collection = aggregate.collections[0]
        print_average_report(
            collection.average_metrics,
            f"{collection.collection_name} [{collection.collection_id}]",
            color,
            title_text="Collection Average Across Runs",
            run_count=collection.run_count,
        )
        return

    print_average_report(
        aggregate.overall_average,
        "All selected collections",
        color,
        title_text="Overall Average Across All Runs",
        run_count=aggregate.run_count,
    )
    for collection in aggregate.collections:
        print_average_report(
            collection.average_metrics,
            f"{collection.collection_name} [{collection.collection_id}]",
            color,
            title_text="Collection Average Across Runs",
            run_count=collection.run_count,
        )
    print_collection_comparison_table(aggregate.collections, color)


def print_collection_list(collections: Sequence[Any], color: bool) -> None:
    print()
    print(colorize("Available Docent Collections", BOLD + GOLD, color))
    print(colorize("─" * 58, SLATE, color))
    for collection in collections:
        collection_id = attr_or_key(collection, "id", "<missing-id>")
        name = attr_or_key(collection, "name", "(unnamed collection)")
        print(
            f"{colorize(str(collection_id), CYAN + BOLD, color)}  "
            f"{colorize(str(name), SLATE, color)}"
        )


def print_run_list(collection_id: str, run_ids: Sequence[str], color: bool) -> None:
    print()
    print(colorize(f"Docent Runs for {collection_id}", BOLD + GOLD, color))
    print(colorize("─" * 58, SLATE, color))
    for run_id in run_ids:
        print(colorize(str(run_id), CYAN + BOLD, color))


def print_selector_header(title: str, subtitle: str | None, color: bool) -> None:
    print()
    print(colorize(title, BOLD + GOLD, color))
    if subtitle:
        print(subtitle)
    print(colorize("Type a number, 'all', or 'q' to quit.", DIM, color))
    print(colorize("─" * 58, SLATE, color))


def prompt_selection(
    *,
    title: str,
    options: Sequence[tuple[str, str]],
    color: bool,
    subtitle: str | None = None,
    all_label: str | None = None,
) -> str:
    if not options:
        raise ValueError("No options are available for selection.")

    while True:
        print_selector_header(title, subtitle, color)
        if all_label:
            print(f"  {colorize('[all]'  , CYAN + BOLD, color)}  {all_label}")
        for index, (value, label) in enumerate(options, start=1):
            option_no = colorize(f"[{index}]", CYAN + BOLD, color)
            print(f"  {option_no:<12}    {label}")

        try:
            raw = input(colorize("\nSelection: ", GREEN + BOLD, color)).strip()
        except EOFError as exc:
            raise UserCancelled("Input stream closed during selection.") from exc

        if not raw:
            print(colorize("Please enter a selection.", RED, color))
            continue

        lowered = raw.lower()
        if lowered in {"q", "quit", "exit"}:
            raise UserCancelled("Selection cancelled by user.")
        if all_label and lowered == "all":
            return "all"
        if raw.isdigit():
            index = int(raw) - 1
            if 0 <= index < len(options):
                return options[index][0]
            print(colorize("That number is out of range.", RED, color))
            continue

        for value, _label in options:
            if raw == value:
                return value

        print(colorize("Unknown selection. Try again.", RED, color))


def prompt_compact_selection(
    *,
    title: str,
    subtitle: str,
    options: Sequence[tuple[str, str]],
    color: bool,
    all_label: str,
    value_label: str,
) -> str:
    if not options:
        raise ValueError("No options are available for selection.")

    max_index = len(options)
    values = {value for value, _label in options}

    while True:
        print()
        print(colorize(title, BOLD + GOLD, color))
        print(subtitle)
        print(f"Type a number (1-{max_index}), 'all', 'id:<{value_label}>', or 'q' to quit.")
        print(colorize("─" * 58, SLATE, color))
        print(f"  {colorize('[all]      ', CYAN + BOLD, color)}  {all_label}")
        print(f"  {colorize('[id:<...>] ', CYAN + BOLD, color)}  Enter a full {value_label} manually")
        print(f"  {colorize('[number]   ', CYAN + BOLD, color)}  Select by the trajectory's number in the list [1-{max_index}]")

        try:
            raw = input(colorize("\nSelection: ", GREEN + BOLD, color)).strip()
        except EOFError as exc:
            raise UserCancelled("Input stream closed during selection.") from exc

        if not raw:
            print(colorize("Please enter a selection.", RED, color))
            continue

        lowered = raw.lower()
        if lowered in {"q", "quit", "exit"}:
            raise UserCancelled("Selection cancelled by user.")
        if lowered == "all":
            return "all"
        if lowered.startswith("id:"):
            manual_value = raw[3:].strip()
            if manual_value in values:
                return manual_value
            print(colorize(f"That {value_label} was not found.", RED, color))
            continue
        if raw.isdigit():
            index = int(raw) - 1
            if 0 <= index < len(options):
                return options[index][0]
            print(colorize("That number is out of range.", RED, color))
            continue
        if raw in values:
            return raw

        print(colorize("Unknown selection. Try again.", RED, color))


def normalize_collection_records(collections: Sequence[Any]) -> list[tuple[str, str]]:
    normalized: list[tuple[str, str]] = []
    for collection in collections:
        collection_id = str(attr_or_key(collection, "id", "<missing-id>"))
        name = str(attr_or_key(collection, "name", "(unnamed collection)"))
        normalized.append((collection_id, f"{name}  {colorize(f'[{collection_id}]', SLATE, True)}"))
    return normalized


def build_run_options(run_ids: Sequence[str]) -> list[tuple[str, str]]:
    return [(run_id, run_id) for run_id in run_ids]


def choose_collection(
    collections: Sequence[Any],
    *,
    color: bool,
    preselected_id: str | None = None,
) -> str:
    options = normalize_collection_records(collections)
    if preselected_id:
        for collection_id, _label in options:
            if collection_id == preselected_id:
                return collection_id
        raise ValueError(f"Collection '{preselected_id}' was not found.")

    return prompt_selection(
        title="Choose a Collection",
        subtitle=f"{len(options)} collections available",
        options=options,
        color=color,
        all_label="Analyze every collection",
    )


def choose_run(
    collection_id: str,
    run_ids: Sequence[str],
    *,
    color: bool,
    preselected_id: str | None = None,
) -> str:
    options = build_run_options(run_ids)
    if preselected_id:
        for run_id, _label in options:
            if run_id == preselected_id:
                return run_id
        raise ValueError(
            f"Run '{preselected_id}' was not found in collection '{collection_id}'."
        )

    return prompt_compact_selection(
        title="Choose a Trajectory",
        subtitle=f"Collection {collection_id} has {len(options)} trajectories",
        options=options,
        color=color,
        all_label="Analyze every trajectory in this collection",
        value_label="run id",
    )


def run_remote_selector(
    *,
    api_key: str | None,
    color: bool,
    collection_id: str | None,
    run_id: str | None,
    all_collections: bool,
    all_runs: bool,
) -> RemoteSelection:
    client = build_docent_client(api_key=api_key)
    collections = list(client.list_collections())
    if not collections:
        raise ValueError("No Docent collections were returned.")

    collection_names = {
        str(attr_or_key(collection, "id", "<missing-id>")): str(
            attr_or_key(collection, "name", "(unnamed collection)")
        )
        for collection in collections
    }
    selected_collection = "all" if all_collections else choose_collection(
        collections,
        color=color,
        preselected_id=collection_id,
    )
    selected_targets: list[RemoteTarget] = []

    if selected_collection == "all":
        for selected_collection_id, _label in normalize_collection_records(collections):
            run_ids = list(client.list_agent_run_ids(selected_collection_id))
            for selected_run_id in run_ids:
                selected_targets.append(
                    RemoteTarget(
                        collection_id=selected_collection_id,
                        collection_name=collection_names[selected_collection_id],
                        run_id=selected_run_id,
                    )
                )
        return RemoteSelection(mode="all_collections", targets=selected_targets)

    run_ids = list(client.list_agent_run_ids(selected_collection))
    if not run_ids:
        raise ValueError(f"Collection '{selected_collection}' has no trajectories.")

    selected_run = "all" if all_runs else choose_run(
        selected_collection,
        run_ids,
        color=color,
        preselected_id=run_id,
    )
    if selected_run == "all":
        return RemoteSelection(
            mode="all_runs",
            targets=[
                RemoteTarget(
                    collection_id=selected_collection,
                    collection_name=collection_names[selected_collection],
                    run_id=selected_run_id,
                )
                for selected_run_id in run_ids
            ],
        )
    return RemoteSelection(
        mode="single",
        targets=[
            RemoteTarget(
                collection_id=selected_collection,
                collection_name=collection_names[selected_collection],
                run_id=selected_run,
            )
        ],
    )


def analyze_remote_targets(
    targets: Sequence[RemoteTarget],
    *,
    api_key: str | None,
    color: bool,
    show_progress: bool,
) -> list[AnalysisResult]:
    client = build_docent_client(api_key=api_key)
    results: list[AnalysisResult] = []
    total = len(targets)
    if show_progress:
        print_progress(0, total, color)

    for index, target in enumerate(targets, start=1):
        results.append(
            analyze_remote_run(
                client,
                target.collection_id,
                target.run_id,
                collection_name=target.collection_name,
            )
        )
        if show_progress:
            print_progress(index, total, color)

    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute message counts for a mini-SWE-agent-v2 trajectory.\n\n"
            "Input modes:\n"
            "  local  : requires PATH to a local JSON trajectory file.\n"
            "  remote : interactive by default; requires Docent authentication via\n"
            "           DOCENT_API_KEY or --api-key. Optional flags can preselect a\n"
            "           collection, a run, or all runs.\n"
            "  runs   : requires --collection-id.\n"
            "  collections : requires Docent authentication only.\n\n"
            "Examples:\n"
            "  trajectory_scanner.py local data/sample_trajectory.json\n"
            "  trajectory_scanner.py --json local data/sample_trajectory.json\n"
            "  trajectory_scanner.py remote\n"
            "  trajectory_scanner.py remote --collection-id COLLECTION_ID\n"
            "  trajectory_scanner.py remote --collection-id COLLECTION_ID --run-id RUN_ID\n"
            "  trajectory_scanner.py remote --collection-id COLLECTION_ID --all-runs\n"
            "  trajectory_scanner.py remote --all-collections\n"
            "  trajectory_scanner.py collections\n"
            "  trajectory_scanner.py runs --collection-id COLLECTION_ID"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Print machine-readable JSON instead of the colorful report.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors in terminal output.",
    )

    subparsers = parser.add_subparsers(dest="source", required=True)

    local_parser = subparsers.add_parser(
        "local",
        help="Analyze a local trajectory JSON file.",
        description=(
            "Analyze a local trajectory JSON file.\n\n"
            "Required input:\n"
            "  PATH  Path to the local JSON trajectory file."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    local_parser.add_argument(
        "path",
        metavar="PATH",
        help="Required. Path to the local trajectory JSON file.",
    )

    remote_parser = subparsers.add_parser(
        "remote",
        help="Analyze remote Docent trajectories, with an interactive selector by default.",
        description=(
            "Analyze remote Docent trajectories.\n\n"
            "Authentication:\n"
            "  Requires DOCENT_API_KEY or --api-key.\n\n"
            "Selection modes:\n"
            "  1. No selection flags:\n"
            "     Opens the interactive selector for collection and trajectory.\n"
            "  2. --collection-id COLLECTION_ID:\n"
            "     Skips collection selection and prompts for a trajectory.\n"
            "  3. --collection-id COLLECTION_ID --run-id RUN_ID:\n"
            "     Analyzes one specific trajectory directly.\n"
            "  4. --collection-id COLLECTION_ID --all-runs:\n"
            "     Analyzes all trajectories in one collection.\n"
            "  5. --all-collections:\n"
            "     Analyzes all trajectories across all collections.\n\n"
            "Mandatory combinations:\n"
            "  --run-id requires --collection-id.\n"
            "  --all-runs requires --collection-id."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    remote_parser.add_argument(
        "--collection-id",
        metavar="COLLECTION_ID",
        help="Optional. Preselect one collection. Required when using --run-id or --all-runs.",
    )
    remote_parser.add_argument(
        "--run-id",
        metavar="RUN_ID",
        help="Optional. Analyze one specific trajectory. Requires --collection-id.",
    )
    remote_parser.add_argument(
        "--all-collections",
        action="store_true",
        help="Optional. Analyze every collection and every trajectory without prompting.",
    )
    remote_parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Optional. Analyze every trajectory in the selected collection. Requires --collection-id.",
    )
    remote_parser.add_argument(
        "--api-key",
        metavar="API_KEY",
        help="Optional. Docent API key. If omitted, DOCENT_API_KEY must be set.",
    )

    collections_parser = subparsers.add_parser(
        "collections",
        help="List available Docent collections.",
        description=(
            "List available Docent collections.\n\n"
            "Authentication:\n"
            "  Requires DOCENT_API_KEY or --api-key."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    collections_parser.add_argument(
        "--api-key",
        metavar="API_KEY",
        help="Optional. Docent API key. If omitted, DOCENT_API_KEY must be set.",
    )

    runs_parser = subparsers.add_parser(
        "runs",
        help="List available run IDs for a Docent collection.",
        description=(
            "List available run IDs for one Docent collection.\n\n"
            "Required input:\n"
            "  --collection-id COLLECTION_ID\n\n"
            "Authentication:\n"
            "  Requires DOCENT_API_KEY or --api-key."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    runs_parser.add_argument(
        "--collection-id",
        required=True,
        metavar="COLLECTION_ID",
        help="Required. Docent collection ID whose run IDs should be listed.",
    )
    runs_parser.add_argument(
        "--api-key",
        metavar="API_KEY",
        help="Optional. Docent API key. If omitted, DOCENT_API_KEY must be set.",
    )

    return parser


def resolve_payload(args: argparse.Namespace) -> tuple[Any, str]:
    if args.source == "local":
        resolved_path = Path(args.path).expanduser().resolve()
        payload = load_local(str(resolved_path))
        try:
            relative_path = resolved_path.relative_to(Path.cwd())
        except ValueError:
            # Path is not relative to cwd (e.g., different mount points on macOS)
            relative_path = resolved_path
        return payload, f"Local file: {relative_path}"

    payload = fetch_docent_run(
        collection_id=args.collection_id,
        run_id=args.run_id,
        api_key=args.api_key,
    )
    return payload, f"Docent run: {args.collection_id}/{args.run_id}"


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.source == "collections":
            collections = list_docent_collections(api_key=args.api_key)
            if args.json_output:
                output = [
                    {
                        "id": attr_or_key(collection, "id"),
                        "name": attr_or_key(collection, "name"),
                    }
                    for collection in collections
                ]
                print(json.dumps(output, indent=2))
            else:
                print_collection_list(collections, color=not args.no_color)
            return 0

        if args.source == "runs":
            run_ids = list_docent_run_ids(
                collection_id=args.collection_id,
                api_key=args.api_key,
            )
            if args.json_output:
                print(
                    json.dumps(
                        {
                            "collection_id": args.collection_id,
                            "run_ids": run_ids,
                        },
                        indent=2,
                    )
                )
            else:
                print_run_list(args.collection_id, run_ids, color=not args.no_color)
            return 0

        if args.source == "remote":
            if args.run_id and not args.collection_id:
                raise ValueError("--run-id requires --collection-id.")
            if args.all_runs and not args.collection_id:
                raise ValueError("--all-runs requires --collection-id.")

            selection = run_remote_selector(
                api_key=args.api_key,
                color=not args.no_color,
                collection_id=args.collection_id,
                run_id=args.run_id,
                all_collections=args.all_collections,
                all_runs=args.all_runs,
            )
            
            # Try to load cached metrics for batch operations
            results: list[AnalysisResult] | None = None
            if selection.is_batch:
                results = try_load_cached_metrics(selection)
                if results:
                    print(colorize("Using cached metrics", DIM, not args.no_color), file=sys.stderr)
            
            # If not using cache, retrieve from remote
            if results is None:
                results = analyze_remote_targets(
                    selection.targets,
                    api_key=args.api_key,
                    color=not args.no_color,
                    show_progress=selection.is_batch,
                )

            if selection.is_batch:
                aggregate = aggregate_batch_results(results, scope=selection.mode)
                export_paths = export_batch_results(selection, aggregate, results)
                if args.json_output:
                    print(
                        json.dumps(
                            {
                                **aggregate.to_dict(),
                                "export_files": [str(path) for path in export_paths],
                            },
                            indent=2,
                        )
                    )
                else:
                    print_aggregate_report(aggregate, color=not args.no_color)
                    print_export_paths(export_paths, color=not args.no_color)
            else:
                result = results[0]
                if args.json_output:
                    print(json.dumps(result.to_dict(), indent=2))
                else:
                    print_report(
                        result.metrics,
                        result.source,
                        color=not args.no_color,
                        trajectory_roles=result.trajectory_roles,
                    )
            return 0

        payload, source_label = resolve_payload(args)
        result = analyze_payload(payload, source_label)
    except (OSError, json.JSONDecodeError, RuntimeError, UserCancelled, ValueError) as exc:
        print(colorize(f"Error: {exc}", RED, not args.no_color), file=sys.stderr)
        return 1

    if args.json_output:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print_report(
            result.metrics,
            result.source,
            color=not args.no_color,
            trajectory_roles=result.trajectory_roles,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
