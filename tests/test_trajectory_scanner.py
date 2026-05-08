from __future__ import annotations

import builtins
import json
import runpy
import sys
import types
from pathlib import Path
from types import SimpleNamespace
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(PROJECT_ROOT))  # pragma: no cover

import trajectory_scanner as ts


class DummyMessage:
    def __init__(
        self,
        *,
        role: str | None = None,
        content: str | None = None,
        message_id: str | None = None,
        metadata: dict | None = None,
        tool_call_id: str | None = None,
        function: str | None = None,
    ) -> None:
        self.id = message_id
        self.role = role
        self.content = content
        self.metadata = metadata
        self.tool_call_id = tool_call_id
        self.function = function


class FakeDocentClient:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key

    def get_agent_run(self, collection_id: str, run_id: str):
        return {
            "transcripts": [
                {
                    "messages": [
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": collection_id},
                        {"role": "assistant", "content": run_id},
                        {"role": "tool", "content": "ok"},
                    ]
                }
            ]
        }

    def list_collections(self):
        return [
            {"id": "col-1", "name": "Alpha"},
            {"id": "col-2", "name": "Beta"},
        ]

    def list_agent_run_ids(self, collection_id: str):
        mapping = {
            "col-1": ["run-a", "run-b"],
            "col-2": ["run-c"],
        }
        return mapping[collection_id]


class StubParser:
    def __init__(self, args):
        self._args = args

    def parse_args(self):
        return self._args


def make_args(**overrides):
    base = {
        "source": "local",
        "json_output": False,
        "no_color": True,
        "path": "unused.json",
        "collection_id": None,
        "run_id": None,
        "all_collections": False,
        "all_runs": False,
        "api_key": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def make_metrics() -> ts.Metrics:
    return ts.Metrics(system=1, user=2, assistant=3, tool=4, total=10)


def make_result(source: str = "sample") -> ts.AnalysisResult:
    return ts.AnalysisResult(
        source=source,
        metrics=make_metrics(),
        collection_id="c",
        collection_name="Collection",
        run_id="r",
    )


def test_analysis_result_to_dict():
    result = make_result("demo")
    assert result.to_dict() == {
        "source": "demo",
        "collection_id": "c",
        "collection_name": "Collection",
        "run_id": "r",
        "metrics": {
            "system": 1,
            "user": 2,
            "assistant": 3,
            "tool": 4,
            "total": 10,
        },
    }


def test_try_load_dotenv_calls_loader(monkeypatch):
    calls: list[str] = []
    fake_module = types.ModuleType("dotenv")
    fake_module.load_dotenv = lambda: calls.append("loaded")
    monkeypatch.setitem(sys.modules, "dotenv", fake_module)

    ts.try_load_dotenv()

    assert calls == ["loaded"]


def test_try_load_dotenv_ignores_import_error(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "dotenv":
            raise ImportError("missing dotenv")
        return original_import(name, *args, **kwargs)  # pragma: no cover

    monkeypatch.setattr(builtins, "__import__", fake_import)

    ts.try_load_dotenv()


def test_load_local_reads_json(out_path):
    payload = {"messages": [{"role": "system"}]}
    file_path = out_path / "sample.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    assert ts.load_local(str(file_path)) == payload


def test_build_docent_client_success_and_wrappers(monkeypatch):
    monkeypatch.setattr(ts, "try_load_dotenv", lambda: None)
    monkeypatch.setenv("DOCENT_API_KEY", "env-key")

    fake_docent_module = types.ModuleType("docent")
    fake_docent_module.Docent = FakeDocentClient
    monkeypatch.setitem(sys.modules, "docent", fake_docent_module)

    client = ts.build_docent_client()

    assert isinstance(client, FakeDocentClient)
    assert client.api_key == "env-key"
    assert ts.fetch_docent_run("col-1", "run-a", api_key="cli-key")["transcripts"][0]["messages"][2]["content"] == "run-a"
    assert ts.list_docent_collections(api_key="cli-key")[0]["id"] == "col-1"
    assert ts.list_docent_run_ids("col-2", api_key="cli-key") == ["run-c"]


def test_build_docent_client_missing_docent_package(monkeypatch):
    monkeypatch.setattr(ts, "try_load_dotenv", lambda: None)
    monkeypatch.delenv("DOCENT_API_KEY", raising=False)
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "docent":
            raise ImportError("missing docent")
        return original_import(name, *args, **kwargs)  # pragma: no cover

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="requires the 'docent' package"):
        ts.build_docent_client()


def test_build_docent_client_requires_api_key(monkeypatch):
    monkeypatch.setattr(ts, "try_load_dotenv", lambda: None)
    monkeypatch.delenv("DOCENT_API_KEY", raising=False)
    fake_docent_module = types.ModuleType("docent")
    fake_docent_module.Docent = FakeDocentClient
    monkeypatch.setitem(sys.modules, "docent", fake_docent_module)

    with pytest.raises(RuntimeError, match="DOCENT_API_KEY"):
        ts.build_docent_client()


def test_extract_helpers_and_normalize_message():
    message = DummyMessage(
        role="assistant",
        content="hello",
        message_id="m1",
        metadata={"ok": True},
        tool_call_id="t1",
    )

    assert ts.extract_mapping({"a": 1}) == {"a": 1}
    assert ts.extract_mapping("x") is None
    assert ts.attr_or_key({"a": 1}, "a", 2) == 1
    assert ts.attr_or_key(SimpleNamespace(a=3), "a", 2) == 3
    assert ts.attr_or_key(SimpleNamespace(), "a", 2) == 2
    assert ts.normalize_message({"role": "user"}) == {"role": "user"}
    assert ts.normalize_message(message) == {
        "id": "m1",
        "role": "assistant",
        "content": "hello",
        "metadata": {"ok": True},
        "tool_call_id": "t1",
    }


def test_flatten_transcripts_and_extract_messages_variants():
    transcript_object = SimpleNamespace(
        messages=[DummyMessage(role="system", content="sys"), DummyMessage(role="user", content="usr")]
    )
    flattened = ts.flatten_transcripts([transcript_object])
    assert flattened == [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]

    direct = ts.extract_messages({"messages": [{"role": "assistant"}]})
    nested = ts.extract_messages({"agent_run": {"messages": [{"role": "tool"}]}})
    transcript = ts.extract_messages({"transcripts": [{"messages": [{"role": "system"}]}]})
    sequence = ts.extract_messages([{}, {"messages": [{"role": "user"}]}])

    assert direct == [{"role": "assistant"}]
    assert nested == [{"role": "tool"}]
    assert transcript == [{"role": "system"}]
    assert sequence == [{"role": "user"}]


def test_extract_messages_errors():
    with pytest.raises(ValueError, match="empty"):
        ts.extract_messages(None)
    with pytest.raises(ValueError, match="does not contain any runs"):
        ts.extract_messages([])
    with pytest.raises(ValueError, match="Unsupported trajectory format"):
        ts.extract_messages([{}])
    with pytest.raises(ValueError, match="Unsupported trajectory format"):
        ts.extract_messages({"transcripts": [{"messages": []}]})


def test_metrics_analysis_and_formatting_helpers():
    messages = [
        {"role": " SYSTEM "},
        {"role": "user"},
        {"role": "assistant"},
        {"role": "tool"},
        {"role": "other"},
    ]
    metrics = ts.compute_metrics(messages)
    result = ts.analyze_payload({"messages": messages}, "payload")

    assert ts.normalize_role({}) == "unknown"
    assert metrics == ts.Metrics(system=1, user=1, assistant=1, tool=1, total=5)
    assert result.source == "payload"
    assert result.metrics == metrics
    assert result.trajectory_roles == ("system", "user", "assistant", "tool", "other")
    assert ts.percent(0, 0) == "0.0%"
    assert ts.percent(1, 4) == "25.0%"
    assert ts.colorize("x", ts.RED, False) == "x"
    assert ts.colorize("x", ts.RED, True).startswith(ts.RED)
    assert ts.format_metric_value(5) == "5"
    assert ts.format_metric_value(5.0) == "5"
    assert ts.format_metric_value(5.25) == "5.2"
    assert ts.draw_bar(0, 0, "system", False).endswith("·" * 20)
    assert "■" in ts.draw_bar(2, 4, "assistant", True)
    assert ts.draw_trajectory_track(("system", "unknown", "tool"), False, width=2) == ["■■", "■"]


def test_print_helpers_emit_expected_output(capsys):
    metrics = make_metrics()
    results = [make_result("one"), make_result("two")]
    average = ts.AverageMetrics(system=1.5, user=2.5, assistant=3.5, tool=4.5, total=12.0)
    aggregate = ts.BatchAggregate(
        scope="all_collections",
        run_count=3,
        overall_average=average,
        collections=[
            ts.CollectionAggregate(
                collection_id="c1",
                collection_name="Collection One",
                short_name="Collect...",
                run_count=2,
                average_metrics=average,
            )
        ],
    )

    ts.print_report(
        metrics,
        "source",
        color=False,
        trajectory_roles=("system", "user", "assistant", "tool"),
    )
    ts.print_average_report(average, "avg-source", color=False, title_text="Average", run_count=2)
    ts.print_batch_header(2, color=False)
    ts.print_batch_report(results, color=False)
    ts.print_collection_list([{"id": "c1", "name": "Name"}], color=False)
    ts.print_run_list("c1", ["r1", "r2"], color=False)
    ts.print_selector_header("Title", "Subtitle", color=False)
    ts.print_progress(1, 2, color=False, label="Progress")
    ts.print_progress(2, 2, color=False, label="Progress")
    ts.print_collection_comparison_table(aggregate.collections, color=False)
    ts.print_aggregate_report(aggregate, color=False)
    ts.print_aggregate_report(
        ts.BatchAggregate(scope="all_runs", run_count=2, overall_average=average, collections=aggregate.collections),
        color=False,
    )

    output = capsys.readouterr().out
    assert "Trajectory Message Analysis" in output
    assert "Trajectory  ■■■■" in output
    assert "Average" in output
    assert "Remote Trajectory Analysis" in output
    assert "Available Docent Collections" in output
    assert "Docent Runs for c1" in output
    assert "Type a number, 'all', or 'q' to quit." in output
    assert "[1/2]" in output
    assert "Collection Comparison" in output
    assert "Progress" in output


def test_prompt_selection_handles_errors_and_valid_paths(monkeypatch, capsys):
    responses = iter(["", "9", "bogus", "2"])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(responses))

    choice = ts.prompt_selection(
        title="Pick",
        subtitle="Subtitle",
        options=[("one", "First"), ("two", "Second")],
        color=False,
        all_label="Everything",
    )

    output = capsys.readouterr().out
    assert choice == "two"
    assert "Please enter a selection." in output
    assert "That number is out of range." in output
    assert "Unknown selection. Try again." in output


def test_prompt_selection_special_cases(monkeypatch):
    with pytest.raises(ValueError, match="No options"):
        ts.prompt_selection(title="x", options=[], color=False)

    monkeypatch.setattr(builtins, "input", lambda _prompt="": "all")
    assert (
        ts.prompt_selection(
            title="x",
            subtitle=None,
            options=[("id", "Label")],
            color=False,
            all_label="everything",
        )
        == "all"
    )

    monkeypatch.setattr(builtins, "input", lambda _prompt="": "id")
    assert ts.prompt_selection(title="x", options=[("id", "Label")], color=False) == "id"

    monkeypatch.setattr(builtins, "input", lambda _prompt="": "q")
    with pytest.raises(ts.UserCancelled, match="cancelled"):
        ts.prompt_selection(title="x", options=[("id", "Label")], color=False)

    def raise_eof(_prompt=""):
        raise EOFError

    monkeypatch.setattr(builtins, "input", raise_eof)
    with pytest.raises(ts.UserCancelled, match="Input stream closed"):
        ts.prompt_selection(title="x", options=[("id", "Label")], color=False)


def test_prompt_compact_selection_paths(monkeypatch, capsys):
    responses = iter(["", "999", "id:missing", "2"])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(responses))

    choice = ts.prompt_compact_selection(
        title="Compact",
        subtitle="500 trajectories available",
        options=[("run-a", "run-a"), ("run-b", "run-b")],
        color=False,
        all_label="Analyze everything",
        value_label="run id",
    )

    output = capsys.readouterr().out
    assert choice == "run-b"
    assert "Type a number (1-2), 'all', 'id:<run id>', or 'q' to quit." in output
    assert "run-a" not in output
    assert "That run id was not found." in output

    monkeypatch.setattr(builtins, "input", lambda _prompt="": "all")
    assert (
        ts.prompt_compact_selection(
            title="Compact",
            subtitle="2 trajectories available",
            options=[("run-a", "run-a")],
            color=False,
            all_label="Analyze everything",
            value_label="run id",
        )
        == "all"
    )

    monkeypatch.setattr(builtins, "input", lambda _prompt="": "id:run-a")
    assert (
        ts.prompt_compact_selection(
            title="Compact",
            subtitle="2 trajectories available",
            options=[("run-a", "run-a")],
            color=False,
            all_label="Analyze everything",
            value_label="run id",
        )
        == "run-a"
    )

    monkeypatch.setattr(builtins, "input", lambda _prompt="": "q")
    with pytest.raises(ts.UserCancelled, match="cancelled"):
        ts.prompt_compact_selection(
            title="Compact",
            subtitle="2 trajectories available",
            options=[("run-a", "run-a")],
            color=False,
            all_label="Analyze everything",
            value_label="run id",
        )

    with pytest.raises(ValueError, match="No options"):
        ts.prompt_compact_selection(
            title="Compact",
            subtitle="empty",
            options=[],
            color=False,
            all_label="Analyze everything",
            value_label="run id",
        )


def test_collection_and_run_selection_helpers(monkeypatch):
    collections = [{"id": "c1", "name": "One"}]
    run_ids = ["r1", "r2"]

    assert ts.normalize_collection_records(collections) == [("c1", f"One  {ts.colorize(f'[c1]', ts.SLATE, True)}")]
    assert ts.build_run_options(run_ids) == [("r1", "r1"), ("r2", "r2")]
    assert ts.choose_collection(collections, color=False, preselected_id="c1") == "c1"
    assert ts.choose_run("c1", run_ids, color=False, preselected_id="r2") == "r2"

    monkeypatch.setattr(ts, "prompt_selection", lambda **_kwargs: "all")
    monkeypatch.setattr(ts, "prompt_compact_selection", lambda **_kwargs: "all")
    assert ts.choose_collection(collections, color=False) == "all"
    assert ts.choose_run("c1", run_ids, color=False) == "all"

    with pytest.raises(ValueError, match="Collection 'missing'"):
        ts.choose_collection(collections, color=False, preselected_id="missing")
    with pytest.raises(ValueError, match="Run 'missing'"):
        ts.choose_run("c1", run_ids, color=False, preselected_id="missing")


def test_average_and_aggregate_helpers():
    average = ts.compute_average_metrics(
        [
            ts.Metrics(system=1, user=2, assistant=3, tool=4, total=10),
            ts.Metrics(system=3, user=4, assistant=5, tool=6, total=18),
        ]
    )
    assert average == ts.AverageMetrics(system=2.0, user=3.0, assistant=4.0, tool=5.0, total=14.0)
    assert ts.shorten_collection_name("VeryLongCollectionName", "c1") == "VeryLon..."

    aggregate = ts.aggregate_batch_results(
        [
            ts.AnalysisResult(
                source="a",
                metrics=ts.Metrics(system=1, user=1, assistant=1, tool=1, total=4),
                collection_id="c1",
                collection_name="Collection One",
                run_id="r1",
            ),
            ts.AnalysisResult(
                source="b",
                metrics=ts.Metrics(system=3, user=1, assistant=1, tool=1, total=6),
                collection_id="c1",
                collection_name="Collection One",
                run_id="r2",
            ),
            ts.AnalysisResult(
                source="c",
                metrics=ts.Metrics(system=2, user=2, assistant=2, tool=2, total=8),
                collection_id="c2",
                collection_name="Collection Two",
                run_id="r3",
            ),
        ],
        scope="all_collections",
    )
    assert aggregate.run_count == 3
    assert aggregate.overall_average.total == pytest.approx(6.0)
    assert [collection.collection_id for collection in aggregate.collections] == ["c1", "c2"]
    assert aggregate.collections[0].average_metrics.total == pytest.approx(5.0)

    with pytest.raises(ValueError, match="empty result set"):
        ts.aggregate_batch_results([], scope="all_runs")
    with pytest.raises(ValueError, match="without any trajectories"):
        ts.compute_average_metrics([])


def test_export_helpers_write_expected_json(out_path, capsys):
    selection_runs = ts.RemoteSelection(
        mode="all_runs",
        targets=[
            ts.RemoteTarget(collection_id="col/1", collection_name="Alpha", run_id="run-a"),
            ts.RemoteTarget(collection_id="col/1", collection_name="Alpha", run_id="run-b"),
        ],
    )
    results_runs = [
        ts.AnalysisResult(
            source="Docent run: col/1/run-a",
            collection_id="col/1",
            collection_name="Alpha",
            run_id="run-a",
            metrics=ts.Metrics(system=1, user=1, assistant=1, tool=1, total=4),
        ),
        ts.AnalysisResult(
            source="Docent run: col/1/run-b",
            collection_id="col/1",
            collection_name="Alpha",
            run_id="run-b",
            metrics=ts.Metrics(system=3, user=1, assistant=1, tool=1, total=6),
        ),
    ]
    aggregate_runs = ts.aggregate_batch_results(results_runs, scope="all_runs")
    exported_runs = ts.export_batch_results(
        selection_runs,
        aggregate_runs,
        results_runs,
        base_dir=out_path,
    )
    assert exported_runs[0].name == "all_runs_col_1_metrics.json"
    runs_payload = json.loads(exported_runs[0].read_text(encoding="utf-8"))
    assert runs_payload["average_metrics"]["total"] == pytest.approx(5.0)
    assert [item["run_id"] for item in runs_payload["trajectory_metrics"]] == ["run-a", "run-b"]

    selection_collections = ts.RemoteSelection(
        mode="all_collections",
        targets=[
            ts.RemoteTarget(collection_id="col-1", collection_name="Alpha", run_id="run-a"),
            ts.RemoteTarget(collection_id="col-1", collection_name="Alpha", run_id="run-b"),
            ts.RemoteTarget(collection_id="col-2", collection_name="Beta", run_id="run-c"),
        ],
    )
    results_collections = [
        ts.AnalysisResult(
            source="Docent run: col-1/run-a",
            collection_id="col-1",
            collection_name="Alpha",
            run_id="run-a",
            metrics=ts.Metrics(system=1, user=1, assistant=1, tool=1, total=4),
        ),
        ts.AnalysisResult(
            source="Docent run: col-1/run-b",
            collection_id="col-1",
            collection_name="Alpha",
            run_id="run-b",
            metrics=ts.Metrics(system=3, user=1, assistant=1, tool=1, total=6),
        ),
        ts.AnalysisResult(
            source="Docent run: col-2/run-c",
            collection_id="col-2",
            collection_name="Beta",
            run_id="run-c",
            metrics=ts.Metrics(system=2, user=2, assistant=2, tool=2, total=8),
        ),
    ]
    aggregate_collections = ts.aggregate_batch_results(results_collections, scope="all_collections")
    exported_collections = ts.export_batch_results(
        selection_collections,
        aggregate_collections,
        results_collections,
        base_dir=out_path,
    )
    collections_payload = json.loads(exported_collections[0].read_text(encoding="utf-8"))
    assert collections_payload["collections"][0]["aggregate_metrics"]["total"] == pytest.approx(5.0)
    assert collections_payload["collections"][1]["trajectory_metrics"][0]["run_id"] == "run-c"

    assert ts.ensure_out_dir(out_path) == out_path / "output"
    assert ts.sanitize_filename_part("co l/1") == "co_l_1"
    assert ts.build_all_runs_export_payload(selection_runs, aggregate_runs, results_runs)["scope"] == "all_runs"
    assert (
        ts.build_all_collections_runs_payload(aggregate_collections, results_collections)["scope"]
        == "all_collections"
    )
    assert ts.write_json_file(out_path / "direct.json", {"ok": True}).exists()
    ts.print_export_paths(exported_collections, color=False)
    assert "Saved JSON exports" in capsys.readouterr().out


def test_run_remote_selector_paths(monkeypatch):
    fake_client = FakeDocentClient("key")
    monkeypatch.setattr(ts, "build_docent_client", lambda api_key=None: fake_client)

    selection = ts.run_remote_selector(
        api_key=None,
        color=False,
        collection_id=None,
        run_id=None,
        all_collections=True,
        all_runs=False,
    )
    assert selection.mode == "all_collections"
    assert [(item.collection_id, item.collection_name, item.run_id) for item in selection.targets] == [
        ("col-1", "Alpha", "run-a"),
        ("col-1", "Alpha", "run-b"),
        ("col-2", "Beta", "run-c"),
    ]

    selection = ts.run_remote_selector(
        api_key=None,
        color=False,
        collection_id="col-1",
        run_id="run-b",
        all_collections=False,
        all_runs=False,
    )
    assert selection.mode == "single"
    assert [(item.collection_id, item.collection_name, item.run_id) for item in selection.targets] == [
        ("col-1", "Alpha", "run-b")
    ]

    selection = ts.run_remote_selector(
        api_key=None,
        color=False,
        collection_id="col-1",
        run_id=None,
        all_collections=False,
        all_runs=True,
    )
    assert selection.mode == "all_runs"
    assert [(item.collection_id, item.collection_name, item.run_id) for item in selection.targets] == [
        ("col-1", "Alpha", "run-a"),
        ("col-1", "Alpha", "run-b"),
    ]


def test_run_remote_selector_error_paths(monkeypatch):
    class EmptyClient(FakeDocentClient):
        def list_collections(self):
            return []

    monkeypatch.setattr(ts, "build_docent_client", lambda api_key=None: EmptyClient(api_key))
    with pytest.raises(ValueError, match="No Docent collections"):
        ts.run_remote_selector(
            api_key=None,
            color=False,
            collection_id=None,
            run_id=None,
            all_collections=False,
            all_runs=False,
        )

    class NoRunsClient(FakeDocentClient):
        def list_agent_run_ids(self, collection_id: str):
            return []

    monkeypatch.setattr(ts, "build_docent_client", lambda api_key=None: NoRunsClient(api_key))
    with pytest.raises(ValueError, match="has no trajectories"):
        ts.run_remote_selector(
            api_key=None,
            color=False,
            collection_id="col-1",
            run_id=None,
            all_collections=False,
            all_runs=False,
        )


def test_analyze_remote_run_and_targets(monkeypatch):
    fake_client = FakeDocentClient("key")
    result = ts.analyze_remote_run(fake_client, "col-1", "run-a", collection_name="Alpha")
    assert result.source == "Docent run: col-1/run-a"
    assert result.collection_name == "Alpha"
    assert result.metrics.total == 4
    assert result.trajectory_roles == ("system", "user", "assistant", "tool")

    progress_calls: list[tuple[int, int, bool, str]] = []
    monkeypatch.setattr(ts, "build_docent_client", lambda api_key=None: fake_client)
    monkeypatch.setattr(
        ts,
        "print_progress",
        lambda current, total, color, label="Processing trajectories": progress_calls.append(
            (current, total, color, label)
        ),
    )
    results = ts.analyze_remote_targets(
        [
            ts.RemoteTarget(collection_id="col-1", collection_name="Alpha", run_id="run-a"),
            ts.RemoteTarget(collection_id="col-2", collection_name="Beta", run_id="run-c"),
        ],
        api_key=None,
        color=False,
        show_progress=True,
    )
    assert [item.run_id for item in results] == ["run-a", "run-c"]
    assert progress_calls == [
        (0, 2, False, "Processing trajectories"),
        (1, 2, False, "Processing trajectories"),
        (2, 2, False, "Processing trajectories"),
    ]


def test_build_parser_and_resolve_payload(out_path, monkeypatch):
    parser = ts.build_parser()
    help_text = parser.format_help()
    assert "Examples:" in help_text
    assert "runs --collection-id COLLECTION_ID" in help_text

    sample = out_path / "payload.json"
    sample.write_text(json.dumps({"messages": [{"role": "system"}]}), encoding="utf-8")
    payload, source = ts.resolve_payload(make_args(source="local", path=str(sample)))
    assert payload["messages"][0]["role"] == "system"
    assert str(sample.resolve()) in source

    monkeypatch.setattr(ts, "fetch_docent_run", lambda **kwargs: {"messages": [{"role": "user"}]})
    payload, source = ts.resolve_payload(
        make_args(source="remote", collection_id="c1", run_id="r1", api_key="key")
    )
    assert payload == {"messages": [{"role": "user"}]}
    assert source == "Docent run: c1/r1"


def test_main_collections_branches(monkeypatch, capsys):
    collections = [{"id": "c1", "name": "One"}]
    monkeypatch.setattr(ts, "build_parser", lambda: StubParser(make_args(source="collections", json_output=True)))
    monkeypatch.setattr(ts, "list_docent_collections", lambda api_key=None: collections)
    assert ts.main() == 0
    assert '"id": "c1"' in capsys.readouterr().out

    printed: list[tuple] = []
    monkeypatch.setattr(ts, "build_parser", lambda: StubParser(make_args(source="collections", json_output=False)))
    monkeypatch.setattr(ts, "print_collection_list", lambda data, color: printed.append((data, color)))
    assert ts.main() == 0
    assert printed == [(collections, False)]


def test_main_runs_branches(monkeypatch, capsys):
    monkeypatch.setattr(
        ts,
        "build_parser",
        lambda: StubParser(make_args(source="runs", json_output=True, collection_id="c1")),
    )
    monkeypatch.setattr(ts, "list_docent_run_ids", lambda collection_id, api_key=None: ["r1"])
    assert ts.main() == 0
    assert '"collection_id": "c1"' in capsys.readouterr().out

    printed: list[tuple] = []
    monkeypatch.setattr(
        ts,
        "build_parser",
        lambda: StubParser(make_args(source="runs", json_output=False, collection_id="c1")),
    )
    monkeypatch.setattr(ts, "print_run_list", lambda collection_id, run_ids, color: printed.append((collection_id, run_ids, color)))
    assert ts.main() == 0
    assert printed == [("c1", ["r1"], False)]


def test_main_remote_branches(monkeypatch, capsys):
    single_selection = ts.RemoteSelection(
        mode="single",
        targets=[ts.RemoteTarget(collection_id="c1", collection_name="One", run_id="r1")],
    )
    monkeypatch.setattr(
        ts,
        "build_parser",
        lambda: StubParser(make_args(source="remote", json_output=True, collection_id="c1", run_id="r1")),
    )
    monkeypatch.setattr(ts, "run_remote_selector", lambda **kwargs: single_selection)
    monkeypatch.setattr(
        ts,
        "analyze_remote_targets",
        lambda targets, api_key=None, color=False, show_progress=False: [make_result("remote-json")],
    )
    assert ts.main() == 0
    assert '"source": "remote-json"' in capsys.readouterr().out

    printed: list[tuple] = []
    monkeypatch.setattr(
        ts,
        "build_parser",
        lambda: StubParser(make_args(source="remote", json_output=False, collection_id="c1", run_id="r1")),
    )
    monkeypatch.setattr(
        ts,
        "print_report",
        lambda metrics, source, color, trajectory_roles=(): printed.append(
            (metrics, source, color, trajectory_roles)
        ),
    )
    assert ts.main() == 0
    assert printed[0][1] == "remote-json"
    assert printed[0][2] is False
    assert printed[0][3] == ()

    aggregate = ts.BatchAggregate(
        scope="all_runs",
        run_count=2,
        overall_average=ts.AverageMetrics(system=1, user=1, assistant=1, tool=1, total=4),
        collections=[
            ts.CollectionAggregate(
                collection_id="c1",
                collection_name="One",
                short_name="One",
                run_count=2,
                average_metrics=ts.AverageMetrics(system=1, user=1, assistant=1, tool=1, total=4),
            )
        ],
    )
    monkeypatch.setattr(
        ts,
        "build_parser",
        lambda: StubParser(make_args(source="remote", json_output=True, collection_id="c1", all_runs=True)),
    )
    monkeypatch.setattr(
        ts,
        "run_remote_selector",
        lambda **kwargs: ts.RemoteSelection(
            mode="all_runs",
            targets=[ts.RemoteTarget(collection_id="c1", collection_name="One", run_id="r1")],
        ),
    )
    monkeypatch.setattr(
        ts,
        "analyze_remote_targets",
        lambda targets, api_key=None, color=False, show_progress=False: [make_result("agg")],
    )
    monkeypatch.setattr(ts, "aggregate_batch_results", lambda results, scope: aggregate)
    monkeypatch.setattr(ts, "export_batch_results", lambda selection, aggregate, results: [Path("output/export.json")])
    assert ts.main() == 0
    output = capsys.readouterr().out
    assert '"scope": "all_runs"' in output
    assert '"export_files": [' in output

    aggregate_printed: list[tuple] = []
    export_printed: list[tuple] = []
    monkeypatch.setattr(
        ts,
        "build_parser",
        lambda: StubParser(make_args(source="remote", json_output=False, collection_id="c1", all_runs=True)),
    )
    monkeypatch.setattr(ts, "print_aggregate_report", lambda batch, color: aggregate_printed.append((batch, color)))
    monkeypatch.setattr(ts, "print_export_paths", lambda paths, color: export_printed.append((paths, color)))
    assert ts.main() == 0
    assert aggregate_printed == [(aggregate, False)]
    assert export_printed == [([Path("output/export.json")], False)]


def test_main_remote_validation_and_error_handling(monkeypatch, capsys):
    monkeypatch.setattr(
        ts,
        "build_parser",
        lambda: StubParser(make_args(source="remote", run_id="r1", collection_id=None)),
    )
    assert ts.main() == 1
    assert "--run-id requires --collection-id." in capsys.readouterr().err

    monkeypatch.setattr(
        ts,
        "build_parser",
        lambda: StubParser(make_args(source="remote", all_runs=True, collection_id=None)),
    )
    assert ts.main() == 1
    assert "--all-runs requires --collection-id." in capsys.readouterr().err

    monkeypatch.setattr(ts, "build_parser", lambda: StubParser(make_args(source="local")))
    monkeypatch.setattr(ts, "resolve_payload", lambda _args: (_ for _ in ()).throw(ValueError("boom")))
    assert ts.main() == 1
    assert "Error: boom" in capsys.readouterr().err


def test_main_local_branches(monkeypatch, capsys):
    monkeypatch.setattr(ts, "build_parser", lambda: StubParser(make_args(source="local", json_output=True)))
    monkeypatch.setattr(ts, "resolve_payload", lambda _args: ({"messages": []}, "src-json"))
    monkeypatch.setattr(ts, "analyze_payload", lambda payload, source: make_result(source))
    assert ts.main() == 0
    assert '"source": "src-json"' in capsys.readouterr().out

    printed: list[tuple] = []
    monkeypatch.setattr(ts, "build_parser", lambda: StubParser(make_args(source="local", json_output=False)))
    monkeypatch.setattr(ts, "resolve_payload", lambda _args: ({"messages": []}, "src-human"))
    monkeypatch.setattr(
        ts,
        "print_report",
        lambda metrics, source, color, trajectory_roles=(): printed.append(
            (metrics, source, color, trajectory_roles)
        ),
    )
    assert ts.main() == 0
    assert printed == [(make_metrics(), "src-human", False, ())]


def test_module_main_entrypoint_executes(out_path, monkeypatch, capsys):
    payload = {"messages": [{"role": "system"}, {"role": "user"}]}
    sample = out_path / "sample.json"
    sample.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["trajectory_scanner.py", "--json", "local", str(sample)])

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("trajectory_scanner", run_name="__main__")

    assert excinfo.value.code == 0
    assert '"total": 2' in capsys.readouterr().out


def test_print_report_multiline_trajectory(capsys):
    """Test print_report with trajectory that spans multiple lines."""
    metrics = make_metrics()
    trajectory_roles = ("system", "user", "assistant", "tool") * 25  # 100 roles
    ts.print_report(
        metrics,
        "multiline test",
        color=False,
        trajectory_roles=trajectory_roles,
    )
    output = capsys.readouterr().out
    assert "Trajectory Message Analysis" in output
    assert output.count("\n") > 5  # Should have multiple lines


def test_print_progress_with_zero_total(capsys):
    """Test print_progress with total <= 0 returns early."""
    ts.print_progress(0, 0, color=False)
    output = capsys.readouterr().out
    assert output == ""

    ts.print_progress(5, 0, color=False)
    output = capsys.readouterr().out
    assert output == ""


def test_print_export_paths_empty(capsys):
    """Test print_export_paths with empty paths list."""
    ts.print_export_paths([], color=False)
    output = capsys.readouterr().out
    assert output == ""


def test_print_collection_comparison_table_empty(capsys):
    """Test print_collection_comparison_table with empty collections."""
    ts.print_collection_comparison_table([], color=False)
    output = capsys.readouterr().out
    assert output == ""


def test_prompt_compact_selection_eof_error(monkeypatch):
    """Test prompt_compact_selection handles EOF during input."""
    def raise_eof(_prompt=""):
        raise EOFError

    monkeypatch.setattr(builtins, "input", raise_eof)
    with pytest.raises(ts.UserCancelled, match="Input stream closed"):
        ts.prompt_compact_selection(
            title="Test",
            subtitle="Testing EOF",
            options=[("id1", "Label 1"), ("id2", "Label 2")],
            color=False,
            all_label="All",
            value_label="id",
        )


def test_prompt_compact_selection_direct_value_match(monkeypatch):
    """Test prompt_compact_selection with direct value (not prefixed with id:)."""
    monkeypatch.setattr(builtins, "input", lambda _prompt="": "id2")
    result = ts.prompt_compact_selection(
        title="Test",
        subtitle="Testing direct match",
        options=[("id1", "Label 1"), ("id2", "Label 2")],
        color=False,
        all_label="All",
        value_label="id",
    )
    assert result == "id2"


def test_prompt_compact_selection_invalid_then_valid(monkeypatch, capsys):
    """Test prompt_compact_selection with invalid input then valid selection."""
    responses = iter(["invalid-value", "all"])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(responses))
    result = ts.prompt_compact_selection(
        title="Test",
        subtitle="Testing invalid then valid",
        options=[("id1", "Label 1"), ("id2", "Label 2")],
        color=False,
        all_label="All",
        value_label="id",
    )
    output = capsys.readouterr().out
    assert "Unknown selection. Try again." in output
    assert result == "all"


def test_load_cached_metrics_all_runs(out_path):
    """Test loading cached metrics for all_runs mode."""
    cached_payload = {
        "scope": "all_runs",
        "collection": {
            "collection_id": "col-1",
            "collection_name": "Collection One",
        },
        "run_count": 2,
        "average_metrics": {"system": 1.0, "user": 1.0, "assistant": 1.0, "tool": 1.0, "total": 4.0},
        "trajectory_metrics": [
            {
                "source": "Docent run: col-1/run-a",
                "collection_id": "col-1",
                "collection_name": "Collection One",
                "run_id": "run-a",
                "metrics": {"system": 1, "user": 1, "assistant": 1, "tool": 1, "total": 4},
            },
            {
                "source": "Docent run: col-1/run-b",
                "collection_id": "col-1",
                "collection_name": "Collection One",
                "run_id": "run-b",
                "metrics": {"system": 2, "user": 2, "assistant": 2, "tool": 2, "total": 8},
            },
        ],
    }
    cache_file = out_path / "output" / "all_runs_col-1_metrics.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(cached_payload), encoding="utf-8")

    results = ts.reconstruct_results_from_cached(cached_payload, is_all_collections=False)
    assert len(results) == 2
    assert results[0].run_id == "run-a"
    assert results[0].metrics.total == 4
    assert results[1].metrics.total == 8


def test_load_cached_metrics_all_collections(out_path):
    """Test loading cached metrics for all_collections mode."""
    cached_payload = {
        "scope": "all_collections",
        "run_count": 3,
        "overall_average": {"system": 1.5, "user": 1.5, "assistant": 1.5, "tool": 1.5, "total": 6.0},
        "collections": [
            {
                "collection_id": "col-1",
                "collection_name": "Collection One",
                "short_name": "Col One",
                "run_count": 2,
                "aggregate_metrics": {"system": 1.0, "user": 1.0, "assistant": 1.0, "tool": 1.0, "total": 4.0},
                "trajectory_metrics": [
                    {
                        "source": "Docent run: col-1/run-a",
                        "collection_id": "col-1",
                        "collection_name": "Collection One",
                        "run_id": "run-a",
                        "metrics": {"system": 1, "user": 1, "assistant": 1, "tool": 1, "total": 4},
                    },
                ],
            },
            {
                "collection_id": "col-2",
                "collection_name": "Collection Two",
                "short_name": "Col Two",
                "run_count": 1,
                "aggregate_metrics": {"system": 2.0, "user": 2.0, "assistant": 2.0, "tool": 2.0, "total": 8.0},
                "trajectory_metrics": [
                    {
                        "source": "Docent run: col-2/run-c",
                        "collection_id": "col-2",
                        "collection_name": "Collection Two",
                        "run_id": "run-c",
                        "metrics": {"system": 2, "user": 2, "assistant": 2, "tool": 2, "total": 8},
                    },
                ],
            },
        ],
    }

    results = ts.reconstruct_results_from_cached(cached_payload, is_all_collections=True)
    assert len(results) == 2
    assert results[0].collection_id == "col-1"
    assert results[1].collection_id == "col-2"
    assert results[0].metrics.total == 4
    assert results[1].metrics.total == 8


def test_try_load_cached_metrics_all_runs(out_path):
    """Test try_load_cached_metrics for all_runs selection."""
    cached_payload = {
        "scope": "all_runs",
        "collection": {"collection_id": "col-1", "collection_name": "Collection One"},
        "run_count": 1,
        "average_metrics": {"system": 1.0, "user": 1.0, "assistant": 1.0, "tool": 1.0, "total": 4.0},
        "trajectory_metrics": [
            {
                "source": "Docent run: col-1/run-a",
                "collection_id": "col-1",
                "collection_name": "Collection One",
                "run_id": "run-a",
                "metrics": {"system": 1, "user": 1, "assistant": 1, "tool": 1, "total": 4},
            },
        ],
    }
    cache_file = out_path / "output" / "all_runs_col-1_metrics.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(cached_payload), encoding="utf-8")

    selection = ts.RemoteSelection(
        mode="all_runs",
        targets=[ts.RemoteTarget(collection_id="col-1", collection_name="Collection One", run_id="run-a")],
    )
    results = ts.try_load_cached_metrics(selection, base_dir=out_path)
    assert results is not None
    assert len(results) == 1
    assert results[0].run_id == "run-a"


def test_try_load_cached_metrics_all_collections(out_path):
    """Test try_load_cached_metrics for all_collections selection."""
    runs_payload = {
        "scope": "all_collections",
        "run_count": 1,
        "overall_average": {"system": 1.0, "user": 1.0, "assistant": 1.0, "tool": 1.0, "total": 4.0},
        "collections": [
            {
                "collection_id": "col-1",
                "collection_name": "Collection One",
                "short_name": "Col One",
                "run_count": 1,
                "aggregate_metrics": {"system": 1.0, "user": 1.0, "assistant": 1.0, "tool": 1.0, "total": 4.0},
                "trajectory_metrics": [
                    {
                        "source": "Docent run: col-1/run-a",
                        "collection_id": "col-1",
                        "collection_name": "Collection One",
                        "run_id": "run-a",
                        "metrics": {"system": 1, "user": 1, "assistant": 1, "tool": 1, "total": 4},
                    },
                ],
            },
        ],
    }
    out_subdir = out_path / "output"
    out_subdir.mkdir(parents=True, exist_ok=True)
    (out_subdir / "all_collections_runs_data.json").write_text(json.dumps(runs_payload), encoding="utf-8")

    selection = ts.RemoteSelection(
        mode="all_collections",
        targets=[ts.RemoteTarget(collection_id="col-1", collection_name="Collection One", run_id="run-a")],
    )
    results = ts.try_load_cached_metrics(selection, base_dir=out_path)
    assert results is not None
    assert len(results) == 1
    assert results[0].collection_id == "col-1"


def test_try_load_cached_metrics_single_mode_not_cached(out_path):
    """Test that try_load_cached_metrics returns None for single mode."""
    selection = ts.RemoteSelection(
        mode="single",
        targets=[ts.RemoteTarget(collection_id="col-1", collection_name="Collection One", run_id="run-a")],
    )
    results = ts.try_load_cached_metrics(selection, base_dir=out_path)
    assert results is None


def test_try_load_cached_metrics_missing_files(out_path):
    """Test that try_load_cached_metrics returns None when cache files are missing."""
    selection = ts.RemoteSelection(
        mode="all_runs",
        targets=[ts.RemoteTarget(collection_id="col-1", collection_name="Collection One", run_id="run-a")],
    )
    results = ts.try_load_cached_metrics(selection, base_dir=out_path)
    assert results is None


def test_load_cached_metrics_corrupted_file(out_path):
    """Test that load_cached_all_runs_metrics handles corrupted JSON."""
    cache_file = out_path / "output" / "all_runs_col-1_metrics.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text("invalid json {", encoding="utf-8")

    result = ts.load_cached_all_runs_metrics("col-1", base_dir=out_path)
    assert result is None


def test_load_cached_all_collections_metrics_missing_files(out_path):
    """Test that load_cached_all_collections_metrics returns None when the file is missing."""
    assert ts.load_cached_all_collections_metrics(base_dir=out_path) is None


def test_try_load_cached_metrics_all_collections_missing_scope(out_path):
    """Test try_load_cached_metrics with all_collections but missing runs data scope."""
    runs_payload = {
        "scope": "single",  # Wrong scope
        "collections": [],
    }
    out_subdir = out_path / "output"
    out_subdir.mkdir(parents=True, exist_ok=True)
    (out_subdir / "all_collections_runs_data.json").write_text(json.dumps(runs_payload), encoding="utf-8")

    selection = ts.RemoteSelection(
        mode="all_collections",
        targets=[ts.RemoteTarget(collection_id="col-1", collection_name="One", run_id="run-a")],
    )
    results = ts.try_load_cached_metrics(selection, base_dir=out_path)
    assert results is None


def test_try_load_cached_metrics_all_collections_missing_runs_data(out_path):
    """Test try_load_cached_metrics with all_collections but no runs data file."""
    selection = ts.RemoteSelection(
        mode="all_collections",
        targets=[ts.RemoteTarget(collection_id="col-1", collection_name="One", run_id="run-a")],
    )
    results = ts.try_load_cached_metrics(selection, base_dir=out_path)
    assert results is None


def test_load_cached_all_collections_metrics_corrupted_runs_file(out_path):
    """Test load_cached_all_collections_metrics when the runs file is corrupted."""
    out_subdir = out_path / "output"
    out_subdir.mkdir(parents=True, exist_ok=True)
    (out_subdir / "all_collections_runs_data.json").write_text(
        "invalid json {", encoding="utf-8"
    )

    assert ts.load_cached_all_collections_metrics(base_dir=out_path) is None


def test_try_load_cached_metrics_unknown_mode():
    """Test try_load_cached_metrics with unknown selection mode."""
    selection = ts.RemoteSelection(
        mode="unknown",
        targets=[ts.RemoteTarget(collection_id="col-1", collection_name="One", run_id="run-a")],
    )
    results = ts.try_load_cached_metrics(selection)
    assert results is None


def test_try_load_cached_metrics_mismatched_scope(out_path):
    """Test try_load_cached_metrics with mismatched scope."""
    cached_payload = {
        "scope": "single",  # Wrong scope
        "collection": {"collection_id": "col-1", "collection_name": "Collection One"},
        "run_count": 1,
        "average_metrics": {"system": 1.0, "user": 1.0, "assistant": 1.0, "tool": 1.0, "total": 4.0},
        "trajectory_metrics": [],
    }
    cache_file = out_path / "output" / "all_runs_col-1_metrics.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(cached_payload), encoding="utf-8")

    selection = ts.RemoteSelection(
        mode="all_runs",
        targets=[ts.RemoteTarget(collection_id="col-1", collection_name="Collection One", run_id="run-a")],
    )
    results = ts.try_load_cached_metrics(selection, base_dir=out_path)
    assert results is None


def test_main_remote_all_runs_with_cache(monkeypatch, capsys, out_path):
    """Test main() function using cached metrics for all_runs."""
    cached_payload = {
        "scope": "all_runs",
        "collection": {"collection_id": "col-1", "collection_name": "One"},
        "run_count": 1,
        "average_metrics": {"system": 1.0, "user": 1.0, "assistant": 1.0, "tool": 1.0, "total": 4.0},
        "trajectory_metrics": [
            {
                "source": "Docent run: col-1/run-a",
                "collection_id": "col-1",
                "collection_name": "One",
                "run_id": "run-a",
                "metrics": {"system": 1, "user": 1, "assistant": 1, "tool": 1, "total": 4},
            },
        ],
    }
    cache_file = out_path / "output" / "all_runs_col-1_metrics.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(cached_payload), encoding="utf-8")

    monkeypatch.setattr(
        ts,
        "build_parser",
        lambda: StubParser(make_args(source="remote", json_output=False, collection_id="col-1", all_runs=True, path=str(out_path))),
    )
    monkeypatch.setattr(
        ts,
        "run_remote_selector",
        lambda **kwargs: ts.RemoteSelection(
            mode="all_runs",
            targets=[ts.RemoteTarget(collection_id="col-1", collection_name="One", run_id="run-a")],
        ),
    )
    monkeypatch.setattr(ts, "ensure_out_dir", lambda base_dir=None: out_path / "output")

    assert ts.main() == 0
    stderr_output = capsys.readouterr().err
    assert "Using cached metrics" in stderr_output
