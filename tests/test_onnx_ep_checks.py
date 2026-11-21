import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import test_onnx_ep as checks


def test_audio_benchmark_covers_entire_track_and_reports_speed(monkeypatch):
    audio = np.vstack([np.arange(10), np.arange(10) + 2]).astype(np.float32)
    chunks = []

    class Session:
        def run(self, outputs, inputs):
            chunk = inputs["input"]
            chunks.append(chunk.copy())
            return [chunk[:, None, :, :]]

    times = iter([10.0, 12.0])
    monkeypatch.setattr(checks.time, "perf_counter", lambda: next(times))
    result = checks.benchmark_audio(Session(), {"input": np.zeros((1, 2, 4), dtype=np.float32)}, audio, 44100)
    reference = audio.mean(axis=0)
    expected = (audio - reference.mean()) / (reference.std(ddof=1) + 1e-8)

    assert len(chunks) == 4
    for chunk, offset in zip(chunks, [0, 3, 6, 9]):
        count = min(4, 10 - offset)
        np.testing.assert_allclose(chunk[0, :, :count], expected[:, offset:offset + count])
        assert not chunk[0, :, count:].any()
    assert result["audio_s"] == pytest.approx(10 / 44100)
    assert result["audio_processing_s"] == 2.0
    assert result["processing_speed"] == pytest.approx(10 / 44100 / 2)


def test_audio_benchmark_rejects_invalid_input_and_output():
    audio = np.zeros((2, 10), dtype=np.float32)
    inputs = {"input": np.zeros((1, 2, 4), dtype=np.float32)}
    session = SimpleNamespace(run=lambda *args: [np.zeros((1, 2, 4), dtype=np.float32)])
    with pytest.raises(ValueError, match="44.1 kHz stereo"):
        checks.benchmark_audio(session, inputs, audio, 48000)
    with pytest.raises(ValueError, match="one .* output"):
        checks.benchmark_audio(session, inputs, audio, 44100)


def test_load_benchmark_audio_rejects_wrong_format(monkeypatch):
    monkeypatch.setitem(checks.sys.modules, "soundfile", SimpleNamespace(
        read=lambda *args, **kwargs: (np.zeros((4, 1), dtype=np.float32), 44100)))
    with pytest.raises(ValueError, match="44.1 kHz stereo"):
        checks.load_benchmark_audio("audio.wav")


def test_main_reports_processing_speed_in_job_summary(monkeypatch, tmp_path, capsys):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("BENCHMARK_AUDIO", "audio.wav")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setenv("TEST_TARGETS", json.dumps([{"name": "CPU", "ep": "CPUExecutionProvider"}]))
    monkeypatch.setattr(checks.sys, "argv", ["test_onnx_ep.py", "model.onnx"])
    monkeypatch.setattr(checks, "create_dummy_inputs", lambda path: {})
    benchmark_input = (np.zeros((2, 4), dtype=np.float32), 44100)
    monkeypatch.setattr(checks, "load_benchmark_audio", lambda path: benchmark_input)

    def run(model, inputs, target, benchmark_input=None):
        assert benchmark_input is not None
        return {"name": "CPU", "status": "PASSED", "compile_s": 1.25, "warmup_s": 0.1,
                "infer_s": 0.2, "processing_speed": 12.1}

    monkeypatch.setattr(checks, "run_single_target", run)
    with pytest.raises(SystemExit) as result:
        checks.main()
    assert result.value.code == 0
    output = capsys.readouterr().out
    assert "Processing speed" in output and "12.10x" in output
    assert "Audio speed" not in output
    text = summary.read_text(encoding="utf-8")
    assert "| CPU | PASSED | 1.250 | 12.10x |" in text
    assert "Processing speed" in text and "Audio speed" not in text
    assert "do not guarantee GPU or Neural Engine placement" in text


def test_summary_columns_align_for_long_names_and_missing_processing_speed(monkeypatch, capsys):
    targets = [{"name": "A very long target name that exceeds the original table width", "ep": "CPUExecutionProvider"},
               {"name": "Short", "ep": "CPUExecutionProvider"}]
    monkeypatch.delenv("BENCHMARK_AUDIO", raising=False)
    monkeypatch.setenv("TEST_TARGETS", json.dumps(targets))
    monkeypatch.setattr(checks.sys, "argv", ["test_onnx_ep.py", "model.onnx"])
    monkeypatch.setattr(checks, "create_dummy_inputs", lambda path: {})

    def run(model, inputs, target, benchmark_input=None):
        return {"name": target["name"], "status": "PASSED", "compile_s": 123456789.123,
                "warmup_s": 0.1, "infer_s": 0.2}

    monkeypatch.setattr(checks, "run_single_target", run)
    with pytest.raises(SystemExit) as result:
        checks.main()
    assert result.value.code == 0
    lines = capsys.readouterr().out.splitlines()
    header_index = next(i for i, line in enumerate(lines) if line.startswith("Target ") and "Processing speed" in line)
    header, separator, *rows = lines[header_index:header_index + 4]
    positions = [i for i, char in enumerate(header) if char == "|"]
    assert len(positions) == 5
    assert [i for i, char in enumerate(separator) if char == "+"] == positions
    assert all([i for i, char in enumerate(row) if char == "|"] == positions for row in rows)
    assert all(row.endswith("N/A") for row in rows)


@pytest.mark.parametrize("platform", ["3720", "4000", "5010", "5020", "6010"])
@pytest.mark.parametrize("log_level", [None, "LOG_TRACE"])
def test_npu_aot_uses_plugin_compiler_and_exports_blob(monkeypatch, platform, log_level):
    calls = []

    class Model:
        def is_dynamic(self):
            return False

    class CompiledModel:
        def export_model(self, stream):
            stream.write(b"compiled blob")

    class Core:
        available_devices = ["CPU"]

        def read_model(self, path):
            assert path == "model.onnx"
            return Model()

        def compile_model(self, model, device_name, config):
            calls.append((device_name, config))
            return CompiledModel()

    monkeypatch.setitem(checks.sys.modules, "openvino", SimpleNamespace(Core=Core, __version__="test"))
    monkeypatch.setattr(checks.sys, "platform", "linux")
    result = checks._npu_aot_compile_impl("model.onnx", "NPU", platform, log_level)

    assert result["status"] == "PASSED (AOT)"
    config = {"NPU_PLATFORM": platform, "NPU_COMPILER_TYPE": "PLUGIN"}
    if log_level:
        config["LOG_LEVEL"] = log_level
    assert calls == [("NPU", config)]


def test_npu_target_passes_diagnostic_settings_to_child(monkeypatch):
    calls = []
    expected = {"name": "NPU 4000", "status": "PASSED (AOT)"}

    def run(args, name, description, log_resources=False):
        calls.append((args, name, description, log_resources))
        return expected

    monkeypatch.setattr(checks, "run_logged_child", run)
    target = {"name": "NPU 4000", "ep": "OpenVINOExecutionProvider", "npu_platform": "4000",
              "npu_log_level": "LOG_TRACE", "log_resources": True}
    assert checks.run_single_target("model.onnx", {}, target) == expected
    assert calls == [(["--npu-aot", "model.onnx", "NPU 4000", "4000", "LOG_TRACE"],
                      "NPU 4000", "NPU AOT compile", True)]


def test_resource_monitor_reports_process_and_cgroup(monkeypatch, capsys):
    calls = []
    finished = SimpleNamespace(wait=lambda interval: len(calls) > 0)
    proc = SimpleNamespace(pid=123, poll=lambda: None)

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(stdout="123 R 80.0 1024 2048 4 01:00 -")

    def read(path, **kwargs):
        from io import StringIO
        return StringIO("100\n200\n")

    monkeypatch.setattr(checks.subprocess, "run", run)
    monkeypatch.setattr(checks, "open", read, raising=False)
    checks.report_child_resources(proc, finished, interval=1)
    output = capsys.readouterr().out
    assert calls[0][0][:3] == ["ps", "-p", "123"]
    assert "%CPU(avg)" in output and "123 R 80.0" in output
    assert "/sys/fs/cgroup/memory.current: 100, 200" in output
    assert "/sys/fs/cgroup/memory.events: 100, 200" in output


def test_resource_monitor_handles_missing_tools_and_cgroups(monkeypatch, capsys):
    events = iter([False, True])
    finished = SimpleNamespace(wait=lambda interval: next(events))
    proc = SimpleNamespace(pid=123, poll=lambda: None)

    def unavailable(*args, **kwargs):
        raise FileNotFoundError("not available")

    monkeypatch.setattr(checks.subprocess, "run", unavailable)
    monkeypatch.setattr(checks, "open", unavailable, raising=False)
    checks.report_child_resources(proc, finished, interval=1)
    assert "Process resource snapshot unavailable" in capsys.readouterr().out


def test_plugin_discovery_failure_reports_hardware(monkeypatch, capsys):
    hardware = SimpleNamespace(type="CPU", vendor="Intel", vendor_id=0x8086, device_id=0, metadata={})
    device = SimpleNamespace(ep_name="CPUExecutionProvider", ep_vendor="Microsoft", ep_metadata={}, device=hardware)
    plugin = SimpleNamespace(get_ep_name=lambda: "OpenVINOExecutionProvider", get_library_path=lambda: "plugin.so")
    monkeypatch.setitem(checks.sys.modules, "onnxruntime_ep_openvino", plugin)
    monkeypatch.setattr(checks, "_registered_plugin_eps", set())
    monkeypatch.setattr(checks.ort, "register_execution_provider_library", lambda *args: None)
    monkeypatch.setattr(checks.ort, "get_ep_devices", lambda: [device])
    levels = []
    monkeypatch.setattr(checks.ort, "set_default_logger_severity", levels.append)
    with pytest.raises(RuntimeError, match="No OpenVINO plugin EP device found"):
        checks.create_plugin_ep_session("model.onnx", None, {"device_type": "CPU"}, verbose=True)
    assert levels == [0, 2]
    output = capsys.readouterr().out
    assert "CPUExecutionProvider" in output and "vendor=Intel" in output and "vendor_id=32902" in output


def test_plugin_verbose_logging_is_enabled_in_child(monkeypatch):
    verbose_values = []
    monkeypatch.setattr(checks.sys, "argv", ["test_onnx_ep.py", "--plugin-ep"])

    def fail_session(*args, verbose=False):
        verbose_values.append(verbose)
        raise RuntimeError("no plugin device")

    monkeypatch.setattr(checks, "create_plugin_ep_session", fail_session)
    target = {"name": "CPU", "ep": "OpenVINOExecutionProvider", "options": {"device_type": "CPU"},
              "plugin_ep": True, "plugin_verbose": True}
    result = checks.run_single_target("model.onnx", {}, target)
    assert verbose_values == [True]
    assert result["status"] == "FAILED" and "no plugin device" in result["error"]


def test_resource_monitor_stops_when_child_exits(monkeypatch, tmp_path):
    child = tmp_path / "child.py"
    child.write_text("print('RESULT_JSON:{\"name\": \"target\", \"status\": \"PASSED\"}')\n", encoding="utf-8")
    monkeypatch.setattr(checks, "__file__", str(child))
    monkeypatch.setattr(checks.sys, "platform", "linux")
    stopped = threading.Event()

    def monitor(proc, finished):
        if finished.wait(5):
            stopped.set()

    monkeypatch.setattr(checks, "report_child_resources", monitor)
    assert checks.run_logged_child([], "target", "child", log_resources=True)["status"] == "PASSED"
    assert stopped.is_set()


def test_npu_aot_rejects_empty_blob(monkeypatch):
    class Core:
        available_devices = ["CPU"]

        def read_model(self, path):
            return SimpleNamespace(is_dynamic=lambda: False)

        def compile_model(self, model, device_name, config):
            return SimpleNamespace(export_model=lambda stream: None)

    monkeypatch.setitem(checks.sys.modules, "openvino", SimpleNamespace(Core=Core, __version__="test"))
    monkeypatch.setattr(checks.sys, "platform", "linux")
    result = checks._npu_aot_compile_impl("model.onnx", "NPU", "3720")

    assert result["status"] == "FAILED"
    assert "empty blob" in result["error"]


def test_npu_6010_compiler_rejection_is_a_failure(monkeypatch):
    class Core:
        available_devices = ["CPU"]

        def read_model(self, path):
            return SimpleNamespace(is_dynamic=lambda: False)

        def compile_model(self, model, device_name, config):
            assert device_name == "NPU"
            assert config == {"NPU_PLATFORM": "6010", "NPU_COMPILER_TYPE": "PLUGIN"}
            raise RuntimeError("Unknown value for platform: 6010")

    monkeypatch.setitem(checks.sys.modules, "openvino", SimpleNamespace(Core=Core, __version__="test"))
    monkeypatch.setattr(checks.sys, "platform", "linux")
    result = checks._npu_aot_compile_impl("model.onnx", "NPU 6010", "6010")

    assert result["status"] == "FAILED"
    assert "Unknown value for platform: 6010" in result["error"]


def test_plugin_cpu_runs_in_isolated_process(monkeypatch):
    target = {"name": "OpenVINO CPU", "ep": "OpenVINOExecutionProvider", "options": {"device_type": "CPU"}, "plugin_ep": True}
    expected = {"name": target["name"], "status": "PASSED"}
    calls = []

    def run(args, name, description):
        calls.append((args, name, description))
        return expected

    monkeypatch.setattr(checks.sys, "argv", ["test_onnx_ep.py", "model.onnx"])
    monkeypatch.setattr(checks, "run_logged_child", run)
    assert checks.run_single_target("model.onnx", {}, target) == expected
    assert calls[0][0][:2] == ["--plugin-ep", "model.onnx"]
    assert json.loads(calls[0][0][2]) == target


def test_child_output_is_streamed_before_exit(monkeypatch, tmp_path):
    child = tmp_path / "child.py"
    release = tmp_path / "release"
    child.write_text(
        "import json, pathlib, sys, time\n"
        "print('child started', flush=True)\n"
        f"release = pathlib.Path({str(release)!r})\n"
        "deadline = time.monotonic() + 10\n"
        "while not release.exists() and time.monotonic() < deadline:\n"
        "    time.sleep(0.01)\n"
        "if not release.exists():\n"
        "    sys.exit(2)\n"
        "print('child warning', file=sys.stderr, flush=True)\n"
        "print('RESULT_JSON:' + json.dumps({'name': 'target', 'status': 'PASSED'}))\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(checks, "__file__", str(child))
    lines = []
    started = threading.Event()

    def capture(*args, **kwargs):
        text = " ".join(str(arg) for arg in args)
        lines.append(text)
        if "child started" in text:
            started.set()

    monkeypatch.setattr(checks, "print", capture, raising=False)
    results = []
    runner = threading.Thread(target=lambda: results.append(checks.run_logged_child([], "target", "child")))
    runner.start()
    try:
        assert started.wait(5), "child output was not streamed"
        assert runner.is_alive()
    finally:
        release.touch()
        runner.join(10)
    assert not runner.is_alive()
    assert results[0]["status"] == "PASSED"
    assert any("child warning" in line for line in lines)


def test_target_heartbeat_stops_after_target(monkeypatch):
    messages = []
    heartbeat = threading.Event()

    def capture(*args, **kwargs):
        text = " ".join(str(arg) for arg in args)
        messages.append(text)
        if "still running" in text:
            heartbeat.set()

    monkeypatch.setattr(checks, "print", capture, raising=False)
    with checks.report_target_progress("NPU 3720", interval=0.01):
        assert heartbeat.wait(2)
    count = len(messages)
    time.sleep(0.05)
    assert len(messages) == count
    assert any("NPU 3720" in message and "not compiler progress" in message for message in messages)


@pytest.mark.parametrize("output", ["nothing", "RESULT_JSON:invalid", "RESULT_JSON:[]"])
def test_child_requires_a_valid_result(monkeypatch, tmp_path, output):
    child = tmp_path / "child.py"
    child.write_text(f"print({output!r})\n", encoding="utf-8")
    monkeypatch.setattr(checks, "__file__", str(child))
    assert checks.run_logged_child([], "target", "child")["status"] == "FAILED"


def test_child_result_after_utf16_stderr_newline(monkeypatch, tmp_path):
    child = tmp_path / "child.py"
    child.write_text(
        "import sys\n"
        "sys.stderr.buffer.write(b'X\\x00\\n\\x00')\n"
        "sys.stderr.flush()\n"
        "print('RESULT_JSON:{\"name\": \"target\", \"status\": \"PASSED\"}')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(checks, "__file__", str(child))
    assert checks.run_logged_child([], "target", "child")["status"] == "PASSED"


@pytest.mark.parametrize("runner,args", [
    (checks.test_npu_aot_compile, ("model.onnx", "NPU", "3720")),
    (checks.run_plugin_ep_in_subprocess, ("model.onnx", {"ep": "OpenVINOExecutionProvider"})),
])
def test_subprocess_failure_is_not_reported_as_success(monkeypatch, tmp_path, runner, args):
    child = tmp_path / "child.py"
    child.write_text(
        "import json, sys\n"
        "print('RESULT_JSON:' + json.dumps({'name': 'target', 'status': 'PASSED'}))\n"
        "sys.exit(1)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(checks, "__file__", str(child))
    assert runner(*args)["status"] == "FAILED"
