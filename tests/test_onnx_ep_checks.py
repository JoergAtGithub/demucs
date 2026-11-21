import json
import threading
import time
from types import SimpleNamespace

import pytest

from scripts import test_onnx_ep as checks


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
