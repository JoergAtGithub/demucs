import ctypes
from contextlib import contextmanager
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import numpy as np

import onnx

if len(sys.argv) < 2 or sys.argv[1] != "--plugin-ep":
    try:
        import openvino  # noqa: F401  (must be imported BEFORE onnxruntime on Linux & Windows to prevent libopenvino symbol collisions)
    except ImportError:
        pass

import onnxruntime as ort


def resolve_shape(shape_proto, default_audio_samples=44100 * 5):
    """Replace dynamic dimensions with fixed test values (required for NPU & CoreML)."""
    resolved = []
    for idx, dim in enumerate(shape_proto.dim):
        if dim.HasField("dim_value") and dim.dim_value > 0:
            resolved.append(dim.dim_value)
        else:
            param = dim.dim_param.lower() if dim.HasField("dim_param") else ""
            if idx == 0 or "batch" in param:
                resolved.append(1)
            elif "channel" in param:
                resolved.append(2)
            elif "time" in param or "sample" in param or "length" in param:
                resolved.append(default_audio_samples)
            else:
                # Fallback for other dynamic axes (e.g. STFT frames)
                resolved.append(256)
    return resolved


def create_dummy_inputs(model_path):
    """Generate matching dummy input tensors for all ONNX model inputs."""
    model = onnx.load(model_path, load_external_data=False)
    inputs = {}
    type_map = {
        onnx.TensorProto.FLOAT: np.float32,
        onnx.TensorProto.FLOAT16: np.float16,
        onnx.TensorProto.INT64: np.int64,
        onnx.TensorProto.INT32: np.int32,
        onnx.TensorProto.BOOL: np.bool_,
    }

    for inp in model.graph.input:
        tensor_type = inp.type.tensor_type
        np_dtype = type_map.get(tensor_type.elem_type, np.float32)
        shape = resolve_shape(tensor_type.shape)
        if np.issubdtype(np_dtype, np.floating):
            inputs[inp.name] = np.random.randn(*shape).astype(np_dtype)
        else:
            inputs[inp.name] = np.ones(shape, dtype=np_dtype)
        print(f"   Input '{inp.name}': Shape={shape}, Dtype={np_dtype.__name__}")
    return inputs


def load_benchmark_audio(path):
    import soundfile as sf

    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != 44100 or audio.shape[1] != 2 or not len(audio):
        raise ValueError(f"Benchmark audio must be nonempty 44.1 kHz stereo; got {sample_rate} Hz and shape {audio.shape}.")
    if not np.isfinite(audio).all():
        raise ValueError("Benchmark audio contains non-finite samples.")
    print(f"🎵 Benchmark audio: {path}, {len(audio) / sample_rate:.3f}s, {sample_rate} Hz stereo")
    return np.ascontiguousarray(audio.T), sample_rate


def benchmark_audio(session, dummy_inputs, audio, sample_rate):
    if len(dummy_inputs) != 1:
        raise ValueError("Audio benchmark requires a single model input.")
    input_name, example = next(iter(dummy_inputs.items()))
    if example.ndim != 3 or example.shape[:2] != (1, 2) or example.shape[2] <= 0 or example.dtype != np.float32:
        raise ValueError(f"Audio benchmark requires a float32 [1, 2, samples] input; got {example.shape}, {example.dtype}.")
    if sample_rate != 44100 or audio.ndim != 2 or audio.shape[0] != 2 or audio.shape[1] == 0:
        raise ValueError("Audio benchmark requires nonempty 44.1 kHz stereo audio.")

    chunk_size = example.shape[2]
    stride = max(1, int(chunk_size * 0.75))
    length = audio.shape[1]
    started = time.perf_counter()
    reference = audio.mean(axis=0)
    mean = reference.mean()
    std = reference.std(ddof=1) if length > 1 else 0.0
    normalized = (audio - mean) / (std + 1e-8)
    weights = np.minimum(np.arange(1, chunk_size + 1), np.arange(chunk_size, 0, -1)).astype(np.float32)
    weights /= weights.max()
    total_weights = np.zeros(length, dtype=np.float32)
    separated = None
    for offset in range(0, length, stride):
        count = min(chunk_size, length - offset)
        chunk = np.zeros((1, 2, chunk_size), dtype=np.float32)
        chunk[0, :, :count] = normalized[:, offset:offset + count]
        outputs = session.run(None, {input_name: chunk})
        if len(outputs) != 1 or outputs[0].ndim != 4 or outputs[0].shape[0] != 1 or outputs[0].shape[2:] != (2, chunk_size):
            raise ValueError("Audio benchmark requires one [1, sources, 2, samples] output matching the input length.")
        output = outputs[0][0]
        if not np.isfinite(output).all():
            raise ValueError("Audio benchmark output contains non-finite samples.")
        if separated is None:
            separated = np.zeros((output.shape[0], 2, length), dtype=np.float32)
        elif separated.shape[0] != output.shape[0]:
            raise ValueError("Audio benchmark output source count changed between chunks.")
        separated[:, :, offset:offset + count] += output[:, :, :count] * weights[:count]
        total_weights[offset:offset + count] += weights[:count]
    separated /= total_weights
    separated *= std + 1e-8
    separated += mean
    elapsed = time.perf_counter() - started
    if elapsed <= 0:
        raise ValueError("Audio benchmark elapsed time must be positive.")
    duration = length / sample_rate
    return {"audio_s": duration, "audio_processing_s": elapsed, "processing_speed": duration / elapsed}


def get_openvino_devices():
    """Return available OpenVINO hardware devices if openvino package is installed."""
    try:
        from openvino import Core
        return Core().available_devices
    except Exception:
        return []


def verify_cudnn_libraries():
    """Verify that NVIDIA cuDNN shared libraries are present and loadable."""
    try:
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
    except Exception:
        pass

    search_roots = [os.path.dirname(ort.__file__)]
    try:
        import nvidia.cudnn
        search_roots.extend(list(nvidia.cudnn.__path__))
    except Exception:
        pass

    cudnn_files = []
    for root in search_roots:
        cudnn_files.extend(glob.glob(os.path.join(root, "**", "*cudnn*"), recursive=True))

    shared_libs = [
        f for f in cudnn_files
        if (".so" in os.path.basename(f) or f.endswith(".dll")) and os.path.isfile(f)
    ]
    if not shared_libs:
        return False, "No cuDNN shared libraries (*.so / *.dll) found in environment."

    # Try loading the primary cuDNN library to verify ABI/symbol linkage
    primary = sorted(shared_libs, key=len)[0]
    try:
        ctypes.CDLL(primary)
    except Exception as e:
        return False, f"Failed to load cuDNN library '{primary}': {e}"

    return True, f"Loaded cuDNN library: {os.path.basename(primary)}"


def provider_is_usable(model_path, provider):
    """Return True if the provider can really be created (libraries and GPU present)."""
    try:
        opts = ort.SessionOptions()
        opts.log_severity_level = 4
        session = ort.InferenceSession(model_path, sess_options=opts, providers=[provider, "CPUExecutionProvider"])
        return provider in session.get_providers()
    except Exception:
        return False


def check_cuda_cudnn_without_gpu(model_path, target_name):
    """Verify CUDA + cuDNN EP registration, cuDNN DLL linkage, and strict ONNX graph compatibility."""
    t1 = time.perf_counter()
    cudnn_ok, cudnn_msg = verify_cudnn_libraries()
    if not cudnn_ok:
        print(f"❌ cuDNN ERROR: {cudnn_msg}")
        return {"name": target_name, "status": "FAILED", "compile_s": 0.0, "warmup_s": 0.0, "infer_s": 0.0, "error": cudnn_msg}

    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        msg = f"CUDAExecutionProvider missing from onnxruntime-gpu build (found: {providers})."
        print(f"❌ ERROR: {msg}")
        return {"name": target_name, "status": "FAILED", "compile_s": 0.0, "warmup_s": 0.0, "infer_s": 0.0, "error": msg}

    model = onnx.load(model_path)
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    opsets = {op.domain or "ai.onnx": op.version for op in inferred.opset_import}
    t_compile = time.perf_counter() - t1

    print(f"ℹ️  {cudnn_msg}")
    print(f"ℹ️  CUDA + cuDNN EP registered in ORT. Verified opsets: {opsets}")
    print("⚠️  No physical NVIDIA GPU on runner (PTX/cuDNN kernels are compatible across Turing/Ampere/Ada/Hopper/Blackwell).")
    print(f"✅ CUDA + cuDNN dry-run & strict graph validation passed in {t_compile:.3f}s!")
    return {"name": target_name, "status": "PASSED (DRY)", "compile_s": t_compile, "warmup_s": 0.0, "infer_s": 0.0, "error": ""}


def find_migraphx_driver():
    """Locate migraphx-driver on PATH or in the usual ROCm install prefixes."""
    import shutil
    found = shutil.which("migraphx-driver")
    if found:
        return found
    for pattern in ("/opt/rocm*/bin/migraphx-driver", "/opt/rocm*/**/migraphx-driver", "/usr/bin/migraphx-driver"):
        hits = sorted(glob.glob(pattern, recursive=True))
        if hits:
            return hits[-1]
    return None


def run_migraphx_offline(model_path, target):
    """Cross-compile ONNX model for a specific AMD GPU gfx architecture using migraphx-driver."""
    name = target.get("name", "MIGraphX")
    gfx_arch = target["gfx_arch"]
    optional_arch = target.get("optional_arch", False)

    print("\n--------------------------------------------------")
    print(f"🎯 Target:              {name}")
    print("🔌 Backend:             AMD MIGraphX AOT Compiler")
    print(f"🧠 Target GPU Arch:     {gfx_arch}")
    print("--------------------------------------------------")

    driver = find_migraphx_driver()
    if driver is None:
        msg = "migraphx-driver not found on PATH or under /opt/rocm*."
        print(f"\u274c ERROR: {msg}")
        return {"name": name, "status": "FAILED", "compile_s": 0.0, "warmup_s": 0.0, "infer_s": 0.0, "error": msg}
    print(f"\u2139\ufe0f  Using {driver}")

    cmd = [driver, "compile", "--gpu", "--gpu-arch", gfx_arch, model_path]
    t1 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    t_compile = time.perf_counter() - t1

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout).strip()
        if optional_arch and ("unknown" in err.lower() or "invalid" in err.lower() or "arch" in err.lower()):
            print(f"⏭️  SKIPPED ({gfx_arch} not supported by installed ROCm version):\n{err[:300]}")
            return {"name": name, "status": "SKIPPED", "compile_s": 0.0, "warmup_s": 0.0, "infer_s": 0.0, "error": err}
        print(f"❌ MIGRAPHX COMPILATION ERROR ({gfx_arch}):\n{err}")
        return {"name": name, "status": "FAILED", "compile_s": t_compile, "warmup_s": 0.0, "infer_s": 0.0, "error": err}

    print(f"⏱️ [Step 1] MIGraphX Graph & MLIR Compile: {t_compile:.3f}s")
    print(f"✅ Offline compilation for AMD {gfx_arch} succeeded!")
    return {"name": name, "status": "PASSED (AOT)", "compile_s": t_compile, "warmup_s": 0.0, "infer_s": 0.0, "error": ""}


def _capture_native_stderr(func):
    """Run func() and capture everything ONNX Runtime writes natively to stderr (fd 2)."""
    sys.stderr.flush()
    tmp = tempfile.TemporaryFile()
    saved = os.dup(2)
    os.dup2(tmp.fileno(), 2)
    result, error = None, None
    try:
        result = func()
    except Exception as e:
        error = e
    finally:
        os.dup2(saved, 2)
        os.close(saved)
    tmp.seek(0)
    raw = tmp.read()
    tmp.close()
    text = raw.decode("utf-16-le", "replace") if b"\x00" in raw[:200] else raw.decode("utf-8", "replace")
    return result, error, text


def diagnose_cpu_fallback(model_path, target_ep, ep_options):
    """Re-create the session with CPU fallback allowed and report which nodes need the CPU."""
    print("\n🔎 CPU-FALLBACK DIAGNOSIS (session re-created with CPU fallback allowed)")
    opts = ort.SessionOptions()
    opts.log_severity_level = 0
    providers = [(target_ep, ep_options) if ep_options else target_ep, "CPUExecutionProvider"]
    session, error, log = _capture_native_stderr(
        lambda: ort.InferenceSession(model_path, sess_options=opts, providers=providers)
    )
    if error is not None:
        print(f"   Diagnostic session could not be created: {error}")
        return

    active = session.get_providers()
    print(f"   Active providers: {active}")
    if target_ep not in active:
        print(f"   ➜ {target_ep} could NOT be created at all, so ALL nodes fall back to the CPU.")
        print("   This is a provider/option problem, not an unsupported-operator problem. ORT messages:")
        for line in log.splitlines():
            if re.search(r"\[(W|E):", line) and ("Failed to create" in line or target_ep in line or "OpenVINO" in line):
                print(f"     {line.strip()[:200]}")
        print("   Check the EP options above (e.g. 'load_config' JSON, device_type) and the provider/runtime versions.")
        return

    op_by_name = {}
    try:
        for n in onnx.load(model_path, load_external_data=False).graph.node:
            op_by_name[n.name] = n.op_type
    except Exception:
        pass

    # Split the verbose log into records (each starts with a timestamp) and pick the placement records
    records = re.split(r"\r?\n(?=\d{4}-\d{2}-\d{2} )", log)
    placement = [r for r in records if "placed on" in r or "Node placements" in r]
    log_lines = log.splitlines()
    cpu_chunks = []
    for i, line in enumerate(log_lines):
        if "placed on [CPUExecutionProvider]" in line:
            chunk = [line]
            for nxt in log_lines[i + 1:i + 400]:
                if "placed on [" in nxt:
                    break
                chunk.append(nxt)
            cpu_chunks.append("\n".join(chunk))
    cpu_text = "\n".join(cpu_chunks) or "\n".join(r for r in placement if "CPUExecutionProvider" in r)
    for r in placement:
        header = re.search(r"(All nodes placed on|Node\(s\) placed on) \[[^\]]+\][^\n]*", r)
        if header:
            print(f"   {header.group(0).strip()}")

    if "CPUExecutionProvider" not in cpu_text:
        print("   No nodes assigned to the CPU EP in the diagnostic session.")
        return

    names = [n for n in op_by_name if n and re.search(rf"(?<![\w.]){re.escape(n)}(?![\w.])", cpu_text)]
    if names:
        counts = {}
        for n in names:
            counts[op_by_name[n]] = counts.get(op_by_name[n], 0) + 1
        print(f"   Op types that need the CPU: {dict(sorted(counts.items(), key=lambda kv: -kv[1]))}")
        print(f"   Nodes that need the CPU ({len(names)}):")
        for n in names[:50]:
            print(f"     - {n} ({op_by_name[n]})")
        if len(names) > 50:
            print(f"     ... and {len(names) - 50} more")
    else:
        print("   Raw ORT placement log for the CPU EP:")
        for line in cpu_text.splitlines()[:60]:
            print(f"     {line.strip()[:200]}")


@contextmanager
def report_target_progress(name, interval=30):
    started = time.monotonic()
    finished = threading.Event()

    def report():
        while not finished.wait(interval):
            elapsed = time.monotonic() - started
            print(f"⏳ {name}: still running after {elapsed:.0f}s (elapsed time, not compiler progress).", flush=True)

    print(f"▶️ Starting {name}.", flush=True)
    reporter = threading.Thread(target=report, daemon=True)
    reporter.start()
    try:
        yield
    finally:
        finished.set()
        reporter.join()


def report_child_resources(proc, finished, interval=60):
    while not finished.wait(interval):
        if proc.poll() is not None:
            break
        try:
            snapshot = subprocess.run(
                ["ps", "-p", str(proc.pid), "-o", "pid=,stat=,%cpu=,rss=,vsz=,nlwp=,etime=,wchan:32="],
                capture_output=True, text=True, timeout=5,
            )
            print(f"🔎 PID STAT %CPU(avg) RSS(KiB) VSZ(KiB) THREADS ELAPSED WCHAN: {snapshot.stdout.strip() or 'unavailable'}", flush=True)
        except (OSError, subprocess.SubprocessError) as e:
            print(f"🔎 Process resource snapshot unavailable: {e}", flush=True)
        for path in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.events"):
            try:
                with open(path, encoding="utf-8") as source:
                    value = source.read().strip().replace("\n", ", ")
                print(f"🔎 {path}: {value}", flush=True)
            except OSError:
                pass


def run_logged_child(args, name, description, log_resources=False):
    proc = subprocess.Popen(
        [sys.executable, "-u", os.path.abspath(__file__), *args],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    result = None
    finished = threading.Event()
    monitor = None
    if log_resources and sys.platform.startswith("linux"):
        monitor = threading.Thread(target=report_child_resources, args=(proc, finished), daemon=True)
        monitor.start()
    try:
        with proc.stdout:
            for line in proc.stdout:
                print(line, end="", flush=True)
                result_line = line.lstrip("\x00")
                if result_line.startswith("RESULT_JSON:"):
                    try:
                        result = json.loads(result_line[len("RESULT_JSON:"):])
                    except json.JSONDecodeError:
                        result = None
        returncode = proc.wait()
    finally:
        finished.set()
        if monitor is not None:
            monitor.join()
    if returncode == 0 and isinstance(result, dict):
        return result
    msg = f"{description} failed or did not return a valid result (exit code {returncode})."
    print(f"❌ {msg}")
    return {"name": name, "status": "FAILED", "compile_s": 0.0, "warmup_s": 0.0, "infer_s": 0.0, "error": msg}


def test_npu_aot_compile(model_path, name, platform, log_level=None, log_resources=False):
    """Run NPU AOT compilation in an isolated process and stream its output."""
    args = ["--npu-aot", model_path, name, str(platform)]
    if log_level:
        args.append(log_level)
    return run_logged_child(args, name, "NPU AOT compile", log_resources=log_resources)


def _npu_ir_check(ov, model_path, model, name, platform, t0):
    """Primary NPU check on runners without NPU compiler/driver: OpenVINO IR conversion & NPU readiness."""
    print(f"ℹ️  No NPU compiler on runner; running OpenVINO IR conversion & NPU readiness check for {platform}.")
    allowed_types = {"f32", "f16", "i8", "u8", "i32", "i64", "boolean"}
    problems = []
    if model.is_dynamic():
        onnx_inputs = onnx.load(model_path, load_external_data=False).graph.input
        shapes = {i.name: resolve_shape(i.type.tensor_type.shape) for i in onnx_inputs}
        try:
            model.reshape(shapes)
        except Exception as e:
            problems.append(f"could not reshape model to static shapes: {e}")
        if not problems and model.is_dynamic():
            problems.append("model still has dynamic shapes (NPU requires static shapes)")
    for port in list(model.inputs) + list(model.outputs):
        et = port.get_element_type().get_type_name()
        if et not in allowed_types:
            problems.append(f"port '{port.get_any_name()}' uses unsupported precision {et}")
    for op in model.get_ordered_ops():
        if "complex" in op.get_type_name().lower():
            problems.append(f"unsupported complex operator {op.get_type_name()} ({op.get_friendly_name()})")
    if problems:
        msg = "; ".join(problems)
        print(f"❌ NPU {platform} READINESS ERROR: {msg}")
        return {"name": name, "status": "FAILED", "compile_s": time.perf_counter() - t0, "warmup_s": 0.0, "infer_s": 0.0, "error": msg}
    with tempfile.TemporaryDirectory() as tmp:
        ov.serialize(model, os.path.join(tmp, "temp_npu_ir.xml"), os.path.join(tmp, "temp_npu_ir.bin"))
    t_total = time.perf_counter() - t0
    print(f"✅ OpenVINO IR serialization & NPU readiness for {platform} PASSED in {t_total:.3f}s (no NPU compiler on runner).")
    return {"name": name, "status": "PASSED (IR)", "compile_s": t_total, "warmup_s": 0.0, "infer_s": 0.0, "error": ""}


def run_openvino_gpu_offline_check(model_path, name):
    """GPU-less runners: verify the Intel GPU plugin binary and OpenVINO IR conversion of the model."""
    print("ℹ️  No physical Intel GPU on runner; running OpenVINO GPU plugin & IR readiness check.")
    t0 = time.perf_counter()
    try:
        import openvino as ov
        ov_dir = os.path.dirname(ov.__file__)
        gpu_plugins = glob.glob(os.path.join(ov_dir, "**", "*intel_gpu_plugin*"), recursive=True)
        if not gpu_plugins:
            raise FileNotFoundError("OpenVINO Intel GPU plugin library is missing from the installation.")
        print(f"📦 GPU plugin: {gpu_plugins[0]}")
        model = ov.Core().read_model(model_path)
        with tempfile.TemporaryDirectory() as tmp:
            ov.serialize(model, os.path.join(tmp, "temp_ov_gpu.xml"), os.path.join(tmp, "temp_ov_gpu.bin"))
    except Exception as e:
        msg = str(e)
        print(f"❌ OpenVINO GPU offline check ERROR: {msg}")
        return {"name": name, "status": "FAILED", "compile_s": time.perf_counter() - t0, "warmup_s": 0.0, "infer_s": 0.0, "error": msg}
    compile_s = round(time.perf_counter() - t0, 3)
    print(f"✅ OpenVINO GPU plugin & IR serialization PASSED in {compile_s:.3f}s. GPU compilation and loading were not tested.")
    return {"name": name, "status": "PASSED (IR)", "compile_s": compile_s, "warmup_s": 0.0, "infer_s": 0.0, "error": ""}


def _npu_aot_compile_impl(model_path, name, platform, log_level=None):
    import io
    try:
        import openvino as ov
    except ImportError as e:
        msg = f"openvino Python package not installed: {e}"
        print(f"❌ ERROR: {msg}")
        return {"name": name, "status": "FAILED", "compile_s": 0.0, "warmup_s": 0.0, "infer_s": 0.0, "error": msg}
    t0 = time.perf_counter()
    try:
        core = ov.Core()
        print(f"ℹ️  OpenVINO {ov.__version__}, devices visible: {core.available_devices}")
        print(f"⏳ NPU {platform}: reading model.", flush=True)
        model = core.read_model(model_path)
        config = {"NPU_PLATFORM": str(platform), "NPU_COMPILER_TYPE": "PLUGIN"}
        if log_level:
            config["LOG_LEVEL"] = log_level
            print(f"🔎 NPU {platform}: compiler log level {log_level}.", flush=True)
        if sys.platform == "win32":
            return _npu_ir_check(ov, model_path, model, name, platform, t0)
        if model.is_dynamic():
            print(f"⏳ NPU {platform}: resolving dynamic input shapes.", flush=True)
            onnx_inputs = onnx.load(model_path, load_external_data=False).graph.input
            model.reshape({i.name: resolve_shape(i.type.tensor_type.shape) for i in onnx_inputs})
        print(f"⏳ NPU {platform}: starting compiler-in-plugin.", flush=True)
        compiled = core.compile_model(model, device_name="NPU", config=config)
        t_compile = time.perf_counter() - t0
        print(f"⏳ NPU {platform}: compilation finished; exporting blob.", flush=True)
        blob = io.BytesIO()
        compiled.export_model(blob)
        if not blob.getbuffer().nbytes:
            raise RuntimeError("NPU compiler exported an empty blob.")
        size_mb = len(blob.getvalue()) / (1024 * 1024)
    except Exception as e:
        msg = str(e)
        print(f"❌ NPU {platform} AOT COMPILE ERROR:\n{msg}")
        return {"name": name, "status": "FAILED", "compile_s": time.perf_counter() - t0, "warmup_s": 0.0, "infer_s": 0.0, "error": msg}
    print(f"✅ NPU {platform} AOT compile PASSED in {t_compile:.3f}s (blob: {size_mb:.2f} MB)")
    return {"name": name, "status": "PASSED (AOT)", "compile_s": t_compile, "warmup_s": 0.0, "infer_s": 0.0, "error": ""}


# Operators unsupported by DirectML or known to cause hard failures/CPU fallbacks
DML_UNSUPPORTED_OPS = {
    "Complex",
    "STFT",
    "InverseSTFT",
    "StringNormalizer",
    "TfIdfVectorizer",
    "TreeEnsembleClassifier",
    "TreeEnsembleRegressor",
}


def run_directml_offline_check(onnx_path, target_name="DirectML (Offline Check)"):
    """
    Validates DirectML runtime availability (DirectML.dll + d3d12.dll) and checks
    ONNX graph compatibility for DirectML on headless Windows runners without a GPU.
    """
    print("-" * 50)
    print(f"🎯 Target:              {target_name}")
    print("🔌 Execution Provider:  DmlExecutionProvider (Offline Runtime & Graph Check)")
    print("⚙️  EP Options:          {}")
    print("-" * 50)

    t0 = time.perf_counter()

    try:
        # 1. Verify that DirectML.dll is bundled with onnxruntime-directml and loadable
        ort_dir = os.path.dirname(ort.__file__)
        dml_dlls = glob.glob(os.path.join(ort_dir, "**", "DirectML.dll"), recursive=True)
        if not dml_dlls:
            raise FileNotFoundError("DirectML.dll was not found inside the onnxruntime-directml package.")

        dml_lib = ctypes.WinDLL(dml_dlls[0])
        d3d12_lib = ctypes.WinDLL("d3d12.dll")

        if not hasattr(dml_lib, "DMLCreateDevice"):
            raise AttributeError("Symbol 'DMLCreateDevice' is missing in DirectML.dll.")
        if not hasattr(d3d12_lib, "D3D12CreateDevice"):
            raise AttributeError("Symbol 'D3D12CreateDevice' is missing in d3d12.dll.")

        # 2. Load ONNX model and validate DirectML opset limits (DirectML supports up to Opset 21)
        model = onnx.load(onnx_path, load_external_data=False)
        opset = max((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), default=0)
        if opset > 21:
            raise ValueError(f"ONNX Opset {opset} exceeds DirectML maximum supported opset (21).")

        # 3. Check all graph nodes against known unsupported DirectML operators
        used_ops = {node.op_type for node in model.graph.node}
        unsupported = used_ops.intersection(DML_UNSUPPORTED_OPS)
        if unsupported:
            raise ValueError(f"Found operators unsupported by DirectML: {sorted(unsupported)}")

        # 4. Ensure no complex tensor types (COMPLEX64=14, COMPLEX128=15) exist in the graph
        value_infos = list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
        for vi in value_infos:
            elem_type = vi.type.tensor_type.elem_type
            if elem_type in (14, 15):
                raise TypeError(f"Tensor '{vi.name}' uses complex dtype ({elem_type}), which DirectML does not support.")

        compile_s = round(time.perf_counter() - t0, 3)

        print(f"📦 Loaded DirectML.dll:  {dml_dlls[0]}")
        print(f"ℹ️  Graph validated:      {len(model.graph.node)} nodes, {len(used_ops)} unique ops, Opset {opset}")
        print(f"⏱️ [Step 1] Validation:  {compile_s:.3f}s")
        print(f"✅ All offline checks for {target_name} passed successfully!")
        return {"name": target_name, "status": "PASSED (Offline)", "compile_s": compile_s, "warmup_s": 0.0, "infer_s": 0.0, "error": ""}

    except Exception as exc:
        compile_s = round(time.perf_counter() - t0, 3)
        error_msg = str(exc)
        print(f"❌ ERROR: {error_msg}")
        return {"name": target_name, "status": "FAILED", "compile_s": compile_s, "warmup_s": 0.0, "infer_s": 0.0, "error": error_msg}


_registered_plugin_eps = set()


def run_plugin_ep_in_subprocess(model_path, target):
    """Keep the standalone EP's bundled OpenVINO libraries isolated from the Python package."""
    name = target.get("name", target["ep"])
    return run_logged_child(["--plugin-ep", model_path, json.dumps(target)], name, "OpenVINO plugin EP process")


def create_plugin_ep_session(model_path, sess_options, ep_options, verbose=False):
    """Create a session using the standalone onnxruntime-ep-openvino plugin EP."""
    import onnxruntime_ep_openvino as openvino_ep
    if verbose:
        ort.set_default_logger_severity(0)
        print("🔎 Enabled verbose ONNX Runtime logging for plugin registration.", flush=True)
    try:
        ep_name = openvino_ep.get_ep_name()
        if ep_name not in _registered_plugin_eps:
            ort.register_execution_provider_library("openvino_ep", openvino_ep.get_library_path())
            _registered_plugin_eps.add(ep_name)
        options = dict(ep_options)
        device_type = options.pop("device_type", "CPU")
        all_devices = ort.get_ep_devices()
        devices = [
            d for d in all_devices
            if d.ep_name.split(".")[0] == ep_name and str(d.ep_metadata.get("ov_device", "")).upper() == device_type.upper()
        ]
        devices.sort(key=lambda d: d.ep_name != ep_name)
        if not devices:
            print(f"🔍 Registered plugin library: {openvino_ep.get_library_path()}")
            print(f"🔍 ORT EP devices visible ({len(all_devices)}):")
            for d in all_devices:
                hardware = d.device
                print(f"   - {d.ep_name}: ep_vendor={d.ep_vendor}, ep_metadata={d.ep_metadata}, "
                      f"type={hardware.type}, vendor={hardware.vendor}, vendor_id={hardware.vendor_id}, "
                      f"device_id={hardware.device_id}, metadata={hardware.metadata}")
            raise RuntimeError(f"No OpenVINO plugin EP device found for {device_type}.")
    finally:
        if verbose:
            ort.set_default_logger_severity(2)
    sess_options.add_provider_for_devices([devices[0]], options)
    return ort.InferenceSession(model_path, sess_options=sess_options)


def run_single_target(model_path, dummy_inputs, target, benchmark_input=None):
    """Run compilation and optional inference for a single EP/device configuration."""
    if target.get("backend") == "migraphx_cli":
        return run_migraphx_offline(model_path, target)

    if target.get("backend") == "directml_offline":
        return run_directml_offline_check(model_path, target.get("name", "DirectML (Offline Check)"))

    if target.get("npu_platform"):
        name = target.get("name", target["ep"])
        print("\n--------------------------------------------------")
        print(f"🎯 Target:              {name}")
        print(f"🧠 Target NPU Platform: {target['npu_platform']} (OpenVINO API AOT compile, no ORT session)")
        print("--------------------------------------------------")
        return test_npu_aot_compile(
            model_path, name, target["npu_platform"],
            log_level=target.get("npu_log_level"),
            log_resources=target.get("log_resources", False),
        )

    name = target.get("name", target["ep"])
    target_ep = target["ep"]
    ep_options = dict(target.get("options", {}))
    npu_platform = target.get("npu_platform")
    compile_only = target.get("compile_only", False)
    skip_if_missing_hw = target.get("skip_if_missing_hw", False)
    optional_platform = target.get("optional_platform", False)
    plugin_ep = target.get("plugin_ep", False)
    if (not plugin_ep and target.get("ep") == "OpenVINOExecutionProvider"
            and "OpenVINOExecutionProvider" not in ort.get_available_providers()):
        import importlib.util
        plugin_ep = importlib.util.find_spec("onnxruntime_ep_openvino") is not None

    if (plugin_ep and ep_options.get("device_type", "CPU") == "CPU"
            and (len(sys.argv) < 2 or sys.argv[1] != "--plugin-ep")):
        return run_plugin_ep_in_subprocess(model_path, target)

    temp_config_path = None

    # Dynamically generate OpenVINO NPU AOT config if npu_platform is specified
    if npu_platform:
        ov_config = {
            "NPU": {
                "NPU_COMPILER_TYPE": "PLUGIN",
                "NPU_PLATFORM": str(npu_platform),
            }
        }
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(ov_config, tmp)
        tmp.close()
        temp_config_path = tmp.name
        ep_options["device_type"] = "NPU"
        ep_options["load_config"] = temp_config_path

    print("\n--------------------------------------------------")
    print(f"🎯 Target:              {name}")
    print(f"🔌 Execution Provider:  {target_ep}")
    if npu_platform:
        print(f"🧠 Target NPU Platform: {npu_platform} (AOT Plugin Compiler)")
    print(f"⚙️  EP Options:          {ep_options}")
    print("--------------------------------------------------")

    try:
        # Handle NVIDIA CUDA + cuDNN dry-run on CPU-only runners
        if target_ep == "CUDAExecutionProvider" and target.get("allow_no_gpu_dry_run", False):
            if not provider_is_usable(model_path, "CUDAExecutionProvider"):
                return check_cuda_cudnn_without_gpu(model_path, name)

        # Check physical hardware availability for OpenVINO GPU (requires OpenCL driver)
        if target_ep == "OpenVINOExecutionProvider" and ep_options.get("device_type") == "GPU":
            ov_devices = get_openvino_devices()
            print(f"🔍 Detected OpenVINO Hardware Devices: {ov_devices}")
            if target.get("force_offline") or (not any(d.startswith("GPU") for d in ov_devices) and skip_if_missing_hw):
                return run_openvino_gpu_offline_check(model_path, name)

        if not plugin_ep and target_ep not in ort.get_available_providers():
            msg = f"{target_ep} is not installed/available in this ONNX Runtime build."
            if target.get("skip_if_unavailable", False):
                print(f"⏭️  SKIPPED: {msg}")
                return {"name": name, "status": "SKIPPED", "compile_s": 0.0, "warmup_s": 0.0, "infer_s": 0.0, "error": msg}
            print(f"❌ ERROR: {msg}")
            return {"name": name, "status": "FAILED", "compile_s": 0.0, "warmup_s": 0.0, "infer_s": 0.0, "error": msg}

        # Per-op CPU fallback is allowed; only a provider that cannot be created at all is an error
        sess_options = ort.SessionOptions()
        if target_ep == "OpenVINOExecutionProvider" and ep_options.get("device_type") == "CPU":
            sess_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        sess_options.log_severity_level = 2

        # 1. Load & compile model (graph validation + EP compilation)
        t1 = time.perf_counter()
        try:
            if plugin_ep:
                session = create_plugin_ep_session(model_path, sess_options, ep_options, verbose=target.get("plugin_verbose", False))
            else:
                providers = [(target_ep, ep_options)] if ep_options else [target_ep]
                session = ort.InferenceSession(
                    model_path,
                    sess_options=sess_options,
                    providers=providers,
                )
        except Exception as e:
            err_str = str(e)
            if optional_platform and ("Unsupported platform" in err_str or "NPU_PLATFORM" in err_str):
                print(f"⏭️  SKIPPED (Platform not supported by installed compiler):\n{e}")
                return {"name": name, "status": "SKIPPED", "compile_s": 0.0, "warmup_s": 0.0, "infer_s": 0.0, "error": err_str}
            print(f"❌ COMPILATION ERROR on {name}:\n{e}")
            if target_ep != "CPUExecutionProvider" and not plugin_ep:
                try:
                    diagnose_cpu_fallback(model_path, target_ep, ep_options)
                except Exception as diag_err:
                    print(f"   Diagnosis failed: {diag_err}")
            return {"name": name, "status": "FAILED", "compile_s": 0.0, "warmup_s": 0.0, "infer_s": 0.0, "error": err_str}

        t_compile = time.perf_counter() - t1
        print(f"⏱️ [Step 1] Session & Compilation:   {t_compile:.3f}s")

        active_providers = session.get_providers()
        if target_ep not in active_providers:
            msg = f"{target_ep} was not activated (active: {active_providers})."
            if ep_options.get("device_type") == "NPU" and not any(d.startswith("NPU") for d in get_openvino_devices()):
                msg = "No Intel NPU device visible to OpenVINO on this runner (skipped)."
                print(f"⏭️  SKIPPED: {msg}")
                return {"name": name, "status": "SKIPPED", "compile_s": t_compile, "warmup_s": 0.0, "infer_s": 0.0, "error": msg}
            print(f"❌ ERROR: {msg}")
            return {"name": name, "status": "FAILED", "compile_s": t_compile, "warmup_s": 0.0, "infer_s": 0.0, "error": msg}

        if target_ep != "CPUExecutionProvider" and not plugin_ep:
            print("⚠️  Note: individual ops may fall back to the CPU EP (informational, not an error).")
            try:
                diagnose_cpu_fallback(model_path, target_ep, ep_options)
            except Exception as diag_err:
                print(f"   Diagnosis failed: {diag_err}")

        if compile_only:
            print(f"✅ Offline compilation (AOT) for {name} succeeded!")
            return {"name": name, "status": "PASSED (AOT)", "compile_s": t_compile, "warmup_s": 0.0, "infer_s": 0.0, "error": ""}

        # 2. Warmup inference (first run compiles shaders/kernels)
        t2 = time.perf_counter()
        try:
            outputs = session.run(None, dummy_inputs)
        except Exception as e:
            print(f"❌ WARMUP INFERENCE ERROR on {name}:\n{e}")
            return {"name": name, "status": "FAILED", "compile_s": t_compile, "warmup_s": 0.0, "infer_s": 0.0, "error": str(e)}
        t_warmup = time.perf_counter() - t2
        print(f"⏱️ [Step 2] 1st Inference (Warmup):  {t_warmup:.3f}s")

        # 3. Second inference (steady-state runtime without init overhead)
        t3 = time.perf_counter()
        try:
            outputs = session.run(None, dummy_inputs)
        except Exception as e:
            print(f"❌ STEADY INFERENCE ERROR on {name}:\n{e}")
            return {"name": name, "status": "FAILED", "compile_s": t_compile, "warmup_s": t_warmup, "infer_s": 0.0, "error": str(e)}
        t_infer = time.perf_counter() - t3
        print(f"⏱️ [Step 3] 2nd Inference (Steady):  {t_infer:.3f}s")

        for idx, out in enumerate(outputs):
            if np.isnan(out).any():
                msg = f"Output[{idx}] contains NaN values!"
                print(f"❌ ERROR: {msg}")
                return {"name": name, "status": "FAILED", "compile_s": t_compile, "warmup_s": t_warmup, "infer_s": t_infer, "error": msg}
            print(f"   Output[{idx}]: Shape={out.shape}, Min={out.min():.4f}, Max={out.max():.4f}")

        result = {"name": name, "status": "PASSED", "compile_s": t_compile, "warmup_s": t_warmup, "infer_s": t_infer, "error": ""}
        if benchmark_input is not None:
            try:
                result.update(benchmark_audio(session, dummy_inputs, *benchmark_input))
            except Exception as e:
                print(f"❌ AUDIO BENCHMARK ERROR on {name}: {e}")
                result.update(status="FAILED", error=str(e))
                return result
            print(f"🎵 Audio processing: {result['audio_processing_s']:.3f}s for {result['audio_s']:.3f}s, processing speed {result['processing_speed']:.2f}x")
        print(f"✅ All checks for {name} passed successfully!")
        return result
    finally:
        if temp_config_path and os.path.exists(temp_config_path):
            os.remove(temp_config_path)


def _result(name, status, t, error=""):
    return {"name": name, "status": status, "compile_s": t, "warmup_s": 0.0, "infer_s": 0.0, "error": error}


def check_onnx_structure(model_path):
    """ONNX checker (full), strict shape inference and a plain ORT CPU session to catch structural errors."""
    name = "ONNX Checker + Shape Inference + ORT CPU"
    print("\n--------------------------------------------------")
    print(f"🎯 Pre-flight check:    {name}")
    print("--------------------------------------------------")
    t0 = time.perf_counter()
    try:
        onnx.checker.check_model(model_path, full_check=True)
        print("ℹ️  onnx.checker.check_model(full_check=True) passed.")
        model = onnx.load(model_path, load_external_data=False)
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
        print(f"ℹ️  Strict shape inference passed ({len(inferred.graph.node)} nodes, opsets "
              f"{ {op.domain or 'ai.onnx': op.version for op in inferred.opset_import} }).")
        dynamic = []
        for vi in list(model.graph.input) + list(model.graph.output):
            dims = vi.type.tensor_type.shape.dim
            if any(d.dim_param or (d.HasField("dim_value") and d.dim_value <= 0) or not (d.dim_param or d.HasField("dim_value")) for d in dims):
                dynamic.append(vi.name)
        if dynamic:
            print(f"⚠️  Inputs/outputs with dynamic or unknown dimensions: {dynamic}")
        else:
            print("ℹ️  All graph inputs/outputs have static shapes.")
        so = ort.SessionOptions()
        so.log_severity_level = 3
        ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])
        print("ℹ️  ORT CPUExecutionProvider session created.")
    except Exception as e:
        print(f"❌ STRUCTURAL CHECK ERROR:\n{e}")
        return _result(name, "FAILED", time.perf_counter() - t0, str(e))
    t = time.perf_counter() - t0
    print(f"✅ Structural checks passed in {t:.3f}s!")
    return _result(name, "PASSED", t)


def find_polygraphy():
	"""Locate the polygraphy CLI on PATH, next to the interpreter or in the user scripts directory."""
	import shutil
	import sysconfig
	exe = shutil.which("polygraphy")
	if exe:
		return exe
	dirs = [os.path.join(os.path.dirname(sys.executable), "Scripts" if sys.platform == "win32" else "bin")]
	try:
		dirs.append(sysconfig.get_path("scripts", "nt_user" if sys.platform == "win32" else "posix_user"))
	except Exception:
		pass
	for d in dirs:
		for cand in ("polygraphy.exe", "polygraphy"):
			p = os.path.join(d, cand)
			if os.path.exists(p):
				return p
	return None


def _run_polygraphy(name, args, max_lines=60, timeout=600):
	"""Run one polygraphy sub-command and return a result dict. A native crash counts as FAILED."""
	print("\n--------------------------------------------------")
	print(f"🎯 Pre-flight check:    {name}")
	print("--------------------------------------------------")
	t0 = time.perf_counter()
	exe = find_polygraphy()
	if exe is None:
		msg = "polygraphy executable not found (pip install colored polygraphy --extra-index-url https://pypi.ngc.nvidia.com)."
		print(f"❌ {msg}")
		return _result(name, "FAILED", time.perf_counter() - t0, msg)
	env = dict(os.environ, PYTHONUTF8="1")
	try:
		proc = subprocess.run(
			[exe] + args, capture_output=True, text=True, encoding="utf-8", errors="replace",
			timeout=timeout, env=env,
		)
	except subprocess.TimeoutExpired:
		msg = f"polygraphy {' '.join(args[:2])} timed out after {timeout}s."
		print(f"❌ {msg}")
		return _result(name, "FAILED", time.perf_counter() - t0, msg)
	out = ((proc.stdout or "") + (proc.stderr or "")).strip()
	lines = out.splitlines()
	print("\n".join(lines[:max_lines]))
	if len(lines) > max_lines:
		print(f"   ... ({len(lines) - max_lines} more lines)")
	t = time.perf_counter() - t0
	if proc.returncode != 0:
		msg = f"polygraphy {' '.join(args[:2])} failed (exit code {proc.returncode})."
		if proc.returncode in (3221225477, -1073741819, -11, 139):
			msg += " Native crash: TensorRT cannot initialize without a CUDA-capable NVIDIA GPU/driver."
		print(f"❌ {msg}")
		return _result(name, "FAILED", t, msg)
	print(f"✅ {name} succeeded in {t:.3f}s!")
	return _result(name, "PASSED", t)


def check_polygraphy_inspect(model_path):
	"""`polygraphy inspect model --show layers`: inputs, outputs and all layers/operators (no GPU needed)."""
	return _run_polygraphy("NVIDIA Polygraphy Inspect Model", ["inspect", "model", model_path, "--show", "layers"])


def check_polygraphy_capability(model_path):
	"""Static TensorRT compatibility check (no GPU needed; `polygraphy inspect capability` requires trt.Builder and a GPU)."""
	name = "Static TensorRT Op/Opset Check"
	print("\n--------------------------------------------------")
	print(f"🎯 Pre-flight check:    {name}")
	print("--------------------------------------------------")
	t0 = time.perf_counter()
	unsupported_ops = {"STFT", "InverseSTFT", "StringNormalizer", "TfIdfVectorizer"}
	complex_types = (onnx.TensorProto.COMPLEX64, onnx.TensorProto.COMPLEX128)
	model = onnx.load(model_path)
	errors = []
	used_ops = {n.op_type for n in model.graph.node}
	opset = max((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), default=0)
	if opset > 21:
		errors.append(f"ONNX opset {opset} is too new for TensorRT 10 (max 21).")
	bad_ops = sorted(used_ops & unsupported_ops)
	if bad_ops:
		errors.append(f"Operators not natively supported by TensorRT: {bad_ops}")
	for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
		if vi.type.tensor_type.elem_type in complex_types:
			errors.append(f"Complex tensor '{vi.name}' is not supported by TensorRT.")
	elapsed = time.perf_counter() - t0
	if errors:
		msg = "; ".join(errors)
		print(f"❌ {name} FAILED: {msg}")
		return {"name": name, "status": "FAILED", "compile_s": elapsed, "warmup_s": 0.0, "infer_s": 0.0, "error": msg}
	print(f"✅ {name} passed in {elapsed:.3f}s ({len(used_ops)} unique ops, opset {opset}).")
	return {"name": name, "status": "PASSED", "compile_s": elapsed, "warmup_s": 0.0, "infer_s": 0.0, "error": ""}


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_onnx_ep.py <path_to_onnx>")
        sys.exit(1)

    model_path = sys.argv[1]

    raw_targets = os.environ.get("TEST_TARGETS")
    if raw_targets:
        targets = json.loads(raw_targets)
    else:
        targets = [
            {"name": "ONNX CPU", "ep": "CPUExecutionProvider"},
            {"name": "OpenVINO CPU", "ep": "OpenVINOExecutionProvider", "options": {"device_type": "CPU"}},
            {"name": "OpenVINO GPU", "ep": "OpenVINOExecutionProvider", "options": {"device_type": "GPU"}, "skip_if_missing_hw": True},
            {"name": "NPU 3720 (Meteor/Arrow Lake)", "ep": "OpenVINOExecutionProvider", "npu_platform": "3720", "compile_only": True},
            {"name": "NPU 5010 (Panther Lake)", "ep": "OpenVINOExecutionProvider", "npu_platform": "5010", "compile_only": True},
        ]

    print("==================================================")
    print(f"🚀 Testing Model:       {model_path}")
    print(f"📋 Available Providers: {ort.get_available_providers()}")
    print(f"🔢 Test Sequence:       {[t.get('name', t.get('ep', 'Target')) for t in targets]}")
    print("==================================================")

    t0 = time.perf_counter()
    dummy_inputs = create_dummy_inputs(model_path)
    print(f"⏱️ Input tensors generated in {time.perf_counter() - t0:.3f}s")

    audio_path = os.environ.get("BENCHMARK_AUDIO")
    benchmark_input = load_benchmark_audio(audio_path) if audio_path else None
    results = []
    if any(t.get("ep") == "CUDAExecutionProvider" for t in targets):
        results.append(check_onnx_structure(model_path))
        results.append(check_polygraphy_inspect(model_path))
        results.append(check_polygraphy_capability(model_path))
    for target in targets:
        with report_target_progress(target.get("name", target.get("ep", "Target"))):
            res = run_single_target(model_path, dummy_inputs, target, benchmark_input=benchmark_input)
        if res["status"] == "SKIPPED":
            print(f"❌ Skipping is disabled: {res['name']} is reported as FAILED ({res.get('error', '')})")
            res["status"] = "FAILED"
        results.append(res)

    headers = ("Target", "Status", "Compile (s)", "Warmup (s)", "Steady (s)", "Processing speed")
    rows = [
        (r["name"], r["status"], f"{r['compile_s']:.3f}", f"{r['warmup_s']:.3f}",
         f"{r['infer_s']:.3f}", f"{r['processing_speed']:.2f}x" if "processing_speed" in r else "N/A")
        for r in results
    ]
    widths = [max(len(str(row[i])) for row in [headers, *rows]) for i in range(len(headers))]

    def format_row(row):
        return " | ".join(str(value).ljust(widths[i]) if i < 2 else str(value).rjust(widths[i])
                          for i, value in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    print("\n" + format_row(headers))
    print(separator)
    for row in rows:
        print(format_row(row))
    has_failure = any(r["status"] == "FAILED" for r in results)
    if audio_path and os.environ.get("GITHUB_STEP_SUMMARY"):
        summary = ["| Target | Status | Compile (s) | Processing speed |", "| --- | --- | ---: | ---: |"]
        for row in rows:
            name, status, compile_time, _, _, speed = row
            summary.append(f"| {name.replace('|', '/')} | {status} | {compile_time} | {speed} |")
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as summary_file:
            summary_file.write("## Real-audio ONNX benchmark\n\n" + "\n".join(summary) + "\n\n")
            summary_file.write("Processing speed is audio duration divided by processing time, excluding session initialization and warmup. "
                               "Processing includes normalization, 25% overlapping model inference and reconstruction; file loading is excluded. "
                               "CoreML compute units permit CPU fallback and do not guarantee GPU or Neural Engine placement.\n")

    sys.exit(1 if has_failure else 0)


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "--npu-aot":
        _res = _npu_aot_compile_impl(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) > 5 else None)
        print("RESULT_JSON:" + json.dumps(_res), flush=True)
        sys.exit(0)
    if len(sys.argv) == 4 and sys.argv[1] == "--plugin-ep":
        _target = json.loads(sys.argv[3])
        _res = run_single_target(sys.argv[2], create_dummy_inputs(sys.argv[2]), _target)
        print("RESULT_JSON:" + json.dumps(_res), flush=True)
        sys.exit(0)
    main()
