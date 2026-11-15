import time
import openvino as ov

core = ov.Core()
core.set_property({"CACHE_DIR": "./npu_cache"})

model = core.read_model("htdemucs.onnx")

print("Starte NPU-Kompilierung mit TRACE-Logs...")
start = time.time()

# Teste zunächst mit LOG_TRACE, um den exakten Pass zu sehen
compiled_model = core.compile_model(
    model, 
    "NPU", 
    {
        "LOG_LEVEL": "LOG_TRACE",
        "NPU_COMPILER_TYPE": "DRIVER"
    }
)

print(f"Fertig kompiliert in {time.time() - start:.2f} Sekunden!")
