#!/usr/bin/env python

import torch
from torch.nn import functional as F
import argparse
from pathlib import Path
from demucs.pretrained import get_model
from demucs.htdemucs import HTDemucs
from demucs import spec as demucs_spec
from demucs.stft import STFT_Process
from demucs.istft import ISTFT_Process

DEMUCS_MODEL = "htdemucs"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Demucs PyTorch models to ONNX')
    parser.add_argument("dest_dir", type=str, help="destination path for the converted model")
    parser.add_argument(
        "--dft",
        choices=["factorized", "dense"],
        default="factorized",
        help="STFT/ISTFT implementation: factorized (two small 64x64 matrix stages, ~16x fewer MACs, "
        "small constants) or dense (one 4096x4098 matrix).",
    )

    args = parser.parse_args()

    if args.dft == "dense":
        demucs_spec.demucs_stft = STFT_Process(factorized=False)
        demucs_spec.demucs_istft = ISTFT_Process(factorized=False)

    dir_out = Path(args.dest_dir)
    dir_out.mkdir(parents=True, exist_ok=True)

    # Load the appropriate model
    model = get_model(DEMUCS_MODEL)
    model_name = DEMUCS_MODEL

    # Check if model is an instance of BagOfModels
    if isinstance(model, HTDemucs):
        core_model = model
    elif hasattr(model, 'models') and isinstance(model.models[0], HTDemucs):
        core_model = model.models[0]  # Select the first model in BagOfModels
    else:
        raise TypeError("Unsupported model type")

    # Set the model to onnx export mode
    core_model.onnx_exportable = True

    # Prepare a dummy input tensor
    dummy_waveform = torch.randn(1, 2, 343980)
    training_length = int(core_model.segment * core_model.samplerate)
    dummy_waveform = F.pad(dummy_waveform, (0, training_length - dummy_waveform.shape[-1]))
    dummy_input = (dummy_waveform)

    # Warm-up run: fills the positional embedding caches so that the export
    # sees them as constants instead of tracing sin/cos/scatter ops.
    core_model.eval()
    with torch.no_grad():
        core_model(dummy_input)

    # Define output file name
    onnx_file_path = dir_out / f"{model_name}.onnx"
    print(f"Converting {model_name} to ONNX format...")

    # Export the core model to ONNX
    try:
        torch.onnx.export(
            core_model,
            dummy_input,
            str(onnx_file_path),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
        )

        # The exporter stores the weights in a separate "<name>.onnx.data" file.        # Embed them so that the single .onnx file is complete.
        import onnx
        from onnx import numpy_helper
        onnx_model = onnx.load(str(onnx_file_path))

        # The exporter adds a `constant_value` input to non-constant Pad nodes.
        # It is ignored by the ONNX spec in these modes, but MIGraphX (AMD) treats
        # the third input of such nodes as `axes` and fails to parse the model.
        for node in onnx_model.graph.node:
            if node.op_type != "Pad" or len(node.input) != 3:
                continue
            mode = next((a.s for a in node.attribute if a.name == "mode"), b"constant")
            if mode != b"constant":
                del node.input[2]

        # The exporter emits attributes that carry no information here but that
        # DirectML rejects: Reshape allowzero=1 (the shapes contain no zeros) and
        # ConvTranspose output_padding=[0].
        init = {i.name: i for i in onnx_model.graph.initializer}
        for node in onnx_model.graph.node:
            for attr in list(node.attribute):
                if node.op_type == "ConvTranspose" and attr.name == "output_padding" and not any(attr.ints):
                    node.attribute.remove(attr)
                elif node.op_type == "Reshape" and attr.name == "allowzero" and node.input[1] in init:
                    if not (numpy_helper.to_array(init[node.input[1]]) == 0).any():
                        node.attribute.remove(attr)

        onnx.save_model(onnx_model, str(onnx_file_path), save_as_external_data=False)
        external_data_path = Path(f"{onnx_file_path}.data")
        if external_data_path.exists():
            external_data_path.unlink()

        print(f"Model successfully converted to ONNX format at {onnx_file_path}")
    except Exception as e:
        print("Error during ONNX export:", e)
