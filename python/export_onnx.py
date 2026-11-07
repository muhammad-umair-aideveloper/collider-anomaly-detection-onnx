"""Export trained autoencoder model to ONNX format with batch size 1.

This script loads the trained PyTorch autoencoder checkpoint, reconstructs the
selected architecture variant (including internal feature selection and
standardization buffers), wraps it in an inference module that computes
per-event reconstruction error, and exports it to ONNX format using opset 17.

PyTorch Version Note:
In PyTorch 2.15.0+cpu, the default ONNX exporter delegates to the new Dynamo/torch.export
pipeline which requires onnxscript and defaults to opset 18+ (attempting a lossy conversion
for lower opsets). Passing `dynamo=False` explicitly selects the stable, deterministic
TorchScript-based exporter, ensuring exact adherence to opset 17 with static shapes [1, 28].
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import onnx
import torch
import torch.nn as nn

# Ensure python/ directory is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

import config
from model import PRE_DECLARED_VARIANTS, Autoencoder


class AnomalyDetectorExportWrapper(nn.Module):
    """Thin wrapper module exposing single-event reconstruction error for ONNX export.

    The wrapper consumes raw 28-dimensional detector events [1, 28] and produces
    the per-event reconstruction error as a 1D tensor [1].
    """

    def __init__(self, model: Autoencoder) -> None:
        """Initialize export wrapper with trained autoencoder.

        Args:
            model: Trained Autoencoder instance.
        """
        super().__init__()
        self.model = model

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward inference pass producing per-event reconstruction error.

        Args:
            features: Raw detector feature tensor of shape [1, 28].

        Returns:
            torch.Tensor: Reconstruction error scalar/tensor of shape [1].
        """
        return self.model.reconstruction_error(features)


def parse_args() -> argparse.Namespace:
    """Parse CLI options for ONNX export script.

    Returns:
        argparse.Namespace: Parsed CLI options.
    """
    parser = argparse.ArgumentParser(
        description="Export trained PyTorch autoencoder to ONNX format."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=config.PYTORCH_MODEL_PATH,
        help=f"Path to input PyTorch checkpoint (default: {config.PYTORCH_MODEL_PATH}).",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=config.ONNX_MODEL_PATH,
        help=f"Destination path for exported ONNX model (default: {config.ONNX_MODEL_PATH}).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=config.ONNX_OPSET,
        help=f"ONNX operator set version (default: {config.ONNX_OPSET}).",
    )
    return parser.parse_args()


def load_model_for_export(model_path: Path) -> Tuple[Autoencoder, str]:
    """Load autoencoder checkpoint and instantiate correct architecture variant.

    Args:
        model_path: Path to autoencoder.pt checkpoint.

    Returns:
        Tuple of (model, variant_name).
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"PyTorch model checkpoint not found at '{model_path}'. Run training first."
        )

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("config", {})
    variant = cfg.get("variant", "V0")

    if variant in PRE_DECLARED_VARIANTS:
        model = Autoencoder.from_variant(variant)
    elif "layers" in cfg:
        sel_features = cfg.get("selected_indices")
        model = Autoencoder(layers=cfg["layers"], selected_features=sel_features)
    else:
        model = Autoencoder()

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, variant


def export_to_onnx(
    model: Autoencoder,
    out_path: Path,
    opset: int = config.ONNX_OPSET,
) -> Path:
    """Export wrapped autoencoder to ONNX model file with static batch size 1.

    Args:
        model: Trained Autoencoder.
        out_path: Target ONNX file path.
        opset: Target ONNX opset version.

    Returns:
        Path: Output path of exported ONNX model.
    """
    wrapper = AnomalyDetectorExportWrapper(model)
    wrapper.eval()

    # Fixed batch size 1 dummy input with all 28 raw features
    dummy_input = torch.zeros(
        (config.ONNX_BATCH_SIZE, config.NUM_FEATURES),
        dtype=torch.float32,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Note on PyTorch 2.15 exporter behavior:
    # PyTorch 2.15 defaults to Dynamo-based export when onnxscript is present, which
    # enforces opset 18+ and fails conversion to opset 17. Setting dynamo=False invokes
    # the robust TorchScript-based exporter, natively producing a valid Opset 17 graph.
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(out_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=[config.ONNX_INPUT_NAME],
        output_names=[config.ONNX_OUTPUT_NAME],
        dynamic_axes=None,  # Fixed batch size 1
        dynamo=False,
    )

    return out_path


def inspect_and_check_onnx_model(onnx_path: Path) -> None:
    """Validate exported ONNX model using onnx.checker and print graph metadata.

    Args:
        onnx_path: Path to ONNX model file.
    """
    if not onnx_path.exists():
        raise FileNotFoundError(f"Exported ONNX file not found at '{onnx_path}'.")

    # 1. Structural integrity check
    model_proto = onnx.load(str(onnx_path))
    onnx.checker.check_model(model_proto)

    # 2. Extract graph attributes
    graph = model_proto.graph
    opset_versions = [
        f"{opset.domain or 'ai.onnx'}: {opset.version}"
        for opset in model_proto.opset_import
    ]

    inputs_info: List[str] = []
    for inp in graph.input:
        shape_dims = [
            dim.dim_value if dim.dim_value > 0 else dim.dim_param
            for dim in inp.type.tensor_type.shape.dim
        ]
        inputs_info.append(f"{inp.name} (shape: {shape_dims})")

    outputs_info: List[str] = []
    for out in graph.output:
        shape_dims = [
            dim.dim_value if dim.dim_value > 0 else dim.dim_param
            for dim in out.type.tensor_type.shape.dim
        ]
        outputs_info.append(f"{out.name} (shape: {shape_dims})")

    file_size_bytes = onnx_path.stat().st_size
    file_size_kb = file_size_bytes / 1024.0

    print("=" * 60)
    print("ONNX EXPORT VERIFICATION & GRAPH INSPECTION")
    print("=" * 60)
    print(f"Model Path:         {onnx_path}")
    print(f"Model File Size:    {file_size_bytes:,} bytes ({file_size_kb:.2f} KB)")
    print(f"Opset Version:      {', '.join(opset_versions)}")
    print(f"Graph Input(s):     {', '.join(inputs_info)}")
    print(f"Graph Output(s):    {', '.join(outputs_info)}")
    print(f"onnx.checker:       PASSED (Valid ONNX graph)")
    print("=" * 60)


def main() -> None:
    """Execute complete ONNX model export workflow."""
    args = parse_args()

    print("=" * 60)
    print("COLLIDER ANOMALY DETECTION: ONNX EXPORT")
    print("=" * 60)
    print(f"PyTorch Checkpoint: {args.model_path}")
    print(f"Target ONNX Path:   {args.out_path}")
    print(f"Requested Opset:    {args.opset}")
    print(f"Fixed Batch Size:   {config.ONNX_BATCH_SIZE}")
    print(f"Raw Input Features: {config.NUM_FEATURES}")

    # 1. Load model
    model, variant = load_model_for_export(args.model_path)
    print(f"Loaded Variant:     {variant} ({PRE_DECLARED_VARIANTS[variant]['description']})")

    # 2. Export model
    export_to_onnx(model, args.out_path, opset=args.opset)
    print(f"Export completed:   {args.out_path}")

    # 3. Inspect and check
    inspect_and_check_onnx_model(args.out_path)


if __name__ == "__main__":
    main()
