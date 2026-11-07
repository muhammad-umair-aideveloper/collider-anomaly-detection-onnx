"""Unit tests for ONNX model structure, I/O signature, and inference parity."""

import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

import config
from export_onnx import load_model_for_export


# Skip module tests if ONNX model has not yet been exported
pytestmark = pytest.mark.skipif(
    not config.ONNX_MODEL_PATH.exists(),
    reason=f"ONNX model not found at '{config.ONNX_MODEL_PATH}'. Run python/export_onnx.py first.",
)


def test_onnx_model_checker() -> None:
    """Verify that the exported ONNX model passes onnx.checker structural validation."""
    model_proto = onnx.load(str(config.ONNX_MODEL_PATH))
    onnx.checker.check_model(model_proto)


def test_onnx_graph_io_signatures() -> None:
    """Verify input name/shape [1, 28] and output name/shape [1] in ONNX graph."""
    model_proto = onnx.load(str(config.ONNX_MODEL_PATH))
    graph = model_proto.graph

    assert len(graph.input) == 1, f"Expected 1 graph input, got {len(graph.input)}"
    assert len(graph.output) == 1, f"Expected 1 graph output, got {len(graph.output)}"

    inp = graph.input[0]
    assert inp.name == config.ONNX_INPUT_NAME, f"Expected input name '{config.ONNX_INPUT_NAME}', got '{inp.name}'"
    inp_shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    assert inp_shape == [config.ONNX_BATCH_SIZE, config.NUM_FEATURES], (
        f"Expected input shape [{config.ONNX_BATCH_SIZE}, {config.NUM_FEATURES}], got {inp_shape}"
    )

    out = graph.output[0]
    assert out.name == config.ONNX_OUTPUT_NAME, f"Expected output name '{config.ONNX_OUTPUT_NAME}', got '{out.name}'"
    out_shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
    assert out_shape in ([1], [config.ONNX_BATCH_SIZE]), f"Expected output shape [1], got {out_shape}"

    # Verify Opset version
    opset_versions = {opset.version for opset in model_proto.opset_import if not opset.domain}
    assert config.ONNX_OPSET in opset_versions, (
        f"Expected opset {config.ONNX_OPSET} in model opset imports, got {opset_versions}"
    )


def test_onnx_vs_pytorch_parity_and_non_negative() -> None:
    """Verify ONNX Runtime output matches PyTorch within 1e-5 on 20 sample events and is non-negative."""
    # 1. Load PyTorch model
    pytorch_model, _ = load_model_for_export(config.PYTORCH_MODEL_PATH)
    pytorch_model.eval()

    # 2. Load ORT session
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(config.ONNX_MODEL_PATH),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    # 3. Load 20 events from sample_events.csv
    data = np.loadtxt(config.SAMPLE_EVENTS_PATH, delimiter=",", max_rows=20, dtype=np.float32)
    features = data[:, 1:]

    # 4. Compare inference on each event
    torch.set_num_threads(1)
    with torch.no_grad():
        tensor = torch.from_numpy(features).float()
        pytorch_errors = pytorch_model.reconstruction_error(tensor).cpu().numpy()

    ort_errors = np.zeros(len(features), dtype=np.float32)
    input_name = config.ONNX_INPUT_NAME
    for i in range(len(features)):
        event_in = features[i : i + 1]  # Shape: [1, 28]
        res = session.run(None, {input_name: event_in})
        ort_errors[i] = float(res[0][0])

    # Assert non-negativity
    assert np.all(ort_errors >= 0.0), f"Found negative error values in ORT outputs: {ort_errors}"
    assert np.all(pytorch_errors >= 0.0), f"Found negative error values in PyTorch outputs: {pytorch_errors}"

    # Assert parity within 1e-5
    abs_diffs = np.abs(pytorch_errors - ort_errors)
    max_diff = float(np.max(abs_diffs))
    assert max_diff < config.VERIFY_TOLERANCE, (
        f"Max abs diff {max_diff:.6e} exceeds tolerance {config.VERIFY_TOLERANCE:.1e}"
    )
