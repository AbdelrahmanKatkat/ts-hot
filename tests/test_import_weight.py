"""The CPU inference image installs only the core dependencies: importing the package must not need torch."""

import subprocess
import sys

HEAVY = ("torch", "torchvision", "onnx", "datasets", "pandas", "pyarrow", "onnxruntime")


def test_import_ts_hot_loads_no_training_or_dataset_framework():
    code = (
        "import sys, ts_hot;"
        f"bad = [m for m in {HEAVY!r} if m in sys.modules];"
        "assert not bad, bad;"
        "assert callable(ts_hot.predict_session) and callable(ts_hot.decode_polygons)"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-800:]


def test_serving_and_evaluation_modules_import_without_torch():
    code = (
        "import sys, ts_hot.serve, ts_hot.evaluate, ts_hot.export, ts_hot.dataset, ts_hot.tiles, ts_hot.decode, ts_hot.georef, ts_hot.postprocess;"
        f"bad = [m for m in {HEAVY!r} if m in sys.modules];"
        "assert not bad, bad"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-800:]
