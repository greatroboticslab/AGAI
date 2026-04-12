"""
RF-DETR detection via subprocess.

Calls detect_images.py inside the rfdetr conda environment so the main
process (running in minigptv) never needs to import the rfdetr package.
"""

import json
import subprocess
import sys

from .config import RFDETR_CONDA_ENV, RFDETR_DETECT_SCRIPT, PROJECT_ROOT


def run_rfdetr(image_paths):
    """Run RF-DETR with per-class best thresholds on a list of images.

    Returns:
        dict keyed by image path, each value containing
        'detected_parts', 'all_detections', 'num_detections'.
        Empty dict on failure.
    """
    cmd = [
        "bash", "-c",
        'eval "$(conda shell.bash hook 2>/dev/null)" && '
        f'conda activate {RFDETR_CONDA_ENV} && '
        f'python "{RFDETR_DETECT_SCRIPT}" --images ' +
        " ".join(f'"{p}"' for p in image_paths)
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0:
        print(f"[RF-DETR] stderr:\n{result.stderr}", file=sys.stderr)
        return {}

    stdout = result.stdout
    json_start = stdout.find("{")
    if json_start == -1:
        print(f"[RF-DETR] no JSON in stdout:\n{stdout}", file=sys.stderr)
        return {}

    return json.loads(stdout[json_start:])
