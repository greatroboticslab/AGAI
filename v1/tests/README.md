# `v1/tests/`: unit tests of v1 (October 2025)

Part of v1, the original system (2025 to April 2026). A small `unittest` suite, added on 21 October 2025 (the
checkpoint name and a `torch.load` argument updated on 21 January 2026), that checks
[`../resnet_classifier.py`](../resnet_classifier.py), the interface of [`../demo_v5.py`](../demo_v5.py), and whether
the MiniGPT-v2 config, weights and data are present on the lab server. The 2026 revival does not use it.

## Files

| File | What it is |
|---|---|
| [`run_tests.py`](run_tests.py) | runner: discovers `test_*.py` in this folder, prints a report (Python version, GPU, missing lab files, counts of tests, failures, errors and skips) and exits with 1 on any failure or error |
| [`test_resnet_classifier.py`](test_resnet_classifier.py) | loading the lab's ResNet-50 checkpoint, `diagnose_or_none()` on an example image and on bad input, the class list and thresholds of `resnet_classifier.py` |
| [`test_model_loading.py`](test_model_loading.py) | the MiniGPT-v2 eval config, stage-2 checkpoint, LLaMA weights, instruction records and the seven training folders exist on the lab server; the model registry; a mocked `torch.load` |
| [`test_demo_interface.py`](test_demo_interface.py) | `demo_v5.parse_args()` and the parameters of `demo_v5.process_chat_with_image()`; two tests (image validation, temperature range) that do not call the application |
| [`test_utils.py`](test_utils.py) | helpers: test images and annotation files, a mock model, processor and chat state, validators for a diagnosis, a report and a config, a GPU check, and the lab-server data paths used by the runner's file check |
| [`__init__.py`](__init__.py) | marks the folder as a package (one comment line) |

## Running

```bash
cd v1                      # demo_v5.py, imported by test_demo_interface.py, looks for eval_configs/ here
python tests/run_tests.py
```

The tests import the code from the parent folder (`v1/`) and need torch, numpy and Pillow; `test_demo_interface.py`
needs everything `demo_v5.py` imports. On the cluster on 18 November 2026 the suite gave 13 tests, 6 skipped and
1 error, the same before and after the move into `v1/` ([lab notebook](../../v2/EXPERIMENT_LOG.md), entry (ad)).

## Known problems

- **Lab-server paths.** Checkpoints, configs, data and example images are named by absolute paths under
  `/data/AGAI/MiniGPT-4/`, and the example folder `/data/AGAI/MiniGPT-4/examples/` is not in the repository. Tests
  whose files are missing skip.
- **The error.** `test_demo_interface.py` imports `demo_v5`, which runs the application's start-up at import time:
  argument parsing, the config check, ResNet-50 and MiniGPT-v2 loading and the construction of the interface (the
  server itself starts only under `__main__`). When the config is not in the working directory or the weights are
  missing, the start-up calls `sys.exit(1)` and unittest reports the module as an error. The import also opens
  `app.log` in the working directory.
- **Stale expectations.** `test_classes_definition` expects the five classes that `resnet_classifier.py` defines at
  import time (drought, frost, healthy, overwatered, root_rot); the v1 system classifies seven conditions, and
  `load_resnet()` takes the class list from the checkpoint. `test_thresholds_definition` imports `_THRESH`, which
  `resnet_classifier.py` does not define (it has `_CLASS_THRESH`), so it always skips. `assert_valid_diagnosis()` in
  `test_utils.py` expects the seven labels.
