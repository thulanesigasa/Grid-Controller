# Face-model input audit

The user's two saved face crops were evaluated against the SHA256-verified
UniFace 4.1.0 MobileGaze ResNet34 model. All inference used local OpenCV DNN;
no face detector, ONNX Runtime session or photo upload was used. The app's
preprocessing and gaze-angle decoding were reproduced, and outputs were matched
by the names `yaw` and `pitch`, rather than assumed output order.

Both baseline predictions matched the supplied diagnostics within 0.001 degree
in pitch and yaw. This confirms reproduction of those two inputs and outputs,
not ground-truth accuracy or equivalence across all inputs/backends.

| Perturbation | P1 change from baseline | P2 change from baseline |
|---|---:|---:|
| Trim 5% from all four sides | 5.80° | 2.17° |
| Trim 5% from left edge | 18.40° | 1.57° |
| Trim 5% from right edge | 5.66° | 0.21° |
| Trim 5% from top edge | 3.89° | 1.28° |
| Trim 5% from bottom edge | 0.82° | 3.72° |
| Square crop with reflected padding | 6.02° | 0.77° |
| Rotate input −5° | 0.49° | 3.36° |
| Rotate input +5° | 4.37° | 5.38° |
| Level the detected eye line | 4.02° | 3.98° |
| Mirror input horizontally | 13.27° | 3.61° |

Changes are angles between projected 2D directions, after undoing any input
rotation or mirror. They are not pitch/yaw error against a known gaze target.
Rotations and square padding use explicitly synthetic reflected border pixels.
Subcrops use actual pixels. The saved inputs have no surrounding context, so
expanding the real face crop or changing the detector was not tested. The
available full photograph is 2048×1547 whereas the diagnostics came from a
4416×3336 image; mixing their coordinates would invalidate this comparison.

## Findings

- Player 2's reported direction persists across these modest input changes.
  A simple crop or roll correction is not substantiated as a solution.
- Player 1 has greater sensitivity to one crop edge. This is further reason to
  measure cropping behavior across multiple labelled photographs.
- Agreement across perturbations does not prove accuracy. Averaging the outputs
  would create another estimate without establishing the correct direction.
- No new correction was applied to the app. The existing disagreement gate still
  only applies to separate-eye predictions, not the whole-face model tested here.

The next stage is an evaluation set with known targets and match-level holdouts.
For a scene model, use separately prepared ball-removed inputs; keep original
ball labels for scoring and watch for removal artifacts. Do not use the guessed
ball position in this photograph as a verified label.

## Reproduce

Install the existing AI dependencies, save the two face PNGs and diagnostics JSON
from `diagnose_gaze.py`, and obtain the official ResNet34 weights referenced by
UniFace 4.1.0:
https://github.com/yakhyo/gaze-estimation/releases/download/weights/resnet34_gaze.onnx

```powershell
.venv\Scripts\python audit_gaze_inputs.py diagnostics.json --crops . --model resnet34_gaze.onnx --output input-audit.json
```

The audit refuses weights with a different SHA256 and refuses to overwrite an
existing output. It does not download models automatically. Full numerical runs
are in `input-audit-results.json`; no photographs are committed to the repo.
