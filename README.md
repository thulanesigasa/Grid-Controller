# Grid Controller — Gaze Lab

A local Python desktop prototype for exploring where two manually estimated gaze rays intersect in a photograph. All coordinates remain in original-image pixels through zoom and pan. Optional local AI detects faces and eye landmarks, then estimates gaze angles to suggest editable lines. An intersection is not proof of a missing ball's location.

## Run on Windows

Install Python 3.10+ with Tcl/Tk support (included in the standard python.org installer), then:

```powershell
git clone https://github.com/thulanesigasa/Grid-Controller.git
cd Grid-Controller
py -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python app.py
```

On Linux/macOS use `python3 -m venv .venv` and `.venv/bin/python` instead. Linux may require the distribution's `python3-tk` package.

## Experiment

1. Open the original photograph. Its actual dimensions appear at the top. The image is not automatically rotated using EXIF, so coordinates follow the decoded pixel array.
2. Select Player 1, click the eye midpoint, then a point along the estimated gaze direction. Repeat for Player 2.
3. Drag either handle to refine it. Clicking elsewhere after completing a pair starts a new pair for that player.
4. Scroll to zoom around the pointer, right-drag or choose Pan to move, and Fit to reset the view.
5. Read the white intersection marker. Magenta dots sample independent angle changes of −θ, 0, and +θ for each ray. The reported range is a sensitivity experiment, not a statistical confidence interval. Invalid backward/parallel intersections are omitted; very shallow crossings can be unstable.
6. Use Inspect to select a recognizable point. Enter the game coordinates of the same point and Compare. Repeat near all corners and the centre, and after resizing. This checks agreement; it does not automatically calibrate offsets or certify the game's mapping.
7. Export JSON saves image dimensions, manual rays, estimated intersection and sensitivity setting. It does not upload your photo or submit entries.

The photograph investigated during development was 4416 × 3336 pixels. The app detects dimensions per file rather than assuming that all photographs share this size. Download the original from:
https://api-dream-drive.injozitech.com/api/images/1791152129901DD_Comp_Image_26.jpg

## Validation

```sh
python -m unittest discover -s tests -v
```

Tests cover known intersections, invalid rays and coordinate transforms under zoom and pan. Real-world gaze accuracy and exact agreement with the external game still require comparison testing. Test estimation on photographs with known ball positions before interpreting results as useful predictions.

## Design

Tkinter supplies the desktop UI; Pillow renders the visible image region. Geometry uses Python math. The optional AI path uses UniFace, OpenCV, NumPy and ONNX Runtime on CPU.


## Automatic eyes and gaze suggestions

From your existing checkout, close the app, then run in PowerShell:

```powershell
git pull --ff-only origin main
.venv\Scripts\python -m pip install -r requirements-ai.txt
.venv\Scripts\python app.py
```

Open the original photo and click **Analyse image (AI)**. The first run downloads RetinaFace and MobileGaze ResNet34 model weights into UniFace's local cache (roughly 100 MB total); later runs reuse them. Internet is needed for those downloads, but photographs are processed locally and are not uploaded. Python 3.11 or 3.12 is recommended for the optional AI dependencies.

The model runs in a background thread. The app detects face boxes and two eye landmarks, uses their midpoint as the ray origin, and estimates pitch/yaw from each full-resolution face crop. It selects the two largest usable faces and labels them left-to-right. With more people, verify that these are your intended subjects. One detected face only produces one suggestion. Near-camera-facing predictions have no stable projected direction and leave that ray empty. No detections or a model/download error preserves existing lines. Changing the image or lines during analysis discards stale results.

Review the boxes and directions, then drag handles to correct them. The tool computes a forward intersection only when the suggested rays permit one; it does not force agreement or invent a target when they diverge. Automatic analysis sets angle sensitivity to ±10° as an illustrative perturbation, **not a calibrated uncertainty interval**. Manual edits are identified in the display and export; exported raw model suggestions remain available for comparison.

### What automatic analysis does not establish

Eye landmarks are estimated eye positions, not precise pupil tracking. A predicted gaze angle is projected into 2D using the upstream visualization convention. There is no calibrated camera geometry, eye-depth measurement, ball detector or learned missing-ball reconstruction. Two projected rays can miss the true target even when facial predictions look plausible. A single image cannot establish a guaranteed ball coordinate. The original image contains no ground-truth ball for evaluating accuracy.

### Model sources

- [UniFace](https://github.com/yakhyo/uniface) — optional pinned dependency, RetinaFace landmarks and MobileGaze inference; MIT code license.
- [MobileGaze](https://github.com/yakhyo/gaze-estimation) — gaze models trained on Gaze360. Follow upstream model and dataset terms for your intended use; weights are downloaded by the dependency, not bundled here.
- [Upstream gaze module documentation](https://yakhyo.github.io/uniface/modules/gaze/).

Unit tests exercise the projection sign convention, unstable camera-facing inputs, non-finite values and the manual geometry. They do not measure gaze accuracy.

### Development smoke test

The full CPU pipeline was run on the original 4416 × 3336 photograph on 25 October 2026. It detected both players, produced eye landmarks and gaze rays, and returned a forward intersection around `(2068, 1152)`. This is an unvalidated model output, not a known ball position. There is no ground-truth ball location in the supplied photograph. Desktop interaction still needs a Windows check; model inference and geometry were tested separately in a headless environment.

## Refined eye placement

AI analysis now follows the coarse face detector with the UniFace **FaceMesh V2 478-point iris model**, using each face's original-resolution pixels. It estimates iris centres (indices 468 and 473) and updates the ray's origin to their midpoint. This additional model is about 4.6 MB and downloads once. The gaze-angle model is unchanged: more appropriate ray origins do not by themselves prove better gaze angles or a more accurate ball position.

White eye rings indicate accepted refinement (or manual correction); yellow rings indicate a coarse fallback. Refinement checks face presence, finite positions, bounds, movement from the coarse landmarks and eye separation. These are sanity checks, not calibrated eye-accuracy scores. Download or inference failure retains coarse eye estimates and marks them as fallback. Restart the app to retry a failed model download.

To correct an individual eye, select **Edit eyes**, zoom in, and drag its small ring onto the visible iris centre. The labels `eye 1` and `eye 2` follow image-left/image-right order, avoiding anatomical left/right ambiguity. The gaze origin and endpoint move together, preserving the current line angle. Raw coarse landmarks, refined/corrected coordinates and per-eye status are included in JSON export.

On the supplied photo, the iris refinement was compared visually against the previous coarse markers on full-resolution face crops and the complete pipeline accepted all four iris estimates. This is a qualitative check on one image, not a measured accuracy benchmark. Blur, occlusion and extreme gaze can still cause errors; increasing zoom does not recover missing image detail.

## Separate gaze direction for each eye (experimental)

Close the app and install the optional backend in PowerShell (Python 3.11/3.12):

```powershell
git pull --ff-only origin main
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements-per-eye.txt
.venv\Scripts\python app.py
```

This installs a pinned version of [ptgaze](https://github.com/hysts/pytorch_mpiigaze_demo), with its MPIIGaze ResNet preact model. Git must be installed. PyTorch and the additional dependencies are a substantial download; model weights download on first use. The default face-level mode continues to work without this extra installation.

1. Open a photograph, check **Per-eye AI (experimental)**, then click **Analyse image (AI)**. The checkbox selects the next analysis mode; changing it alone does not replace existing results.
2. Each usable eye receives its own solid **1-screen-pixel** ray, estimated from its separately normalized grayscale eye crop and head pose. Cyan is Player 1, orange is Player 2. These are not copies of the face-level direction.
3. Toggle **Boundary lines** for two thin dashed lines at ± the angle setting around each eye's central ray. Boundaries start at the iris, not the eyelid corners: eye width does not define gaze uncertainty. The angle is an adjustable image-plane tolerance, not a measured confidence interval.
4. A green region shows where all four forward cones overlap within the image. Missing directions, disagreement or zero tolerance produce no shared area; the tool does not force an intersection. One player alone cannot produce a four-eye region.
5. Select **Eye directions** to drag individual endpoint circles. **Edit eyes** moves iris origins while preserving their directions. To return to face-level analysis, uncheck Per-eye AI and analyse again.
6. Export JSON schema 2 includes per-eye directions, model/camera assumptions and the shared-region polygon in original-image coordinates. The face-level estimated intersection is null in per-eye mode to avoid presenting it as the four-eye result.

The photograph's camera calibration is unknown. This backend assumes focal length equal to image width, an image-centred principal point and zero lens distortion; it uses a generic face model. Eye predictions share head pose and a trained model, so their errors are correlated. Four lines do not mean four independent measurements or proven higher accuracy. Sports photographs may differ substantially from training images. No ball detector or ground-truth validation has been added.

Model code and model details: [ptgaze](https://github.com/hysts/pytorch_mpiigaze_demo), [MPIIGaze weights](https://huggingface.co/hysts/ptgaze-mpiigaze-resnet-preact). Review upstream model/data terms for your use. Missing mesh, unstable projection and inference errors leave directions unavailable and are reported in the status/export instead of being replaced with duplicate face rays.

### Per-eye validation status

Geometry/projection and per-eye editing tests pass, including a known shared target, diverging cones, missing rays, image bounds and preserving direction when moving an iris. End-to-end per-eye inference and Windows desktop interaction have **not yet been validated**. The development environment lacked MediaPipe's native `libGLESv2.so.2`; Linux may need its distribution's GLES package (for example `libgles2` on Debian/Ubuntu). Automatic approval review also stopped an unexpected telemetry request during the full-photo test. ONNX Runtime telemetry is now explicitly disabled before creating inference sessions; the photo test was not retried. These implementation checks do not measure gaze or ball-location accuracy.

## Validation mode: measure against a visible ball

This mode collects evaluation records; it does **not** train or update the AI.
Use photographs where the ball is visible and the selected players are watching it.
A guessed position in a missing-ball competition image is not ground truth.

1. Open the original photo and run AI analysis (or draw your manual estimate).
2. Click **Freeze prediction** before labelling the ball. This captures the source,
   rays, model output, angle setting and point/overlap region independently of later edits.
   While AI is busy, freezing is disabled by a status message.
3. The app switches to **Mark known ball**. Click the visible ball centre. A pink
   marker shows the label; zoom and pan preserve original-image coordinates.
4. Check **Ball is visible / centre verified** after checking your label. Optionally
   enter a **Match/group** identifier so related photographs can stay together in
   future training/test splits. Verification is your annotation, not independent certification.
5. Click **Save validation JSON**. Save each trial under its own filename, e.g.
   `match01-frame03-ai.json` and `match01-frame03-manual.json`.

The record includes the decoded RGB image hash, image dimensions, UTC timestamps,
source/model information, a frozen prediction, the user-labelled ball centre and
error in pixels and as a percentage of image diagonal. Photos are not embedded or
uploaded. Keep originals alongside records for future evaluation/training.

A missing forward intersection has a null point error, not zero. Per-eye mode has
no point prediction: its metric is whether the frozen overlap region contains the
known ball, at the recorded tolerance. Empty regions remain misses. Region coverage
alone does not establish accuracy because increasing tolerance enlarges regions.

Changing lines, rerunning AI or changing tolerance does not change the frozen trial.
**Freeze prediction** starts a new trial and clears its ball label; opening another
image or **Reset lines** clears validation. Save before replacing a trial. The regular
**Export JSON** remains an export of current analysis, distinct from validation JSON.

For fair evaluation, record failures as well as successes and use identical ball
labels for AI/manual comparisons. Do not tune lines after seeing the answer and
present that as an independent manual prediction. The face/per-eye AI uses face crops, but
verify that a visible ball does not appear inside those crops; this mode does not
mask it automatically. No calibrated reliability score is added. Training and held-out
accuracy measurements are still future work; do not train and test on the same match
or near-duplicate frames.


## Scene gaze: full photograph and head boxes

**Analyse scene (AI)** uses pretrained [Gaze-LLE](https://github.com/fkryan/gazelle)
with a DINOv2 backbone to estimate where each selected person may be looking.
It considers the full photograph with a head-box input. It does not explicitly
predict actions, detect a missing ball, train on your corrections, or guarantee
that two players are watching the same target. Existing face/per-eye modes remain
available through **Analyse image**.

In PowerShell, from the project folder:

```powershell
# Skip this first command if CPU torch and torchvision are already installed.
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements-scene.txt
.venv\Scripts\python app.py
```

The small Gaze-LLE inference modules are bundled with their upstream MIT license
and revision in `gazelle/`; this avoids upstream packaging omissions.
The first scene analysis downloads the pinned DINOv2 code/backbone and Gaze-LLE
weights, then caches them. Allow several minutes and several hundred MB of disk
space. Inference runs on CPU locally; later runs reuse the loaded model.

1. Open your original photograph.
2. Select **Head 1**, then click two opposite corners around the first player's
   entire head, including hair. Repeat with **Head 2** for the other player.
   One head is also supported. Boxes use original-image coordinates at any zoom.
3. Click **Analyse scene (AI)**. Alternatively, **Auto head boxes** clears manual
   boxes; the next analysis proposes the two largest detected faces, expanded
   to approximate heads and ordered left to right. Inspect these proposals:
   they can miss hair or select the wrong person. Manual boxes are preferable
   when proposals are inaccurate.
4. Use **Heatmap: P1 / P2 / Both / Hidden** to compare the suggestions. Cyan is
   P1, orange P2, and green marks shared high relative activation. No overlap
   means no shared region is reported; the app never forces an intersection.
5. Correct a head box and analyse again to recompute. Box changes do not alter
   the current heatmap until analysis succeeds. **Reset lines** clears analysis.

Heatmaps are 64 by 64 cells resized onto the image, not pixel-accurate directions.
Colours show each map's relative intensity, not probability of finding a ball.
Green requires both normalized maps to be at least 0.6 of their own range.
An in-frame score below 0.5 or a flat map suppresses that person's displayed
location. These are display rules, not calibrated confidence or accuracy claims.
JSON exports include raw maps, head boxes, model revisions, scores and individual
cell-centre peaks; the combined ball coordinate remains null.

Validation records report each available peak's error and shared-region coverage
against a known target. **Scene inference sees the whole image, including any
visible ball.** Success on a visible-ball photograph does not establish accuracy
on missing-ball photographs. A controlled missing-ball evaluation needs separately
prepared inputs with the ball removed and labels retained from the original,
plus held-out matches; removal artifacts can themselves leak the answer.
Do not use your guessed coordinates as verified labels.

Scene tests cover coordinate mapping, invalid outputs, absent/disagreeing maps,
layer separation, frozen validation and stale-result rejection. These checks do
not measure prediction accuracy. Windows desktop interaction still needs testing
on your computer.


Development verification: all 31 automated tests passed. The pinned backbone and
checkpoint loaded successfully on Linux CPU (Python 3.12, torch 2.14.1+cpu,
torchvision 0.29.1+cpu), and scene-only inference returned two 64x64 maps on the
2048x1547 example photograph with manually supplied head boxes. Both in-frame
scores fell below the 0.5 display gate, so no region was displayed. This is an
integration check, not evidence of improved accuracy. Automatic head proposals
were not rerun on that photograph; Windows UI interaction remains unverified.
Missing xFormers warnings are expected for the CPU path; it is not required.
