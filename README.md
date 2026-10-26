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
