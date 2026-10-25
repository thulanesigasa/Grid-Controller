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
