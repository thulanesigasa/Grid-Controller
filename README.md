# Grid Controller — Gaze Lab

A local Python desktop prototype for exploring where two manually estimated gaze rays intersect in a photograph. All coordinates remain in original-image pixels through zoom and pan. No automatic gaze model is included. An intersection is not proof of a missing ball's location.

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

Tkinter supplies the desktop UI; Pillow decodes and renders the visible image region. Geometry uses Python's standard math library. OpenCV/NumPy are not necessary for this manual prototype; a future automatic gaze-estimation experiment can add them after the manual baseline is evaluated.
