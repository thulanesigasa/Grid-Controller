"""Optional MPIIGaze eye-crop model; assumes uncalibrated camera intrinsics."""
import math
from pathlib import Path


def screen_ray(origin, projected, length):
    """Anchor a projected camera-space direction at the visible iris."""
    a, b = projected
    values = (*origin, *a, *b, length)
    if not all(math.isfinite(float(v)) for v in values):
        return None
    dx, dy = b[0]-a[0], b[1]-a[1]
    norm = math.hypot(dx, dy)
    if norm < 1.:
        return None
    return [list(origin), [float(origin[0]+length*dx/norm), float(origin[1]+length*dy/norm)]]


# Display-space heuristic, not an anatomical limit or calibrated confidence.
MAX_PAIR_ANGLE_DEGREES = 60.0


def assess_eye_pair(rays):
    """Reject unusable or strongly conflicting projected directions as a pair.

    Raw rays remain untouched. Passing only means this heuristic found no large
    disagreement; it does not establish correctness or a common fixation point.
    """
    vectors = []
    if len(rays) != 2:
        return dict(usable=False, angle_degrees=None, reason='Missing eye direction')
    for ray in rays:
        try:
            if ray is None or len(ray) != 2 or any(len(p) != 2 for p in ray):
                raise ValueError
            if not all(math.isfinite(float(v)) for p in ray for v in p):
                raise ValueError
            dx, dy = ray[1][0]-ray[0][0], ray[1][1]-ray[0][1]
            norm = math.hypot(dx, dy)
            if norm < 1e-8 or not math.isfinite(norm):
                raise ValueError
            vectors.append((dx/norm, dy/norm))
        except (ValueError, TypeError, OverflowError):
            return dict(usable=False, angle_degrees=None, reason='Missing or invalid eye direction')
    dot = sum(a*b for a, b in zip(*vectors))
    angle = math.degrees(math.acos(max(-1., min(1., dot))))
    usable = angle <= MAX_PAIR_ANGLE_DEGREES
    return dict(usable=usable, angle_degrees=angle,
                reason='No large directional conflict detected' if usable else 'Conflicting eye directions')


def prepare_landmarks(landmarks):
    """OpenCV solvePnP requires packed floating-point XY coordinates."""
    import numpy as np
    points = np.asarray(landmarks)
    if points.shape != (468, 2) or not np.isfinite(points).all():
        raise ValueError('Expected 468 finite XY face landmarks for head pose.')
    # Slicing XY from an XYZ mesh leaves gaps between rows. np.asarray alone
    # preserves those strides, which OpenCV's checkVector rejects.
    return np.ascontiguousarray(points, dtype=np.float64)


class PerEyeGaze:
    def __init__(self, width, height):
        try:
            import ptgaze
            from omegaconf import OmegaConf
            from ptgaze.config_schema import AppConfig
            from ptgaze.gaze_estimator import GazeEstimator
            from ptgaze.utils import download_mpiigaze_model, download_mediapipe_face_landmarker
        except ImportError as exc:
            raise RuntimeError('Install per-eye dependencies: python -m pip install -r requirements-per-eye.txt') from exc
        root = Path(ptgaze.__file__).parent
        cfg = OmegaConf.merge(OmegaConf.structured(AppConfig), OmegaConf.load(root/'data/configs/mpiigaze.yaml'))
        cfg.PACKAGE_ROOT = str(root)
        cfg.device = 'cpu'
        cfg.gaze_estimator.use_dummy_camera_params = True
        cfg.gaze_estimator.dummy_camera_size = [width, height]
        cfg.gaze_estimator.checkpoint = str(download_mpiigaze_model())
        download_mediapipe_face_landmarker()
        cfg.face_detector.mediapipe_static_image_mode = True
        cfg.face_detector.mediapipe_model_path = str(Path(cfg.face_detector.mediapipe_model_path).expanduser())
        try:
            self.engine = GazeEstimator(cfg)
        except OSError as exc:
            raise RuntimeError('Per-eye native dependency failed. On Linux, install the system GLES library '
                               '(for example libgles2 on Debian/Ubuntu). Details: ' + str(exc)) from exc
        self.size = (width, height)

    def analyse_face(self, bgr, box, landmarks, eyes):
        import numpy as np
        from ptgaze.common import Face
        if landmarks is None:
            return [None, None], 'No reliable face mesh for eye normalization.'
        face = Face(np.asarray(box).reshape(2, 2), prepare_landmarks(landmarks))
        self.engine.estimate_gaze(bgr, face)
        rays = []
        for eye, origin in zip((face.reye, face.leye), eyes):
            points = np.array([eye.center, eye.center + .05*eye.gaze_vector])
            if not np.isfinite(points).all() or (points[:, 2] <= 0).any():
                rays.append(None)
            else:
                projected = self.engine.camera.project_points(points)
                rays.append(screen_ray(origin, projected, min(self.size)*.15))
        return rays, 'Separate eye crops; shared head pose; assumed camera.'

    def close(self):
        self.engine.close()
