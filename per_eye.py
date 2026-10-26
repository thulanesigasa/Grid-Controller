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
        face = Face(np.asarray(box).reshape(2, 2), np.asarray(landmarks))
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
