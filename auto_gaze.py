"""Optional local face/eye detection and model-based gaze suggestions."""
import math


def projected_ray(origin, pitch, yaw, length):
    """MobileGaze's screen projection (same signs as upstream draw_gaze).

    This is an orthographic direction cue, not calibrated 3D triangulation.
    """
    if not all(math.isfinite(v) for v in (*origin, pitch, yaw, length)):
        raise ValueError('Non-finite gaze prediction')
    dx = -math.sin(yaw)*math.cos(pitch)
    dy = -math.sin(pitch)
    norm = math.hypot(dx, dy)
    if norm < .08:
        return None  # Nearly camera-facing; projected direction is unstable.
    return [list(origin), [origin[0]+length*dx/norm, origin[1]+length*dy/norm]]


class AutoGaze:
    def __init__(self):
        try:
            from uniface.detection import RetinaFace
            from uniface.gaze import MobileGaze
            from uniface.constants import GazeWeights
        except ImportError as exc:
            raise RuntimeError('Install AI dependencies first: python -m pip install -r requirements-ai.txt') from exc
        # UniFace downloads and verifies its model files, then caches them locally.
        self.detector = RetinaFace(input_size=(1280, 1280), providers=['CPUExecutionProvider'])
        self.estimator = MobileGaze(model_name=GazeWeights.RESNET34, providers=['CPUExecutionProvider'])

    def analyse(self, image):
        import cv2
        import numpy as np
        rgb = np.asarray(image.convert('RGB'))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        height, width = bgr.shape[:2]
        # Bound detector cost; gaze inference uses original-resolution face crops.
        scale = min(1., 1600/max(width, height))
        small = cv2.resize(bgr, (round(width*scale), round(height*scale)))
        sx, sy = small.shape[1]/width, small.shape[0]/height
        faces = self.detector.detect(small)
        candidates = []
        for face in faces:
            box = np.asarray(face.bbox, dtype=float)[:4] / [sx, sy, sx, sy]
            if not np.isfinite(box).all():
                continue
            x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
            x2, y2 = min(width, int(box[2])), min(height, int(box[3]))
            if x2-x1 < 20 or y2-y1 < 20:
                continue
            landmarks = getattr(face, 'landmarks', None)
            if landmarks is None:
                continue
            points = np.asarray(landmarks, dtype=float).reshape(-1, 2) / [sx, sy]
            if len(points) < 2 or not np.isfinite(points[:2]).all():
                continue
            eyes = points[:2]
            if not all(x1 <= x <= x2 and y1 <= y <= y2 for x, y in eyes):
                continue
            candidates.append((face, (x1, y1, x2, y2), eyes))
        candidates.sort(key=lambda item: (item[1][2]-item[1][0])*(item[1][3]-item[1][1]), reverse=True)
        count = len(candidates)
        selected = sorted(candidates[:2], key=lambda item: item[1][0])
        results = []
        for face, box, eyes in selected:
            x1, y1, x2, y2 = box
            prediction = self.estimator.estimate(bgr[y1:y2, x1:x2])
            pitch, yaw = float(prediction.pitch), float(prediction.yaw)
            origin = eyes.mean(axis=0).tolist()
            ray = projected_ray(origin, pitch, yaw, min(width, height)*.15)
            results.append(dict(box=list(box), eyes=eyes.tolist(), ray=ray,
                                pitch_radians=pitch, yaw_radians=yaw))
        return dict(faces=results, detected_count=count, model='UniFace 4.1.0 / MobileGaze ResNet34',
                    projection='uncalibrated 2D projection of estimated gaze',
                    selection='two largest usable faces, ordered left to right')
