"""Optional local face/eye detection and model-based gaze suggestions."""
import math
import copy


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


def refine_eyes(coarse, landmarks, score, box):
    """Validate each mesh iris independently; preserve coarse eyes on failure.

    Indices 468/473 are V2 iris centres. Presence is not eye-position confidence.
    """
    import numpy as np
    eyes = np.asarray(coarse, dtype=float).copy()
    status = ['coarse fallback', 'coarse fallback']
    pts = np.asarray(landmarks, dtype=float)
    if not math.isfinite(float(score)) or score < .8 or pts.ndim != 2 or pts.shape[0] < 478 or pts.shape[1] < 2:
        return eyes, status
    x1, y1, x2, y2 = box
    size = max(x2-x1, y2-y1)
    for i, idx in enumerate((468, 473)):
        point = pts[idx, :2]
        if (np.isfinite(point).all() and x1 <= point[0] <= x2 and y1 <= point[1] <= y2
                and np.linalg.norm(point-eyes[i]) <= .3*size):
            eyes[i] = point
            status[i] = 'iris refined'
    if not .15*(x2-x1) <= np.linalg.norm(eyes[1]-eyes[0]) <= .9*(x2-x1) or eyes[0, 0] >= eyes[1, 0]:
        return np.asarray(coarse, dtype=float).copy(), ['coarse fallback']*2
    return eyes, status


class AutoGaze:
    def __init__(self):
        try:
            import onnxruntime
            onnxruntime.disable_telemetry_events()
            from uniface.detection import RetinaFace
            from uniface.gaze import MobileGaze
            from uniface.constants import GazeWeights
        except ImportError as exc:
            raise RuntimeError('Install AI dependencies first: python -m pip install -r requirements-ai.txt') from exc
        # UniFace downloads and verifies its model files, then caches them locally.
        self.detector = RetinaFace(input_size=(1280, 1280), providers=['CPUExecutionProvider'])
        self.eye_mesh = None
        self.eye_warning = None
        self.estimator = MobileGaze(model_name=GazeWeights.RESNET34, providers=['CPUExecutionProvider'])

    def analyse(self, image, per_eye=False):
        import cv2
        import numpy as np
        if self.eye_mesh is None and self.eye_warning is None:
            try:
                from uniface.landmark import FaceMesh
                from uniface.constants import FaceMeshWeights
                self.eye_mesh = FaceMesh(model_name=FaceMeshWeights.V2_478, providers=['CPUExecutionProvider'])
            except Exception as exc:
                self.eye_warning = 'Iris refinement unavailable: ' + str(exc)
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
        eye_engine = None
        if per_eye and selected:
            from per_eye import PerEyeGaze
            eye_engine = getattr(self, 'per_eye_engine', None)
            if eye_engine is None or eye_engine.size != (width, height):
                if eye_engine is not None:
                    eye_engine.close()
                self.per_eye_engine = None
                eye_engine = self.per_eye_engine = PerEyeGaze(width, height)
        results = []
        for face, box, eyes in selected:
            x1, y1, x2, y2 = box
            coarse_eyes = eyes.copy()
            eye_status = ['coarse fallback']*2
            eye_warning = self.eye_warning
            mesh_points = None
            if self.eye_mesh is not None:
                try:
                    mesh = self.eye_mesh.predict(bgr, bboxes=[np.asarray(box)], keypoints=[eyes])[0]
                    eyes, eye_status = refine_eyes(eyes, mesh.landmarks, mesh.score, box)
                    pts = np.asarray(mesh.landmarks)
                    if mesh.score >= .8 and pts.shape[0] >= 468 and np.isfinite(pts[:468, :2]).all():
                        mesh_points = pts[:468, :2]
                except Exception as exc:
                    eye_warning = 'Iris refinement failed: ' + str(exc)
            prediction = self.estimator.estimate(bgr[y1:y2, x1:x2])
            pitch, yaw = float(prediction.pitch), float(prediction.yaw)
            origin = eyes.mean(axis=0).tolist()
            ray = projected_ray(origin, pitch, yaw, min(width, height)*.15)
            results.append(dict(box=list(box), eyes=eyes.tolist(), coarse_eyes=coarse_eyes.tolist(),
                                eye_status=eye_status, eye_warning=eye_warning, ray=ray,
                                pitch_radians=pitch, yaw_radians=yaw))
            if eye_engine is not None:
                try:
                    rays, warning = eye_engine.analyse_face(bgr, box, mesh_points, eyes)
                except Exception as exc:
                    rays, warning = [None, None], 'Per-eye inference failed: ' + str(exc)
                results[-1].update(eye_rays=rays, raw_eye_rays=copy.deepcopy(rays), per_eye_note=warning)
        return dict(faces=results, detected_count=count, per_eye_enabled=per_eye,
                    per_eye_model='ptgaze MPIIGaze ResNet preact' if per_eye else None,
                    camera='assumed focal length=image width; principal point=image centre; zero distortion' if per_eye else None, model='UniFace 4.1.0 / MobileGaze ResNet34',
                    projection='uncalibrated 2D projection of estimated gaze',
                    selection='two largest usable faces, ordered left to right')
