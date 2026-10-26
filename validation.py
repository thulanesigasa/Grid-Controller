"""Frozen predictions and user-labelled ground truth; no training or inference."""
import copy
import hashlib
import math
from datetime import datetime, timezone


def freeze_prediction(image, image_name, source, rays, ai_analysis, point, region, tolerance):
    rgb = image.convert('RGB')
    digest = hashlib.sha256(f'{rgb.width}x{rgb.height}:RGB:'.encode() + rgb.tobytes()).hexdigest()
    return copy.deepcopy(dict(
        schema_version=1, captured_at=datetime.now(timezone.utc).isoformat(),
        image=dict(name=image_name, width=rgb.width, height=rgb.height, rgb_sha256=digest),
        prediction=dict(source=source, rays=rays, ai_analysis=ai_analysis,
                        point=point, region=region, tolerance_degrees=tolerance),
        ground_truth=None, metrics=None))


def region_contains(polygon, point):
    """Boundary-inclusive membership for a non-degenerate convex overlap region."""
    if not polygon or len(polygon) < 3:
        return False
    area = sum(a[0]*b[1]-b[0]*a[1] for a, b in zip(polygon, polygon[1:]+polygon[:1]))
    if abs(area) < 1e-8:
        return False
    crosses = [(b[0]-a[0])*(point[1]-a[1])-(b[1]-a[1])*(point[0]-a[0])
               for a, b in zip(polygon, polygon[1:]+polygon[:1])]
    return all(c >= -1e-8 for c in crosses) or all(c <= 1e-8 for c in crosses)


def label_record(snapshot, target, group='', verified=False):
    w, h = snapshot['image']['width'], snapshot['image']['height']
    if len(target) != 2 or not all(math.isfinite(v) for v in target) or not (0 <= target[0] < w and 0 <= target[1] < h):
        raise ValueError('Ball centre must be inside the original image.')
    record = copy.deepcopy(snapshot)
    point = record['prediction']['point']
    region = record['prediction']['region']
    error = math.dist(point, target) if point is not None else None
    record['ground_truth'] = dict(point=list(target), source='user marked visible ball centre',
                                  user_verified=bool(verified), match_group=group.strip(),
                                  labelled_at=datetime.now(timezone.utc).isoformat())
    record['metrics'] = dict(point_prediction_available=point is not None,
                             error_pixels=error,
                             error_percent_diagonal=100*error/math.hypot(w, h) if error is not None else None,
                             region_available=bool(region),
                             target_in_region=region_contains(region, target) if region is not None else None)
    return record
