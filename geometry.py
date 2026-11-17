"""Geometry in original-image pixels; independent of display zoom."""
import math


def intersection(a, b):
    """Return the forward-ray intersection, or None for parallel/backward rays."""
    (x, y), (u, v) = a
    (p, q), (r, s) = b
    dx, dy, ex, ey = u-x, v-y, r-p, s-q
    det = dx*ey-dy*ex
    norm = math.hypot(dx, dy)*math.hypot(ex, ey)
    if norm == 0 or abs(det)/norm < 1e-4:
        return None
    t = ((p-x)*ey-(q-y)*ex)/det
    z = ((p-x)*dy-(q-y)*dx)/det
    return (x+t*dx, y+t*dy) if t >= 0 and z >= 0 else None


def rotate(ray, degrees):
    (x, y), (u, v) = ray
    a = math.radians(degrees)
    dx, dy = u-x, v-y
    return ((x, y), (x+dx*math.cos(a)-dy*math.sin(a),
                      y+dx*math.sin(a)+dy*math.cos(a)))


def sensitivity(a, b, degrees):
    """Sample angle perturbations. This is sensitivity, not a confidence interval."""
    return [p for i in (-degrees, 0, degrees) for j in (-degrees, 0, degrees)
            if (p := intersection(rotate(a, i), rotate(b, j))) is not None]


def image_point(x, y, scale, offset):
    return ((x-offset[0])/scale, (y-offset[1])/scale)


def cone_overlap(rays, degrees, width, height):
    """Clip image to forward angular cones. Tolerance is not confidence."""
    if not rays or not 0 < degrees < 89 or width <= 1 or height <= 1:
        return []
    polygon = [(0., 0.), (width-1., 0.), (width-1., height-1.), (0., height-1.)]
    def clip(poly, distance):
        result = []
        if not poly:
            return result
        previous = poly[-1]
        dp = distance(previous)
        for current in poly:
            dc = distance(current)
            if (dc >= 0) != (dp >= 0):
                t = dp/(dp-dc)
                result.append(tuple(previous[k]+t*(current[k]-previous[k]) for k in (0, 1)))
            if dc >= 0:
                result.append(current)
            previous, dp = current, dc
        return result
    for ray in rays:
        if not ray or len(ray) != 2 or not all(math.isfinite(v) for p in ray for v in p) or math.dist(*ray) < 1e-8:
            return []
        for angle, sign in ((-degrees, 1), (degrees, -1)):
            a, b = rotate(ray, angle)
            dx, dy = b[0]-a[0], b[1]-a[1]
            polygon = clip(polygon, lambda p: sign*(dx*(p[1]-a[1])-dy*(p[0]-a[0])))
    return polygon if len(polygon) >= 3 else []


def fit_four_rays(rays, degrees, width, height, require_tolerance=True):
    """Least-squares 2D point, accepted only inside all four forward cones.

    Equal weights; RMS is perpendicular pixel disagreement, not target accuracy.
    Nearly parallel sets are rejected even if a large overlap polygon exists.
    With require_tolerance=False, return a flagged approximate fit outside the
    angular tolerance; forward direction and image bounds are still required.
    """
    if len(rays) != 4 or not 0 <= degrees <= 15 or width <= 1 or height <= 1:
        return None
    rows = []
    for ray in rays:
        if not ray or len(ray) != 2 or any(len(p) != 2 for p in ray) or not all(math.isfinite(v) for p in ray for v in p):
            return None
        a, b = ray
        length = math.dist(a, b)
        if length < 1e-8:
            return None
        dx, dy = (b[0]-a[0])/length, (b[1]-a[1])/length
        nx, ny = -dy, dx
        rows.append((a, dx, dy, nx, ny, nx*a[0]+ny*a[1]))
    aa = sum(r[3]**2 for r in rows)
    ab = sum(r[3]*r[4] for r in rows)
    bb = sum(r[4]**2 for r in rows)
    ac = sum(r[3]*r[5] for r in rows)
    bc = sum(r[4]*r[5] for r in rows)
    det = aa*bb-ab*ab
    if det/(aa+bb)**2 < 1e-4:
        return None
    x, y = (ac*bb-bc*ab)/det, (bc*aa-ac*ab)/det
    if not (0 <= x <= width-1 and 0 <= y <= height-1):
        return None
    residuals = []
    angles = []
    for a, dx, dy, nx, ny, c in rows:
        vx, vy = x-a[0], y-a[1]
        forward = dx*vx+dy*vy
        perpendicular = nx*x+ny*y-c
        if forward < 0:
            return None
        angles.append(math.degrees(math.atan2(abs(perpendicular), forward)))
        residuals.append(perpendicular**2)
    within = max(angles) <= degrees+1e-7
    if require_tolerance and not within:
        return None
    return dict(point=[x, y], rms_pixels=math.sqrt(sum(residuals)/4),
                within_tolerance=within, max_angle_degrees=max(angles))
