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
