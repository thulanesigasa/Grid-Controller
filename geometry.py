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
