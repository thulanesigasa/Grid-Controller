"""Compare a prediction with a reference point without changing the prediction."""
import math
import re
from coordinate_grid import map_point


def parse_reference(text, image_size, coordinate_size=None):
    values = re.split(r'[,\s]+', text.strip())
    if len(values) != 2:
        raise ValueError('Enter X, Y, for example 2177, 1208, or clear.')
    try:
        point = tuple(float(v) for v in values)
    except ValueError:
        raise ValueError('X and Y must be finite numbers.') from None
    size = coordinate_size or image_size
    if not all(math.isfinite(v) and 0 <= v < size[i] for i, v in enumerate(point)):
        raise ValueError('The judges\' point must be inside the selected coordinate grid.')
    return map_point(point, size, image_size) if coordinate_size else point


def comparison_text(point, reference, image_size, coordinate_size=None):
    predicted = map_point(point, image_size, coordinate_size)
    target = map_point(reference, image_size, coordinate_size)
    dx, dy = (target[i]-predicted[i] for i in (0, 1))
    units = 'competition px' if coordinate_size else 'photo px'
    return (f"Judges' point: X={target[0]:.1f}, Y={target[1]:.1f}\n"
            f"Judges minus estimate: dX={dx:+.1f}, dY={dy:+.1f}\n"
            f"Distance: {math.hypot(dx, dy):.1f} {units}")
