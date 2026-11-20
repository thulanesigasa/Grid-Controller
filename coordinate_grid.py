"""Map an uncropped photograph's pixels to a competition's coordinate grid."""
import math
import re


def validate_grid(image_size, grid_size):
    for size in (image_size, grid_size):
        if len(size) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v < 2 for v in size):
            raise ValueError('Width and height must be whole numbers of at least 2 pixels.')
    ratio = (grid_size[0]/grid_size[1])/(image_size[0]/image_size[1])
    if abs(ratio-1) > .005:
        raise ValueError('Image and grid aspect ratios differ. Use the same uncropped photo and check the official dimensions.')
    return tuple(grid_size)


def parse_grid(text, image_size):
    if text.strip().lower() == 'image':
        return None
    match = re.fullmatch(r'\s*(\d+)\s*[xX×,]\s*(\d+)\s*', text)
    if not match:
        raise ValueError('Enter width x height, for example 4416 x 3336, or image to use photo pixels.')
    return validate_grid(image_size, tuple(map(int, match.groups())))


def map_point(point, image_size, grid_size=None):
    if len(point) != 2 or not all(math.isfinite(v) for v in point):
        raise ValueError('Coordinates must be finite X/Y values.')
    if grid_size is None:
        return tuple(point)
    grid_size = validate_grid(image_size, grid_size)
    # Match the website convention x/source_width, not x/(width-1).
    return tuple(point[i]*grid_size[i]/image_size[i] for i in (0, 1))
