"""Render a shareable prediction in original-image coordinates."""
import math
import json
from coordinate_grid import map_point
from PIL import ImageDraw, ImageFont
from geometry import rotate, cone_overlap


def prediction_caption(point, approximate=False, image_size=None, coordinate_size=None):
    x, y = map_point(point, image_size, coordinate_size)
    grid = (f'Competition grid {coordinate_size[0]} x {coordinate_size[1]}\n' if coordinate_size else
            f'Image pixels {image_size[0]} x {image_size[1]}\n' if image_size else '')
    return (f'X: {x:.1f}   Y: {y:.1f}\n' + grid
            + ('Approximate fit - lines disagree' if approximate else 'Lines agree - estimate only')
            + '\nby thulanesigasa')


def _font(size):
    for name in ('arial.ttf', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def render_prediction(image, rays, point, approximate=False, degrees=2., boundaries=True, coordinate_size=None, reference=None):
    """Copy the photo; render original-pixel lines, marker and a white caption."""
    if len(point) != 2 or not all(math.isfinite(v) for v in point):
        raise ValueError('Prediction must contain finite X/Y coordinates.')
    w, h = image.size
    if not (0 <= point[0] <= w-1 and 0 <= point[1] <= h-1):
        raise ValueError('Prediction lies outside the image.')
    if len(rays) not in (2, 4) or any(len(r) != 2 or any(len(p) != 2 for p in r)
                                       or not all(math.isfinite(v) for p in r for v in p)
                                       or math.dist(*r) < 1e-8 for r in rays):
        raise ValueError('Complete, nonzero prediction lines are required.')
    result = image.convert('RGB').copy()
    colors = ('#00d8ff', '#ffb340')
    polygon = cone_overlap(rays, degrees, w, h)
    if polygon:
        from PIL import Image
        overlay = Image.new('RGBA', (w, h))
        ImageDraw.Draw(overlay).polygon(polygon, fill=(163, 230, 53, 50))
        result = Image.alpha_composite(result.convert('RGBA'), overlay).convert('RGB')
    draw = ImageDraw.Draw(result)
    extent = math.hypot(w, h)*2
    for i, ray in enumerate(rays):
        color = colors[i//2 if len(rays) == 4 else i]
        variants = [(ray, False)]
        if boundaries:
            variants += [(rotate(ray, d), True) for d in (-degrees, degrees)]
        for (a, b), dashed in variants:
            length = math.dist(a, b)
            dx, dy = (b[0]-a[0])/length, (b[1]-a[1])/length
            if dashed:
                for start in range(0, math.ceil(extent), 12):
                    draw.line([(a[0]+dx*start, a[1]+dy*start),
                               (a[0]+dx*(start+5), a[1]+dy*(start+5))], fill=color, width=1)
            else:
                draw.line([tuple(a), (a[0]+dx*extent, a[1]+dy*extent)], fill=color, width=1)
    x, y = point
    radius = max(7, round(min(w, h)/140))
    color = '#fbbf24' if approximate else '#a3e635'
    draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color, outline='black', width=2)
    draw.line((x-radius*1.5, y, x+radius*1.5, y), fill='white', width=2)
    draw.line((x, y-radius*1.5, x, y+radius*1.5), fill='white', width=2)
    draw.ellipse((x-2, y-2, x+2, y+2), fill='black')
    caption = prediction_caption(point, approximate, image.size, coordinate_size)
    size = max(12, round(min(w, h)/65))
    pad = max(6, round(size*.5))
    while True:
        font = _font(size)
        box = draw.multiline_textbbox((0, 0), caption, font=font, spacing=4)
        bw, bh = box[2]-box[0]+pad*2, box[3]-box[1]+pad*2
        if bw <= w-8 and bh <= h-8:
            break
        if size <= 6:
            raise ValueError('Image is too small for a readable coordinate label.')
        size -= 1
    left = max(4, min(x+radius*2, w-bw-4))
    top = max(4, min(y-radius-bh, h-bh-4))
    draw.rectangle((left, top, left+bw, top+bh), fill='white', outline='black', width=1)
    draw.multiline_text((left+pad-box[0], top+pad-box[1]), caption,
                        font=font, fill='black', spacing=4)
    if reference is not None:
        from comparison import comparison_text, parse_reference
        # Validate in photo pixels, then compare in the selected output grid.
        parse_reference(f'{reference[0]}, {reference[1]}', image.size)
        rx, ry = reference
        draw.line((x, y, rx, ry), fill='#ef4444', width=2)
        draw.rectangle((rx-radius, ry-radius, rx+radius, ry+radius), outline='#ef4444', width=3)
        text = comparison_text(point, reference, image.size, coordinate_size)
        box2 = draw.multiline_textbbox((0, 0), text, font=font, spacing=4)
        rw, rh = box2[2]-box2[0]+pad*2, box2[3]-box2[1]+pad*2
        # A separate panel at the bottom preserves the estimate's original caption.
        if rw > w-8 or rh > h-8:
            raise ValueError('Image is too small for a readable comparison label.')
        left2, top2 = 4, h-rh-4
        draw.rectangle((left2, top2, left2+rw, top2+rh), fill='white', outline='#ef4444', width=2)
        draw.multiline_text((left2+pad-box2[0], top2+pad-box2[1]), text, font=font, fill='black', spacing=4)
    result.info['GridController'] = json.dumps(dict(image_size=list(image.size),
        coordinate_size=list(coordinate_size or image.size), point_image=list(point),
        point_grid=list(map_point(point, image.size, coordinate_size)),
        reference_image=list(reference) if reference is not None else None,
        reference_grid=list(map_point(reference, image.size, coordinate_size)) if reference is not None else None,
        approximate=approximate, green_means='lines agree, accuracy unverified', credit='by thulanesigasa'), allow_nan=False)
    return result
