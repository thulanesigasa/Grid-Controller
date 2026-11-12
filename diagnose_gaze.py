"""Save the actual face-model inputs and raw outputs for local troubleshooting."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import zipfile
from PIL import Image


def save_diagnostics(image, analysis, destination):
    """Write lossless crops in player order alongside the unmodified model output."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f'Refusing to replace {destination}; choose another output filename.')
    report = dict(image_size=list(image.size), analysis=analysis, players=[],
                  note='Diagnostic outputs, not verified gaze directions. Crops contain player faces.')
    crops = []
    for i, face in enumerate(analysis['faces'], 1):
        box = face['box']
        if len(box) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box):
            raise ValueError('Invalid face crop box.')
        x1, y1, x2, y2 = box
        if not all(int(v) == v for v in box) or not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
            raise ValueError('Diagnostic crop must match the original integer model crop.')
        name = f'player-{i}-face-input.png'
        crops.append((name, image.crop(tuple(map(int, box))).convert('RGB')))
        report['players'].append(dict(player=i, crop=name, box=box,
                                      pitch_degrees=math.degrees(face['pitch_radians']),
                                      yaw_degrees=math.degrees(face['yaw_radians']),
                                      crop_size=[x2-x1, y2-y1]))
    # Serialize before writing, so invalid numeric model outputs cannot leave a partial report.
    payload = json.dumps(report, indent=2, allow_nan=False)
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('diagnostics.json', payload)
        from io import BytesIO
        for name, crop in crops:
            buffer = BytesIO()
            crop.save(buffer, format='PNG')
            archive.writestr(name, buffer.getvalue())
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', nargs='?', help='Original photograph. If omitted, a file picker opens.')
    parser.add_argument('--per-eye', action='store_true', help='Also run the optional separate-eye model.')
    parser.add_argument('--output', type=Path, help='New ZIP filename; existing files are never overwritten.')
    args = parser.parse_args()
    path = args.image
    if not path:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(title='Choose the original photograph for gaze diagnostics')
        root.destroy()
    if not path:
        return
    with Image.open(path) as original:
        image = original.convert('RGB')
    from auto_gaze import AutoGaze
    print(f'Analysing original {image.width} x {image.height} pixels; model downloads may be needed.', flush=True)
    data = AutoGaze().analyse(image, per_eye=args.per_eye)
    if not data['faces']:
        raise RuntimeError('No usable faces were detected; no diagnostic report was created.')
    output = args.output or Path('gaze-diagnostics-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')+'.zip')
    print('Saved:', save_diagnostics(image, data, output).resolve())
    print('ZIP contains player face crops and raw angles. These results do not establish gaze accuracy.')


if __name__ == '__main__':
    main()
