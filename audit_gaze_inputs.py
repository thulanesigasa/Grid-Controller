"""Reproducible crop/roll stability audit of the pinned face gaze model.

Runs local OpenCV DNN inference only. Images are never sent to a server.
The model must already exist at --model and match the pinned SHA256.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import cv2
import numpy as np
from auto_gaze import projected_ray

MODEL_SHA256 = 'c8e6b14f6095d2425241b9302aa663d9a23b7dfb9d43941352b718c91dc7f2cf'


def direction(pitch, yaw, roll=0., mirrored=False):
    ray = projected_ray([0., 0.], pitch, yaw, 1.)
    if ray is None:
        return None
    x, y = ray[1]
    if mirrored:
        x = -x
    # Inverse of cv2.getRotationMatrix2D's image-coordinate rotation.
    angle = math.radians(roll)
    return [math.cos(angle)*x-math.sin(angle)*y,
            math.sin(angle)*x+math.cos(angle)*y]


def angle_between(a, b):
    if a is None or b is None:
        return None
    return math.degrees(math.atan2(abs(a[0]*b[1]-a[1]*b[0]), a[0]*b[0]+a[1]*b[1]))


def variants(image, eye_roll=0.):
    h, w = image.shape[:2]
    yield 'baseline', image, 0., False
    # Genuine subcrops only: no invented image context outside the saved face.
    for name, left, top, right, bottom in (
        ('trim-all-5pct', .05, .05, .05, .05),
        ('trim-left-5pct', .05, 0, 0, 0), ('trim-right-5pct', 0, 0, .05, 0),
        ('trim-top-5pct', 0, .05, 0, 0), ('trim-bottom-5pct', 0, 0, 0, .05)):
        a, b, c, d = round(left*w), round(top*h), w-round(right*w), h-round(bottom*h)
        yield name, image[b:d, a:c], 0., False
    # Aspect-ratio experiment: synthetic reflected borders, labelled explicitly.
    side = max(w, h)
    padded = cv2.copyMakeBorder(image, (side-h)//2, side-h-(side-h)//2,
                               (side-w)//2, side-w-(side-w)//2, cv2.BORDER_REFLECT_101)
    yield 'square-reflected-padding', padded, 0., False
    for name, roll in [('roll-minus-5', -5.), ('roll-plus-5', 5.), ('eye-line-levelled', eye_roll)]:
        transform = cv2.getRotationMatrix2D(((w-1)/2, (h-1)/2), roll, 1.)
        rotated = cv2.warpAffine(image, transform, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        yield name, rotated, roll, False
    yield 'horizontal-flip', image[:, ::-1].copy(), 0., True


class LocalModel:
    def __init__(self, path):
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != MODEL_SHA256:
            raise ValueError('Model does not match pinned UniFace 4.1.0 ResNet34 weights.')
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.names = self.net.getUnconnectedOutLayersNames()
        if set(self.names) != {'yaw', 'pitch'}:
            raise ValueError('Expected named yaw/pitch outputs.')

    def predict(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        a = cv2.resize(rgb, (448, 448)).astype(np.float32)/255.
        a = (a-np.array([.485, .456, .406], np.float32))/np.array([.229, .224, .225], np.float32)
        self.net.setInput(a.transpose(2, 0, 1)[None])
        decoded = {}
        for name, output in zip(self.names, self.net.forward(self.names)):
            logits = output.reshape(-1)
            if logits.size != 90 or not np.isfinite(logits).all():
                raise ValueError('Invalid gaze logits.')
            probs = np.exp(logits-logits.max())
            probs /= probs.sum()
            decoded[name] = math.radians(float(np.sum(probs*np.arange(90))*4-180))
        return decoded['pitch'], decoded['yaw']


def audit(image, face, predict):
    eye1, eye2 = face['eyes']
    roll = math.degrees(math.atan2(eye2[1]-eye1[1], eye2[0]-eye1[0]))
    runs = []
    for name, crop, rotation, flipped in variants(image, roll):
        pitch, yaw = predict(crop)
        runs.append(dict(variant=name, crop_size=[crop.shape[1], crop.shape[0]],
                         pitch_degrees=math.degrees(pitch), yaw_degrees=math.degrees(yaw),
                         restored_direction=direction(pitch, yaw, rotation, flipped)))
    base = runs[0]
    for run in runs:
        run['direction_change_degrees'] = angle_between(base['restored_direction'], run['restored_direction'])
    raw_error = max(abs(base['pitch_degrees']-math.degrees(face['pitch_radians'])),
                    abs(base['yaw_degrees']-math.degrees(face['yaw_radians'])))
    return dict(baseline_matches_diagnostics=raw_error <= .1, baseline_max_angle_error_degrees=raw_error,
                eye_line_roll_degrees=roll, runs=runs,
                note='Perturbation stability is not gaze accuracy. No transformation selected as the correct gaze.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('diagnostics', type=Path)
    p.add_argument('--crops', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new output filename.')
    cv2.setNumThreads(2)
    model = LocalModel(args.model)
    data = json.loads(args.diagnostics.read_text())
    reports = []
    for i, face in enumerate(data['analysis']['faces'], 1):
        image = cv2.imread(str(args.crops/f'player-{i}-face-input.png'))
        if image is None or [image.shape[1], image.shape[0]] != [face['box'][2]-face['box'][0], face['box'][3]-face['box'][1]]:
            raise ValueError('Missing crop or dimensions do not match diagnostics.')
        reports.append(dict(player=i, **audit(image, face, model.predict)))
    result = dict(model_sha256=MODEL_SHA256, backend='OpenCV DNN CPU', opencv_version=cv2.__version__, players=reports)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    for r in reports:
        print('Player', r['player'], 'baseline match:', r['baseline_matches_diagnostics'])
        for run in r['runs']:
            print(' ', run['variant'], 'direction change:', run['direction_change_degrees'])


if __name__ == '__main__':
    main()
