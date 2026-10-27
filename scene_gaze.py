"""Scene-conditioned gaze heatmaps; not ball detection or action prediction."""
import math

DINO_REPO = 'facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8'
MODEL = 'gazelle_dinov2_vitb14_inout'
CHECKPOINT = f'https://github.com/fkryan/gazelle/releases/download/v1.0.0/{MODEL}.pt'
GAZELLE_REVISION = '2d78f9f3bd2a5db360354954ec1a4c526ee8ab55'


def normalize_boxes(boxes, size):
    w, h = size
    result = []
    for box in boxes:
        if len(box) != 4 or not all(math.isfinite(v) for v in box):
            raise ValueError('Head boxes must contain four finite coordinates.')
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise ValueError('Head boxes must be non-empty and inside the image.')
        normalized = (x1/w, y1/h, x2/w, y2/h)
        if round(normalized[0]*32) >= round(normalized[2]*32) or round(normalized[1]*32) >= round(normalized[3]*32):
            raise ValueError('Head box is too small at model resolution. Enclose the entire head or use a closer photograph.')
        result.append(normalized)
    if not 1 <= len(result) <= 2:
        raise ValueError('Select one or two head boxes.')
    return result


def summarize_map(heatmap, in_frame, size):
    import numpy as np
    a = np.asarray(heatmap, dtype=float)
    if a.shape != (64, 64) or not np.isfinite(a).all() or (a < 0).any() or (a > 1).any():
        raise ValueError('Expected a finite 64x64 gaze score map in [0,1].')
    if not math.isfinite(in_frame) or not 0 <= in_frame <= 1:
        raise ValueError('Invalid in-frame score.')
    y, x = np.unravel_index(a.argmax(), a.shape)
    # Cell-centre coordinates. Resolution is coarse, not pixel-accurate.
    peak = [(x+.5)*size[0]/64, (y+.5)*size[1]/64]
    usable = in_frame >= .5 and float(a.max()-a.min()) > 1e-6
    return dict(heatmap=a.tolist(), in_frame_score=float(in_frame),
                peak=peak if usable else None, spatially_usable=usable,
                note='Uncalibrated in-frame score; 0.5 display gate, not measured reliability.')


def display_map(person):
    import numpy as np
    a = np.asarray(person['heatmap'], dtype=float)
    if not person['spatially_usable']:
        return np.zeros_like(a)
    return (a-a.min())/(a.max()-a.min())


def shared_mask(people, threshold=.6):
    """Shared high relative activation, never a joint probability."""
    import numpy as np
    if len(people) != 2 or not all(p['spatially_usable'] for p in people):
        return np.zeros((64, 64), dtype=bool)
    return (display_map(people[0]) >= threshold) & (display_map(people[1]) >= threshold)


def overlay_image(people, view='Both'):
    import numpy as np
    from PIL import Image
    rgba = np.zeros((64, 64, 4), dtype=np.uint8)
    colors = [(0, 216, 255), (255, 179, 64)]
    for i, p in enumerate(people):
        if view not in ('Both', f'P{i+1}'):
            continue
        a = display_map(p)
        active = a >= .25
        # Composite per-player colours with relative intensity; suppress low tail.
        alpha = np.where(active, a*.48, 0)
        old_alpha = rgba[:, :, 3]/255.
        new_alpha = alpha+old_alpha*(1-alpha)
        rgb = (np.array(colors[i])[None, None, :]*alpha[:, :, None]
               + rgba[:, :, :3]*old_alpha[:, :, None]*(1-alpha[:, :, None]))
        rgba[:, :, :3] = (rgb/np.maximum(new_alpha[:, :, None], 1e-8)).astype(np.uint8)
        rgba[:, :, 3] = (new_alpha*255).astype(np.uint8)
    if view == 'Both':
        mask = shared_mask(people)
        rgba[mask] = [163, 230, 53, 160]
    return Image.fromarray(rgba)


class SceneGaze:
    def __init__(self):
        self.model = None
        self.detector = None

    def load_model(self):
        if self.model is not None:
            return
        try:
            import torch
            from gazelle.model import GazeLLE
            from gazelle.backbone import DinoV2Backbone, Backbone
        except ImportError as exc:
            raise RuntimeError('Install scene dependencies: python -m pip install -r requirements-scene.txt') from exc
        class PinnedBackbone(DinoV2Backbone):
            def __init__(self):
                Backbone.__init__(self)
                self.model = torch.hub.load(DINO_REPO, 'dinov2_vitb14', trust_repo=True)
        backbone = PinnedBackbone()
        model = GazeLLE(backbone, inout=True)
        state = torch.hub.load_state_dict_from_url(CHECKPOINT, map_location='cpu', weights_only=True)
        expected = {k for k in model.state_dict() if not k.startswith('backbone.')}
        if set(state) != expected:
            raise RuntimeError('Scene checkpoint keys do not match the pinned model; refusing partial weights.')
        full = model.state_dict()
        full.update(state)
        model.load_state_dict(full, strict=True)
        self.transform = backbone.get_transform((448, 448))
        self.model = model.to('cpu').eval()

    def detect_heads(self, image):
        import numpy as np
        import cv2
        if self.detector is None:
            import onnxruntime
            onnxruntime.disable_telemetry_events()
            from uniface.detection import RetinaFace
            self.detector = RetinaFace(input_size=(1280, 1280), providers=['CPUExecutionProvider'])
        w, h = image.size
        scale = min(1., 1600/max(w, h))
        small = image.resize((round(w*scale), round(h*scale)))
        faces = self.detector.detect(cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2BGR))
        boxes = []
        for face in faces:
            x1, y1, x2, y2 = np.asarray(face.bbox, dtype=float)[:4]/[small.width/w, small.height/h, small.width/w, small.height/h]
            if not np.isfinite([x1, y1, x2, y2]).all() or x2-x1 < 20 or y2-y1 < 20:
                continue
            bw, bh = x2-x1, y2-y1
            # Face detector is a proposal only: include forehead/hair around face.
            box = [max(0., x1-.15*bw), max(0., y1-.3*bh), min(float(w), x2+.15*bw), min(float(h), y2+.1*bh)]
            if box[0] < box[2] and box[1] < box[3]:
                boxes.append(box)
        boxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
        return sorted(boxes[:2], key=lambda b: b[0])

    def analyse(self, image, boxes=None):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError('Install scene dependencies: python -m pip install -r requirements-scene.txt') from exc
        image = image.convert('RGB')
        manual = boxes is not None
        boxes = boxes if manual else self.detect_heads(image)
        if not boxes:
            raise RuntimeError('No head proposals found. Draw Head 1 / Head 2 boxes and try Analyse scene again.')
        normalized = normalize_boxes(boxes, image.size)
        self.load_model()
        with torch.inference_mode():
            out = self.model({'images': self.transform(image).unsqueeze(0), 'bboxes': [normalized]})
        maps, scores = out['heatmap'][0].cpu().numpy(), out['inout'][0].cpu().numpy()
        if len(maps) != len(boxes) or len(scores) != len(boxes):
            raise RuntimeError('Scene model returned the wrong number of predictions.')
        people = [dict(box=list(box), **summarize_map(hm, float(score), image.size))
                  for box, hm, score in zip(boxes, maps, scores)]
        return dict(scene_enabled=True, per_eye_enabled=False, faces=[dict(box=p['box'], eyes=[], ray=None) for p in people],
                    people=people, model=MODEL, gazelle_revision=GAZELLE_REVISION, dino_repo=DINO_REPO,
                    head_source='manual boxes' if manual else 'two largest expanded face proposals, left to right',
                    shared_relative_threshold=.6, in_frame_display_gate=.5,
                    shared_cell_count=int(shared_mask(people).sum()),
                    note='Scene gaze only; no action prediction, ball detection, training or calibrated confidence.')
