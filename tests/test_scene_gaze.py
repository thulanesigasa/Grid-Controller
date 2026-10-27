import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
from scene_gaze import normalize_boxes, summarize_map, shared_mask, overlay_image, SceneGaze


def person(x, y, score=.9):
    a = np.zeros((64, 64))
    a[y, x] = 1
    return summarize_map(a, score, (640, 320))


class SceneTests(unittest.TestCase):
    def test_boxes_and_mapping(self):
        self.assertEqual(normalize_boxes([[64, 32, 128, 64]], (640, 320)), [(.1, .1, .2, .2)])
        self.assertEqual(person(63, 0)['peak'], [635, 2.5])
        for b in ([], [[0, 0, 0, 10]], [[-1, 0, 10, 10]], [[0, 0, 1000, 20]], [[0, 0, 1, 1]]):
            with self.assertRaises(ValueError):
                normalize_boxes(b, (640, 320))

    def test_no_forced_overlap_or_low_score_peak(self):
        a, b = person(1, 2), person(30, 40)
        self.assertFalse(shared_mask([a, b]).any())
        self.assertEqual(shared_mask([a, a]).sum(), 1)
        low = person(1, 2, .1)
        self.assertIsNone(low['peak'])
        self.assertFalse(shared_mask([a, low]).any())
        self.assertIsNone(summarize_map(np.ones((64, 64)), .9, (640, 320))['peak'])
        self.assertFalse(shared_mask([a]).any())

    def test_bad_output_rejected(self):
        for a in (np.zeros((32, 32)), np.full((64, 64), np.nan), np.full((64, 64), 2)):
            with self.assertRaises(ValueError):
                summarize_map(a, .9, (640, 320))

    def test_separate_layers_and_green_shared_support(self):
        a, b = person(1, 2), person(30, 40)
        im = np.asarray(overlay_image([a, b], 'P1'))
        self.assertGreater(im[2, 1, 3], 0)
        self.assertEqual(im[40, 30, 3], 0)
        self.assertEqual(list(np.asarray(overlay_image([a, a]))[2, 1]), [163, 230, 53, 160])

    def test_inference_contract_without_downloads(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Optional torch not installed')
        engine = SceneGaze()
        seen = []
        def model(inputs):
            seen.append(inputs)
            a = torch.zeros((2, 64, 64))
            a[0, 10, 20] = 1
            a[1, 30, 40] = 1
            return {'heatmap': [a], 'inout': [torch.tensor([.9, .2])]}
        engine.model = model
        engine.transform = lambda image: torch.zeros((3, 448, 448))
        result = engine.analyse(Image.new('RGB', (640, 320)), [[10, 20, 30, 40], [100, 20, 130, 40]])
        self.assertEqual(tuple(seen[0]['images'].shape), (1, 3, 448, 448))
        self.assertEqual(len(result['people']), 2)
        self.assertIsNone(result['people'][1]['peak'])
        self.assertEqual(result['shared_cell_count'], 0)

    def test_validation_keeps_individual_errors_not_fake_intersection(self):
        from validation import freeze_prediction, label_record
        ai = {'scene_enabled': True, 'people': [person(1, 2), person(1, 2, .1)]}
        s = freeze_prediction(Image.new('RGB', (640, 320)), 'x', 'Scene gaze AI', [], ai, None, None, 10)
        r = label_record(s, [15, 12.5])
        self.assertIsNone(r['metrics']['error_pixels'])
        self.assertEqual(r['metrics']['scene_people'][0]['error_pixels'], 0)
        self.assertIsNone(r['metrics']['scene_people'][1]['error_pixels'])
        self.assertFalse(r['metrics']['scene_shared_contains_target'])

    def test_stale_result_preserves_existing_analysis(self):
        import queue
        from app import App
        app = App.__new__(App)
        app.ai_queue = queue.Queue()
        app.image = object()
        app.ai_data = {'old': True}
        app.ai_button = app.scene_button = SimpleNamespace(configure=lambda **kw: None)
        messages = []
        app.ai_status = SimpleNamespace(set=messages.append)
        app.analysis_state = lambda: 'new boxes'
        app.ai_queue.put((app.image, 'old boxes', {'scene_enabled': True}, None))
        app.poll_ai()
        self.assertEqual(app.ai_data, {'old': True})
        self.assertIn('discarded', messages[0])
