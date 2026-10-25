import math
import unittest
from auto_gaze import projected_ray

class ProjectionTests(unittest.TestCase):
    def test_upstream_signs(self):
        ray = projected_ray((100, 100), math.pi/6, math.pi/6, 50)
        self.assertLess(ray[1][0], 100)
        self.assertLess(ray[1][1], 100)
        self.assertAlmostEqual(math.dist(*ray), 50)
    def test_camera_facing_has_no_reliable_projection(self):
        self.assertIsNone(projected_ray((100, 100), 0, 0, 50))
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            projected_ray((100, 100), float('nan'), 0, 50)

class PipelineTests(unittest.TestCase):
    def test_resize_coordinates_return_to_original_pixels(self):
        try:
            import numpy as np
            import cv2
        except ImportError:
            self.skipTest('Optional AI dependencies are not installed')
        from types import SimpleNamespace
        from PIL import Image
        from auto_gaze import AutoGaze
        face = SimpleNamespace(bbox=np.array([100, 100, 200, 200]),
                               landmarks=np.array([[125, 130], [175, 130], [150, 150]]))
        engine = AutoGaze.__new__(AutoGaze)
        engine.detector = SimpleNamespace(detect=lambda image: [face])
        crops = []
        def estimate(crop):
            crops.append(crop.shape)
            return SimpleNamespace(pitch=.5, yaw=.2)
        engine.estimator = SimpleNamespace(estimate=estimate)
        result = engine.analyse(Image.new('RGB', (3200, 1600)))
        self.assertEqual(result['faces'][0]['box'], [200, 200, 400, 400])
        self.assertEqual(result['faces'][0]['ray'][0], [300, 260])
        self.assertEqual(crops, [(200, 200, 3)])
