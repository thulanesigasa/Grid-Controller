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
        engine.eye_mesh = None
        engine.eye_warning = 'Disabled for isolated test'
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


class EyeRefinementTests(unittest.TestCase):
    def test_refinement_and_per_eye_fallback(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest('Optional AI dependencies are not installed')
        from auto_gaze import refine_eyes
        coarse = np.array([[30., 40.], [70., 40.]])
        mesh = np.zeros((478, 3))
        mesh[468, :2] = [32, 37]
        mesh[473, :2] = [999, 999]
        eyes, status = refine_eyes(coarse, mesh, .99, [0, 0, 100, 100])
        np.testing.assert_equal(eyes, [[32, 37], [70, 40]])
        self.assertEqual(status, ['iris refined', 'coarse fallback'])
        eyes, _ = refine_eyes(coarse, mesh, .2, [0, 0, 100, 100])
        np.testing.assert_equal(eyes, coarse)
        eyes, _ = refine_eyes(coarse, mesh, float('nan'), [0, 0, 100, 100])
        np.testing.assert_equal(eyes, coarse)

class EyeEditingTests(unittest.TestCase):
    def test_drag_eye_moves_ray_without_changing_direction(self):
        try:
            from app import App
        except ImportError:
            self.skipTest('GUI dependencies unavailable')
        from types import SimpleNamespace
        app = App.__new__(App)
        app.drag = ('eye', 0, 1)
        app.ai_data = {'faces': [{'eyes': [[20, 30], [60, 30]], 'eye_status': ['iris refined']*2}]}
        app.rays = [[[40, 30], [80, 10]], []]
        app.inside = lambda p: True
        app.point = lambda e: (70, 40)
        app.draw = lambda: None
        app.move(SimpleNamespace())
        self.assertEqual(app.rays[0], [[45, 35], [85, 15]])
        self.assertEqual(app.ai_data['faces'][0]['eye_status'][1], 'manually corrected')
