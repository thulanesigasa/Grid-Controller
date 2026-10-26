import math
import unittest
from geometry import cone_overlap
from per_eye import screen_ray


class PerEyeTests(unittest.TestCase):
    def test_projection_anchors_iris_preserves_direction(self):
        ray = screen_ray([20, 30], [[100, 100], [103, 96]], 50)
        self.assertEqual(ray, [[20, 30], [50, -10]])
        self.assertIsNone(screen_ray([20, 30], [[1, 1], [1, 1]], 50))
        self.assertIsNone(screen_ray([20, 30], [[1, 1], [float('nan'), 2]], 50))

    def test_four_cones_share_known_target(self):
        rays = [[[10, 90], [50, 20]], [[20, 90], [50, 20]],
                [[80, 90], [50, 20]], [[90, 90], [50, 20]]]
        poly = cone_overlap(rays, 5, 100, 100)
        self.assertGreaterEqual(len(poly), 3)
        self.assertLess(min(p[0] for p in poly), 50)
        self.assertGreater(max(p[0] for p in poly), 50)
        self.assertLess(min(p[1] for p in poly), 20)
        self.assertGreater(max(p[1] for p in poly), 20)
        for x, y in poly:
            self.assertTrue(0 <= x <= 99 and 0 <= y <= 99)
            for a, b in rays:
                d = (b[0]-a[0], b[1]-a[1])
                v = (x-a[0], y-a[1])
                self.assertGreaterEqual(sum(d[k]*v[k] for k in (0, 1)), 0)
                self.assertLessEqual(abs(math.atan2(d[0]*v[1]-d[1]*v[0], sum(d[k]*v[k] for k in (0, 1)))), math.radians(5)+1e-8)

    def test_missing_backward_and_zero_tolerance_no_region(self):
        rays = [[[20, 50], [10, 50]], [[80, 50], [90, 50]]]
        self.assertEqual(cone_overlap(rays, 10, 100, 100), [])
        self.assertEqual(cone_overlap([rays[0], None], 10, 100, 100), [])
        self.assertEqual(cone_overlap(rays, 0, 100, 100), [])

    def test_eye_drag_preserves_own_ray_and_other_eye(self):
        from app import App
        app = App.__new__(App)
        face = {'eyes': [[20, 30], [60, 30]], 'eye_status': ['iris refined']*2,
                'eye_rays': [[[20, 30], [50, 10]], [[60, 30], [50, 10]]]}
        app.ai_data = {'faces': [face]}
        app.rays = [[[40, 30], [50, 10]], []]
        app.drag = ('eye', 0, 0)
        app.point = lambda e: (25, 35)
        app.inside = lambda p: True
        app.draw = lambda: None
        app.move(None)
        self.assertEqual(face['eye_rays'], [[[25, 35], [55, 15]], [[60, 30], [50, 10]]])
        app.drag = ('eye_direction', 0, 1)
        app.move(None)
        self.assertEqual(face['eye_rays'][1], [[60, 30], [25, 35]])
