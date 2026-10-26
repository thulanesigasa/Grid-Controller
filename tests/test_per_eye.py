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


class HeadPoseInputTests(unittest.TestCase):
    def test_xyz_slice_reproduces_error_then_recovers_pose(self):
        import numpy as np
        import cv2
        from per_eye import prepare_landmarks
        rng = np.random.default_rng(42)
        model = rng.uniform(-.1, .1, (468, 3))
        camera = np.array([[1000., 0, 500], [0, 1000, 400], [0, 0, 1]])
        expected = np.array([0., 0., 1.])
        xy, _ = cv2.projectPoints(model, np.zeros(3), expected, camera, np.zeros(5))
        mesh = np.zeros((478, 3))
        mesh[:468, :2] = xy.reshape(-1, 2)
        sliced = mesh[:468, :2]
        self.assertFalse(sliced.flags.c_contiguous)
        with self.assertRaises(cv2.error):
            cv2.solvePnP(model, sliced, camera, np.zeros(5))
        ok, rotation, translation = cv2.solvePnP(model, prepare_landmarks(sliced), camera, np.zeros(5))
        self.assertTrue(ok)
        np.testing.assert_allclose(translation.ravel(), expected, atol=1e-6)
        np.testing.assert_allclose(rotation.ravel(), 0, atol=1e-6)

    def test_invalid_mesh_is_rejected_before_opencv(self):
        import numpy as np
        from per_eye import prepare_landmarks
        for points in (np.zeros((467, 2)), np.full((468, 2), np.nan)):
            with self.assertRaises(ValueError):
                prepare_landmarks(points)
