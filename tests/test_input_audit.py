import math
import unittest
import numpy as np
from audit_gaze_inputs import direction, angle_between, variants, audit


class InputAuditTests(unittest.TestCase):
    def test_flip_and_rotation_restore_original_coordinate_system(self):
        # A pure leftward ray becomes rightward when a mirrored input is restored.
        a = direction(0, math.pi/2, mirrored=True)
        np.testing.assert_allclose(a, [1, 0], atol=1e-8)
        # An upward ray in an image rotated CCW 90 degrees restores rightward.
        b = direction(math.pi/2, 0, roll=90)
        np.testing.assert_allclose(b, [1, 0], atol=1e-8)
        self.assertIsNone(direction(0, 0))

    def test_direction_distance_handles_wraparound(self):
        def vector(degrees):
            x = math.radians(degrees)
            return [math.cos(x), math.sin(x)]
        self.assertAlmostEqual(angle_between(vector(179), vector(-179)), 2)
        self.assertAlmostEqual(angle_between([1, 0], [-1, 0]), 180)
        self.assertIsNone(angle_between(None, [1, 0]))

    def test_subcrops_preserve_actual_pixels(self):
        image = np.arange(100*80*3, dtype=np.uint8).reshape(100, 80, 3)
        crops = {name: crop for name, crop, _, _ in variants(image)}
        np.testing.assert_array_equal(crops['baseline'], image)
        np.testing.assert_array_equal(crops['trim-left-5pct'], image[:, 4:])
        np.testing.assert_array_equal(crops['trim-all-5pct'], image[5:95, 4:76])
        np.testing.assert_array_equal(crops['horizontal-flip'], image[:, ::-1])
        self.assertEqual(crops['square-reflected-padding'].shape, (100, 100, 3))

    def test_baseline_mismatch_reported_and_not_corrected(self):
        face = dict(eyes=[[10, 10], [20, 10]], pitch_radians=.4, yaw_radians=.3)
        result = audit(np.zeros((100, 80, 3), np.uint8), face, lambda crop: (.2, .1))
        self.assertFalse(result['baseline_matches_diagnostics'])
        self.assertAlmostEqual(result['runs'][0]['pitch_degrees'], math.degrees(.2))
        self.assertEqual(len(result['runs']), 11)
