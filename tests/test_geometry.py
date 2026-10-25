import unittest
from geometry import intersection, image_point, sensitivity

class GeometryTests(unittest.TestCase):
    def test_known_crossing(self):
        self.assertEqual(intersection(((0, 0), (1, 1)), ((10, 0), (9, 1))), (5, 5))
    def test_parallel(self):
        self.assertIsNone(intersection(((0, 0), (1, 1)), ((2, 0), (3, 1))))
    def test_backward(self):
        self.assertIsNone(intersection(((0, 0), (-1, -1)), ((10, 0), (9, 1))))
    def test_zero_length(self):
        self.assertIsNone(intersection(((0, 0), (0, 0)), ((10, 0), (9, 1))))
    def test_zoom_pan_round_trip(self):
        for scale in (.05, 1, 8):
            for p in ((0, 0), (2208, 1668), (4415, 3335)):
                off = (-100, 250)
                actual = image_point(p[0]*scale+off[0], p[1]*scale+off[1], scale, off)
                for got, expected in zip(actual, p):
                    self.assertAlmostEqual(got, expected, places=8)
    def test_sensitivity(self):
        points = sensitivity(((0, 0), (1, 1)), ((10, 0), (9, 1)), 2)
        self.assertEqual(len(points), 9)
        self.assertIn((5, 5), points)

if __name__ == '__main__':
    unittest.main()
