import unittest
from PIL import Image
from validation import freeze_prediction, label_record, region_contains


class ValidationTests(unittest.TestCase):
    def snapshot(self, point=(10, 20), region=None):
        return freeze_prediction(Image.new('RGB', (300, 400)), 'test.png', 'AI suggestion',
                                 [[[0, 0], [1, 1]]], {'faces': []}, point, region, 10)

    def test_known_error_and_diagonal_normalization(self):
        record = label_record(self.snapshot(), [13, 24], 'match-1', True)
        self.assertEqual(record['metrics']['error_pixels'], 5)
        self.assertEqual(record['metrics']['error_percent_diagonal'], 1)
        self.assertTrue(record['ground_truth']['user_verified'])

    def test_snapshot_survives_edits_and_label_does_not_modify_it(self):
        rays = [[[0, 0], [1, 1]]]
        ai = {'faces': [{'eye_rays': rays}]}
        snapshot = freeze_prediction(Image.new('RGB', (30, 40)), 'x', 'Manual', rays, ai, [10, 20], None, 10)
        rays[0][1][0] = 99
        ai['faces'].clear()
        self.assertEqual(snapshot['prediction']['rays'][0][1][0], 1)
        self.assertEqual(len(snapshot['prediction']['ai_analysis']['faces']), 1)
        record = label_record(snapshot, [13, 24])
        self.assertIsNone(snapshot['ground_truth'])
        record['prediction']['point'][0] = 100
        self.assertEqual(snapshot['prediction']['point'][0], 10)

    def test_no_prediction_not_counted_as_zero_error(self):
        record = label_record(self.snapshot(None), [13, 24])
        self.assertIsNone(record['metrics']['error_pixels'])
        self.assertIsNone(record['metrics']['error_percent_diagonal'])
        self.assertFalse(record['metrics']['point_prediction_available'])

    def test_overlap_inside_outside_edge_and_empty(self):
        polygon = [[0, 0], [10, 0], [10, 10], [0, 10]]
        self.assertTrue(region_contains(polygon, [5, 5]))
        self.assertTrue(region_contains(polygon, [0, 5]))
        self.assertTrue(region_contains(polygon[::-1], [5, 5]))
        self.assertFalse(region_contains(polygon, [20, 5]))
        self.assertFalse(region_contains([], [5, 5]))
        self.assertFalse(region_contains([[0, 0], [1, 1], [2, 2]], [1, 1]))
        result = label_record(self.snapshot(None, polygon), [5, 5])
        self.assertTrue(result['metrics']['target_in_region'])
        self.assertIsNone(result['metrics']['error_pixels'])

    def test_reject_out_of_bounds_and_nonfinite_labels(self):
        for point in ([-1, 0], [300, 0], [0, 400], [float('nan'), 0]):
            with self.assertRaises(ValueError):
                label_record(self.snapshot(), point)

    def test_pixel_fingerprint_distinguishes_images(self):
        a = self.snapshot()['image']['rgb_sha256']
        self.assertEqual(a, self.snapshot()['image']['rgb_sha256'])
        b = freeze_prediction(Image.new('RGB', (300, 400), 'white'), 'test.png', 'Manual', [], None, None, None, 10)
        self.assertNotEqual(a, b['image']['rgb_sha256'])
