import copy
import json
import unittest
from unittest.mock import patch
from PIL import Image
from app import App
from comparison import parse_reference, comparison_text
from prediction_image import render_prediction


class ComparisonTests(unittest.TestCase):
    def test_comp26_reference_and_distance(self):
        reference = parse_reference('2177, 1208', (2400, 1813), (4416, 3336))
        self.assertAlmostEqual(reference[0], 1183.1521739130435)
        text = comparison_text((1111.4, 568.4), reference, (2400, 1813), (4416, 3336))
        self.assertIn('dX=+132.0, dY=+162.1', text)
        self.assertIn('209.1 competition px', text)
        for value in ('nan, 3', 'inf, 2', '-1, 20', '4416, 2', '1, 3336', '1, 2, 3'):
            with self.assertRaises(ValueError):
                parse_reference(value, (2400, 1813), (4416, 3336))

    def test_dialog_preserves_lines_and_invalid_reference(self):
        a = App.__new__(App)
        a.image = Image.new('RGB', (400, 300))
        a.coordinate_size = (800, 600)
        a.root = None
        a.draw = lambda: None
        a.manual_rays = [[[10, 250], [200, 100]]]*4
        before = copy.deepcopy(a.manual_rays)
        with patch('app.simpledialog.askstring', return_value='500, 300'):
            a.set_judges_point()
        self.assertEqual(a.judges_point, (250, 150))
        with patch('app.simpledialog.askstring', return_value='900, 300'), patch('app.messagebox.showerror'):
            a.set_judges_point()
        self.assertEqual(a.judges_point, (250, 150))
        self.assertEqual(a.manual_rays, before)
        with patch('app.simpledialog.askstring', return_value='clear'):
            a.set_judges_point()
        self.assertIsNone(a.judges_point)

    def test_export_keeps_estimate_and_adds_reference_metadata(self):
        image = Image.new('RGB', (800, 600), 'blue')
        rays = [[[20, 550], [400, 200]], [[40, 550], [400, 200]],
                [[750, 550], [400, 200]], [[770, 550], [400, 200]]]
        before = copy.deepcopy(rays)
        output = render_prediction(image, rays, (400, 200), coordinate_size=(1600, 1200), reference=(450, 260))
        data = json.loads(output.info['GridController'])
        self.assertEqual(data['point_image'], [400, 200])
        self.assertEqual(data['reference_grid'], [900, 520])
        self.assertEqual(output.getpixel((443, 253)), (239, 68, 68))
        self.assertEqual(rays, before)
        self.assertEqual(image.getpixel((443, 253)), (0, 0, 255))
