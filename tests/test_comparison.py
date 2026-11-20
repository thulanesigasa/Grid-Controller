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
        a.workflow = 'Manual'
        a.draw = lambda: None
        a.manual_fit = lambda **kw: {'point': (200, 100)}
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

    def test_reopened_photo_can_compare_entered_prediction_without_lines(self):
        from types import SimpleNamespace
        a = App.__new__(App)
        a.image = Image.new('RGB', (2400, 1813))
        a.coordinate_size = (4416, 3336)
        a.root = None
        a.workflow = 'Manual'
        a.draw = lambda: None
        a.manual_fit = lambda **kw: None
        a.manual_rays = [[], [], [], []]
        with patch('app.simpledialog.askstring', side_effect=['2177, 1208', '2045, 1045.9']):
            a.set_judges_point()
        self.assertAlmostEqual(a.coordinate_point(a.previous_estimate)[0], 2045)
        self.assertEqual(a.manual_rays, [[], [], [], []])
        calls = []
        a.canvas = SimpleNamespace(create_line=lambda *v, **kw: calls.append('line'),
            create_rectangle=lambda *v, **kw: calls.append('rectangle'),
            create_text=lambda *v, **kw: calls.append(kw['text']),
            winfo_width=lambda: 1000, winfo_height=lambda: 700)
        a.screen = lambda p: p
        a.result = SimpleNamespace(get=lambda: '', set=lambda v: None)
        a.draw_comparison(None)
        self.assertIn('rectangle', calls)
        self.assertTrue(any('No prediction loaded' in c for c in calls))
        output = render_prediction(a.image, [], a.previous_estimate, approximate=True,
            coordinate_size=a.coordinate_size, reference=a.judges_point, previous_estimate=True)
        metadata = json.loads(output.info['GridController'])
        self.assertTrue(metadata['previous_estimate'])
        self.assertAlmostEqual(metadata['point_grid'][0], 2045)

    def test_eye_lines_uses_intersection_but_not_heatmap_or_separate_rays(self):
        a = App.__new__(App)
        a.workflow = 'Eye lines'
        a.ai_data = None
        a.scene_active = lambda: False
        a.estimate = lambda: (100, 50)
        a.inside = lambda p: True
        self.assertEqual(a.comparison_estimate(), (100, 50))
        a.ai_data = {'per_eye_enabled': True}
        self.assertIsNone(a.comparison_estimate())
        a.ai_data = None
        a.scene_active = lambda: True
        self.assertIsNone(a.comparison_estimate())
