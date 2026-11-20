import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from PIL import Image
from app import App
from coordinate_grid import map_point, parse_grid
from prediction_image import render_prediction, prediction_caption


class CoordinateGridTests(unittest.TestCase):
    def test_competition_26_scaling_and_round_trip(self):
        image, grid = (2400, 1813), (4416, 3336)
        mapped = map_point((1111.4, 568.4), image, grid)
        self.assertAlmostEqual(mapped[0], 2044.976)
        self.assertAlmostEqual(mapped[1], 1045.881081081081)
        original = map_point(mapped, grid, image)
        self.assertAlmostEqual(original[0], 1111.4)
        self.assertAlmostEqual(original[1], 568.4)
        self.assertEqual(map_point((0, 0), image, grid), (0, 0))
        self.assertEqual(map_point((10.5, 20.3), image), (10.5, 20.3))

    def test_parser_validates_integer_sizes_and_same_uncropped_aspect_ratio(self):
        self.assertEqual(parse_grid('4416 × 3336', (2400, 1813)), (4416, 3336))
        self.assertIsNone(parse_grid('image', (2400, 1813)))
        for text in ('1 x 1', 'nan x 10', '4416 x 4416', '-4 x 8', '2400.5 x 1813'):
            with self.assertRaises(ValueError): parse_grid(text, (2400, 1813))
        with self.assertRaises(ValueError): map_point((math.nan, 1), (400, 300))

    def test_export_maps_caption_but_keeps_marker_and_photo_dimensions(self):
        image = Image.new('RGB', (400, 300), 'blue')
        rays = [[[20, 250], [200, 100]], [[40, 250], [200, 100]],
                [[350, 250], [200, 100]], [[370, 250], [200, 100]]]
        output = render_prediction(image, rays, [200, 100], coordinate_size=(800, 600))
        self.assertEqual(output.size, (400, 300))
        self.assertEqual(output.getpixel((200, 100)), (0, 0, 0))
        caption = prediction_caption([200, 100], image_size=image.size, coordinate_size=(800, 600))
        self.assertIn('X: 400.0   Y: 200.0', caption)
        self.assertIn('Competition grid 800 x 600', caption)
        self.assertIn('Lines agree - estimate only', caption)
        self.assertIn('by thulanesigasa', caption)
        metadata = json.loads(output.info['GridController'])
        self.assertEqual(metadata['point_image'], [200, 100])
        self.assertEqual(metadata['point_grid'], [400, 200])

    def test_setting_grid_does_not_move_lines_and_invalid_grid_preserves_setting(self):
        a = App.__new__(App)
        a.image = Image.new('RGB', (400, 300))
        a.coordinate_size = None
        a.root = None
        a.manual_rays = [[[20, 250], [200, 100]]]*4
        before = json.dumps(a.manual_rays)
        a.cursor = a.ai_status = SimpleNamespace(set=lambda text: None)
        a.draw = lambda: None
        with patch('app.simpledialog.askstring', return_value='800 x 600'):
            a.set_coordinate_grid()
        self.assertEqual(a.coordinate_point([200, 100]), (400, 200))
        self.assertEqual(json.dumps(a.manual_rays), before)
        with patch('app.simpledialog.askstring', return_value='800 x 800'), patch('app.messagebox.showerror') as error:
            a.set_coordinate_grid()
            error.assert_called_once()
        self.assertEqual(a.coordinate_size, (800, 600))
        with patch('app.simpledialog.askstring', return_value='image'):
            a.set_coordinate_grid()
        self.assertIsNone(a.coordinate_size)

    def test_app_png_save_preserves_grid_metadata(self):
        a = App.__new__(App)
        a.image = Image.new('RGB', (400, 300), 'blue')
        a.ai_busy = False
        a.workflow = 'Manual'
        a.coordinate_size = (800, 600)
        a.manual_rays = [[[20, 250], [200, 100]], [[40, 250], [200, 100]],
                         [[350, 250], [200, 100]], [[370, 250], [200, 100]]]
        a.manual_fit = lambda **kw: dict(point=[200, 100], within_tolerance=True)
        a.degrees = lambda: 2
        a.boundaries = SimpleNamespace(get=lambda: False)
        a.ai_status = SimpleNamespace(set=lambda text: None)
        with tempfile.TemporaryDirectory() as directory:
            a.path = str(Path(directory)/'photo.png')
            saved = Path(directory)/'prediction.png'
            with patch('app.filedialog.asksaveasfilename', return_value=str(saved)):
                a.save_prediction_image()
            with Image.open(saved) as image:
                info = json.loads(image.info['GridController'])
                self.assertEqual(image.size, (400, 300))
                self.assertEqual(info['coordinate_size'], [800, 600])
                self.assertEqual(info['point_grid'], [400, 200])
