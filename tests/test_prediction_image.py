import tempfile
from pathlib import Path
import unittest
from PIL import Image
from prediction_image import render_prediction, prediction_caption


class PredictionImageTests(unittest.TestCase):
    def test_original_pixels_and_dimensions_preserved_with_white_caption(self):
        original = Image.new('RGB', (800, 600), '#4477aa')
        before = original.tobytes()
        rays = [[[100, 550], [400, 200]], [[150, 550], [400, 200]],
                [[650, 550], [400, 200]], [[700, 550], [400, 200]]]
        result = render_prediction(original, rays, [400, 200], boundaries=False)
        self.assertEqual(original.tobytes(), before)
        self.assertEqual(result.size, original.size)
        self.assertGreater(sum(count for count, color in result.getcolors(800*600) if color == (255, 255, 255)), 100)
        self.assertEqual(result.getpixel((400, 200)), (0, 0, 0))
        with tempfile.TemporaryDirectory() as directory:
            for ext in ('png', 'jpg'):
                path = Path(directory)/('prediction.'+ext)
                result.save(path)
                with Image.open(path) as saved:
                    self.assertEqual(saved.size, original.size)

    def test_caption_contains_exact_coordinates_credit_and_uncertainty(self):
        self.assertEqual(prediction_caption([123.45, 67.89]),
                         'X: 123.5   Y: 67.9\nLines agree - estimate only\nby thulanesigasa')
        self.assertIn('Approximate fit - lines disagree', prediction_caption([10, 20], True))

    def test_edge_target_label_fits_and_invalid_position_rejected(self):
        image = Image.new('RGB', (400, 300), 'blue')
        rays = [[[20, 200], [1, 1]], [[100, 250], [1, 1]]]
        result = render_prediction(image, rays, [1, 1], approximate=True)
        self.assertEqual(result.size, image.size)
        with self.assertRaises(ValueError):
            render_prediction(image, rays, [-1, 1])
        with self.assertRaises(ValueError):
            render_prediction(image, [], [10, 10])

    def test_manual_save_button_writes_a_prediction_image(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from app import App
        app = App.__new__(App)
        app.image = Image.new('RGB', (400, 300), 'blue')
        app.ai_busy = False
        app.workflow = 'Manual'
        app.manual_rays = [[[20, 250], [200, 100]], [[40, 250], [200, 100]],
                           [[350, 250], [200, 100]], [[370, 250], [200, 100]]]
        app.manual_fit = lambda **kw: dict(point=[200, 100], within_tolerance=False)
        app.degrees = lambda: 2
        app.boundaries = SimpleNamespace(get=lambda: False)
        messages = []
        app.ai_status = SimpleNamespace(set=messages.append)
        with tempfile.TemporaryDirectory() as directory:
            app.path = str(Path(directory)/'original.png')
            destination = Path(directory)/'prediction.png'
            with patch('app.filedialog.asksaveasfilename', return_value=str(destination)):
                app.save_prediction_image()
            self.assertTrue(destination.exists())
            self.assertIn('saved', messages[0])
            self.assertFalse(Path(app.path).exists())
