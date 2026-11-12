import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from io import BytesIO
from PIL import Image
from diagnose_gaze import save_diagnostics


class DiagnosticsTests(unittest.TestCase):
    def test_crops_match_exact_pixels_and_keep_player_order(self):
        image = Image.new('RGB', (100, 60), 'red')
        image.paste('blue', (50, 0, 100, 60))
        analysis = {'faces': [dict(box=[55, 5, 80, 30], pitch_radians=.2, yaw_radians=-.3),
                              dict(box=[5, 5, 30, 30], pitch_radians=.1, yaw_radians=.3)]}
        with tempfile.TemporaryDirectory() as directory:
            path = save_diagnostics(image, analysis, Path(directory)/'report.zip')
            with zipfile.ZipFile(path) as archive:
                report = json.loads(archive.read('diagnostics.json'))
                self.assertEqual(report['image_size'], [100, 60])
                self.assertEqual(report['analysis'], analysis)
                crop = Image.open(BytesIO(archive.read('player-1-face-input.png')))
                self.assertEqual(crop.size, (25, 25))
                self.assertEqual(crop.getpixel((0, 0)), (0, 0, 255))
            with self.assertRaises(FileExistsError):
                save_diagnostics(image, analysis, path)
