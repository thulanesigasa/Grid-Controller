import copy
import math
import unittest
from types import SimpleNamespace
from per_eye import assess_eye_pair
from app import App


def ray(angle, origin=(20, 80)):
    a = math.radians(angle)
    return [list(origin), [origin[0]+30*math.cos(a), origin[1]+30*math.sin(a)]]


class ConsistencyTests(unittest.TestCase):
    def test_opposed_and_large_disagreement_rejected(self):
        for angles in ((-90, 90), (-90, 170), (0, 61)):
            self.assertFalse(assess_eye_pair([ray(a) for a in angles])['usable'])

    def test_small_disagreement_and_wraparound_pass(self):
        for angles in ((-100, -80), (179, -179), (0, 59)):
            self.assertTrue(assess_eye_pair([ray(a) for a in angles])['usable'])

    def test_missing_degenerate_nonfinite_rejected(self):
        for pair in ([], [ray(0)], [ray(0), None], [ray(0), [[1, 1], [1, 1]]],
                     [ray(0), [[0, 0], [math.nan, 1]]]):
            self.assertFalse(assess_eye_pair(pair)['usable'])

    def test_rejected_pair_never_enters_overlap_and_raw_preserved(self):
        a = App.__new__(App)
        pairs = [[ray(-90), ray(90)], [ray(-85), ray(-95)]]
        a.ai_data = {'faces': [{'eye_rays': p, 'raw_eye_rays': copy.deepcopy(p)} for p in pairs]}
        original = copy.deepcopy(a.ai_data)
        a.image = SimpleNamespace(width=200, height=200)
        a.degrees = lambda: 15
        rays, polygon = a.eye_overlay()
        self.assertEqual(rays[:2], [None, None])
        self.assertEqual(polygon, [])
        self.assertEqual(a.ai_data, original)
        # A deliberate edit is re-evaluated, but the original prediction remains.
        a.ai_data['faces'][0]['eye_rays'][1] = ray(-90)
        self.assertTrue(all(r is not None for r in a.eye_overlay()[0]))
        self.assertEqual(a.ai_data['faces'][0]['raw_eye_rays'], original['faces'][0]['raw_eye_rays'])

    def test_conflict_display_is_short_grey_without_boundaries(self):
        a = App.__new__(App)
        pair = [ray(-90), ray(90)]
        a.ai_data = {'faces': [{'eye_rays': pair}]}
        a.image = SimpleNamespace(width=200, height=200)
        a.degrees = lambda: 10
        a.screen = lambda p: p
        a.boundaries = SimpleNamespace(get=lambda: True)
        a.ray_source = 'Per-eye AI suggestion'
        lines, messages = [], []
        a.canvas = SimpleNamespace(create_line=lambda *args, **kw: lines.append((args, kw)),
                                   create_oval=lambda *args, **kw: None, create_text=lambda *args, **kw: None)
        a.result = SimpleNamespace(set=messages.append)
        a.draw_eye_overlay()
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0][0], tuple(pair[0][0]+pair[0][1]))
        self.assertTrue(all(kw['fill'] == '#9ca3af' and kw['dash'] for args, kw in lines))
        self.assertIn('unreliable', messages[0])
        self.assertIn('withheld', messages[0])
