import copy
import math
from types import SimpleNamespace
import unittest
from geometry import fit_four_rays, cone_overlap
from app import App


class Value:
    def __init__(self, value): self.value = value
    def get(self): return self.value
    def set(self, value): self.value = value


def target_rays(target=(50, 20)):
    return [[[10, 90], list(target)], [[20, 90], list(target)],
            [[80, 90], list(target)], [[90, 90], list(target)]]


class ManualEyesTests(unittest.TestCase):
    def test_four_lines_fit_known_target(self):
        fit = fit_four_rays(target_rays(), 2, 100, 100)
        self.assertAlmostEqual(fit['point'][0], 50)
        self.assertAlmostEqual(fit['point'][1], 20)
        self.assertLess(fit['rms_pixels'], 1e-8)
        self.assertEqual(len(cone_overlap(target_rays(), 2, 100, 100)) >= 3, True)

    def test_all_four_lines_change_fit(self):
        rays = target_rays()
        rays[3][1][0] += 5
        fit = fit_four_rays(rays, 10, 100, 100)
        self.assertIsNotNone(fit)
        self.assertGreater(abs(fit['point'][0]-50), .1)
        self.assertGreater(fit['rms_pixels'], 0)

    def test_no_coordinate_for_incomplete_degenerate_parallel_backward_or_outside(self):
        examples = [target_rays()[:3], target_rays()[:3]+[[]],
                    target_rays()[:3]+[[[90, 90], [90, 90]]],
                    [[[x, 90], [x, 20]] for x in (10, 20, 80, 90)],
                    target_rays((50, -20))]
        backward = target_rays()
        backward[0][1] = [-30, 160]
        examples.append(backward)
        for rays in examples:
            self.assertIsNone(fit_four_rays(rays, 10, 100, 100))

    def test_bad_fit_rejected_by_angular_tolerance(self):
        rays = target_rays()
        rays[3][1] = [10, 80]
        self.assertIsNone(fit_four_rays(rays, 2, 100, 100))

    def app(self):
        a = App.__new__(App)
        a.image = SimpleNamespace(width=100, height=100, size=(100, 100))
        a.workflow = 'Manual'
        a.mode = Value('Manual eye 1')
        a.ai_status = Value('')
        a.manual_rays = [[], [], [], []]
        a.rays = [[], []]
        a.ai_data = None
        a.point = lambda e: [e.x/2, e.y/2]
        a.screen = lambda p: [p[0]*2, p[1]*2]
        a.inside = lambda p: 0 <= p[0] < 100 and 0 <= p[1] < 100
        a.draw = lambda: None
        a.drag = None
        return a

    def test_each_eye_draws_and_drags_independently_in_image_coordinates(self):
        a = self.app()
        for i in range(4):
            a.mode.set(f'Manual eye {i+1}')
            a.down(SimpleNamespace(x=20+20*i, y=180))
            a.down(SimpleNamespace(x=100, y=40))
        before = copy.deepcopy(a.manual_rays)
        a.mode.set('Manual eye 3')
        a.down(SimpleNamespace(x=100, y=40))
        self.assertEqual(a.drag, ('manual_handle', 2, 1))
        a.move(SimpleNamespace(x=110, y=44))
        self.assertEqual(a.manual_rays[2][1], [55, 22])
        for i in (0, 1, 3): self.assertEqual(a.manual_rays[i], before[i])
        self.assertEqual(a.rays, [[], []])

    def test_render_four_thin_lines_and_fit_without_any_ai(self):
        a = self.app()
        a.manual_rays = target_rays()
        a.degrees = lambda: 2
        a.boundaries = Value(False)
        a.clicked = None
        a.result = Value('')
        lines = []
        a.canvas = SimpleNamespace(create_polygon=lambda *args, **kw: None,
                                   create_line=lambda *args, **kw: lines.append(kw),
                                   create_oval=lambda *args, **kw: None,
                                   create_text=lambda *args, **kw: None,
                                   create_rectangle=lambda *args, **kw: None,
                                   winfo_width=lambda: 800, winfo_height=lambda: 600)
        a.draw_manual()
        self.assertEqual(sum(line['width'] == 1 for line in lines), 4)
        self.assertIn('4/4', a.result.get())
        self.assertIn('X=50.0', a.result.get())
        self.assertIsNone(a.ai_data)

    def test_approximate_fit_is_flagged_without_bypassing_forward_bounds(self):
        rays = target_rays()
        rays[3][1][0] += 8
        self.assertIsNone(fit_four_rays(rays, .1, 100, 100))
        approximate = fit_four_rays(rays, .1, 100, 100, require_tolerance=False)
        self.assertIsNotNone(approximate)
        self.assertFalse(approximate['within_tolerance'])
        self.assertGreater(approximate['max_angle_degrees'], .1)
        self.assertIsNone(fit_four_rays(target_rays((50, -20)), 2, 100, 100, require_tolerance=False))
        rays[0][1] = [-30, 160]
        self.assertIsNone(fit_four_rays(rays, 2, 100, 100, require_tolerance=False))

    def test_approximate_marker_does_not_become_supported_estimate(self):
        a = self.app()
        a.manual_rays = target_rays()
        a.manual_rays[3][1][0] += 8
        a.degrees = lambda: .1
        a.boundaries = Value(False)
        a.clicked = None
        a.result = Value('')
        colors, labels = [], []
        a.canvas = SimpleNamespace(create_polygon=lambda *args, **kw: None,
                                   create_line=lambda *args, **kw: None,
                                   create_oval=lambda *args, **kw: colors.append(kw.get('fill')),
                                   create_rectangle=lambda *args, **kw: None,
                                   create_text=lambda *args, **kw: labels.append(kw['text']),
                                   winfo_width=lambda: 800, winfo_height=lambda: 600)
        a.draw_manual()
        self.assertIn('#fbbf24', colors)
        self.assertTrue(any('APPROXIMATE FIT' in text for text in labels))
        self.assertIn('AMBER', a.result.get())
        self.assertIsNone(a.estimate())
