import unittest
from types import SimpleNamespace
from auto_gaze import match_heads
from app import App


class LinkedHeadsTests(unittest.TestCase):
    def test_player_order_follows_selected_heads_not_size(self):
        left = ('left', [10, 10, 30, 30], [])
        right = ('right', [100, 10, 140, 50], [])
        other = ('other', [200, 0, 400, 200], [])
        self.assertEqual(match_heads([other, left, right], [[90, 0, 150, 60], [0, 0, 40, 40]]), [right, left])

    def test_missing_head_does_not_substitute_unrelated_face(self):
        with self.assertRaises(ValueError):
            match_heads([('left', [10, 10, 30, 30], [])], [[0, 0, 40, 40], [90, 0, 150, 60]])

    def test_same_face_cannot_fill_two_heads(self):
        with self.assertRaises(ValueError):
            match_heads([('left', [10, 10, 30, 30], [])], [[0, 0, 40, 40]]*2)

    def test_adjust_direction_supports_face_ray(self):
        a = App.__new__(App)
        a.image = object()
        a.mode = SimpleNamespace(get=lambda: 'Eye directions')
        a.point = lambda e: [e.x, e.y]
        a.inside = lambda p: True
        a.screen = lambda p: p
        a.ai_data = {'per_eye_enabled': False}
        a.rays = [[[10, 10], [50, 50]], []]
        a.draw = lambda: None
        a.drag = None
        a.down(SimpleNamespace(x=50, y=50))
        self.assertEqual(a.drag, ('handle', 0, 1))
        a.move(SimpleNamespace(x=60, y=40))
        self.assertEqual(a.rays[0], [[10, 10], [60, 40]])
