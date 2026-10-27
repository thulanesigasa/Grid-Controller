import unittest
from types import SimpleNamespace
from app import App


class Value:
    def __init__(self, value=None):
        self.value = value
    def get(self):
        return self.value
    def set(self, value):
        self.value = value


class WorkspaceTests(unittest.TestCase):
    def app(self):
        a = App.__new__(App)
        a.workflow = 'Scene AI'
        a.workflow_results = {}
        a.ai_data = {'scene_enabled': True}
        a.rays = [[], []]
        a.ray_source = 'Scene gaze AI'
        a.ai_busy = False
        a.mode = Value('Inspect')
        a.ai_status = Value()
        a.line_settings = SimpleNamespace(pack_forget=lambda: None, pack=lambda **kw: None)
        a.help_label = object()
        a.update_help = lambda: None
        return a

    def test_tabs_restore_results_without_mixing_methods(self):
        a = self.app()
        selected = Value('Manual')
        a.tabs = SimpleNamespace(select=lambda: None, tab=lambda *args: selected.get())
        a.change_workflow()
        self.assertIsNone(a.ai_data)
        self.assertEqual(a.mode.get(), 'Player 1')
        a.rays = [[[1, 2], [3, 4]], []]
        selected.set('Scene AI')
        a.change_workflow()
        self.assertTrue(a.ai_data['scene_enabled'])
        self.assertEqual(a.rays, [[], []])
        selected.set('Manual')
        a.change_workflow()
        self.assertEqual(a.rays[0], [[1, 2], [3, 4]])

    def test_busy_analysis_prevents_method_switch(self):
        a = self.app()
        a.ai_busy = True
        selections = []
        a.tabs = SimpleNamespace(select=lambda *args: selections.extend(args), tab=lambda *args: 'Manual')
        a.change_workflow()
        self.assertEqual(a.workflow, 'Scene AI')
        self.assertEqual(selections, [0])

    def test_hiding_validation_exits_ball_marking(self):
        a = self.app()
        a.details_visible = True
        a.mode.set('Mark ball')
        a.details = SimpleNamespace(pack_forget=lambda: None)
        a.details_button = SimpleNamespace(configure=lambda **kw: None)
        a.toggle_details()
        self.assertFalse(a.details_visible)
        self.assertEqual(a.mode.get(), 'Inspect')
