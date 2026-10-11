"""Replay AUTO readiness through the real scheduler without a device."""

import ast
import json
from pathlib import Path
import re
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

from maa.custom_action import CustomAction
from maa.custom_recognition import CustomRecognition
from maa.resource import Resource
from maa.tasker import Tasker

from test_activity_entry import ROOT, ScreenController
from test_review_fixes import task_result


ENTRY = "Global.AutoCombat.Enable"


def enable_function(shared):
    # Exercise the production nested function, including task-result handling.
    tree = ast.parse((ROOT / "agent/custom/action/auto_combat.py").read_text(encoding="utf-8"))
    function = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "auto_combat")
    namespace = dict(info_share=shared, logger=Mock(),
                     self=NS(toolbar_roi=[0, 0, 640, 66]),
                     act_mgr=NS(task_succeeded=lambda r: r is not None and r.status.succeeded))
    exec(compile(ast.Module(body=[function], type_ignores=[]), "auto-enable", "exec"), namespace)
    return namespace


class AutoController(ScreenController):
    def __init__(self, initial, hidden_frames, ignore_click):
        super().__init__()
        self.state = initial
        self.hidden_frames = hidden_frames
        self.ignore_click = ignore_click
        self.captures = 0
        self.clicks = []

    def screencap(self):
        self.captures += 1
        image = super().screencap()
        # Recognition reads the captured frame, not the controller's live state.
        image[0, 0, 0] = 0 if self.captures <= self.hidden_frames else self.state
        return image

    def click(self, x, y):
        self.clicks.append((x, y))
        if not self.ignore_click:
            self.state = 2
        return True

    def touch_down(self, contact, x, y, pressure):
        return self.click(x, y)

    def touch_up(self, contact):
        return True


class AutoScreen(CustomRecognition):
    def analyze(self, context, argv):
        text = {0: "三", 1: "OFF", 2: "ON"}[int(argv.image[0, 0, 0])]
        patterns = json.loads(argv.custom_recognition_param)
        if any(re.search(pattern, text) for pattern in patterns):
            return [402, 35, 28, 19]
        return None


class EnableAction(CustomAction):
    def __init__(self, shared):
        super().__init__()
        self.namespace = enable_function(shared)

    def run(self, context, argv):
        self.namespace["context"] = context
        return self.namespace["auto_combat"]()


class AutoEnableTests(unittest.TestCase):
    def replay(self, initial=1, hidden_frames=0, ignore_click=False):
        source = json.loads((ROOT / "assets/resource/base/pipeline/combat/global_combat.json")
                            .read_text(encoding="utf-8"))
        self.assertIn(ENTRY, source)
        pipeline = {name: node for name, node in source.items()
                    if name == ENTRY or name.startswith(ENTRY + ".")}
        for node in pipeline.values():
            # Keep routing and Click intact; replace only OCR and scale waits.
            recognition = node.get("recognition", {})
            if recognition.get("type") == "OCR":
                node["recognition"] = {"type": "Custom", "param": {
                    "custom_recognition": "AutoScreen",
                    "custom_recognition_param": recognition["param"]["expected"],
                }}
            node.update(timeout=max(1, node.get("timeout", 20000) // 100),
                        rate_limit=5, pre_delay=0, post_delay=0)
        pipeline["Replay"] = {"pre_delay": 0, "post_delay": 0, "action": {
            "type": "Custom", "param": {"custom_action": "EnableAction"}}}
        shared = NS(auto_combat_mode=False)
        controller = AutoController(initial, hidden_frames, ignore_click)
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            (bundle / "pipeline").mkdir()
            (bundle / "pipeline/test.json").write_text(json.dumps(pipeline), encoding="utf-8")
            resource = Resource()
            resource.register_custom_recognition("AutoScreen", AutoScreen())
            resource.register_custom_action("EnableAction", EnableAction(shared))
            self.assertTrue(resource.post_bundle(bundle).wait().status.succeeded)
            self.assertTrue(controller.post_connection().wait().status.succeeded)
            tasker = Tasker()
            self.assertTrue(tasker.bind(resource, controller))
            timed_out = threading.Event()

            def stop_loop():
                timed_out.set()
                tasker.post_stop()

            watchdog = threading.Timer(5, stop_loop)
            watchdog.start()
            try:
                succeeded = tasker.post_task("Replay").wait().status.succeeded
            finally:
                watchdog.cancel()
            self.assertFalse(timed_out.is_set())
        self.assertEqual(shared.auto_combat_mode, succeeded)
        return controller, succeeded

    def test_intro_frames_are_recaptured_then_off_is_clicked_and_on_confirmed(self):
        controller, succeeded = self.replay(hidden_frames=4)
        self.assertTrue(succeeded)
        self.assertGreater(controller.captures, 5)
        self.assertEqual(len(controller.clicks), 1)
        x, y = controller.clicks[0]
        self.assertTrue(402 <= x < 430 and 35 <= y < 54)

    def test_already_on_does_not_toggle_off(self):
        controller, succeeded = self.replay(initial=2)
        self.assertTrue(succeeded)
        self.assertEqual(controller.clicks, [])

    def test_ignored_click_does_not_report_enabled(self):
        controller, succeeded = self.replay(ignore_click=True)
        self.assertFalse(succeeded)
        self.assertEqual(len(controller.clicks), 1)
        self.assertGreater(controller.captures, 3)

    def test_missing_auto_times_out_without_click(self):
        controller, succeeded = self.replay(hidden_frames=10000)
        self.assertFalse(succeeded)
        self.assertEqual(controller.clicks, [])
        self.assertGreater(controller.captures, 3)

    def test_cancelled_task_cannot_mark_auto_enabled(self):
        shared = NS(auto_combat_mode=False)
        namespace = enable_function(shared)
        context = Mock()
        context.tasker.stopping = True
        context.run_task.return_value = task_result(True)
        namespace["context"] = context
        self.assertFalse(namespace["auto_combat"]())
        self.assertFalse(shared.auto_combat_mode)


if __name__ == "__main__":
    unittest.main()
