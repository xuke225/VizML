import base64
import copy
import unittest
from unittest.mock import patch

import numpy as np
from flask import Flask

from backend.arena.api import blueprint
from backend.arena.engine import generate
from backend.arena.observation import fit_observation
from backend.arena.timeline import record_timeline


class TimelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = dict(points=generate({"count": 48})["points"], seed=42, version=4, steps=3,
                           params={"mlp": {"hidden_layer_sizes": "8,4"}, "knn": {"weights": "distance"}})
        cls.record = record_timeline(cls.payload)

    def test_all_six_tracks_have_measured_monotone_durations(self):
        self.assertEqual(set(self.record["tracks"]), {"svm", "knn", "tree", "forest", "mlp", "bayes"})
        for track in self.record["tracks"].values():
            elapsed = track["setup_ms"]
            self.assertGreaterEqual(elapsed, 0)
            for frame in track["frames"]:
                self.assertGreaterEqual(frame["duration_ms"], 0)
                elapsed += frame["duration_ms"]
                self.assertAlmostEqual(frame["elapsed_ms"], elapsed)
                self.assertAlmostEqual(frame["model"]["observation"]["timing"]["step_ms"], frame["duration_ms"])
            self.assertAlmostEqual(track["duration_ms"], elapsed)
            self.assertGreaterEqual(track["recording_ms"], track["duration_ms"])
        self.assertGreaterEqual(self.record["timeline"]["recording_ms"],
                                sum(t["recording_ms"] for t in self.record["tracks"].values()))
        self.assertEqual(self.record["timeline"]["duration_ms"], max(t["duration_ms"] for t in self.record["tracks"].values()))
        self.assertFalse(self.record["timeline"]["ranked"])

    def test_every_frame_matches_real_manual_checkpoints(self):
        for step in range(1, 4):
            expected = fit_observation(dict(self.payload, step=step))
            for key, track in self.record["tracks"].items():
                frame = track["frames"][min(step-1, len(track["frames"])-1)]
                actual = frame["model"]
                bits = np.unpackbits(np.frombuffer(base64.b64decode(actual["grid_bits"]), dtype=np.uint8))[:14400]
                np.testing.assert_array_equal(bits, expected["models"][key]["grid"])
                self.assertEqual(actual["predictions"], expected["models"][key]["predictions"])
                self.assertEqual(actual["test"], expected["models"][key]["test"])
                if key == "knn":
                    for pid, explanation in actual["explanations"].items():
                        np.testing.assert_allclose(explanation["partial_probabilities"], expected["models"][key]["explanations"][pid]["partial_probabilities"])

    def test_real_epoch_and_incremental_forest_state_are_preserved(self):
        for i, frame in enumerate(self.record["tracks"]["mlp"]["frames"], 1):
            self.assertEqual(frame["model"]["fit_info"]["iterations"], i)
            self.assertEqual(len(frame["model"]["fit_info"]["loss_curve"]), i)
        for i, frame in enumerate(self.record["tracks"]["forest"]["frames"], 1):
            self.assertTrue(frame["model"]["effective_params"]["warm_start"])
            self.assertEqual(frame["model"]["fit_info"]["tree_count"], i)
        for frame in self.record["tracks"]["svm"]["frames"]:
            self.assertEqual(frame["model"]["effective_params"]["max_iter"], frame["step"])
            self.assertIn("完整求解", frame["model"]["observation"]["note"])

    def test_encoding_cost_does_not_enter_step_clock(self):
        from backend.arena import timeline
        actual_encoder = timeline.describe_stage
        counter = [0.0]

        def now():
            counter[0] += 0.001
            return counter[0]

        def expensive_encoding(*args, **kwargs):
            counter[0] += 500  # 500 seconds of simulated prediction/encoding.
            return actual_encoder(*args, **kwargs)

        with patch.object(timeline, 'perf_counter', side_effect=now), patch.object(timeline, 'describe_stage', side_effect=expensive_encoding):
            result = record_timeline(dict(self.payload, steps=2))
        for track in result["tracks"].values():
            self.assertGreater(track["recording_ms"], 500000)
            self.assertAlmostEqual(track["setup_ms"], 1)
            for frame in track["frames"]:
                self.assertAlmostEqual(frame["duration_ms"], 1)
            self.assertAlmostEqual(track["duration_ms"], 1 + len(track["frames"]))

    def test_limits_and_data_validation_are_not_silent(self):
        for value in (0, -1, 1.5, True, 1001, "10"):
            with self.assertRaises(ValueError):
                record_timeline(dict(self.payload, steps=value))
        payload = copy.deepcopy(self.payload)
        payload["params"]["svm"] = {"unknown": 2}
        with self.assertRaises(ValueError):
            record_timeline(payload)

    def test_tracks_stop_at_their_real_limits_not_uniform_step_count(self):
        result = record_timeline(dict(self.payload, steps=8, params={"forest": {"n_estimators": 3}, "mlp": {"hidden_layer_sizes": "8"}}))
        self.assertEqual(len(result["tracks"]["forest"]["frames"]), 3)
        self.assertEqual(len(result["tracks"]["knn"]["frames"]), 5)
        self.assertEqual(len(result["tracks"]["bayes"]["frames"]), 5)
        self.assertEqual(len(result["tracks"]["mlp"]["frames"]), 8)
        self.assertFalse(result["tracks"]["mlp"]["complete"])

    def test_alternative_solvers_and_network_types(self):
        for config in ({"solver": "lbfgs"}, {"solver": "sgd"}, {"network_type": "perceptron"}):
            payload = copy.deepcopy(self.payload)
            payload["params"]["mlp"].update(config)
            result = record_timeline(dict(payload, steps=2))
            info = result["tracks"]["mlp"]["frames"][-1]["model"]["fit_info"]
            self.assertEqual(info["solver"], config.get("solver", "perceptron"))
            if config.get("solver") == "lbfgs" or config.get("network_type") == "perceptron":
                self.assertEqual(info["loss_curve"], [])

    def test_api_includes_timing_metadata_and_clear_errors(self):
        app = Flask(__name__)
        app.register_blueprint(blueprint)
        with app.test_client() as client:
            response = client.post('/api/arena/timeline', json=dict(self.payload, steps=1))
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertIn('mesh_prediction', data["timeline"]["excludes"])
            self.assertEqual(data["version"], 4)
            self.assertEqual(data["timeline"]["max_steps"], 1000)
            response = client.post('/api/arena/timeline', json=dict(
                self.payload, steps=1000, selected_models=["knn", "bayes"]))
            self.assertEqual(response.status_code, 200)
            extended = response.get_json()
            self.assertEqual(extended["timeline"]["steps"], 1000)
            self.assertEqual(len(extended["tracks"]["knn"]["frames"]), 5)
            self.assertEqual(len(extended["tracks"]["bayes"]["frames"]), 5)
            self.assertEqual(client.post('/api/arena/timeline', json=dict(self.payload, steps=1001)).status_code, 400)


if __name__ == '__main__':
    unittest.main()
