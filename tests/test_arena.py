import copy
import unittest

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from backend.arena.engine import BOUNDS, MODEL_KEYS, fit_experiment, generate, prepare


class ArenaEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input = dict(points=generate({"kind": "moons", "count": 120, "noise": 0.19})["points"], seed=42)
        cls.result, cls.models, cls.X, cls.mesh = fit_experiment(cls.input, keep_models=True)

    def test_shared_split_and_coordinate_range(self):
        r = self.result
        self.assertEqual(r["model_keys"], list(MODEL_KEYS))
        self.assertEqual(set(r["models"]), set(MODEL_KEYS))
        for model in r["models"].values():
            self.assertEqual(model["train_ids"], r["split"]["train_ids"])
            self.assertEqual(model["test_ids"], r["split"]["test_ids"])
        self.assertFalse(set(r["split"]["train_ids"]) & set(r["split"]["test_ids"]))
        self.assertEqual(len(r["split"]["train_ids"]) + len(r["split"]["test_ids"]), 120)
        self.assertEqual(r["grid"]["bounds"], BOUNDS)
        np.testing.assert_allclose(self.mesh[0], [-3, 3])
        np.testing.assert_allclose(self.mesh[-1], [3, -3])

    def test_grid_and_sample_predictions_are_real(self):
        for key, model in self.models.items():
            np.testing.assert_array_equal(model.predict(self.mesh), self.result["models"][key]["grid"])
            prediction = model.predict(self.X)
            self.assertEqual([int(v) for v in prediction],
                             list(self.result["models"][key]["predictions"].values()))
            self.assertEqual(len(self.result["models"][key]["grid"]), 14400)

    def test_metrics_are_holdout_predictions(self):
        r = self.result
        labels = {p["id"]: p["clean_label"] for p in r["points"]}
        y = [labels[i] for i in r["split"]["test_ids"]]
        for m in r["models"].values():
            pred = [m["predictions"][i] for i in r["split"]["test_ids"]]
            self.assertAlmostEqual(m["test"]["accuracy"], accuracy_score(y, pred))
            self.assertAlmostEqual(m["test"]["f1"], f1_score(y, pred, average="macro"))
            self.assertAlmostEqual(m["test"]["precision"], precision_score(y, pred, average="macro", zero_division=0))
            self.assertAlmostEqual(m["test"]["recall"], recall_score(y, pred, average="macro"))
            self.assertGreaterEqual(m["train_ms"], 0)

    def test_standardization_fits_training_only(self):
        train = set(self.result["split"]["train_ids"])
        expected = np.array([[p["x"], p["y"]] for p in self.result["points"] if p["id"] in train]).mean(axis=0)
        for key in ("svm", "knn", "mlp"):
            np.testing.assert_allclose(self.models[key].steps[0][1].mean_, expected)

    def test_reproducible_and_independent_of_client_order(self):
        changed = copy.deepcopy(self.input)
        changed["points"].reverse()
        other = fit_experiment(changed)
        self.assertEqual(other["split"], self.result["split"])
        for key in self.models:
            self.assertEqual(other["models"][key]["grid"], self.result["models"][key]["grid"])

    def test_train_drag_keeps_holdout_and_changes_boundaries(self):
        changed = copy.deepcopy(self.input)
        ids = set(self.result["split"]["train_ids"][:10])
        for point in changed["points"]:
            if point["id"] in ids:
                point["x"], point["y"] = 2.7, 2.4
        other = fit_experiment(changed)
        self.assertEqual(other["split"], self.result["split"])
        self.assertEqual(other["holdout_fingerprint"], self.result["holdout_fingerprint"])
        self.assertNotEqual(other["fingerprint"], self.result["fingerprint"])
        self.assertTrue(all(other["models"][k]["grid"] != self.result["models"][k]["grid"] for k in self.models))

    def test_test_drag_invalidates_comparability(self):
        changed = copy.deepcopy(self.input)
        pid = self.result["split"]["test_ids"][0]
        next(p for p in changed["points"] if p["id"] == pid)["x"] = 2.9
        other = prepare(changed)[0]
        self.assertEqual(other["split"], self.result["split"])
        self.assertNotEqual(other["holdout_fingerprint"], self.result["holdout_fingerprint"])

    def test_challenges_never_change_test_coordinates_or_labels(self):
        changed = copy.deepcopy(self.input)
        changed["challenge"] = {"noise": 0.4, "corruption": 0.3}
        other = fit_experiment(changed)
        self.assertEqual(other["split"], self.result["split"])
        self.assertEqual(other["holdout_fingerprint"], self.result["holdout_fingerprint"])
        original = {p["id"]: p for p in self.result["points"]}
        for p in other["points"]:
            if p["role"] == "test":
                self.assertEqual(p, original[p["id"]])
            if p["corrupted"]:
                self.assertEqual(p["role"], "train")
                self.assertNotEqual(p["label"], p["clean_label"])
        self.assertEqual(sum(p["corrupted"] for p in other["points"]),
                         int(len(other["split"]["train_ids"]) * 0.3))

    def test_mechanism_explanations_are_model_outputs(self):
        r = self.result
        for i, p in enumerate(r["points"][:12]):
            pid = p["id"]
            self.assertAlmostEqual(r["models"]["svm"]["explanations"][pid]["decision_score"],
                                   self.models["svm"].decision_function(self.X[i:i+1])[0])
            k = r["models"]["knn"]["explanations"][pid]["neighbors"]
            scaled = self.models["knn"].steps[0][1].transform(self.X[i:i+1])
            distances, indices = self.models["knn"].steps[-1][1].kneighbors(scaled)
            self.assertEqual([n["id"] for n in k], [r["split"]["train_ids"][j] for j in indices[0]])
            np.testing.assert_allclose([n["distance"] for n in k], distances[0])
            tree = r["models"]["tree"]["explanations"][pid]
            self.assertEqual(tree["leaf"], self.models["tree"].apply(self.X[i:i+1])[0])
            for step in tree["path"]:
                condition = self.X[i, step["feature"]] <= step["threshold"]
                self.assertEqual(step["direction"], "≤" if condition else ">")

    def test_new_model_probabilities_and_forest_aggregation(self):
        r = self.result
        for key in ("forest", "mlp", "bayes"):
            actual = self.models[key].predict_proba(self.X)
            for i, p in enumerate(r["points"]):
                reported = r["models"][key]["explanations"][p["id"]]["probabilities"]
                np.testing.assert_allclose(reported, actual[i])
                self.assertAlmostEqual(sum(reported), 1)
                self.assertEqual(r["models"][key]["predictions"][p["id"]], int(np.argmax(reported)))
        forest = self.models["forest"]
        mean_probabilities = np.mean([tree.predict_proba(self.X) for tree in forest.estimators_], axis=0)
        np.testing.assert_allclose(forest.predict_proba(self.X), mean_probabilities)
        actual_b = np.sum([tree.predict(self.X) == 1 for tree in forest.estimators_], axis=0)
        for i, p in enumerate(r["points"]):
            self.assertEqual(r["models"]["forest"]["explanations"][p["id"]]["tree_votes"],
                             [len(forest.estimators_) - int(actual_b[i]), int(actual_b[i])])
        self.assertEqual(r["models"]["forest"]["fit_info"]["aggregation"], "mean_probability")

    def test_mlp_actual_iterations_architecture_and_loss(self):
        network = self.models["mlp"].steps[-1][1]
        info = self.result["models"]["mlp"]["fit_info"]
        self.assertEqual(info["network"], [2, 100, 50, 1])
        self.assertEqual(info["iterations"], network.n_iter_)
        self.assertEqual(info["iterations"], len(info["loss_curve"]))
        self.assertAlmostEqual(info["final_loss"], network.loss_)
        np.testing.assert_allclose(info["loss_curve"], network.loss_curve_)

    def test_iteration_limit_reports_warning_not_fake_convergence(self):
        limited = fit_experiment(dict(self.input, params={"mlp": {"width": 4, "layers": 1, "iterations": 100}}))
        info = limited["models"]["mlp"]["fit_info"]
        self.assertTrue(info["iteration_limit_reached"])
        self.assertEqual(info["iterations"], 100)
        self.assertEqual(len(limited["models"]["mlp"]["grid"]), 14400)

    def test_bayes_distribution_is_fitted_on_training_only(self):
        info = self.result["models"]["bayes"]["fit_info"]
        train_ids = set(self.result["split"]["train_ids"])
        for label in (0, 1):
            coordinates = [[p["x"], p["y"]] for p in self.result["points"]
                           if p["id"] in train_ids and p["label"] == label]
            np.testing.assert_allclose(info["means"][label], np.mean(coordinates, axis=0))
        np.testing.assert_allclose(info["priors"], self.models["bayes"].class_prior_)

    def test_generators_honor_count_and_seed(self):
        for kind in ("moons", "circles", "xor"):
            p = {"kind": kind, "count": 125, "seed": 19}
            a, b = generate(p), generate(p)
            self.assertEqual(a, b)
            self.assertEqual(len(a["points"]), 125)
            self.assertEqual({p["label"] for p in a["points"]}, {0, 1})

    def test_invalid_data_rejected(self):
        bad = [[], self.input["points"][:23], [dict(p, label=0) for p in self.input["points"]],
               self.input["points"] + [self.input["points"][0]],
               [dict(p, x=float("nan")) for p in self.input["points"]],
               [dict(p, label=True) for p in self.input["points"]],
               [dict(p, x=4) for p in self.input["points"]]]
        for points in bad:
            with self.subTest(points=len(points)), self.assertRaises(ValueError):
                prepare(dict(points=points))

    def test_computation_budgets_and_parameters(self):
        cases = [{"params": {"knn": {"k": 32}}}, {"params": {"tree": {"depth": 16}}},
                 {"params": {"svm": {"C": float("inf")}}}, {"seed": -1},
                 {"challenge": {"corruption": 0.9}}, {"test_size": 0.01},
                 {"params": {"forest": {"trees": 121}}}, {"params": {"forest": {"features": "bad"}}},
                 {"params": {"mlp": {"width": 257}}}, {"params": {"mlp": {"layers": 9}}},
                 {"params": {"mlp": {"iterations": 2001}}}, {"params": {"mlp": {"alpha": -1}}},
                 {"params": {"bayes": {"smoothing": 0}}}, {"params": {"forest": []}}]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                fit_experiment(dict(self.input, **case))
        with self.assertRaises(ValueError):
            generate({"count": 601})

    def test_all_models_participate_without_selection(self):
        self.assertEqual(self.result["model_keys"], list(MODEL_KEYS))
        selected=fit_experiment(dict(self.input, selected_models=["svm", "knn", "tree"]))
        self.assertEqual(set(selected["models"]), {"svm", "knn", "tree"})

    def test_coincident_samples_return_clear_error_not_nan(self):
        points = [dict(p, x=0, y=0) for p in self.input["points"]]
        with self.assertRaisesRegex(ValueError, "重合"):
            fit_experiment(dict(points=points))

    def test_minority_holdout_too_small_is_rejected(self):
        points = copy.deepcopy(self.input["points"])
        for i, p in enumerate(points):
            p["label"] = int(i < 6)
        with self.assertRaisesRegex(ValueError, "测试样本不足"):
            prepare(dict(points=points, test_size=0.2))

    def test_maximum_sample_budget_completes(self):
        data = generate({"kind": "circles", "count": 600, "noise": 0.3})
        result = fit_experiment(dict(points=data["points"]))
        self.assertEqual(len(result["points"]), 600)
        self.assertEqual(len(result["models"]["knn"]["grid"]), 14400)

    def test_ranking_uses_f1_not_timing(self):
        r = self.result
        for a in self.models:
            for b in self.models:
                if r["models"][a]["test"]["f1"] > r["models"][b]["test"]["f1"]:
                    self.assertLess(r["ranks"][a], r["ranks"][b])


class ArenaApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app import app
        cls.client = app.test_client()

    def test_homepage_keeps_labs_and_replaces_old_module(self):
        for route in ('/', '/html/index.html'):
            with self.subTest(route=route):
                response = self.client.get(route)
                html = response.get_data(as_text=True)
                response.close()
                self.assertEqual(response.status_code, 200)
                self.assertEqual(html.count('id="ml-arena"'), 1)
                self.assertIn('/html/linear_regression.html', html)
                self.assertNotIn("{% include", html)
                self.assertNotIn("forge", html.lower())
        self.assertEqual(self.client.post("/api/ai-forge/run", json={}).status_code, 404)

    def test_api_success_and_validation(self):
        points = self.client.post("/api/arena/generate", json={"kind": "xor", "count": 80}).json["points"]
        r = self.client.post("/api/arena/run", json={"points": points, "version": 7})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json["version"], 7)
        self.assertEqual(set(r.json["models"]), set(MODEL_KEYS))
        self.assertEqual(self.client.post("/api/arena/run", json={"points": []}).status_code, 400)
        self.assertEqual(self.client.post("/api/arena/run", data="{bad").status_code, 400)
        self.assertEqual(self.client.post("/api/arena/run", json=[]).status_code, 400)
        self.assertEqual(self.client.post("/api/arena/run", data=" " * 180001).status_code, 413)


if __name__ == "__main__":
    unittest.main()
