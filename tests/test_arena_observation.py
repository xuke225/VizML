import copy
import unittest

import numpy as np
from threadpoolctl import threadpool_limits

from backend.arena.engine import generate, models_for, prepare
from backend.arena.observation import fit_observation


class ObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = dict(points=generate({"count": 96, "noise": 0.2})["points"], seed=42, version=8,
                           params={"mlp": {"hidden_layer_sizes": "8,4", "max_iter": 100},
                                   "forest": {"n_estimators": 12}, "tree": {"scale_features": True}})
        cls.first, cls.first_models, cls.X, cls.mesh = fit_observation(dict(cls.payload, step=1), True)
        cls.ten, cls.ten_models, _, _ = fit_observation(dict(cls.payload, step=10), True)

    def test_six_real_meshes_shared_holdout_no_rank(self):
        self.assertEqual(self.ten["ranks"], {})
        self.assertFalse(self.ten["observation"]["ranked"])
        self.assertEqual(self.first["split"], self.ten["split"])
        self.assertEqual(self.ten["version"], 8)
        for key, model in self.ten_models.items():
            np.testing.assert_array_equal(model.predict(self.mesh), self.ten["models"][key]["grid"])
            self.assertEqual(self.ten["models"][key]["test_ids"], self.ten["split"]["test_ids"])

    def test_mlp_is_ten_actual_optimizer_preserving_epochs(self):
        data, X, y, train, _ = prepare(self.payload)
        manual = models_for(self.payload["params"], data["seed"], len(train))["mlp"]
        transformed = manual.steps[0][1].fit_transform(X[train])
        with threadpool_limits(limits=1):
            for _ in range(10):
                manual.steps[-1][1].partial_fit(transformed, y[train], classes=[0, 1])
        fitted = self.ten_models["mlp"].steps[-1][1]
        for actual, expected in zip(fitted.coefs_, manual.steps[-1][1].coefs_):
            np.testing.assert_allclose(actual, expected)
        info = self.ten["models"]["mlp"]["fit_info"]
        self.assertEqual(info["iterations"], 10)  # n_iter_ alone is 1 in sklearn 1.3.
        self.assertEqual(len(info["loss_curve"]), 10)
        self.assertEqual(len(self.first["models"]["mlp"]["fit_info"]["loss_curve"]), 1)
        self.assertFalse(np.allclose(fitted.coefs_[0], self.first_models["mlp"].steps[-1][1].coefs_[0]))

    def test_sgd_and_perceptron_epochs_without_invented_loss(self):
        for config in ({"solver": "sgd", "activation": "tanh", "batch_size": 8}, {"network_type": "perceptron"}):
            payload = copy.deepcopy(self.payload)
            payload["params"]["mlp"].update(config)
            r = fit_observation(dict(payload, step=3))["models"]["mlp"]
            self.assertEqual(r["fit_info"]["iterations"], 3)
            self.assertEqual(r["observation"]["mode"], "incremental_epochs")
            if config.get("network_type") == "perceptron":
                self.assertEqual(r["fit_info"]["loss_curve"], [])
                self.assertIsNone(r["fit_info"]["final_loss"])
                self.assertNotIn("probabilities", next(iter(r["explanations"].values())))

    def test_svm_and_lbfgs_truthfully_use_independent_budgets(self):
        for r, budget in ((self.first, 1), (self.ten, 10)):
            info = r["models"]["svm"]
            self.assertEqual(info["effective_params"]["max_iter"], budget)
            self.assertEqual(info["configured_params"]["max_iter"], 1000)
            self.assertEqual(info["observation"]["mode"], "independent_budget")
            self.assertLessEqual(info["observation"]["actual_iterations"], budget)
            self.assertTrue(set(info["observation"]["support_ids"]) <= set(r["split"]["train_ids"]))
        payload = copy.deepcopy(self.payload)
        payload["params"]["mlp"]["solver"] = "lbfgs"
        r = fit_observation(dict(payload, step=10))["models"]["mlp"]
        self.assertEqual(r["observation"]["mode"], "independent_budget")
        self.assertEqual(r["effective_params"]["max_iter"], 10)
        self.assertEqual(r["fit_info"]["loss_curve"], [])

    def test_forest_ten_trees_are_full_forest_prefix(self):
        data, X, y, train, _ = prepare(self.payload)
        full = models_for(self.payload["params"], 42, len(train))["forest"].fit(X[train], y[train])
        prefix = self.ten_models["forest"]
        for i, tree in enumerate(prefix.estimators_):
            np.testing.assert_array_equal(tree.predict(self.mesh), full.estimators_[i].predict(self.mesh))
        self.assertEqual(len(self.first_models["forest"].estimators_), 1)
        self.assertEqual(len(prefix.estimators_), 10)

    def test_tree_root_and_final_prefix_are_real_node_distributions(self):
        prefix = self.first_models["tree"].steps[-1][1]
        self.assertEqual(prefix.expanded_nodes, {0})
        stage = self.first["models"]["tree"]["observation"]
        self.assertEqual(stage["mode"], "tree_structure")
        self.assertEqual(len(stage["segments"]), 1)
        self.assertTrue(all(len(e["path"]) <= 1 for e in self.first["models"]["tree"]["explanations"].values()))
        scaler = self.first_models["tree"].steps[0][1]
        axis = int(prefix.tree_.feature[0])
        self.assertAlmostEqual(stage["segments"][0]["threshold"], prefix.tree_.threshold[0]*scaler.scale_[axis]+scaler.mean_[axis])
        final, models, X, _ = fit_observation(dict(self.payload, step=100), True)
        tree = models["tree"]
        np.testing.assert_array_equal(tree.predict(X), tree.steps[-1][1].estimator.predict(tree.steps[0][1].transform(X)))
        raw = tree.steps[-1][1]
        probes = tree.steps[0][1].transform(self.mesh)
        probes[:3, axis] = raw.tree_.threshold[0] + np.array([-1e-9, 0, 1e-9])
        np.testing.assert_array_equal(raw.predict(probes), raw.estimator.predict(probes))
        self.assertTrue(final["models"]["tree"]["observation"]["complete"])

    def test_unsplittable_tree_reports_zero_real_splits(self):
        payload = dict(points=generate({"count": 24})["points"], params={"tree": {"min_samples_leaf": 20}})
        info = fit_observation(dict(payload, step=1))["models"]["tree"]["observation"]
        self.assertEqual(info["current"], 0)
        self.assertEqual(info["total"], 0)
        self.assertEqual(info["segments"], [])
        self.assertTrue(info["complete"])

    def test_knn_boundary_static_partial_votes_end_at_real_probability(self):
        self.assertEqual(self.first["models"]["knn"]["grid"], self.ten["models"]["knn"]["grid"])
        self.assertTrue(all(len(e["revealed_neighbors"]) == 1 for e in self.first["models"]["knn"]["explanations"].values()))
        for weights in ("uniform", "distance"):
            payload = copy.deepcopy(self.payload)
            payload["params"]["knn"] = {"weights": weights}
            r, models, X, _ = fit_observation(dict(payload, step=10), True)
            proba = models["knn"].predict_proba(X)
            for i, point in enumerate(r["points"]):
                np.testing.assert_allclose(r["models"]["knn"]["explanations"][point["id"]]["partial_probabilities"], proba[i])

    def test_bayes_uses_only_cumulative_training_examples(self):
        first = self.first["models"]["bayes"]["observation"]
        last = self.ten["models"]["bayes"]["observation"]
        self.assertEqual(first["seen_count"], 8)
        self.assertTrue(set(first["seen_ids"]) <= set(last["seen_ids"]))
        self.assertEqual(set(last["seen_ids"]), set(self.ten["split"]["train_ids"]))
        self.assertFalse(set(first["seen_ids"]) & set(self.ten["split"]["test_ids"]))
        data, X, y, train, _ = prepare(self.payload)
        full = models_for(self.payload["params"], 42, len(train))["bayes"].fit(X[train], y[train])
        np.testing.assert_allclose(self.ten_models["bayes"].theta_, full.theta_)
        np.testing.assert_allclose(self.ten_models["bayes"].var_, full.var_)
        np.testing.assert_array_equal(self.ten_models["bayes"].predict(X), full.predict(X))

    def test_bayes_scaler_is_fixed_training_pool_only(self):
        payload = copy.deepcopy(self.payload)
        payload["params"]["bayes"] = {"scale_features": True, "priors": [0.3, 0.7]}
        first, models, X, _ = fit_observation(dict(payload, step=1), True)
        _, _, _, train, _ = prepare(payload)
        np.testing.assert_allclose(models["bayes"].steps[0][1].mean_, X[train].mean(axis=0))
        info = first["models"]["bayes"]["observation"]
        np.testing.assert_allclose(info["gaussians"][0]["mean"],
                                   models["bayes"].steps[-1][1].theta_[0] * models["bayes"].steps[0][1].scale_ + X[train].mean(axis=0))
        self.assertIn("整个训练池", info["note"])

    def test_replay_is_deterministic_and_limits_are_explicit(self):
        replay = fit_observation(dict(self.payload, step=10))
        for key in self.ten["models"]:
            self.assertEqual(replay["models"][key]["grid"], self.ten["models"][key]["grid"])
        capped = fit_observation(dict(self.payload, step=10000))
        for key in ("mlp", "forest", "tree", "knn", "bayes"):
            info = capped["models"][key]["observation"]
            self.assertEqual(info["current"], info["total"])
            self.assertTrue(info["complete"])
        for invalid in (0, -1, 1.5, True, 10001, "10"):
            with self.assertRaises(ValueError):
                fit_observation(dict(self.payload, step=invalid))

    def test_endpoint_errors_and_success(self):
        from backend.arena.api import blueprint
        from flask import Flask
        app = Flask(__name__)
        app.register_blueprint(blueprint)
        with app.test_client() as client:
            response = client.post('/api/arena/observe', json=dict(self.payload, step=1))
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json()["observation"]["step"], 1)
            self.assertEqual(client.post('/api/arena/observe', json=dict(self.payload, step=0)).status_code, 400)


if __name__ == "__main__":
    unittest.main()
