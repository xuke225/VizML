"""Parameter parity with the native labs, and proof of estimator passthrough."""
import copy
from html.parser import HTMLParser
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from backend.arena.engine import fit_experiment, generate, models_for, estimator_input
from backend.arena.parameters import PARAMETERS, create_classifier, normalize_parameters


ROOT = Path(__file__).resolve().parents[1]


class SelectOptions(HTMLParser):
    def __init__(self, control):
        super().__init__()
        self.control, self.inside, self.values = control, False, set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "select":
            self.inside = attrs.get("id") == self.control
        if tag == "option" and self.inside:
            self.values.add(attrs.get("value", ""))

    def handle_endtag(self, tag):
        if tag == "select":
            self.inside = False


class ParameterParityTests(unittest.TestCase):
    def test_native_select_options_are_all_available(self):
        bindings = {
            "svm": {"kernel": "kernel"},
            "knn": {"distanceMetric": "metric", "weights": "weights"},
            "tree": {"criterion": "criterion", "maxFeatures": "max_features"},
            "forest": {"maxFeatures": "max_features"},
            "mlp": {"networkType": "network_type", "activation": "activation", "solver": "solver",
                    "hiddenLayers": "hidden_layer_sizes"},
        }
        aliases = {"None": "all", "1.0": "all"}
        for model, controls in bindings.items():
            source = ROOT / "frontend/templates" / PARAMETERS[model]["lab"].split("/")[-1]
            for control, name in controls.items():
                parser = SelectOptions(control)
                parser.feed(source.read_text(encoding="utf-8"))
                self.assertTrue(parser.values, control)
                spec = next(f for f in PARAMETERS[model]["fields"] if f["name"] == name)
                choices = {v for v, _ in spec["options"]}
                if spec["type"] == "layers":
                    choices.add("custom")
                for value in parser.values:
                    with self.subTest(model=model, field=name, option=value):
                        self.assertIn(aliases.get(value, value), choices)

    def test_every_model_has_original_backend_training_parameters(self):
        required = {
            "svm": {"kernel", "C", "gamma", "degree", "coef0", "scale_features", "max_iter", "tol"},
            "knn": {"n_neighbors", "weights", "metric", "p", "scale_features"},
            "tree": {"max_depth", "min_samples_split", "min_samples_leaf", "max_features", "criterion", "scale_features"},
            "forest": {"n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features"},
            "mlp": {"network_type", "hidden_layer_sizes", "activation", "solver", "learning_rate_init", "max_iter", "batch_size", "alpha"},
            "bayes": {"var_smoothing", "priors", "scale_features"},
        }
        for key, names in required.items():
            self.assertEqual({f["name"] for f in PARAMETERS[key]["fields"]}, names)

    def test_all_four_svm_kernels_pass_through(self):
        for kernel in ("linear", "rbf", "poly", "sigmoid"):
            params = dict(kernel=kernel, C=8, gamma=0.3, degree=4, coef0=1.1,
                          max_iter=2300, tol=0.0002, scale_features=False)
            model = models_for({"svm": params}, 17, 90)["svm"]
            effective = model.get_params()
            self.assertEqual(effective["kernel"], kernel)
            for name in ("C", "max_iter", "tol"):
                self.assertEqual(effective[name], params[name])
            if kernel != "linear":
                self.assertEqual(effective["gamma"], 0.3)
            if kernel == "poly":
                self.assertEqual(effective["degree"], 4)
            if kernel in ("poly", "sigmoid"):
                self.assertEqual(effective["coef0"], 1.1)
            self.assertFalse(hasattr(model, "steps"))

    def test_complete_tree_and_forest_parameters_pass_through(self):
        tree = models_for({"tree": dict(criterion="entropy", max_depth=None, min_samples_split=17,
                                       min_samples_leaf=5, max_features="log2")}, 42, 90)["tree"]
        self.assertEqual(tree.get_params()["criterion"], "entropy")
        self.assertIsNone(tree.max_depth)
        self.assertEqual((tree.min_samples_split, tree.min_samples_leaf, tree.max_features), (17, 5, "log2"))
        for features in ("sqrt", "log2", "auto", "all"):
            forest = models_for({"forest": dict(n_estimators=7, max_depth=20, min_samples_split=11,
                                               min_samples_leaf=3, max_features=features)}, 42, 90)["forest"]
            self.assertEqual((forest.n_estimators, forest.max_depth, forest.min_samples_split), (7, 20, 11))
            self.assertEqual(forest.min_samples_leaf, 3)
            self.assertEqual(forest.max_features, None if features == "all" else "sqrt" if features == "auto" else features)

    def test_network_presets_solvers_activations_and_learning_rate(self):
        for hidden in ("50", "100,50", "100,100,50", "100,100,100,100,50", "64,32,16"):
            for solver in ("adam", "sgd", "lbfgs"):
                for activation in ("relu", "tanh", "logistic"):
                    model = models_for({"mlp": dict(hidden_layer_sizes=hidden, solver=solver,
                                                   activation=activation, learning_rate_init=0.03,
                                                   max_iter=2000, alpha=0.2, batch_size=12)}, 3, 90)["mlp"].steps[-1][1]
                    self.assertEqual(model.hidden_layer_sizes, tuple(map(int, hidden.split(","))))
                    self.assertEqual((model.solver, model.activation, model.max_iter, model.alpha), (solver, activation, 2000, 0.2))
                    if solver != "lbfgs":
                        self.assertEqual((model.learning_rate_init, model.batch_size), (0.03, 12))

    def test_bayes_priors_and_scaling_passthrough(self):
        model = models_for({"bayes": dict(priors=[0.7, 0.3], var_smoothing=1e-8, scale_features=True)}, 42, 90)["bayes"]
        self.assertEqual(model.steps[-1][1].priors, [0.7, 0.3])
        self.assertEqual(model.steps[-1][1].var_smoothing, 1e-8)

    def test_feature_aliases_and_arena_normalization_limits(self):
        for model in ("tree", "forest"):
            for value in (None, "None", "1.0", 1.0):
                self.assertEqual(normalize_parameters(model, {"max_features": value})["max_features"], "all")
        # Validation bounds public Arena input, not raw estimator construction.
        self.assertEqual(create_classifier("mlp", {"hidden_layer_sizes": "512"}).hidden_layer_sizes, (512,))
        with self.assertRaises(ValueError):
            normalize_parameters("mlp", {"hidden_layer_sizes": "512"})

    def test_unknown_and_invalid_parameters_not_silently_ignored(self):
        bad = [("svm", {"tollerance": .01}), ("svm", {"kernel": "unknown"}),
               ("bayes", {"priors": [0.8, 0.8]}), ("mlp", {"hidden_layer_sizes": "50,0"}),
               ("mlp", {"hidden_layer_sizes": [257]}), ("mlp", {"hidden_layer_sizes": [128]*9}),
               ("mlp", {"solver": "unknown"}), ("knn", {"metric": "unknown"})]
        for key, config in bad:
            with self.subTest(model=key, config=config), self.assertRaises(ValueError):
                normalize_parameters(key, config)
        with self.assertRaisesRegex(ValueError, "批大小"):
            models_for({"mlp": {"batch_size": 100}}, 42, 50)


class ActualParameterExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input = dict(points=generate({"kind": "moons", "count": 100})["points"],
                         params={"mlp": {"hidden_layer_sizes": "8", "max_iter": 100}})

    def test_poly_and_sigmoid_grids_are_real_predictions(self):
        for kernel in ("poly", "sigmoid"):
            payload = copy.deepcopy(self.input)
            payload["params"]["svm"] = dict(kernel=kernel, degree=4, coef0=0.4, gamma="scale", tol=0.002, max_iter=1500)
            result, models, X, mesh = fit_experiment(payload, keep_models=True)
            np.testing.assert_array_equal(result["models"]["svm"]["grid"], models["svm"].predict(mesh))
            effective = result["models"]["svm"]["effective_params"]
            self.assertEqual((effective["kernel"], effective["max_iter"], effective["tol"]), (kernel, 1500, 0.002))

    def test_scaled_tree_paths_are_reported_in_world_coordinates(self):
        payload = copy.deepcopy(self.input)
        payload["params"]["tree"] = dict(scale_features=True, criterion="entropy", max_features="all")
        result, models, X, _ = fit_experiment(payload, keep_models=True)
        for i, p in enumerate(result["points"]):
            path = result["models"]["tree"]["explanations"][p["id"]]["path"]
            for step in path:
                self.assertEqual(step["direction"], "≤" if X[i, step["feature"]] <= step["threshold"] else ">")
        self.assertTrue(result["models"]["tree"]["preprocessing"]["standardize"])

    def test_knn_raw_manhattan_neighbors_match_selected_metric(self):
        payload = copy.deepcopy(self.input)
        payload["params"]["knn"] = dict(metric="manhattan", scale_features=False, n_neighbors=3, weights="distance")
        result, models, X, _ = fit_experiment(payload, keep_models=True)
        distances, indices = models["knn"].kneighbors(X)
        for i, p in enumerate(result["points"]):
            neighbors = result["models"]["knn"]["explanations"][p["id"]]["neighbors"]
            np.testing.assert_allclose([n["distance"] for n in neighbors], distances[i])
        self.assertFalse(result["models"]["knn"]["preprocessing"]["standardize"])

    def test_lbfgs_does_not_fabricate_a_loss_curve(self):
        payload = copy.deepcopy(self.input)
        payload["params"]["mlp"].update(solver="lbfgs", activation="tanh", hidden_layer_sizes="8,4")
        result = fit_experiment(payload)
        info = result["models"]["mlp"]["fit_info"]
        self.assertEqual(info["solver"], "lbfgs")
        self.assertEqual(info["network"], [2, 8, 4, 1])
        self.assertEqual(info["loss_curve"], [])
        self.assertGreaterEqual(info["final_loss"], 0)

    def test_perceptron_has_decision_scores_not_fake_probabilities(self):
        payload = copy.deepcopy(self.input)
        payload["params"]["mlp"].update(network_type="perceptron", learning_rate_init=0.02)
        result = fit_experiment(payload)
        self.assertEqual(result["models"]["mlp"]["fit_info"]["network"], [2, 1])
        self.assertEqual(result["models"]["mlp"]["effective_params"]["eta0"], 0.02)
        for explanation in result["models"]["mlp"]["explanations"].values():
            self.assertIn("decision_score", explanation)
            self.assertNotIn("probabilities", explanation)


class ModuleIsolationTests(unittest.TestCase):
    def test_native_labs_do_not_depend_on_arena_constructor(self):
        from backend.algorithms import svm, knn, decision_tree, bayesian_classification, neural_network, ensemble
        for module in (svm, knn, decision_tree, bayesian_classification, neural_network, ensemble):
            self.assertFalse(hasattr(module, 'create_classifier'))

    def test_arena_svm_honors_iteration_limit_and_tolerance(self):
        points = generate({"count": 40})["points"]
        X, y = np.array([[p["x"], p["y"]] for p in points]), np.array([p["label"] for p in points])
        model = models_for({'svm': dict(kernel="poly", gamma="scale", degree=4,
                                       coef0=0.3, max_iter=2200, tol=0.002)}, 42, len(X))['svm']
        model.fit(X, y)
        estimator, _, _ = estimator_input(model, X)
        self.assertEqual((estimator.kernel, estimator.max_iter, estimator.tol),
                         ("poly", 2200, 0.002))


if __name__ == "__main__":
    unittest.main()
