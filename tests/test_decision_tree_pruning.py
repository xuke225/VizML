import unittest
import json
from dataclasses import replace

import numpy as np

from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from app import app, session_data
from backend.algorithms.decision_tree_pruning import (
    DecisionTreePruningEngine,
    PruningConfig,
)


class DecisionTreePruningEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = make_classification(
            n_samples=180,
            n_features=6,
            n_informative=4,
            n_redundant=0,
            class_sep=0.9,
            flip_y=0.08,
            random_state=42,
        )
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            cls.X,
            cls.y,
            test_size=0.25,
            stratify=cls.y,
            random_state=42,
        )

    def make_engine(self):
        return DecisionTreePruningEngine(random_state=42)

    def test_pre_pruning_respects_complexity_limits(self):
        model, info = self.make_engine().fit(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            PruningConfig(
                method='pre',
                max_depth=2,
                max_leaf_nodes=3,
                min_samples_leaf=4,
                cv_folds=3,
            ),
        )
        self.assertLessEqual(model.get_depth(), 2)
        self.assertLessEqual(model.get_n_leaves(), 3)
        self.assertEqual(info['method'], 'pre')
        self.assertEqual(len(info['comparison']), 3)

    def test_auto_post_pruning_returns_selection_path(self):
        model, info = self.make_engine().fit(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            PruningConfig(method='post_auto', cv_folds=3),
        )
        self.assertGreaterEqual(info['selected_alpha'], 0)
        self.assertGreater(len(info['path']), 1)
        self.assertEqual(sum(point.get('selected', False) for point in info['path']), 1)
        self.assertEqual(model.tree_.node_count, info['selected_model']['node_count'])
        self.assertIn('one-standard-error', info['selection_rule'])

    def test_auto_selection_uses_standard_error(self):
        _, info = self.make_engine().fit(
            self.X_train, self.y_train, self.X_test, self.y_test,
            PruningConfig(method='post_auto', cv_folds=3),
        )
        best = max(info['path'], key=lambda point: point['cv_mean'])
        self.assertAlmostEqual(best['cv_se'], best['cv_std'] / np.sqrt(3))
        threshold = best['cv_mean'] - best['cv_se']
        expected = max(point['alpha'] for point in info['path']
                       if point['cv_mean'] >= threshold)
        self.assertAlmostEqual(info['selected_alpha'], expected)

    def test_test_labels_do_not_influence_auto_selection(self):
        config = PruningConfig(method='post_auto', cv_folds=3)
        _, original = self.make_engine().fit(
            self.X_train, self.y_train, self.X_test, self.y_test, config,
        )
        _, changed = self.make_engine().fit(
            self.X_train, self.y_train, self.X_test, 1 - self.y_test, config,
        )
        self.assertEqual(original['selected_alpha'], changed['selected_alpha'])

    def test_legacy_pre_predictions_match_sklearn(self):
        model, info = self.make_engine().fit(
            self.X_train, self.y_train, self.X_test, self.y_test,
            PruningConfig(method='pre', max_depth=3, cv_folds=3),
        )
        legacy = DecisionTreeClassifier(max_depth=3, random_state=42)
        legacy.fit(self.X_train, self.y_train)
        np.testing.assert_array_equal(model.predict(self.X_test), legacy.predict(self.X_test))
        self.assertEqual(info['selected_alpha'], 0)

    def test_path_complexity_decreases_with_alpha(self):
        _, info = self.make_engine().fit(
            self.X_train, self.y_train, self.X_test, self.y_test,
            PruningConfig(method='post_auto', cv_folds=3),
        )
        sizes = [point['node_count'] for point in info['path']]
        self.assertEqual(sizes, sorted(sizes, reverse=True))

    def test_rare_class_falls_back_without_nan_or_warning(self):
        _, info = self.make_engine().fit(
            np.arange(20).reshape(10, 2), np.array([0] * 9 + [1]),
            np.array([[1, 2], [3, 4]]), np.array([0, 1]),
            PruningConfig(method='post_auto'),
        )
        self.assertEqual(info['cv_folds'], 0)
        self.assertEqual(info['selected_alpha'], 0)
        self.assertIsNone(info['path'][0]['cv_mean'])

    def test_invalid_parameters_are_rejected(self):
        for field, value in [('ccp_alpha', np.nan), ('ccp_alpha', np.inf),
                             ('ccp_alpha', -0.1), ('max_leaf_nodes', 2.5),
                             ('cv_folds', 11), ('min_samples_leaf', 0)]:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.make_engine().fit(
                    self.X_train, self.y_train, self.X_test, self.y_test,
                    replace(PruningConfig(), **{field: value}),
                )

    def test_manual_post_pruning_uses_requested_alpha(self):
        _, info = self.make_engine().fit(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            PruningConfig(method='post_manual', ccp_alpha=0.01, cv_folds=3),
        )
        self.assertAlmostEqual(info['selected_alpha'], 0.01)
        self.assertEqual(info['method_label'], '手动后剪枝')

    def test_manual_alpha_is_an_exact_curve_point(self):
        model, info = self.make_engine().fit(
            self.X_train, self.y_train, self.X_test, self.y_test,
            PruningConfig(method='post_manual', ccp_alpha=0.012345, cv_folds=3),
        )
        selected = next(point for point in info['path'] if point.get('selected'))
        self.assertEqual(selected['alpha'], 0.012345)
        self.assertEqual(selected['node_count'], model.tree_.node_count)


class PruningTreeComparisonTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_classification(
            n_samples=240, n_features=5, n_informative=4, n_redundant=0,
            flip_y=0.08, random_state=17,
        )
        self.baseline = DecisionTreeClassifier(random_state=42).fit(self.X, self.y)

    def compare(self, selected, method='post_manual'):
        result = DecisionTreePruningEngine._tree_comparison(self.baseline, selected, method)
        json.dumps(result, allow_nan=False)
        nodes = {node['id']: node for node in result['nodes']}
        self.assertEqual(len(nodes), self.baseline.tree_.node_count)
        self.assertEqual(sum(result['counts'].values()), len(nodes))
        actual = result['selected_nodes']
        self.assertEqual(len(actual), selected.tree_.node_count)
        self.assertEqual(sorted(n['id'] for n in actual), list(range(selected.tree_.node_count)))
        for node in actual:
            i = node['id']
            self.assertEqual(node['selected_id'], i)
            self.assertEqual(node['samples'], selected.tree_.n_node_samples[i])
            self.assertAlmostEqual(node['impurity'], selected.tree_.impurity[i])
            self.assertEqual(node['is_leaf'], selected.tree_.children_left[i] == -1)
            if node['parent'] is not None:
                children = selected.tree_.children_left if node['side'] == 'left' else selected.tree_.children_right
                self.assertEqual(children[node['parent']], i)
            if not node['is_leaf']:
                self.assertEqual(node['feature'], selected.tree_.feature[i])
                self.assertAlmostEqual(node['threshold'], selected.tree_.threshold[i])
        for node in nodes.values():
            if node['parent'] is not None:
                self.assertEqual(node['depth'], nodes[node['parent']]['depth'] + 1)
            if node['status'] == 'removed':
                self.assertIn(nodes[node['parent']]['status'], ('cut', 'removed'))
                self.assertIsNone(node['selected_id'])
        return result

    def test_no_pruning_has_no_gray_nodes(self):
        result = self.compare(self.baseline, 'none')
        self.assertEqual(result['counts']['removed'], 0)
        self.assertEqual(result['counts']['cut'], 0)
        self.assertTrue(result['exact_subtree'])
        self.assertEqual(result['counts']['retained'], self.baseline.tree_.node_count)

    def test_post_pruning_maps_renumbered_nodes_and_exact_counts(self):
        for alpha in (0, 0.005, 0.02, 1.0):
            with self.subTest(alpha=alpha):
                selected = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha).fit(self.X, self.y)
                before = selected.predict(self.X)
                result = self.compare(selected)
                self.assertTrue(result['exact_subtree'])
                self.assertEqual(result['counts']['removed'],
                                 self.baseline.tree_.node_count - selected.tree_.node_count)
                mapped = [n['selected_id'] for n in result['nodes'] if n['selected_id'] is not None]
                self.assertEqual(sorted(mapped), list(range(selected.tree_.node_count)))
                np.testing.assert_array_equal(before, selected.predict(self.X))

    def test_root_pruned_to_leaf_keeps_root_and_grays_descendants(self):
        selected = DecisionTreeClassifier(random_state=42, ccp_alpha=1).fit(self.X, self.y)
        result = self.compare(selected)
        self.assertEqual(result['nodes'][0]['status'], 'cut')
        self.assertEqual(result['selected_node_count'], 1)
        self.assertEqual(result['counts']['removed'], self.baseline.tree_.node_count - 1)

    def test_pre_pruning_explains_stopped_growth(self):
        selected = DecisionTreeClassifier(random_state=42, max_depth=1).fit(self.X, self.y)
        result = self.compare(selected, 'pre')
        self.assertEqual(result['removed_label'], '未继续生长')
        self.assertEqual(result['cut_label'], '停止分裂')
        self.assertTrue(result['exact_subtree'])
        self.assertEqual(result['counts']['removed'], self.baseline.tree_.node_count - 3)

    def test_changed_split_is_not_falsely_called_pruning(self):
        selected = DecisionTreeClassifier(random_state=42).fit(self.X, self.y)
        selected.tree_.threshold[0] += 0.125
        result = self.compare(selected, 'pre')
        self.assertFalse(result['exact_subtree'])
        self.assertEqual(result['nodes'][0]['status'], 'changed')
        self.assertEqual(result['counts']['removed'], 0)
        self.assertEqual(result['counts']['unmatched'], self.baseline.tree_.node_count - 1)

    def test_best_first_pre_pruning_does_not_match_by_node_id(self):
        selected = DecisionTreeClassifier(random_state=42, max_leaf_nodes=8).fit(self.X, self.y)
        result = self.compare(selected, 'pre')
        for node in result['nodes']:
            if node['selected_id'] is not None:
                self.assertEqual(node['samples'], selected.tree_.n_node_samples[node['selected_id']])
                if node['status'] == 'retained' and not node['is_leaf']:
                    self.assertEqual(node['feature'], selected.tree_.feature[node['selected_id']])

    def test_single_leaf_baseline_is_valid(self):
        self.baseline = DecisionTreeClassifier().fit(self.X, np.zeros(len(self.X), dtype=int))
        result = self.compare(self.baseline, 'none')
        self.assertEqual(len(result['nodes']), 1)
        self.assertEqual(result['counts']['removed'], 0)
        self.assertTrue(result['nodes'][0]['is_leaf'])

    def test_reported_91_to_29_case_exports_all_29_not_only_four_matches(self):
        X, y = load_iris(return_X_y=True)
        X_train, _, y_train, _ = train_test_split(X[:, :2], y, test_size=0.2, random_state=42)
        self.baseline = DecisionTreeClassifier(random_state=42, max_features='sqrt').fit(X_train, y_train)
        selected = DecisionTreeClassifier(random_state=42, max_features='sqrt',
                                          max_depth=5, max_leaf_nodes=15).fit(X_train, y_train)
        result = self.compare(selected, 'pre')
        self.assertEqual(result['baseline_node_count'], 91)
        self.assertEqual(result['selected_node_count'], 29)
        self.assertEqual(result['counts']['retained'], 4)
        self.assertEqual(result['counts']['removed'], 0)
        self.assertFalse(result['exact_subtree'])
        self.assertEqual(len(result['selected_nodes']), 29)
        self.assertEqual(sum(n['baseline_id'] is None for n in result['selected_nodes']), 25)

    def test_changed_root_still_exports_the_complete_selected_tree(self):
        selected = DecisionTreeClassifier(random_state=42, max_depth=3).fit(self.X, self.y)
        selected.tree_.threshold[0] += 0.125
        result = self.compare(selected, 'pre')
        self.assertEqual(result['counts']['retained'], 0)
        self.assertEqual(len(result['selected_nodes']), selected.tree_.node_count)
        self.assertTrue(all(n['baseline_id'] is None for n in result['selected_nodes']))


class DecisionTreePruningApiTests(unittest.TestCase):
    def setUp(self):
        app.config.update(TESTING=True)
        self.client = app.test_client()
        session_data['datasets'].clear()
        session_data['models'].clear()
        X, y = make_classification(
            n_samples=120,
            n_features=4,
            n_informative=3,
            n_redundant=0,
            random_state=7,
        )
        session_data['datasets']['pruning_test'] = {
            'X': X.tolist(),
            'y': y.tolist(),
            'feature_names': ['特征A', '特征B', '特征C', '特征D'],
        }

    def test_training_api_exposes_pruning_analysis(self):
        response = self.client.post('/api/decision_tree/train', json={
            'session_key': 'pruning_test',
            'params': {
                'pruning_method': 'post_auto',
                'cv_folds': 3,
                'test_size': 0.25,
                'random_state': 42,
            },
        })
        self.assertEqual(response.status_code, 200)
        results = response.get_json()['results']
        self.assertIn('pruning_info', results)
        self.assertEqual(results['pruning_info']['method'], 'post_auto')
        self.assertEqual(len(results['pruning_info']['comparison']), 3)
        self.assertIn('node_count', results['training_info'])
        graph = results['pruning_info']['tree_comparison']
        self.assertEqual(len(graph['nodes']), graph['baseline_node_count'])
        self.assertEqual(graph['selected_node_count'], results['training_info']['node_count'])
        self.assertEqual(len(graph['selected_nodes']), results['training_info']['node_count'])
        self.assertTrue(graph['exact_subtree'])

    def test_page_contains_pruning_workbench(self):
        with self.client.get('/html/decision_tree.html') as response:
            self.assertEqual(response.status_code, 200)
            page = response.get_data(as_text=True)
        self.assertIn('id="pruningLabTitle"', page)
        self.assertIn('value="post_auto"', page)
        self.assertIn('id="pruningPathCanvas"', page)
        self.assertIn('/static/js/decision-tree-pruning.js', page)

    def test_all_modes_support_prediction_and_decision_path(self):
        for method in ['none', 'pre', 'post_auto', 'post_manual']:
            with self.subTest(method=method):
                trained = self.client.post('/api/decision_tree/train', json={
                    'session_key': 'pruning_test',
                    'params': {'pruning_method': method, 'max_depth': 3,
                               'ccp_alpha': 0.01, 'cv_folds': 3},
                })
                self.assertEqual(trained.status_code, 200)
                payload = trained.get_json()
                X = session_data['datasets']['pruning_test']['X'][:2]
                request = {'model_key': payload['model_key'], 'X': X}
                predicted = self.client.post('/api/decision_tree/predict', json=request)
                path = self.client.post('/api/decision_tree/decision_path', json=request)
                self.assertEqual(predicted.status_code, 200)
                self.assertEqual(path.status_code, 200)
                self.assertEqual(predicted.get_json()['predictions'], path.get_json()['predictions'])
                info = payload['results']['pruning_info']
                self.assertEqual(sum(item['selected'] for item in info['comparison']), 1)

    def test_api_rejects_invalid_new_parameters(self):
        for params in [{'pruning_method': 'wrong'}, {'max_leaf_nodes': 2.5},
                       {'ccp_alpha': -1}, {'min_impurity_decrease': -0.1}]:
            with self.subTest(params=params):
                response = self.client.post('/api/decision_tree/train', json={
                    'session_key': 'pruning_test', 'params': params,
                })
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.get_json()['status'], 'error')


if __name__ == '__main__':
    unittest.main()
