import unittest
import numpy as np
from sklearn.datasets import make_blobs
from backend.algorithms.clustering import ClusteringAlgorithms


class ClusteringHistoryTests(unittest.TestCase):
    def setUp(self):
        self.X, _ = make_blobs(n_samples=50, centers=3, random_state=42)
        self.alg = ClusteringAlgorithms()

    def check_history(self, algorithm, **params):
        model, labels, centers, info = self.alg.train_model(self.X, algorithm, record_history=True, **params)
        history = info['history']
        self.assertGreater(len(history), 2)
        np.testing.assert_array_equal(history[-1]['labels'], labels)
        if centers is not None:
            np.testing.assert_allclose(history[-1]['centers'], centers)
        self.assertTrue(all(np.isfinite(f['metric']) and len(f['labels']) == len(self.X) for f in history))
        return model, history

    def test_kmeans_iteration_limit_does_not_add_hidden_update(self):
        model, h = self.check_history('kmeans', max_iter=1, init='random')
        update = [f for f in h if f['title'] == '更新质心'][-1]
        np.testing.assert_allclose(h[-1]['centers'], update['centers'])
        np.testing.assert_array_equal(model.predict(self.X), h[-1]['labels'])
        self.assertEqual(model.n_iter_, 1)

    def test_dbscan_expansion_and_noise(self):
        for eps in [.001, 1.5, 100]:
            with self.subTest(eps=eps):
                model, h = self.check_history('dbscan', eps=eps)
                self.assertEqual(h[0]['labels'], [-2] * len(self.X))
                self.assertTrue(all(b['metric'] >= a['metric'] for a, b in zip(h, h[1:])))
                self.assertEqual(h[-1]['metric'], int(np.sum(model.labels_ >= 0)))

    def test_hierarchical_actual_merges(self):
        for linkage in ['ward', 'complete', 'average', 'single']:
            with self.subTest(linkage=linkage):
                _, h = self.check_history('hierarchical', n_clusters=3, linkage=linkage)
                self.assertEqual([f['metric'] for f in h[:-1]], list(range(len(self.X), 2, -1)))
                self.assertEqual(h[-2]['labels'], h[-1]['labels'])

    def test_gmm_preserves_sklearn_fit(self):
        for covariance_type in ['full', 'tied', 'diag', 'spherical']:
            with self.subTest(covariance_type=covariance_type):
                params = dict(covariance_type=covariance_type, max_iter=10, init_params='random')
                model, h = self.check_history('gmm', **params)
                reference, labels, centers, _ = self.alg.train_model(self.X, 'gmm', **params)
                np.testing.assert_allclose(model.means_, reference.means_)
                np.testing.assert_array_equal(h[-1]['labels'], labels)
                self.assertTrue(all(np.asarray(f['covariances']).shape == (3, 2, 2) for f in h))
                self.assertTrue(all(b['metric'] >= a['metric'] - 1e-7 for a, b in zip(h, h[1:])))


if __name__ == '__main__':
    unittest.main()
