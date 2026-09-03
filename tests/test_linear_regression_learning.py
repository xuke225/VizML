import unittest

import numpy as np

from backend.algorithms.linear_regression import LinearRegressionAlgorithm


class LinearRegressionLearningDataTests(unittest.TestCase):
    def setUp(self):
        self.X = np.linspace(-3, 3, 24).reshape(-1, 1)
        self.y = 2.5 * self.X[:, 0] - 1.25

    def train(self, **kwargs):
        return LinearRegressionAlgorithm().train_model(
            self.X,
            self.y,
            algorithm='linear',
            optimizer='ols',
            test_size=0.25,
            **kwargs,
        )

    def test_point_results_are_aligned_to_original_data(self):
        result = self.train()
        points = result['point_results']

        self.assertEqual(len(points), len(self.X))
        self.assertEqual([point['index'] for point in points], list(range(len(self.X))))
        self.assertEqual({point['split'] for point in points}, {'train', 'test'})

        for index, point in enumerate(points):
            self.assertEqual(point['features'], self.X[index].tolist())
            self.assertAlmostEqual(point['actual'], self.y[index])
            self.assertAlmostEqual(
                point['residual'],
                point['actual'] - point['predicted'],
            )
            self.assertAlmostEqual(point['squared_error'], point['residual'] ** 2)

    def test_loss_summary_matches_point_results(self):
        result = self.train()
        points = result['point_results']
        summary = result['loss_summary']
        squared_errors = np.array([point['squared_error'] for point in points])

        self.assertAlmostEqual(summary['sse'], squared_errors.sum())
        self.assertAlmostEqual(summary['mse'], squared_errors.mean())
        self.assertAlmostEqual(summary['mean_target'], self.y.mean())
        self.assertAlmostEqual(summary['tss'], np.sum((self.y - self.y.mean()) ** 2))
        self.assertAlmostEqual(summary['r2'], 1.0)

    def test_full_predictions_remain_aligned_when_features_are_scaled(self):
        result = self.train(scale_features=True)

        for point in result['point_results']:
            self.assertAlmostEqual(point['predicted'], point['actual'], places=9)

    def test_feature_names_are_preserved_in_multivariable_equation(self):
        X = np.column_stack([self.X[:, 0], self.X[:, 0] ** 2])
        result = LinearRegressionAlgorithm().train_model(
            X,
            self.y,
            algorithm='linear',
            optimizer='ols',
            feature_info={'features_used': ['长度', '长度平方']},
        )
        self.assertEqual(result['training_info']['feature_names'], ['长度', '长度平方'])
        self.assertIn('长度', result['model_equation'])


if __name__ == '__main__':
    unittest.main()
