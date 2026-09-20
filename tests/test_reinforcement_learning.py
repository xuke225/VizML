"""Regression checks for the grid-world training data used by the UI."""

import unittest

from backend.algorithms.reinforcement_learning import (
    ReinforcementLearningAlgorithm,
    build_environment,
)


class ReinforcementLearningTests(unittest.TestCase):
    def test_wind_uses_the_current_column(self):
        env = build_environment('windy_grid')
        self.assertEqual(env.step((3, 3), 1), ((2, 4), -1.0, False))
        self.assertEqual(env.step((3, 6), 1), ((1, 7), -1.0, False))
        self.assertEqual(env.wind, [0, 0, 0, 1, 1, 1, 2, 2, 1, 0])

    def test_checkpoints_have_their_own_policy_and_metrics(self):
        algorithm = ReinforcementLearningAlgorithm()
        result = algorithm.train('windy_grid', episodes=500, alpha=0.3)
        checkpoints = result['checkpoints']
        self.assertEqual(checkpoints[0]['episode'], 1)
        self.assertEqual(checkpoints[-1]['episode'], 500)
        self.assertEqual(result['params']['epsilon_start'], 1.0)
        self.assertTrue(result['stats']['path_reached_goal'])
        self.assertNotEqual(checkpoints[0]['policy'], checkpoints[-1]['policy'])
        for checkpoint in checkpoints:
            self.assertEqual(len(checkpoint['policy']), result['grid']['rows'])
            self.assertTrue(0 <= checkpoint['success_rate'] <= 1)
            self.assertTrue(0 <= checkpoint['epsilon'] <= 1)
            self.assertTrue(checkpoint['greedy_path'])


if __name__ == '__main__':
    unittest.main()
