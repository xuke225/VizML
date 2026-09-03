"""
机器学习算法模块
重构后实现前后端一一对应的高内聚低耦合架构
"""

from .base import BaseAlgorithm, BaseClassificationAlgorithm, BaseRegressionAlgorithm

# 独立的分类算法模块
from .svm import SVMAlgorithm
from .decision_tree import DecisionTreeAlgorithm
from .knn import KNNAlgorithm
from .bayesian_classification import BayesianClassificationAlgorithm

# 独立的回归算法模块
from .linear_regression import LinearRegressionAlgorithm
from .sgd import SGDAlgorithm

# 保持现有的算法模块
from .clustering import ClusteringAlgorithms
from .ensemble import EnsembleAlgorithms
from .neural_network import NeuralNetworkAlgorithms
from .dimensionality_reduction import DimensionalityReductionAlgorithms

__all__ = [
    # 基类
    'BaseAlgorithm',
    'BaseClassificationAlgorithm', 
    'BaseRegressionAlgorithm',
    
    # 独立分类算法
    'SVMAlgorithm',
    'DecisionTreeAlgorithm',
    'KNNAlgorithm',
    'BayesianClassificationAlgorithm',
    
    # 独立回归算法
    'LinearRegressionAlgorithm',
    'SGDAlgorithm',
    
    # 现有算法模块
    'ClusteringAlgorithms',
    'EnsembleAlgorithms',
    'NeuralNetworkAlgorithms',
    'DimensionalityReductionAlgorithms'
]