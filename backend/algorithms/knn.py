"""
K近邻(KNN)算法模块
基于 sklearn 实现 KNN 分类算法
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import logging

from .base import BaseClassificationAlgorithm

logger = logging.getLogger(__name__)


class KNNAlgorithm(BaseClassificationAlgorithm):
    """KNN算法类"""
    
    def __init__(self):
        super().__init__()
    
    def train_model(self, X, y, n_neighbors=5, weights='uniform', metric='minkowski', 
                   p=2, scale_features=True, **kwargs):
        """
        训练KNN模型
        
        Args:
            X: 特征矩阵
            y: 标签向量
            n_neighbors: 邻居数量
            weights: 权重函数 ('uniform', 'distance')
            metric: 距离度量 ('minkowski', 'euclidean', 'manhattan', 'chebyshev')
            p: Minkowski度量的参数 (p=1为曼哈顿距离，p=2为欧几里得距离)
            scale_features: 是否标准化特征
            **kwargs: 其他参数
            
        Returns:
            result: 训练结果字典
        """
        logger.info(f"训练KNN模型: n_neighbors={n_neighbors}, weights={weights}, metric={metric}")
        
        # 数据分割
        test_size = kwargs.get('test_size', 0.2)
        indices = np.arange(len(X))
        X_train, X_test, y_train, y_test, train_indices, test_indices = \
            self._split_data_with_indices(X, y, indices, test_size=test_size)
        
        # 数据标准化（KNN对特征尺度敏感，通常需要标准化）
        if scale_features:
            X_train, X_test = self.scale_features(X_train, X_test)
        
        # 创建并训练模型
        model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            p=p
        )
        model.fit(X_train, y_train)
        
        # 预测和评估
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # 计算评估指标
        train_metrics = self.evaluate_classification(y_train, y_pred_train)
        test_metrics = self.evaluate_classification(y_test, y_pred_test)
        
        metrics = {
            f'train_{k}': v for k, v in train_metrics.items()
        }
        metrics.update({
            f'test_{k}': v for k, v in test_metrics.items()
        })
        
        # 获取KNN特有信息
        training_info = {
            'n_neighbors': n_neighbors,
            'weights': weights,
            'metric': metric,
            'p': p,
            'effective_metric': model.effective_metric_,
            'effective_metric_params': model.effective_metric_params_
        }
        
        # 获取模型参数和决策边界（用于可视化）
        model_params = self.get_model_params(model)
        decision_boundary = None
        if X.shape[1] == 2:  # 只有2D数据才计算决策边界
            decision_boundary = self.get_decision_boundary(model, X, y)
        
        # 计算混淆矩阵
        train_confusion = self.get_confusion_matrix(y_train, y_pred_train)
        test_confusion = self.get_confusion_matrix(y_test, y_pred_test)
        
        # 创建带标识的完整数据集用于前端显示
        from .base import create_full_data_with_labels
        full_data_with_labels = create_full_data_with_labels(
            X, y, train_indices, test_indices, y_pred_train, y_pred_test
        )
        
        # 返回完整的训练结果
        result = {
            'model': model,
            'metrics': metrics,
            'predictions': {
                'train': y_pred_train.tolist(),
                'test': y_pred_test.tolist()
            },
            'full_data_with_labels': full_data_with_labels,
            'training_info': training_info,
            'algorithm_params': model_params,
            'decision_boundary': decision_boundary,
            'confusion_matrices': {
                'train': train_confusion,
                'test': test_confusion
            },
            'data_info': {
                'train_size': len(X_train),
                'test_size': len(X_test),
                'n_features': X.shape[1],
                'n_classes': len(np.unique(y))
            },
            # 保存完整数据和分割索引以供后续使用
            'train_data': {'X': X_train.tolist(), 'y': y_train.tolist()},
            'test_data': {'X': X_test.tolist(), 'y': y_test.tolist()},
            'full_data': {'X': X.tolist(), 'y': y.tolist()},
            'scaler': self.scaler if scale_features else None
        }
        
        return result
    
    def _split_data_with_indices(self, X, y, indices, test_size=0.2):
        """数据分割并返回索引"""
        from sklearn.model_selection import train_test_split
        return train_test_split(X, y, indices, test_size=test_size, random_state=42)
    
    def get_model_params(self, model):
        """
        获取KNN模型参数信息
        
        Args:
            model: 训练好的KNN模型
            
        Returns:
            params: 模型参数字典
        """
        params = {
            'n_neighbors': model.n_neighbors,
            'weights': model.weights,
            'metric': model.metric,
            'p': getattr(model, 'p', None),
            'effective_metric': model.effective_metric_,
            'effective_metric_params': model.effective_metric_params_
        }
        
        return params
    
    def predict_with_proba(self, model, X, scaler=None):
        """
        使用训练好的KNN模型进行预测
        
        Args:
            model: 训练好的模型
            X: 特征矩阵
            scaler: 数据标准化器（如果之前使用过）
            
        Returns:
            result: 预测结果字典
        """
        # 数据标准化（如果之前使用过）
        if scaler is not None:
            X = scaler.transform(X)
        
        # 预测
        predictions = model.predict(X)
        probabilities = None
        
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(X)
            except Exception as e:
                logger.warning(f"无法获取预测概率: {e}")
        
        result = {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist() if probabilities is not None else None
        }
        
        return result
    
    def get_neighbors_info(self, model, X, k=None, scaler=None):
        """
        获取指定样本的邻居信息
        
        Args:
            model: 训练好的KNN模型
            X: 查询样本的特征矩阵
            k: 返回的邻居数量（默认使用模型的n_neighbors）
            scaler: 数据标准化器
            
        Returns:
            neighbors_info: 邻居信息字典
        """
        if scaler is not None:
            X = scaler.transform(X)
        
        if k is None:
            k = model.n_neighbors
        
        # 获取最近邻距离和索引
        distances, indices = model.kneighbors(X, n_neighbors=k)
        
        neighbors_info = []
        for i, sample in enumerate(X):
            sample_neighbors = {
                'sample_index': i,
                'distances': distances[i].tolist(),
                'neighbor_indices': indices[i].tolist(),
                'neighbor_labels': []
            }
            
            # 获取邻居的标签
            neighbor_labels = []
            for neighbor_idx in indices[i]:
                neighbor_label = model._y[neighbor_idx]  # 获取邻居的真实标签
                neighbor_labels.append(int(neighbor_label))
            
            sample_neighbors['neighbor_labels'] = neighbor_labels
            neighbors_info.append(sample_neighbors)
        
        return neighbors_info
    
    def evaluate_k_values(self, X_train, y_train, X_test, y_test, k_range=None, scaler=None):
        """
        评估不同k值的性能
        
        Args:
            X_train: 训练集特征
            y_train: 训练集标签
            X_test: 测试集特征
            y_test: 测试集标签
            k_range: k值范围
            scaler: 数据标准化器
            
        Returns:
            k_evaluation: k值评估结果
        """
        if k_range is None:
            max_k = min(20, len(X_train) // 2)
            k_range = range(1, max_k + 1, 2)  # 奇数k值
        
        if scaler is not None:
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        
        k_results = []
        
        for k in k_range:
            # 创建并训练模型
            temp_model = KNeighborsClassifier(n_neighbors=k)
            temp_model.fit(X_train, y_train)
            
            # 预测和评估
            y_pred_train = temp_model.predict(X_train)
            y_pred_test = temp_model.predict(X_test)
            
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            
            k_results.append({
                'k': k,
                'train_accuracy': float(train_accuracy),
                'test_accuracy': float(test_accuracy),
                'overfitting': float(train_accuracy - test_accuracy)
            })
        
        return {
            'k_range': list(k_range),
            'results': k_results,
            'best_k': max(k_results, key=lambda x: x['test_accuracy'])['k']
        }
    
    def visualize_decision_regions(self, model, X, y, resolution=100, scaler=None):
        """
        可视化KNN决策区域（针对2D数据）
        
        Args:
            model: 训练好的KNN模型
            X: 特征矩阵
            y: 标签向量
            resolution: 网格分辨率
            scaler: 数据标准化器
            
        Returns:
            visualization_data: 可视化数据
        """
        if X.shape[1] != 2:
            logger.warning("只支持2D数据的决策区域可视化")
            return None
        
        if scaler is not None:
            X = scaler.transform(X)
        
        # 创建网格
        margin = 0.5
        x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
        y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
        
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution)
        )
        
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        # 预测网格点
        predictions = model.predict(grid_points)
        probabilities = model.predict_proba(grid_points)
        
        visualization_data = {
            'xx': xx.tolist(),
            'yy': yy.tolist(),
            'predictions': predictions.reshape(xx.shape).tolist(),
            'probabilities': probabilities.reshape(xx.shape[0], xx.shape[1], -1).tolist(),
            'x_range': [float(x_min), float(x_max)],
            'y_range': [float(y_min), float(y_max)],
            'resolution': resolution
        }
        
        return visualization_data