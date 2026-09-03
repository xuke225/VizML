"""
支持向量机(SVM)算法模块
基于 sklearn 实现 SVM 分类算法
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import logging

from .base import BaseClassificationAlgorithm

logger = logging.getLogger(__name__)


class SVMAlgorithm(BaseClassificationAlgorithm):
    """SVM算法类"""
    
    def __init__(self):
        super().__init__()
    
    def train_model(self, X, y, kernel='rbf', C=1.0, gamma='scale', degree=3, coef0=0.0, 
                   scale_features=True, **kwargs):
        """
        训练SVM模型
        
        Args:
            X: 特征矩阵
            y: 标签向量
            kernel: 核函数类型 ('linear', 'poly', 'rbf', 'sigmoid')
            C: 正则化参数
            gamma: 核函数系数
            degree: 多项式核的次数
            coef0: 核函数中的独立项
            scale_features: 是否标准化特征
            **kwargs: 其他参数
            
        Returns:
            model: 训练好的模型
            training_info: 训练信息
        """
        logger.info(f"训练SVM模型: kernel={kernel}, C={C}, gamma={gamma}")
        
        # 数据分割
        test_size = kwargs.get('test_size', 0.2)
        indices = np.arange(len(X))
        X_train, X_test, y_train, y_test, train_indices, test_indices = \
            self._split_data_with_indices(X, y, indices, test_size=test_size)
        
        # 数据标准化（如果需要）
        if scale_features:
            X_train, X_test = self.scale_features(X_train, X_test)
        
        # 处理gamma参数
        if isinstance(gamma, str) and gamma == 'scale':
            gamma = 1.0 / (X.shape[1] * X.var())
        elif isinstance(gamma, str) and gamma == 'auto':
            gamma = 1.0 / X.shape[1]
        
        # 创建并训练模型
        model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            degree=int(degree),
            coef0=float(coef0),
            probability=True,  # 启用概率预测
            random_state=42
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
        
        # 添加SVM特有指标
        if hasattr(model, 'support_'):
            metrics['n_support_vectors'] = int(len(model.support_))
            metrics['support_vector_ratio'] = float(len(model.support_) / len(X_train))
        
        # 获取支持向量的详细信息
        support_vectors = X_train[model.support_] if hasattr(model, 'support_') else None
        training_info = {
            'n_support_vectors': int(len(model.support_)) if hasattr(model, 'support_') else 0,
            'support_vectors': support_vectors.tolist() if support_vectors is not None else None,
            'support_indices': model.support_.tolist() if hasattr(model, 'support_') else None,
            'n_support': model.n_support_.tolist() if hasattr(model, 'n_support_') else None,
            'kernel': kernel,
            'C': C,
            'gamma': gamma,
            'degree': degree,
            'coef0': coef0
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
        获取SVM模型参数信息
        
        Args:
            model: 训练好的SVM模型
            
        Returns:
            params: 模型参数字典
        """
        # 获取支持向量的完整信息
        support_vectors = model.support_vectors_.tolist() if hasattr(model, 'support_vectors_') else None
        support_indices = model.support_.tolist() if hasattr(model, 'support_') else None
        
        params = {
            'kernel': model.kernel,
            'C': model.C,
            'gamma': model.gamma,
            'degree': getattr(model, 'degree', None),
            'coef0': getattr(model, 'coef0', None),
            'support_vectors': support_vectors,
            'support_indices': support_indices,
            'n_support_vectors': len(support_indices) if support_indices else 0,
            'n_support_per_class': model.n_support_.tolist() if hasattr(model, 'n_support_') else None,
            'dual_coef': model.dual_coef_.tolist() if hasattr(model, 'dual_coef_') else None
        }
        
        return params
    
    def predict_with_proba(self, model, X, scaler=None):
        """
        使用训练好的SVM模型进行预测
        
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
    
    def get_feature_importance(self, model):
        """
        获取SVM特征重要性（仅限线性核）
        
        Args:
            model: 训练好的SVM模型
            
        Returns:
            feature_importance: 特征重要性数组
        """
        if hasattr(model, 'coef_') and model.coef_ is not None:
            return np.abs(model.coef_[0])
        else:
            logger.warning("SVM非线性核不支持特征重要性计算")
            return None