"""
决策树算法模块
基于 sklearn 实现决策树分类算法
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
import logging

from .base import BaseClassificationAlgorithm

logger = logging.getLogger(__name__)


class DecisionTreeAlgorithm(BaseClassificationAlgorithm):
    """决策树算法类"""
    
    def __init__(self):
        super().__init__()
    
    def train_model(self, X, y, max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                   max_features=None, criterion='gini', scale_features=False, use_full_features=False, **kwargs):
        """
        训练决策树模型
        
        Args:
            X: 特征矩阵
            y: 标签向量
            max_depth: 最大深度
            min_samples_split: 分割内部节点所需的最小样本数
            min_samples_leaf: 叶子节点所需的最小样本数
            max_features: 寻找最佳分割时要考虑的特征数量
            criterion: 分割质量的测量标准 ('gini', 'entropy')
            scale_features: 是否标准化特征
            use_full_features: 是否使用完整维度特征（需要配合feature_names）
            **kwargs: 其他参数
            
        Returns:
            result: 训练结果字典
        """
        logger.info(f"训练决策树模型: max_depth={max_depth}, criterion={criterion}")
        
        # 数据分割
        test_size = kwargs.get('test_size', 0.2)
        indices = np.arange(len(X))
        X_train, X_test, y_train, y_test, train_indices, test_indices = \
            self._split_data_with_indices(X, y, indices, test_size=test_size)
        
        # 数据标准化（决策树通常不需要，但保留选项）
        if scale_features:
            X_train, X_test = self.scale_features(X_train, X_test)
        
        # 创建并训练模型
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            criterion=criterion,
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
        
        # 获取决策树特有信息
        feature_names = kwargs.get('feature_names', [f'特征{i}' for i in range(X.shape[1])])
        training_info = {
            'max_depth': int(model.get_depth()),
            'n_leaves': int(model.get_n_leaves()),
            'feature_importances': model.feature_importances_.tolist(),
            'feature_names': feature_names,
            'n_features': X.shape[1],
            'tree_structure': self._get_tree_structure(model, feature_names=feature_names)
        }
        
        # 获取模型参数和决策边界（用于可视化）
        model_params = self.get_model_params(model, feature_names)
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
    
    def get_model_params(self, model, feature_names=None):
        """
        获取决策树模型参数信息
        
        Args:
            model: 训练好的决策树模型
            feature_names: 特征名称列表
            
        Returns:
            params: 模型参数字典
        """
        # 获取实际树深度，使用完整深度进行可视化
        actual_depth = int(model.get_depth())
        
        if feature_names is None:
            feature_names = [f'特征{i}' for i in range(len(model.feature_importances_))]
        
        params = {
            'max_depth': actual_depth,
            'n_leaves': int(model.get_n_leaves()),
            'feature_importances': model.feature_importances_.tolist(),
            'feature_names': feature_names,
            'tree_structure': self._get_tree_structure(model, feature_names=feature_names)
        }
        
        return params
    
    def _get_tree_structure(self, model, max_depth=None, feature_names=None):
        """
        获取决策树结构（完整版，用于可视化）
        
        Args:
            model: 决策树模型
            max_depth: 最大深度，None表示使用完整深度
            feature_names: 特征名称列表
            
        Returns:
            tree_structure: 树结构字典
        """
        tree = model.tree_
        
        if feature_names is None:
            feature_names = [f'特征{i}' for i in range(tree.n_features)]
        
        def recurse(node_id, depth=0):
            # 检查是否为叶子节点（左右子节点相同表示叶子）
            is_leaf = tree.children_left[node_id] == tree.children_right[node_id]
            
            # 如果设置了最大深度限制且达到限制，或者本身就是叶子节点
            if (max_depth is not None and depth >= max_depth) or is_leaf:
                # 叶子节点
                value = tree.value[node_id][0].tolist()  # 获取类别分布
                predicted_class = int(value.index(max(value)))  # 预测的类别
                
                return {
                    'is_leaf': True,
                    'value': [value],  # 保持与原格式兼容
                    'samples': int(tree.n_node_samples[node_id]),
                    'predicted_class': predicted_class,
                    'impurity': float(tree.impurity[node_id])
                }
            else:
                # 内部节点
                left_child = tree.children_left[node_id]
                right_child = tree.children_right[node_id]
                
                feature_idx = int(tree.feature[node_id])
                return {
                    'is_leaf': False,
                    'feature': feature_idx,
                    'feature_name': feature_names[feature_idx] if feature_idx < len(feature_names) else f'特征{feature_idx}',
                    'threshold': float(tree.threshold[node_id]),
                    'samples': int(tree.n_node_samples[node_id]),
                    'impurity': float(tree.impurity[node_id]),
                    'left': recurse(left_child, depth + 1),
                    'right': recurse(right_child, depth + 1)
                }
        
        return recurse(0)
    
    def get_feature_importance(self, model):
        """
        获取决策树特征重要性
        
        Args:
            model: 训练好的决策树模型
            
        Returns:
            feature_importance: 特征重要性数组
        """
        return model.feature_importances_
    
    def predict_with_proba(self, model, X, scaler=None):
        """
        使用训练好的决策树模型进行预测
        
        Args:
            model: 训练好的模型
            X: 特征矩阵
            scaler: 数据标准化器（决策树通常不需要）
            
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
    
    def get_decision_path(self, model, X):
        """
        获取决策路径
        
        Args:
            model: 训练好的决策树模型
            X: 特征矩阵
            
        Returns:
            decision_paths: 决策路径信息
        """
        decision_paths = model.decision_path(X)
        leaf_indices = model.apply(X)
        
        paths_info = []
        for i, sample in enumerate(X):
            path = []
            # 获取该样本经过的节点
            nodes = decision_paths[i].indices
            
            for node_id in nodes:
                tree = model.tree_
                if tree.children_left[node_id] == tree.children_right[node_id]:
                    # 叶子节点
                    value = tree.value[node_id][0]
                    predicted_class = np.argmax(value)
                    path.append({
                        'node_id': int(node_id),
                        'is_leaf': True,
                        'predicted_class': int(predicted_class),
                        'samples': int(tree.n_node_samples[node_id]),
                        'value': value.tolist()
                    })
                else:
                    # 内部节点
                    path.append({
                        'node_id': int(node_id),
                        'is_leaf': False,
                        'feature': int(tree.feature[node_id]),
                        'threshold': float(tree.threshold[node_id]),
                        'samples': int(tree.n_node_samples[node_id])
                    })
            
            paths_info.append({
                'sample_index': i,
                'leaf_index': int(leaf_indices[i]),
                'path': path
            })
        
        return paths_info