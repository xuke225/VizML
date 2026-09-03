"""
朴素贝叶斯分类算法模块
基于 sklearn 实现朴素贝叶斯分类算法
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.preprocessing import StandardScaler
import logging
import warnings

from .base import BaseClassificationAlgorithm

logger = logging.getLogger(__name__)


def safe_log_computation(values, epsilon=1e-15):
    """安全的对数计算，避免数值不稳定性"""
    # 将值限制在有效范围内
    safe_values = np.clip(values, epsilon, 1.0 - epsilon)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.log(safe_values)


def clean_numerical_values(data):
    """清理数值中的NaN和inf值"""
    if isinstance(data, dict):
        return {k: clean_numerical_values(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_numerical_values(item) for item in data]
    elif isinstance(data, np.ndarray):
        # 替换NaN和inf为None
        cleaned = data.copy().astype(object)
        mask = ~np.isfinite(data)
        cleaned[mask] = None
        return cleaned.tolist()
    elif isinstance(data, (float, np.floating)):
        if not np.isfinite(data):
            return None
        return float(data)
    elif isinstance(data, (int, np.integer)):
        if not np.isfinite(data):
            return None
        return int(data)
    else:
        return data


class BayesianClassificationAlgorithm(BaseClassificationAlgorithm):
    """朴素贝叶斯分类算法类"""
    
    def __init__(self):
        super().__init__()
    
    def train_model(self, X, y, var_smoothing=1e-9, priors=None, scale_features=False, **kwargs):
        """
        训练高斯朴素贝叶斯模型
        
        Args:
            X: 特征矩阵
            y: 标签向量
            var_smoothing: 高斯朴素贝叶斯的方差平滑参数
            priors: 先验概率
            scale_features: 是否标准化特征
            **kwargs: 其他参数
            
        Returns:
            result: 训练结果字典
        """
        logger.info("训练高斯朴素贝叶斯模型")
        
        # 数据分割
        test_size = kwargs.get('test_size', 0.2)
        indices = np.arange(len(X))
        X_train, X_test, y_train, y_test, train_indices, test_indices = \
            self._split_data_with_indices(X, y, indices, test_size=test_size)
        
        # 数据预处理（高斯朴素贝叶斯）
        if scale_features:
            X_train, X_test = self.scale_features(X_train, X_test)
        
        # 创建高斯朴素贝叶斯模型
        model = GaussianNB(var_smoothing=var_smoothing, priors=priors)
        
        # 训练模型
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
        
        # 获取高斯朴素贝叶斯信息
        training_info = {
            'algorithm_type': 'gaussian',
            'classes': model.classes_.tolist()
        }
        
        # 安全地获取属性
        if hasattr(model, 'class_count_'):
            training_info['class_count'] = model.class_count_.tolist()
        if hasattr(model, 'class_prior_'):
            training_info['class_prior'] = model.class_prior_.tolist()
        
        # 添加高斯朴素贝叶斯特定参数
        # Handle different sklearn versions - use var_ (new) or sigma_ (old)
        variance_attr = getattr(model, 'var_', getattr(model, 'sigma_', None))
        training_info.update({
            'theta': clean_numerical_values(model.theta_),
            'sigma': clean_numerical_values(variance_attr) if variance_attr is not None else None,
            'feature_count': model.theta_.shape[1],
            'var_smoothing': var_smoothing
        })
        
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
        
        # 返回完整的训练结果（清理数值）
        result = {
            'model': model,
            'metrics': clean_numerical_values(metrics),
            'predictions': {
                'train': y_pred_train.tolist(),
                'test': y_pred_test.tolist()
            },
            'full_data_with_labels': full_data_with_labels,
            'training_info': clean_numerical_values(training_info),
            'algorithm_params': clean_numerical_values(model_params),
            'decision_boundary': clean_numerical_values(decision_boundary) if decision_boundary is not None else None,
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
            'train_data': {'X': clean_numerical_values(X_train), 'y': y_train.tolist()},
            'test_data': {'X': clean_numerical_values(X_test), 'y': y_test.tolist()},
            'full_data': {'X': clean_numerical_values(X), 'y': y.tolist()},
            'scaler': self.scaler if scale_features else None
        }
        
        return result
    
    def _split_data_with_indices(self, X, y, indices, test_size=0.2):
        """数据分割并返回索引"""
        from sklearn.model_selection import train_test_split
        return train_test_split(X, y, indices, test_size=test_size, random_state=42)
    
    def get_model_params(self, model):
        """
        获取朴素贝叶斯模型参数信息
        
        Args:
            model: 训练好的朴素贝叶斯模型
            
        Returns:
            params: 模型参数字典
        """
        # 基础参数
        params = {
            'classes': model.classes_.tolist()
        }
        
        # 安全地获取属性，不同的贝叶斯模型有不同的属性名
        if hasattr(model, 'class_prior_'):
            params['class_prior'] = model.class_prior_.tolist()
        if hasattr(model, 'class_count_'):
            params['class_count'] = model.class_count_.tolist()
        
        # 根据模型类型添加特定参数（添加数值稳定性处理）
        if isinstance(model, GaussianNB):
            # Handle different sklearn versions - use var_ (new) or sigma_ (old)
            variance_attr = getattr(model, 'var_', getattr(model, 'sigma_', None))
            params.update({
                'theta': clean_numerical_values(model.theta_),
                'sigma': clean_numerical_values(variance_attr) if variance_attr is not None else None,
                'feature_count': model.theta_.shape[1],
                'var_smoothing': getattr(model, 'var_smoothing', 1e-9),
                'algorithm_type': 'gaussian'
            })
        elif isinstance(model, MultinomialNB):
            params.update({
                'feature_log_prob': clean_numerical_values(model.feature_log_prob_),
                'feature_count': clean_numerical_values(model.feature_count_),
                'alpha': getattr(model, 'alpha', 1.0),
                'fit_prior': getattr(model, 'fit_prior', True),
                'algorithm_type': 'multinomial'
            })
        elif isinstance(model, BernoulliNB):
            params.update({
                'feature_log_prob': clean_numerical_values(model.feature_log_prob_),
                'feature_count': clean_numerical_values(model.feature_count_) if hasattr(model, 'feature_count_') else None,
                'alpha': getattr(model, 'alpha', 1.0),
                'binarize': getattr(model, 'binarize', None),
                'fit_prior': getattr(model, 'fit_prior', True),
                'algorithm_type': 'bernoulli'
            })
        
        return params
    
    
    def get_bayesian_probabilities(self, model, X_point, scaler=None):
        """
        获取朴素贝叶斯模型对特定点的详细概率计算
        
        Args:
            model: 训练好的朴素贝叶斯模型
            X_point: 单个数据点或数据点数组
            scaler: 数据标准化器
            
        Returns:
            详细的概率计算信息
        """
        if not hasattr(model, 'predict_proba'):
            return None
            
        # 确保输入是二维数组
        if X_point.ndim == 1:
            X_point = X_point.reshape(1, -1)
        
        # 数据标准化（如果之前使用过）
        if scaler is not None:
            X_point = scaler.transform(X_point)
            
        # 获取预测概率
        probabilities = model.predict_proba(X_point)
        predictions = model.predict(X_point)
        
        detailed_probs = []
        
        for i, point in enumerate(X_point):
            point_probs = {
                'point_features': point.tolist(),
                'predicted_class': int(predictions[i]),
                'class_probabilities': probabilities[i].tolist(),
                'class_details': []
            }
            
            # 计算每个类别的详细概率
            for class_idx, class_label in enumerate(model.classes_):
                # 先验概率 - 安全获取
                if hasattr(model, 'class_prior_'):
                    prior = model.class_prior_[class_idx]
                else:
                    # 对于某些模型，可能需要从class_count_计算
                    if hasattr(model, 'class_count_'):
                        total_count = np.sum(model.class_count_)
                        prior = model.class_count_[class_idx] / total_count
                    else:
                        prior = 1.0 / len(model.classes_)  # 均匀先验
                
                # 根据模型类型计算似然性
                likelihood = 1.0
                feature_likelihoods = []
                
                if isinstance(model, GaussianNB):
                    # 高斯朴素贝叶斯
                    for feature_idx, feature_value in enumerate(point):
                        mean = model.theta_[class_idx][feature_idx]
                        # Handle different sklearn versions - use var_ (new) or sigma_ (old)
                        variance_attr = getattr(model, 'var_', getattr(model, 'sigma_', None))
                        var = variance_attr[class_idx][feature_idx] if variance_attr is not None else 1e-9
                        
                        # 高斯概率密度函数
                        feature_likelihood = (1.0 / np.sqrt(2 * np.pi * var)) * \
                                           np.exp(-0.5 * ((feature_value - mean) ** 2) / var)
                        
                        feature_likelihoods.append({
                            'feature_index': feature_idx,
                            'feature_value': float(feature_value),
                            'mean': float(mean),
                            'variance': float(var),
                            'likelihood': float(feature_likelihood)
                        })
                        
                        likelihood *= feature_likelihood
                        
                elif isinstance(model, (MultinomialNB, BernoulliNB)):
                    # 多项式或伯努利朴素贝叶斯（添加数值稳定性处理）
                    for feature_idx, feature_value in enumerate(point):
                        # 使用对数概率来计算，添加数值稳定性保护
                        log_prob = model.feature_log_prob_[class_idx][feature_idx]
                        
                        # 安全计算 exp(log_prob)，避免数值溢出
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            exp_log_prob = np.exp(np.clip(log_prob, -500, 0))  # 限制范围避免溢出
                            
                        # 计算特征似然性，添加数值保护
                        if feature_value > 0:
                            feature_likelihood = exp_log_prob
                        else:
                            # 使用数值稳定的计算方式
                            feature_likelihood = max(1 - exp_log_prob, 1e-15)
                        
                        # 确保似然性为有限值
                        if not np.isfinite(feature_likelihood) or feature_likelihood <= 0:
                            feature_likelihood = 1e-15
                        
                        feature_likelihoods.append({
                            'feature_index': feature_idx,
                            'feature_value': float(feature_value),
                            'log_prob': float(log_prob) if np.isfinite(log_prob) else None,
                            'likelihood': float(feature_likelihood)
                        })
                        
                        likelihood *= feature_likelihood
                
                # 后验概率 (未归一化)
                posterior_unnormalized = prior * likelihood
                
                # 确保所有概率值都是有限的
                prior_safe = float(prior) if np.isfinite(prior) else 0.0
                likelihood_safe = float(likelihood) if np.isfinite(likelihood) else 1e-15
                posterior_unnorm_safe = float(posterior_unnormalized) if np.isfinite(posterior_unnormalized) else 0.0
                posterior_prob_safe = float(probabilities[i][class_idx]) if np.isfinite(probabilities[i][class_idx]) else 0.0
                
                point_probs['class_details'].append({
                    'class_label': int(class_label),
                    'prior_probability': prior_safe,
                    'likelihood': likelihood_safe,
                    'posterior_unnormalized': posterior_unnorm_safe,
                    'posterior_probability': posterior_prob_safe,
                    'feature_likelihoods': feature_likelihoods
                })
            
            detailed_probs.append(point_probs)
        
        return detailed_probs[0] if len(detailed_probs) == 1 else detailed_probs
    
    
