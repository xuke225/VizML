"""
机器学习算法基类模块
提供公共的基类和工具函数
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error,
    classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class BaseAlgorithm:
    """算法基类"""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def train_model(self, X, y, **kwargs):
        """
        训练模型的抽象方法
        
        Args:
            X: 特征矩阵
            y: 目标向量
            **kwargs: 算法参数
            
        Returns:
            model: 训练好的模型
            training_info: 训练信息
        """
        raise NotImplementedError("子类必须实现train_model方法")
    
    def predict(self, model, X):
        """
        预测方法
        
        Args:
            model: 训练好的模型
            X: 特征矩阵
            
        Returns:
            predictions: 预测结果
        """
        return model.predict(X)
    
    def evaluate_classification(self, y_true, y_pred):
        """
        分类模型评估
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            
        Returns:
            metrics: 评估指标字典
        """
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        }
        return metrics
    
    def evaluate_regression(self, y_true, y_pred):
        """
        回归模型评估
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            metrics: 评估指标字典
        """
        metrics = {
            'mse': float(mean_squared_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'r2': float(r2_score(y_true, y_pred))
        }
        return metrics
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        数据分割工具方法
        
        Args:
            X: 特征矩阵
            y: 目标向量
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            X_train, X_test, y_train, y_test: 分割后的数据
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def scale_features(self, X_train, X_test=None):
        """
        特征缩放工具方法
        
        Args:
            X_train: 训练集特征
            X_test: 测试集特征（可选）
            
        Returns:
            X_train_scaled: 缩放后的训练集
            X_test_scaled: 缩放后的测试集（如果提供）
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def get_confusion_matrix(self, y_true, y_pred):
        """
        获取混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            
        Returns:
            confusion_matrix: 混淆矩阵
        """
        return confusion_matrix(y_true, y_pred).tolist()
    
    def get_classification_report(self, y_true, y_pred):
        """
        获取分类报告
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            
        Returns:
            report: 分类报告字典
        """
        return classification_report(y_true, y_pred, output_dict=True)


class BaseClassificationAlgorithm(BaseAlgorithm):
    """分类算法基类"""
    
    def get_decision_boundary(self, model, X, y, resolution=100):
        """
        获取决策边界数据点
        
        Args:
            model: 训练好的模型
            X: 特征矩阵
            y: 标签向量
            resolution: 网格分辨率
            
        Returns:
            boundary_data: 决策边界数据
        """
        if X.shape[1] != 2:
            logger.warning("只支持2D数据的决策边界可视化")
            return None
        
        # 创建网格，增加边距
        margin = 0.5
        x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
        y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
        
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution)
        )
        
        # 预测网格点
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        try:
            predictions = model.predict(grid_points)
            
            # 获取决策函数值（距离超平面的距离）
            decision_scores = None
            if hasattr(model, 'decision_function'):
                decision_scores = model.decision_function(grid_points)
            
            # 获取预测概率（如果支持）
            probabilities = None
            if hasattr(model, 'predict_proba'):
                try:
                    probabilities = model.predict_proba(grid_points)
                except Exception as e:
                    logger.warning(f"无法获取预测概率: {e}")
            
            boundary_data = {
                'xx': xx.tolist(),
                'yy': yy.tolist(),
                'predictions': predictions.reshape(xx.shape).tolist(),
                'decision_scores': decision_scores.reshape(xx.shape).tolist() if decision_scores is not None else None,
                'probabilities': probabilities.reshape(xx.shape[0], xx.shape[1], -1).tolist() if probabilities is not None else None,
                'x_range': [float(x_min), float(x_max)],
                'y_range': [float(y_min), float(y_max)],
                'resolution': resolution
            }
            
            return boundary_data
            
        except Exception as e:
            logger.error(f"决策边界计算失败: {e}")
            return None


class BaseRegressionAlgorithm(BaseAlgorithm):
    """回归算法基类"""
    
    def generate_prediction_curve(self, model, X_range, **kwargs):
        """
        生成回归曲线数据点
        
        Args:
            model: 训练好的模型
            X_range: X轴范围
            **kwargs: 其他参数
            
        Returns:
            curve_data: 曲线数据
        """
        if isinstance(X_range, tuple):
            x_min, x_max = X_range
            X_curve = np.linspace(x_min, x_max, 100).reshape(-1, 1)
        else:
            X_curve = np.array(X_range).reshape(-1, 1)
        
        y_pred = model.predict(X_curve)
        
        curve_data = {
            'x': X_curve[:, 0].tolist() if X_curve.shape[1] > 0 else [],
            'y': y_pred.tolist()
        }
        
        return curve_data
    
    def calculate_residuals(self, model, X, y):
        """
        计算残差
        
        Args:
            model: 训练好的模型
            X: 特征矩阵
            y: 真实目标值
            
        Returns:
            residuals_data: 残差数据
        """
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        residuals_data = {
            'predictions': y_pred.tolist(),
            'residuals': residuals.tolist(),
            'absolute_residuals': np.abs(residuals).tolist()
        }
        
        return residuals_data


def create_full_data_with_labels(X, y, train_indices, test_indices, y_pred_train, y_pred_test):
    """
    创建带标识的完整数据集用于前端显示
    
    Args:
        X: 完整特征矩阵
        y: 完整标签向量
        train_indices: 训练集索引
        test_indices: 测试集索引
        y_pred_train: 训练集预测结果
        y_pred_test: 测试集预测结果
        
    Returns:
        full_data_with_labels: 带标识的完整数据
    """
    full_data_with_labels = []
    for i, (x, y_true) in enumerate(zip(X, y)):
        full_data_with_labels.append({
            'index': i,
            'features': x.tolist(),
            'label': int(y_true),
            'data_type': 'unknown'
        })
    
    # 为训练集数据添加预测结果
    for i, (idx, pred) in enumerate(zip(train_indices, y_pred_train)):
        full_data_with_labels[idx]['data_type'] = 'train'
        full_data_with_labels[idx]['prediction'] = int(pred)
        full_data_with_labels[idx]['misclassified'] = bool(pred != full_data_with_labels[idx]['label'])
    
    # 为测试集数据添加预测结果
    for i, (idx, pred) in enumerate(zip(test_indices, y_pred_test)):
        full_data_with_labels[idx]['data_type'] = 'test'
        full_data_with_labels[idx]['prediction'] = int(pred)
        full_data_with_labels[idx]['misclassified'] = bool(pred != full_data_with_labels[idx]['label'])
        
    return full_data_with_labels