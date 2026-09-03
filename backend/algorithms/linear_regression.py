"""
线性回归算法模块
基于 sklearn 实现线性回归和多项式回归算法
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from itertools import combinations_with_replacement
import logging

from .base import BaseRegressionAlgorithm

logger = logging.getLogger(__name__)


class LinearRegressionAlgorithm(BaseRegressionAlgorithm):
    """线性回归算法类"""
    
    def __init__(self):
        super().__init__()
        self.polynomial_transformer = None
    
    def train_model(self, X, y, algorithm='linear', polynomial_degree=1, alpha=1.0, 
                   fit_intercept=True, scale_features=False, optimizer='auto', feature_info=None, **kwargs):
        """
        训练线性回归模型
        
        Args:
            X: 特征矩阵
            y: 目标向量
            algorithm: 算法类型 ('linear', 'ridge', 'lasso', 'polynomial')
            polynomial_degree: 多项式次数（用于多项式回归）
            alpha: 正则化参数（用于Ridge和Lasso）
            fit_intercept: 是否拟合截距
            scale_features: 是否标准化特征
            optimizer: 优化方法 ('auto', 'ols', 'sgd', 'normal_equation')
            **kwargs: 其他参数
            
        Returns:
            result: 训练结果字典
        """
        logger.info(f"训练线性回归模型: algorithm={algorithm}, polynomial_degree={polynomial_degree}")

        input_feature_names = None
        if feature_info and feature_info.get('features_used'):
            candidate_names = list(feature_info['features_used'])
            if len(candidate_names) == X.shape[1]:
                input_feature_names = candidate_names
        if input_feature_names is None:
            input_feature_names = [f'x{i + 1}' for i in range(X.shape[1])]
        processed_feature_names = input_feature_names
        
        # 多项式特征变换
        X_processed = X
        if polynomial_degree > 1 or algorithm == 'polynomial':
            try:
                self.polynomial_transformer = PolynomialFeatures(
                    degree=polynomial_degree, 
                    include_bias=False,
                    interaction_only=False
                )
                X_processed = self.polynomial_transformer.fit_transform(X)
                processed_feature_names = self.polynomial_transformer.get_feature_names_out(
                    input_feature_names
                ).tolist()
                logger.info(f"应用多项式特征变换，次数={polynomial_degree}，原特征: {X.shape[1]}，新特征: {X_processed.shape[1]}")
                
                # 检查特征数量是否合理
                expected_features = sum([len(list(combinations_with_replacement(range(X.shape[1]), d))) for d in range(1, polynomial_degree + 1)])
                if X_processed.shape[1] != expected_features:
                    logger.warning(f"特征数量不匹配，预期: {expected_features}, 实际: {X_processed.shape[1]}")
                    
            except Exception as e:
                logger.error(f"多项式特征变换失败: {e}")
                # 如果变换失败，使用原始特征
                self.polynomial_transformer = None
                X_processed = X
        else:
            self.polynomial_transformer = None
        
        # 数据分割
        test_size = kwargs.get('test_size', 0.2)
        original_indices = np.arange(len(X_processed))
        X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
            X_processed,
            y,
            original_indices,
            test_size=test_size,
            random_state=42
        )
        
        # 数据标准化（如果需要）
        if scale_features:
            X_train, X_test = self.scale_features(X_train, X_test)
        
        # 根据算法类型和优化器创建模型
        if algorithm in ['linear', 'polynomial']:
            if optimizer == 'sgd':
                from sklearn.linear_model import SGDRegressor
                model = SGDRegressor(
                    loss='squared_error',  # 明确指定损失函数
                    fit_intercept=fit_intercept, 
                    random_state=42,
                    max_iter=kwargs.get('max_iter', 1000),
                    tol=kwargs.get('tol', 1e-3),
                    learning_rate=kwargs.get('learning_rate', 'invscaling'),
                    eta0=kwargs.get('eta0', 0.01)
                )
            elif optimizer == 'normal_equation':
                # 使用正则方程的自定义实现
                model = self._create_normal_equation_model(fit_intercept)
            else:  # 'auto' or 'ols'
                model = LinearRegression(fit_intercept=fit_intercept)
        elif algorithm == 'ridge':
            if optimizer == 'sgd':
                from sklearn.linear_model import SGDRegressor
                model = SGDRegressor(
                    loss='squared_error',
                    penalty='l2', alpha=alpha,
                    fit_intercept=fit_intercept, 
                    random_state=42,
                    max_iter=kwargs.get('max_iter', 1000),
                    tol=kwargs.get('tol', 1e-3)
                )
            else:
                model = Ridge(alpha=alpha, fit_intercept=fit_intercept, random_state=42)
        elif algorithm == 'lasso':
            if optimizer == 'sgd':
                from sklearn.linear_model import SGDRegressor
                model = SGDRegressor(
                    loss='squared_error',
                    penalty='l1', alpha=alpha,
                    fit_intercept=fit_intercept, 
                    random_state=42,
                    max_iter=kwargs.get('max_iter', 1000),
                    tol=kwargs.get('tol', 1e-3)
                )
            else:
                model = Lasso(alpha=alpha, fit_intercept=fit_intercept, random_state=42)
        else:
            raise ValueError(f"不支持的线性回归算法: {algorithm}")
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测和评估
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # 生成与原始数据顺序严格对齐的逐点结果。此前前端只能拿到被
        # train_test_split 打乱后的训练集预测，无法可靠地将误差画回原数据点。
        X_all_for_prediction = X_processed
        if scale_features and hasattr(self, 'scaler') and self.scaler is not None:
            X_all_for_prediction = self.scaler.transform(X_processed)
        y_pred_all = model.predict(X_all_for_prediction)
        residuals_all = y - y_pred_all

        split_by_index = np.full(len(X), 'test', dtype=object)
        split_by_index[train_indices] = 'train'
        point_results = [
            {
                'index': int(index),
                'features': np.asarray(X[index]).astype(float).tolist(),
                'actual': float(y[index]),
                'predicted': float(y_pred_all[index]),
                'residual': float(residuals_all[index]),
                'squared_error': float(residuals_all[index] ** 2),
                'split': str(split_by_index[index])
            }
            for index in range(len(X))
        ]

        full_sse = float(np.sum(residuals_all ** 2))
        full_mse = float(np.mean(residuals_all ** 2))
        mean_target = float(np.mean(y))
        total_sum_squares = float(np.sum((y - mean_target) ** 2))
        loss_summary = {
            'sse': full_sse,
            'mse': full_mse,
            'mean_target': mean_target,
            'tss': total_sum_squares,
            'r2': float(r2_score(y, y_pred_all))
        }
        
        # 计算评估指标
        train_metrics = self.evaluate_regression(y_train, y_pred_train)
        test_metrics = self.evaluate_regression(y_test, y_pred_test)
        
        metrics = {
            f'train_{k}': v for k, v in train_metrics.items()
        }
        metrics.update({
            f'test_{k}': v for k, v in test_metrics.items()
        })
        
        # 计算调整R²
        n_features = X_processed.shape[1]
        metrics['train_adjusted_r2'] = self._calculate_adjusted_r2(y_train, y_pred_train, n_features)
        metrics['test_adjusted_r2'] = self._calculate_adjusted_r2(y_test, y_pred_test, n_features)
        
        # 获取训练信息
        training_info = {
            'input_features': X_processed.shape[1],
            'training_samples': X_processed.shape[0],
            'polynomial_degree': polynomial_degree,
            'coefficients': model.coef_.tolist() if hasattr(model, 'coef_') else [],
            'intercept': float(model.intercept_) if hasattr(model, 'intercept_') else 0.0,
            'algorithm': algorithm,
            'optimizer': optimizer,
            'input_feature_names': input_feature_names,
            'feature_names': processed_feature_names,
        }
        
        # 处理SGD模型的特殊情况
        if optimizer == 'sgd':
            # SGDRegressor的coef_可能是1D数组，需要正确处理
            if hasattr(model, 'coef_') and model.coef_ is not None:
                if model.coef_.ndim == 1:
                    training_info['coefficients'] = model.coef_.tolist()
                else:
                    training_info['coefficients'] = model.coef_[0].tolist() if len(model.coef_) > 0 else []
            
            if hasattr(model, 'intercept_') and model.intercept_ is not None:
                if isinstance(model.intercept_, (list, np.ndarray)):
                    training_info['intercept'] = float(model.intercept_[0]) if len(model.intercept_) > 0 else 0.0
                else:
                    training_info['intercept'] = float(model.intercept_)
        
        # 添加算法特定信息
        if algorithm == 'ridge':
            training_info['alpha'] = alpha
        elif algorithm == 'lasso':
            training_info['alpha'] = alpha
            training_info['n_iter'] = getattr(model, 'n_iter_', None)
        
        # 添加SGD特定信息
        if optimizer == 'sgd' and hasattr(model, 'n_iter_'):
            training_info['n_iter'] = getattr(model, 'n_iter_', None)
            training_info['learning_rate'] = kwargs.get('learning_rate', 'invscaling')
            training_info['eta0'] = kwargs.get('eta0', 0.01)
        
        # 单特征数据可直接生成拟合曲线；多特征结果由前端使用诊断图展示。
        prediction_curve = None
        try:
            if X.shape[1] == 1:
                # 1D数据：标准预测曲线
                x_min, x_max = X[:, 0].min(), X[:, 0].max()
                x_range = np.linspace(x_min - 0.5, x_max + 0.5, 100).reshape(-1, 1)
                
                # 应用相同的多项式变换
                if self.polynomial_transformer is not None:
                    x_range_poly = self.polynomial_transformer.transform(x_range)
                else:
                    x_range_poly = x_range
                
                # 应用相同的标准化
                if scale_features and hasattr(self, 'scaler') and self.scaler is not None:
                    x_range_scaled = self.scaler.transform(x_range_poly)
                else:
                    x_range_scaled = x_range_poly
                
                y_curve = model.predict(x_range_scaled)
                prediction_curve = {
                    'x': x_range[:, 0].tolist(),
                    'y': y_curve.tolist(),
                    'curve_type': '1d_curve'
                }
                logger.info(f"生成1D预测曲线，点数: {len(y_curve)}")
                
        except Exception as e:
            logger.error(f"预测曲线生成失败: {e}")
            prediction_curve = None
        
        # 计算残差
        residuals_data = self.calculate_residuals(model, X_train, y_train)
        
        # 获取模型方程
        feature_names = processed_feature_names
        try:
            model_equation = self.get_model_equation(
                model, feature_names, algorithm, degree=polynomial_degree
            )
        except Exception as e:
            logger.error(f"生成模型方程失败: {e}")
            model_equation = "y = ?"
        
        # 返回完整的训练结果
        result = {
            'model': model,
            'metrics': metrics,
            'predictions': {
                'train': y_pred_train.tolist(),
                'test': y_pred_test.tolist()
            },
            'training_info': training_info,
            'prediction_curve': prediction_curve,
            'residuals': residuals_data,
            'point_results': point_results,
            'loss_summary': loss_summary,
            'model_equation': model_equation,
            'data_info': {
                'train_size': len(X_train),
                'test_size': len(X_test),
                'n_features': X.shape[1],
                'n_polynomial_features': X_processed.shape[1]
            },
            # 保存完整数据以供后续使用
            'train_data': {'X': X_train.tolist(), 'y': y_train.tolist()},
            'test_data': {'X': X_test.tolist(), 'y': y_test.tolist()},
            'full_data': {'X': X_processed.tolist(), 'y': y.tolist()},
            'original_data': {'X': X.tolist(), 'y': y.tolist()},
            'scaler': self.scaler if scale_features else None,
            'polynomial_transformer': self.polynomial_transformer
        }
        
        return result
    
    def _calculate_adjusted_r2(self, y_true, y_pred, n_features):
        """
        计算调整R²
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            n_features: 特征数量
            
        Returns:
            adjusted_r2: 调整R²
        """
        n = len(y_true)
        r2 = r2_score(y_true, y_pred)
        
        if n > n_features + 1:
            adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
        else:
            adjusted_r2 = r2
            
        return float(adjusted_r2)
    
    def get_model_equation(self, model, feature_names=None, algorithm='linear', degree=1):
        """
        获取回归方程的字符串表示
        
        Args:
            model: 训练好的模型
            feature_names: 特征名称列表
            algorithm: 算法类型
            degree: 多项式次数
            
        Returns:
            equation: 方程字符串
        """
        try:
            if not hasattr(model, 'coef_') or not hasattr(model, 'intercept_'):
                return "无法获取模型方程"
            
            coef = model.coef_
            intercept = model.intercept_
            
            # 处理SGD模型的特殊格式
            if hasattr(intercept, '__iter__') and not isinstance(intercept, str):
                intercept = float(intercept[0]) if len(intercept) > 0 else 0.0
            else:
                intercept = float(intercept)
            
            if hasattr(coef, 'ndim') and coef.ndim > 1:
                coef = coef[0] if len(coef) > 0 else []
            
            # 处理系数为空的情况
            if len(coef) == 0:
                return f"y = {intercept:.4f}"
            
            # 生成特征名称
            if feature_names is None:
                if algorithm in ['polynomial'] or degree > 1:
                    # 单变量多项式：生成 x, x^2, x^3, ... 形式的特征名
                    feature_names = []
                    for i in range(len(coef)):
                        power = i + 1
                        if power == 1:
                            feature_names.append('x')
                        else:
                            feature_names.append(f'x^{power}')
                else:
                    feature_names = [f'x{i+1}' for i in range(len(coef))]
            
            # 确保特征名称数量与系数数量匹配
            if len(feature_names) != len(coef):
                feature_names = [f'x{i}' for i in range(len(coef))]
            
            # 构建方程字符串
            equation_parts = [f"{intercept:.4f}"]
            
            for i, (coef_val, feature_name) in enumerate(zip(coef, feature_names)):
                coef_val = float(coef_val)  # 确保是浮点数
                
                if abs(coef_val) < 1e-10:  # 忽略极小的系数
                    continue
                
                sign = '+' if coef_val >= 0 else '-'
                coef_abs = abs(coef_val)
                
                if i == 0 and coef_val >= 0:
                    equation_parts.append(f" + {coef_abs:.4f}*{feature_name}")
                else:
                    equation_parts.append(f" {sign} {coef_abs:.4f}*{feature_name}")
            
            equation = "y = " + "".join(equation_parts)
            
            # 清理多余的符号
            equation = equation.replace("y =  + ", "y = ").replace("y =  - ", "y = -")
            
            return equation
        
        except Exception as e:
            logger.error(f"生成模型方程失败: {e}")
            return "y = ?"
    
    def predict_new_data(self, model, X, scaler=None, polynomial_transformer=None):
        """
        使用训练好的模型预测新数据
        
        Args:
            model: 训练好的模型
            X: 新的特征矩阵
            scaler: 数据标准化器
            polynomial_transformer: 多项式变换器
            
        Returns:
            predictions: 预测结果
        """
        X_processed = X
        
        # 应用多项式变换（如果之前使用过）
        if polynomial_transformer is not None:
            X_processed = polynomial_transformer.transform(X_processed)
        
        # 应用标准化（如果之前使用过）
        if scaler is not None:
            X_processed = scaler.transform(X_processed)
        
        predictions = model.predict(X_processed)
        
        return {
            'predictions': predictions.tolist(),
            'processed_features': X_processed.tolist()
        }
    
    def analyze_coefficients(self, model, feature_names=None):
        """
        分析回归系数
        
        Args:
            model: 训练好的模型
            feature_names: 特征名称列表
            
        Returns:
            coefficient_analysis: 系数分析结果
        """
        if not hasattr(model, 'coef_'):
            return None
        
        coef = model.coef_
        intercept = model.intercept_
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(coef))]
        
        # 系数重要性（绝对值）
        coef_importance = np.abs(coef)
        sorted_indices = np.argsort(coef_importance)[::-1]
        
        coefficient_analysis = {
            'intercept': float(intercept),
            'coefficients': [
                {
                    'feature_name': feature_names[i],
                    'coefficient': float(coef[i]),
                    'abs_coefficient': float(abs(coef[i])),
                    'importance_rank': int(rank + 1)
                }
                for rank, i in enumerate(sorted_indices)
            ],
            'positive_coefficients': len(coef[coef > 0]),
            'negative_coefficients': len(coef[coef < 0]),
            'zero_coefficients': len(coef[coef == 0])
        }
        
        return coefficient_analysis
    
    def cross_validate_polynomial_degree(self, X, y, max_degree=5, cv_folds=5):
        """
        交叉验证选择最佳多项式次数
        
        Args:
            X: 特征矩阵
            y: 目标向量
            max_degree: 最大多项式次数
            cv_folds: 交叉验证折数
            
        Returns:
            degree_validation: 多项式次数验证结果
        """
        from sklearn.model_selection import cross_val_score
        
        degree_results = []
        
        for degree in range(1, max_degree + 1):
            # 多项式特征变换
            if degree > 1:
                poly = PolynomialFeatures(degree=degree, include_bias=False)
                X_poly = poly.fit_transform(X)
            else:
                X_poly = X
            
            # 创建模型
            model = LinearRegression()
            
            # 交叉验证
            cv_scores = cross_val_score(model, X_poly, y, cv=cv_folds, 
                                      scoring='neg_mean_squared_error')
            cv_r2_scores = cross_val_score(model, X_poly, y, cv=cv_folds, 
                                         scoring='r2')
            
            degree_results.append({
                'degree': degree,
                'n_features': X_poly.shape[1],
                'cv_mse': float(-cv_scores.mean()),
                'cv_mse_std': float(cv_scores.std()),
                'cv_r2': float(cv_r2_scores.mean()),
                'cv_r2_std': float(cv_r2_scores.std())
            })
        
        # 找到最佳次数（基于R²）
        best_degree = max(degree_results, key=lambda x: x['cv_r2'])['degree']
        
        return {
            'degree_results': degree_results,
            'best_degree': best_degree,
            'validation_metric': 'cross_validated_r2'
        }
    
    def feature_importance_analysis(self, model, X, y, feature_names=None):
        """
        特征重要性分析（基于系数绝对值）
        
        Args:
            model: 训练好的模型
            X: 特征矩阵
            y: 目标向量
            feature_names: 特征名称列表
            
        Returns:
            importance_analysis: 特征重要性分析
        """
        if not hasattr(model, 'coef_'):
            return None
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        # 标准化系数（标准化后的系数更具可比性）
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 重新训练模型以获得标准化系数
        temp_model = LinearRegression()
        temp_model.fit(X_scaled, y)
        
        standardized_coef = temp_model.coef_
        importance_scores = np.abs(standardized_coef)
        
        # 排序
        sorted_indices = np.argsort(importance_scores)[::-1]
        
        importance_analysis = {
            'feature_importance': [
                {
                    'feature_name': feature_names[i],
                    'feature_index': int(i),
                    'raw_coefficient': float(model.coef_[i]),
                    'standardized_coefficient': float(standardized_coef[i]),
                    'importance_score': float(importance_scores[i]),
                    'rank': int(rank + 1)
                }
                for rank, i in enumerate(sorted_indices)
            ],
            'total_importance': float(importance_scores.sum()),
            'max_importance': float(importance_scores.max()),
            'importance_distribution': {
                'mean': float(importance_scores.mean()),
                'std': float(importance_scores.std()),
                'min': float(importance_scores.min()),
                'max': float(importance_scores.max())
            }
        }
        
        return importance_analysis
    
    def _create_normal_equation_model(self, fit_intercept=True):
        """
        创建使用正则方程的自定义模型
        
        Args:
            fit_intercept: 是否拟合截距
            
        Returns:
            model: 自定义模型对象
        """
        class NormalEquationModel:
            def __init__(self, fit_intercept=True):
                self.fit_intercept = fit_intercept
                self.coef_ = None
                self.intercept_ = 0.0
            
            def fit(self, X, y):
                """使用正则方程训练模型"""
                if self.fit_intercept:
                    # 添加偏置项
                    X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
                    # 正则方程: theta = (X^T * X)^-1 * X^T * y
                    try:
                        XtX = X_with_intercept.T @ X_with_intercept
                        # 添加正则化防止奇异矩阵
                        XtX += np.eye(XtX.shape[0]) * 1e-10
                        theta = np.linalg.solve(XtX, X_with_intercept.T @ y)
                        self.intercept_ = theta[0]
                        self.coef_ = theta[1:]
                    except np.linalg.LinAlgError:
                        # 如果正则方程失败，使用伪逆
                        theta = np.linalg.pinv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
                        self.intercept_ = theta[0]
                        self.coef_ = theta[1:]
                else:
                    # 不拟合截距
                    try:
                        XtX = X.T @ X
                        XtX += np.eye(XtX.shape[0]) * 1e-10
                        self.coef_ = np.linalg.solve(XtX, X.T @ y)
                    except np.linalg.LinAlgError:
                        self.coef_ = np.linalg.pinv(X.T @ X) @ X.T @ y
                    self.intercept_ = 0.0
                
                return self
            
            def predict(self, X):
                """预测"""
                return X @ self.coef_ + self.intercept_
        
        return NormalEquationModel(fit_intercept)
    
    def get_supported_optimizers(self):
        """
        获取支持的优化器列表
        
        Returns:
            optimizers: 支持的优化器列表
        """
        return {
            'auto': '自动选择（sklearn默认）',
            'ols': '普通最小二乘法',
            'sgd': '随机梯度下降',
            'normal_equation': '正则方程直接解'
        }
