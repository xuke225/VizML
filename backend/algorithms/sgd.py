"""
随机梯度下降(SGD)算法模块
包含多种现代优化器和实时训练功能的SGD算法实现
"""

import numpy as np
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import logging
import time
import copy
from typing import Dict, List, Tuple, Optional

from .base import BaseRegressionAlgorithm

logger = logging.getLogger(__name__)


class BaseOptimizer:
    """优化器基类"""
    
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        
    def update(self, params, gradients):
        """更新参数"""
        raise NotImplementedError("子类必须实现update方法")
        
    def get_state(self):
        """获取优化器状态"""
        return {'learning_rate': self.learning_rate}


class SGDOptimizer(BaseOptimizer):
    """标准SGD优化器"""
    
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)
        
    def update(self, params, gradients):
        """SGD参数更新"""
        new_params = params - self.learning_rate * gradients
        return new_params


class MomentumOptimizer(BaseOptimizer):
    """带动量的SGD优化器"""
    
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = None
        
    def update(self, params, gradients):
        """Momentum参数更新"""
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
            
        self.velocity = self.momentum * self.velocity - self.learning_rate * gradients
        new_params = params + self.velocity
        return new_params
        
    def get_state(self):
        state = super().get_state()
        state.update({
            'momentum': self.momentum,
            'velocity_norm': float(np.linalg.norm(self.velocity)) if self.velocity is not None else 0.0
        })
        return state


class AdaGradOptimizer(BaseOptimizer):
    """AdaGrad优化器"""
    
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.accumulated_gradients = None
        
    def update(self, params, gradients):
        """AdaGrad参数更新"""
        if self.accumulated_gradients is None:
            self.accumulated_gradients = np.zeros_like(params)
            
        self.accumulated_gradients += gradients ** 2
        adapted_lr = self.learning_rate / (np.sqrt(self.accumulated_gradients) + self.epsilon)
        new_params = params - adapted_lr * gradients
        return new_params
        
    def get_state(self):
        state = super().get_state()
        state.update({
            'epsilon': self.epsilon,
            'accumulated_grad_norm': float(np.linalg.norm(self.accumulated_gradients)) if self.accumulated_gradients is not None else 0.0
        })
        return state


class RMSpropOptimizer(BaseOptimizer):
    """RMSprop优化器"""
    
    def __init__(self, learning_rate=0.01, decay_rate=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.moving_avg_squared = None
        
    def update(self, params, gradients):
        """RMSprop参数更新"""
        if self.moving_avg_squared is None:
            self.moving_avg_squared = np.zeros_like(params)
            
        self.moving_avg_squared = (self.decay_rate * self.moving_avg_squared + 
                                 (1 - self.decay_rate) * gradients ** 2)
        adapted_lr = self.learning_rate / (np.sqrt(self.moving_avg_squared) + self.epsilon)
        new_params = params - adapted_lr * gradients
        return new_params
        
    def get_state(self):
        state = super().get_state()
        state.update({
            'decay_rate': self.decay_rate,
            'epsilon': self.epsilon,
            'moving_avg_norm': float(np.linalg.norm(self.moving_avg_squared)) if self.moving_avg_squared is not None else 0.0
        })
        return state


class AdamOptimizer(BaseOptimizer):
    """Adam优化器（改进版）"""
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2  
        self.epsilon = max(epsilon, 1e-10)  # 确保epsilon不会太小
        self.m = None  # 一阶矩估计
        self.v = None  # 二阶矩估计
        self.t = 0     # 时间步
        
    def update(self, params, gradients):
        """Adam参数更新"""
        self.t += 1
        
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
            
        # 更新一阶和二阶矩估计
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradients ** 2
        
        # 偏差校正（添加数值稳定性保护）
        m_corrected = self.m / max(1 - self.beta1 ** self.t, self.epsilon)
        v_corrected = self.v / max(1 - self.beta2 ** self.t, self.epsilon)
        
        # 参数更新
        new_params = params - self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)
        return new_params
        
    def get_state(self):
        state = super().get_state()
        
        # 安全计算范数，防止无效值
        m_norm = 0.0
        v_norm = 0.0
        
        if self.m is not None:
            m_norm_val = np.linalg.norm(self.m)
            m_norm = float(m_norm_val) if np.isfinite(m_norm_val) else 0.0
            
        if self.v is not None:
            v_norm_val = np.linalg.norm(self.v)
            v_norm = float(v_norm_val) if np.isfinite(v_norm_val) else 0.0
        
        state.update({
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon,
            't': self.t,
            'm_norm': m_norm,
            'v_norm': v_norm,
            'bias_correction_1': max(1 - self.beta1 ** self.t, self.epsilon) if self.t > 0 else 1.0,
            'bias_correction_2': max(1 - self.beta2 ** self.t, self.epsilon) if self.t > 0 else 1.0
        })
        return state


class NesterovOptimizer(BaseOptimizer):
    """Nesterov动量优化器"""
    
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = None
        
    def update(self, params, gradients):
        """Nesterov动量参数更新"""
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
            
        # Nesterov动量更新
        prev_velocity = copy.deepcopy(self.velocity)
        self.velocity = self.momentum * self.velocity - self.learning_rate * gradients
        new_params = params + self.momentum * (self.velocity - prev_velocity) + self.velocity
        return new_params
        
    def get_state(self):
        state = super().get_state()
        state.update({
            'momentum': self.momentum,
            'velocity_norm': float(np.linalg.norm(self.velocity)) if self.velocity is not None else 0.0
        })
        return state




class SGDAlgorithm(BaseRegressionAlgorithm):
    """随机梯度下降算法类"""
    
    def __init__(self):
        super().__init__()
        self.optimizers = {
            'sgd': SGDOptimizer,
            'momentum': MomentumOptimizer,
            'adagrad': AdaGradOptimizer,
            'rmsprop': RMSpropOptimizer,
            'adam': AdamOptimizer,
            'nesterov': NesterovOptimizer
        }
        # 训练状态管理
        self.training_states = {}  # 存储多个训练会话的状态
        self.polynomial_features = None  # 多项式特征生成器
    
    def train_model(self, X, y, model_type='linear', optimizer='sgd', learning_rate=0.01, max_iter=100, 
                   momentum=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8, 
                   decay_rate=0.9, batch_size=None, scale_features=True, 
                   polynomial_degree=2, **kwargs):
        """
        使用指定优化器训练模型
        
        Args:
            X: 特征矩阵
            y: 目标向量
            model_type: 模型类型 ('linear', 'polynomial')
            optimizer: 优化器类型 ('sgd', 'momentum', 'adagrad', 'rmsprop', 'adam', 'nesterov')
            learning_rate: 学习率
            max_iter: 最大迭代次数
            momentum: 动量参数 (用于momentum和nesterov)
            beta1, beta2: Adam参数
            epsilon: 数值稳定性参数
            decay_rate: RMSprop衰减率
            batch_size: 批次大小，None表示使用全部数据
            scale_features: 是否标准化特征
            polynomial_degree: 多项式度数（当model_type='polynomial'时使用）
            **kwargs: 其他参数
            
        Returns:
            result: 训练结果字典
        """
        logger.info(f"训练SGD模型: model_type={model_type}, optimizer={optimizer}, learning_rate={learning_rate}")
        logger.debug(f"输入数据形状: X={X.shape}, y={y.shape}")
        
        # 数据预处理 - 特征变换（多项式扩展等）
        X_processed, y_processed = self._preprocess_data(X, y, model_type, polynomial_degree)
        logger.debug(f"预处理后数据形状: X_processed={X_processed.shape}, y_processed={y_processed.shape}")
        
        # 数据分割
        test_size = kwargs.get('test_size', 0.2)
        X_train, X_test, y_train, y_test = self.split_data(X_processed, y_processed, test_size=test_size)
        
        # 特征标准化（在数据分割后进行）
        if scale_features:
            X_train, X_test = self.scale_features(X_train, X_test)
            logger.info(f"特征标准化完成，训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")
        
        # 创建优化器
        optimizer_params = {'learning_rate': learning_rate}
        if optimizer in ['momentum', 'nesterov']:
            optimizer_params['momentum'] = momentum
        elif optimizer == 'adagrad':
            optimizer_params['epsilon'] = epsilon
        elif optimizer == 'rmsprop':
            optimizer_params['decay_rate'] = decay_rate
            optimizer_params['epsilon'] = epsilon
        elif optimizer == 'adam':
            optimizer_params.update({'beta1': beta1, 'beta2': beta2, 'epsilon': epsilon})
            
        optimizer_obj = self.optimizers[optimizer](**optimizer_params)
        
        # 训练模型
        model, training_history = self._train_with_optimizer(
            X_train, y_train, optimizer_obj, max_iter, batch_size, 
            model_type=model_type, **kwargs
        )
        
        # 预测和评估
        y_pred_train = self._predict_model(model, X_train, model_type)
        y_pred_test = self._predict_model(model, X_test, model_type)
        
        # 计算评估指标
        train_metrics = self.evaluate_regression(y_train, y_pred_train)
        test_metrics = self.evaluate_regression(y_test, y_pred_test)
        
        metrics = {f'train_{k}': v for k, v in train_metrics.items()}
        metrics.update({f'test_{k}': v for k, v in test_metrics.items()})
        
        # 生成预测曲线
        prediction_curve = None
        if X.shape[1] == 1:
            prediction_curve = self._generate_prediction_curve(
                model, X, model_type, polynomial_degree, 
                scale_features, self.scaler if scale_features else None
            )
        
        # 训练信息
        training_info = {
            'model_type': model_type,
            'optimizer': optimizer,
            'learning_rate': learning_rate,
            'max_iter': max_iter,
            'actual_iter': len(training_history),
            'batch_size': batch_size or len(X_train)
        }
        
        if model_type == 'linear':
            training_info.update({
                'final_coefficients': model['weights'].tolist(),
                'final_intercept': float(model['bias'])
            })
        elif model_type == 'polynomial':
            training_info.update({
                'polynomial_degree': polynomial_degree,
                'final_coefficients': model['weights'].tolist(),
                'final_intercept': float(model['bias'])
            })
        
        # 添加优化器特定参数
        training_info.update(optimizer_params)
        
        result = {
            'model': model,
            'metrics': metrics,
            'predictions': {'train': y_pred_train.tolist(), 'test': y_pred_test.tolist()},
            'training_history': training_history,
            'training_info': training_info,
            'prediction_curve': prediction_curve,
            'data_info': {
                'train_size': len(X_train),
                'test_size': len(X_test),
                'n_features': X.shape[1]
            },
            'train_data': {'X': X_train.tolist(), 'y': y_train.tolist()},
            'test_data': {'X': X_test.tolist(), 'y': y_test.tolist()},
            'scaler': self.scaler if scale_features else None
        }
        
        return result
    
    def _preprocess_data(self, X, y, model_type, polynomial_degree=2):
        """
        数据预处理 - 仅处理特征变换，不包括标准化
        
        Args:
            X: 原始特征矩阵
            y: 目标向量
            model_type: 模型类型
            polynomial_degree: 多项式度数
            
        Returns:
            X_processed: 处理后的特征矩阵
            y_processed: 处理后的目标向量
        """
        X_processed = X.copy()
        y_processed = y.copy()
        
        if model_type == 'polynomial':
            # 生成多项式特征
            self.polynomial_features = PolynomialFeatures(
                degree=polynomial_degree, 
                include_bias=False  # 我们会手动添加偏置项
            )
            X_processed = self.polynomial_features.fit_transform(X_processed)
            logger.info(f"多项式特征生成完成，特征数量: {X.shape[1]} -> {X_processed.shape[1]}")
        else:
            # 确保线性模型时清空多项式特征生成器
            self.polynomial_features = None
        
        return X_processed, y_processed
    
    def _predict_model(self, model, X, model_type):
        """根据模型类型进行预测"""
        # 线性或多项式模型
        return X @ model['weights'] + model['bias']
    
    def _train_with_optimizer(self, X, y, optimizer, max_iter, batch_size=None, 
                             model_type='linear', **kwargs):
        """
        使用指定优化器训练模型
        
        Args:
            X: 特征矩阵
            y: 目标向量
            optimizer: 优化器实例
            max_iter: 最大迭代次数
            batch_size: 批次大小
            model_type: 模型类型
            
        Returns:
            model: 训练好的模型字典
            training_history: 训练历史
        """
        m, n = X.shape
        
        # 初始化线性或多项式模型
        weights = np.random.normal(0, 0.01, n)
        bias = 0.0
        
        # 训练历史记录
        training_history = []
        
        # 批次设置
        if batch_size is None:
            batch_size = m
        
        for epoch in range(max_iter):
            epoch_loss = 0
            epoch_start_time = time.time()
            
            # 随机打乱数据
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # 分批训练
            for i in range(0, m, batch_size):
                batch_end = min(i + batch_size, m)
                X_batch = X_shuffled[i:batch_end]
                y_batch = y_shuffled[i:batch_end]
                
                # 线性/多项式模型训练
                predictions = X_batch @ weights + bias
                loss = np.mean((predictions - y_batch) ** 2)
                epoch_loss += loss * len(X_batch)
                
                # 反向传播：计算梯度
                batch_size_actual = len(X_batch)
                weight_gradients = 2 / batch_size_actual * X_batch.T @ (predictions - y_batch)
                bias_gradient = 2 / batch_size_actual * np.sum(predictions - y_batch)
                
                # 合并梯度
                gradients = np.concatenate([weight_gradients, [bias_gradient]])
                current_params = np.concatenate([weights, [bias]])
                
                # 使用优化器更新参数
                new_params = optimizer.update(current_params, gradients)
                weights = new_params[:-1]
                bias = new_params[-1]
            
            # 计算整个epoch的平均损失
            epoch_loss /= m
            
            # 计算R²
            all_predictions = X @ weights + bias
            r2 = 1 - (np.sum((y - all_predictions) ** 2) / np.sum((y - np.mean(y)) ** 2))
            
            # 记录训练历史
            epoch_time = time.time() - epoch_start_time
            history_entry = {
                'epoch': epoch + 1,
                'loss': float(epoch_loss),
                'r2': float(r2),
                'gradient_norm': float(np.linalg.norm(gradients)),
                'epoch_time': float(epoch_time),
                'optimizer_state': optimizer.get_state()
            }
            
            history_entry['weights'] = weights.copy().tolist()
            history_entry['bias'] = float(bias)
                
            training_history.append(history_entry)
            
            # 早停条件
            if epoch > 10 and abs(training_history[epoch]['loss'] - training_history[epoch-1]['loss']) < 1e-8:
                logger.info(f"训练在第 {epoch+1} 轮后收敛")
                break
        
        model = {
            'weights': weights,
            'bias': bias,
            'optimizer': optimizer,
            'model_type': model_type
        }
        
        return model, training_history
    
    def _predict_linear(self, model, X):
        """使用线性模型进行预测（保留以兼容旧代码）"""
        return self._predict_model(model, X, model.get('model_type', 'linear'))
    
    def _generate_prediction_curve(self, model, X, model_type, polynomial_degree=2, scale_features=True, scaler=None):
        """
        生成预测曲线数据 - 确保与训练时的预处理流程完全一致
        
        Args:
            model: 训练好的模型
            X: 原始特征矩阵 (1D)
            model_type: 模型类型 ('linear' 或 'polynomial')
            polynomial_degree: 多项式度数
            scale_features: 是否使用标准化
            scaler: StandardScaler实例
        
        Returns:
            curve_data: 预测曲线数据 {'x': [...], 'y': [...]} 或 None
        """
        if X.shape[1] != 1:
            logger.warning("预测曲线生成仅支持1维特征数据")
            return None
            
        try:
            # 生成预测点的x轴范围
            x_min, x_max = X[:, 0].min(), X[:, 0].max()
            x_range = np.linspace(x_min - 0.5, x_max + 0.5, 100).reshape(-1, 1)
            logger.debug(f"生成预测曲线x范围: [{x_min:.2f}, {x_max:.2f}] -> [{x_min-0.5:.2f}, {x_max+0.5:.2f}]")
            
            # 步骤1：应用与训练时相同的特征变换
            x_range_processed = x_range.copy()
            
            # 多项式特征扩展（如果需要）
            if model_type == 'polynomial':
                if hasattr(self, 'polynomial_features') and self.polynomial_features is not None:
                    x_range_processed = self.polynomial_features.transform(x_range_processed)
                    logger.debug(f"多项式特征扩展: {x_range.shape} -> {x_range_processed.shape}")
                else:
                    logger.error(f"多项式模型但未找到polynomial_features实例")
                    return None
            
            # 步骤2：标准化处理（如果需要）
            if scale_features and scaler is not None:
                original_shape = x_range_processed.shape
                x_range_processed = scaler.transform(x_range_processed)
                logger.debug(f"特征标准化: {original_shape} -> {x_range_processed.shape}")
            elif scale_features and scaler is None:
                logger.warning("要求标准化但未提供scaler实例")
            
            # 步骤3：生成预测
            y_curve = self._predict_model(model, x_range_processed, model_type)
            
            curve_data = {
                'x': x_range[:, 0].tolist(),
                'y': y_curve.tolist()
            }
            
            logger.debug(f"预测曲线生成成功，数据点数量: {len(curve_data['x'])}")
            return curve_data
            
        except Exception as e:
            logger.error(f"生成预测曲线失败: {type(e).__name__}: {str(e)}")
            # 添加更详细的错误信息
            if hasattr(e, '__cause__') and e.__cause__:
                logger.error(f"根本原因: {type(e.__cause__).__name__}: {str(e.__cause__)}")
            return None
    
    def start_training_session(self, session_key, X, y, optimizers_config, scale_features=True, **kwargs):
        """
        启动多优化器训练会话
        
        Args:
            session_key: 会话标识
            X: 特征矩阵
            y: 目标向量
            optimizers_config: 优化器配置字典
            scale_features: 是否标准化特征
            
        Returns:
            session_info: 会话信息
        """
        logger.info(f"启动训练会话: {session_key}")
        
        # 获取模型配置（从第一个优化器配置中提取，因为所有优化器应该使用相同的模型类型）
        first_optimizer_config = next(iter(optimizers_config.values())) if optimizers_config else {}
        model_type = first_optimizer_config.get('model_type', 'linear')
        polynomial_degree = first_optimizer_config.get('polynomial_degree', 2)
        
        # 数据预处理 - 特征变换（多项式扩展等）
        X_processed, y_processed = self._preprocess_data(X, y, model_type, polynomial_degree)
        
        # 数据分割
        test_size = kwargs.get('test_size', 0.2)
        X_train, X_test, y_train, y_test = self.split_data(X_processed, y_processed, test_size=test_size)
        
        # 特征标准化（在数据分割后进行）
        if scale_features:
            X_train, X_test = self.scale_features(X_train, X_test)
            logger.info(f"训练会话特征标准化完成，训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")
        
        # 初始化训练状态
        training_state = {
            'session_key': session_key,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'original_X': X,  # 保存原始数据用于曲线生成
            'original_X_shape': X.shape,
            'model_type': model_type,  # 保存模型类型
            'polynomial_degree': polynomial_degree,  # 保存多项式度数
            'scale_features': scale_features,
            'scaler': self.scaler if scale_features and hasattr(self, 'scaler') else None,  # 保存缩放器实例
            'polynomial_features': self.polynomial_features if model_type == 'polynomial' and hasattr(self, 'polynomial_features') else None,  # 保存多项式特征转换器
            'optimizers': {},
            'training_history': {},
            'current_epoch': 0,
            'is_training': False,
            'start_time': time.time()
        }
        
        # 初始化优化器
        for opt_name, config in optimizers_config.items():
            if opt_name in self.optimizers:
                optimizer_class = self.optimizers[opt_name]
                # 过滤出优化器专用参数，排除模型相关参数
                model_keys = {'model_type', 'polynomial_degree', 'hidden_size', 'max_iter'}
                optimizer_params = {k: v for k, v in config.items() if k not in model_keys}
                optimizer = optimizer_class(**optimizer_params)
                
                # 初始化模型参数 - 使用预处理后的特征维度
                n_features = X_train.shape[1]  # 这是预处理后的特征数量
                model_type_for_init = config.get('model_type', model_type)  # 使用配置中的模型类型
                
                # 线性和多项式模型，都使用线性参数结构（多项式的非线性通过特征扩展实现）
                weights = np.random.normal(0, 0.01, n_features)
                bias = 0.0
                model = {'weights': weights, 'bias': bias, 'model_type': model_type_for_init}
                
                training_state['optimizers'][opt_name] = {
                    'optimizer': optimizer,
                    'model': model,
                    'config': config
                }
                training_state['training_history'][opt_name] = []
        
        # 存储训练状态
        self.training_states[session_key] = training_state
        
        return {
            'session_key': session_key,
            'optimizers': list(optimizers_config.keys()),
            'data_shape': X_train.shape,
            'status': 'initialized'
        }
    
    def train_step(self, session_key, batch_size=None):
        """
        执行一步训练（所有优化器同时训练一个epoch）
        
        Args:
            session_key: 会话标识
            batch_size: 批次大小
            
        Returns:
            step_result: 训练步骤结果
        """
        if session_key not in self.training_states:
            raise ValueError(f"训练会话不存在: {session_key}")
        
        state = self.training_states[session_key]
        state['current_epoch'] += 1
        epoch = state['current_epoch']
        
        X_train = state['X_train']
        y_train = state['y_train']
        m = len(X_train)
        
        if batch_size is None:
            batch_size = m
        
        step_results = {}
        
        # 为所有优化器执行一个epoch的训练
        for opt_name, opt_data in state['optimizers'].items():
            optimizer = opt_data['optimizer']
            model = opt_data['model']
            model_type = model.get('model_type', 'linear')
            
            epoch_loss = 0
            epoch_start_time = time.time()
            
            # 随机打乱数据
            indices = np.random.permutation(m)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # 分批训练
            for i in range(0, m, batch_size):
                batch_end = min(i + batch_size, m)
                X_batch = X_shuffled[i:batch_end]
                y_batch = y_shuffled[i:batch_end]
                
                # 线性/多项式模型训练
                predictions = X_batch @ model['weights'] + model['bias']
                loss = np.mean((predictions - y_batch) ** 2)
                epoch_loss += loss * len(X_batch)
                
                batch_size_actual = len(X_batch)
                weight_gradients = 2 / batch_size_actual * X_batch.T @ (predictions - y_batch)
                bias_gradient = 2 / batch_size_actual * np.sum(predictions - y_batch)
                
                gradients = np.concatenate([weight_gradients, [bias_gradient]])
                current_params = np.concatenate([model['weights'], [model['bias']]])
                
                new_params = optimizer.update(current_params, gradients)
                model['weights'] = new_params[:-1]
                model['bias'] = new_params[-1]
            
            # 计算指标
            epoch_loss /= m
            all_predictions = X_train @ model['weights'] + model['bias']
            r2 = 1 - (np.sum((y_train - all_predictions) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2))
            
            # 记录历史
            epoch_time = time.time() - epoch_start_time
            history_entry = {
                'epoch': epoch,
                'loss': float(epoch_loss),
                'r2': float(r2),
                'gradient_norm': float(np.linalg.norm(gradients)),
                'epoch_time': float(epoch_time),
                'optimizer_state': optimizer.get_state()
            }
            
            history_entry['weights'] = model['weights'].copy().tolist()
            history_entry['bias'] = float(model['bias'])
            
            state['training_history'][opt_name].append(history_entry)
            step_results[opt_name] = history_entry
        
        return {
            'epoch': epoch,
            'results': step_results,
            'status': 'training'
        }
    
    def get_training_state(self, session_key):
        """获取训练状态"""
        if session_key not in self.training_states:
            return {'status': 'not_found'}
        
        state = self.training_states[session_key]
        
        # 计算当前预测曲线（仅支持1D特征数据）
        prediction_curves = {}
        original_shape = state.get('original_X_shape', (0, 0))
        # 检查原始数据是否为1D特征（第二个维度为1）
        if len(original_shape) >= 2 and original_shape[1] == 1:
            # 临时设置多项式特征转换器（如果存在）
            if state.get('polynomial_features'):
                self.polynomial_features = state['polynomial_features']
                
            for opt_name, opt_data in state['optimizers'].items():
                model = opt_data['model']
                model_type = state.get('model_type', model.get('model_type', 'linear'))
                polynomial_degree = state.get('polynomial_degree', 2)
                original_X = state.get('original_X')
                scale_features = state.get('scale_features', False)
                scaler = state.get('scaler', None)
                
                curve = self._generate_prediction_curve(
                    model, 
                    original_X, 
                    model_type, 
                    polynomial_degree,
                    scale_features,
                    scaler
                )
                prediction_curves[opt_name] = curve
        
        return {
            'session_key': session_key,
            'current_epoch': state['current_epoch'],
            'training_history': state['training_history'],
            'prediction_curves': prediction_curves,
            'data_info': {
                'train_size': len(state['X_train']),
                'test_size': len(state['X_test']),
                'n_features': state['X_train'].shape[1]
            },
            'status': 'active'
        }
    
    def compare_optimizers(self, X, y, optimizers_config, max_iter=50, **kwargs):
        """
        比较多个优化器的性能
        
        Args:
            X: 特征矩阵
            y: 目标向量
            optimizers_config: 优化器配置
            max_iter: 最大迭代次数
            
        Returns:
            comparison_result: 比较结果
        """
        logger.info("开始优化器性能比较")
        
        results = {}
        
        for opt_name, config in optimizers_config.items():
            try:
                # 为每个优化器单独训练
                config_copy = config.copy()
                config_copy['max_iter'] = max_iter
                
                # 处理参数冲突，优先使用优化器特定配置
                merged_params = {**kwargs}
                merged_params.update(config_copy)  # 优化器配置覆盖全局配置
                
                result = self.train_model(X, y, optimizer=opt_name, **merged_params)
                
                results[opt_name] = {
                    'final_train_loss': result['training_history'][-1]['loss'],
                    'final_test_loss': result['metrics']['test_mse'],
                    'final_r2': result['metrics']['test_r2'],
                    'convergence_epochs': len(result['training_history']),
                    'training_history': result['training_history'],
                    'prediction_curve': result.get('prediction_curve'),
                    'final_params': {
                        'weights': result['model']['weights'].tolist(),
                        'bias': result['model']['bias']
                    }
                }
                
            except Exception as e:
                logger.error(f"优化器 {opt_name} 训练失败: {e}")
                results[opt_name] = {'error': str(e)}
        
        # 找到最佳优化器（基于测试R²）
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        best_optimizer = None
        if valid_results:
            best_optimizer = max(valid_results.keys(), key=lambda k: valid_results[k]['final_r2'])
        
        return {
            'results': results,
            'best_optimizer': best_optimizer,
            'comparison_metric': 'test_r2',
            'total_optimizers': len(optimizers_config),
            'successful_optimizers': len(valid_results)
        }
    
    def _train_sgd_regression(self, X, y, learning_rate=0.01, max_iter=100, 
                             loss='squared_loss', penalty='l2', alpha=0.0001, **kwargs):
        """训练SGD回归模型"""
        model = SGDRegressor(
            learning_rate='constant',
            eta0=learning_rate,
            max_iter=max_iter,
            loss=loss,
            penalty=penalty,
            alpha=alpha,
            random_state=42,
            warm_start=True  # 允许增量训练
        )
        
        # 记录训练历史
        training_history = []
        
        for i in range(max_iter):
            # 设置当前迭代次数
            model.max_iter = i + 1
            model.fit(X, y)
            
            # 计算当前损失
            y_pred = model.predict(X)
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            training_history.append({
                'iteration': i + 1,
                'mse': float(mse),
                'r2': float(r2),
                'coefficients': model.coef_.tolist(),
                'intercept': float(model.intercept_)
            })
            
            # 早停条件
            if i > 10 and abs(training_history[i]['mse'] - training_history[i-1]['mse']) < 1e-6:
                logger.info(f"SGD在第 {i+1} 次迭代后收敛")
                break
        
        return model, training_history
    
    def _train_custom_gradient_descent(self, X, y, learning_rate=0.01, max_iter=100, **kwargs):
        """
        自定义梯度下降实现
        
        Args:
            X: 特征矩阵
            y: 目标值向量
            learning_rate: 学习率
            max_iter: 最大迭代次数
            
        Returns:
            model: 自定义模型对象
            training_history: 训练历史
        """
        # 添加偏置项
        X_bias = np.c_[np.ones(X.shape[0]), X]
        
        # 初始化参数
        theta = np.random.normal(0, 0.01, X_bias.shape[1])
        m = len(y)
        
        training_history = []
        
        for i in range(max_iter):
            # 前向传播
            predictions = X_bias @ theta
            
            # 计算损失
            mse = np.mean((predictions - y) ** 2)
            r2 = 1 - (np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2))
            
            # 计算梯度
            gradients = 2/m * X_bias.T @ (predictions - y)
            
            # 更新参数
            theta -= learning_rate * gradients
            
            # 记录历史
            training_history.append({
                'iteration': i + 1,
                'mse': float(mse),
                'r2': float(r2),
                'coefficients': theta[1:].tolist(),
                'intercept': float(theta[0]),
                'gradients': gradients.tolist(),
                'gradient_norm': float(np.linalg.norm(gradients))
            })
            
            # 早停条件
            if i > 0 and abs(training_history[i]['mse'] - training_history[i-1]['mse']) < 1e-8:
                logger.info(f"梯度下降在第 {i+1} 次迭代后收敛")
                break
        
        # 创建自定义模型对象
        model = self._create_custom_model(theta[1:], theta[0])
        
        return model, training_history
    
    def _train_normal_equation(self, X, y, **kwargs):
        """
        使用正规方程求解线性回归
        
        Args:
            X: 特征矩阵
            y: 目标值向量
            **kwargs: 其他参数
            
        Returns:
            model: 自定义模型对象
            training_history: 训练历史（单次求解）
        """
        logger.info("使用正规方程求解线性回归")
        
        # 添加偏置项
        X_bias = np.c_[np.ones(X.shape[0]), X]
        
        # 正规方程: θ = (X^T * X)^(-1) * X^T * y
        try:
            theta = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y
        except np.linalg.LinAlgError:
            # 如果矩阵不可逆，使用伪逆
            logger.warning("矩阵不可逆，使用伪逆求解")
            theta = np.linalg.pinv(X_bias.T @ X_bias) @ X_bias.T @ y
        
        # 计算预测和损失
        predictions = X_bias @ theta
        mse = np.mean((predictions - y) ** 2)
        r2 = 1 - (np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2))
        
        # 训练历史（单次求解）
        training_history = [{
            'iteration': 1,
            'mse': float(mse),
            'r2': float(r2),
            'coefficients': theta[1:].tolist(),
            'intercept': float(theta[0]),
            'method': 'normal_equation'
        }]
        
        # 创建自定义模型对象
        model = self._create_custom_model(theta[1:], theta[0])
        
        return model, training_history
    
    def _train_least_squares(self, X, y, **kwargs):
        """
        最小二乘法求解（与正规方程等价）
        
        Args:
            X: 特征矩阵
            y: 目标值向量
            **kwargs: 其他参数
            
        Returns:
            model: sklearn LinearRegression 模型
            training_history: 训练历史（单次求解）
        """
        from sklearn.linear_model import LinearRegression
        
        logger.info("使用最小二乘法求解线性回归")
        
        # 使用 sklearn 的 LinearRegression（内部使用最小二乘法）
        model = LinearRegression(fit_intercept=True)
        model.fit(X, y)
        
        # 计算预测和损失
        predictions = model.predict(X)
        mse = np.mean((predictions - y) ** 2)
        r2 = r2_score(y, predictions)
        
        # 训练历史（单次求解）
        training_history = [{
            'iteration': 1,
            'mse': float(mse),
            'r2': float(r2),
            'coefficients': model.coef_.tolist(),
            'intercept': float(model.intercept_),
            'method': 'least_squares'
        }]
        
        return model, training_history
    
    def _create_custom_model(self, coefficients, intercept):
        """创建自定义模型对象"""
        class CustomLinearModel:
            def __init__(self, coefficients, intercept):
                self.coef_ = np.array(coefficients)
                self.intercept_ = intercept
                
            def predict(self, X):
                return X @ self.coef_ + self.intercept_
        
        return CustomLinearModel(coefficients, intercept)
    
    def analyze_convergence(self, training_history):
        """
        分析训练收敛情况
        
        Args:
            training_history: 训练历史
            
        Returns:
            convergence_analysis: 收敛分析结果
        """
        if len(training_history) < 2:
            return {'status': 'insufficient_data'}
        
        # 提取损失值
        losses = [hist['mse'] for hist in training_history]
        r2_scores = [hist['r2'] for hist in training_history]
        
        # 计算损失变化率
        loss_changes = [abs(losses[i] - losses[i-1]) for i in range(1, len(losses))]
        
        # 判断收敛状态
        final_change = loss_changes[-1] if loss_changes else float('inf')
        avg_change_last_10 = np.mean(loss_changes[-10:]) if len(loss_changes) >= 10 else np.mean(loss_changes)
        
        convergence_status = 'converged' if final_change < 1e-6 else 'not_converged'
        
        convergence_analysis = {
            'status': convergence_status,
            'total_iterations': len(training_history),
            'final_loss': float(losses[-1]),
            'final_r2': float(r2_scores[-1]),
            'loss_reduction': float(losses[0] - losses[-1]),
            'loss_reduction_ratio': float((losses[0] - losses[-1]) / losses[0]) if losses[0] != 0 else 0,
            'final_change_rate': float(final_change),
            'avg_change_last_10': float(avg_change_last_10),
            'convergence_threshold': 1e-6,
            'training_stable': final_change < 1e-4
        }
        
        return convergence_analysis
    
    def predict_new_data(self, model, X, scaler=None):
        """
        使用训练好的模型预测新数据
        
        Args:
            model: 训练好的模型
            X: 新的特征矩阵
            scaler: 数据标准化器
            
        Returns:
            predictions: 预测结果
        """
        X_processed = X
        
        # 应用标准化（如果之前使用过）
        if scaler is not None:
            X_processed = scaler.transform(X_processed)
        
        predictions = model.predict(X_processed)
        
        return {
            'predictions': predictions.tolist(),
            'processed_features': X_processed.tolist()
        }
    
    def compare_algorithms(self, X, y, algorithms=None, **common_params):
        """
        比较不同SGD算法的性能
        
        Args:
            X: 特征矩阵
            y: 目标向量
            algorithms: 要比较的算法列表
            **common_params: 通用参数
            
        Returns:
            comparison_results: 算法比较结果
        """
        if algorithms is None:
            algorithms = ['sgd_regression', 'gradient_descent', 'normal_equation', 'least_squares']
        
        results = {}
        
        for algorithm in algorithms:
            try:
                logger.info(f"比较算法: {algorithm}")
                result = self.train_model(X, y, algorithm=algorithm, **common_params)
                
                results[algorithm] = {
                    'final_train_mse': result['metrics']['train_mse'],
                    'final_test_mse': result['metrics']['test_mse'],
                    'final_train_r2': result['metrics']['train_r2'],
                    'final_test_r2': result['metrics']['test_r2'],
                    'training_time': len(result['training_history']),
                    'convergence': self.analyze_convergence(result['training_history'])
                }
                
            except Exception as e:
                logger.error(f"算法 {algorithm} 训练失败: {e}")
                results[algorithm] = {'error': str(e)}
        
        # 找到最佳算法
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            best_algorithm = min(valid_results.keys(), 
                               key=lambda k: valid_results[k]['final_test_mse'])
        else:
            best_algorithm = None
        
        return {
            'algorithm_results': results,
            'best_algorithm': best_algorithm,
            'comparison_metric': 'test_mse'
        }
    
    def get_available_models(self):
        """
        获取可用的模型类型
        
        Returns:
            models: 模型信息字典
        """
        return {
            'linear': {
                'name': '线性回归',
                'description': '简单的线性模型，y = wx + b',
                'parameters': []
            },
            'polynomial': {
                'name': '多项式回归',
                'description': '多项式特征扩展的线性模型',
                'parameters': ['polynomial_degree']
            }
        }