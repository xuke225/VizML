#!/usr/bin/env python3
"""
机器学习算法可视化平台 - 后端API服务
基于Flask + scikit-learn实现稳定的算法逻辑
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import logging
import os
import uuid
import time
from datetime import date, datetime
from io import BytesIO

from openpyxl import load_workbook

# 导入数据生成模块
from backend.data_generator import DataGenerator
# 导入机器学习算法模块
from backend.algorithms.clustering import ClusteringAlgorithms
from backend.algorithms.neural_network import NeuralNetworkAlgorithms
from backend.algorithms.ensemble import EnsembleAlgorithms
from backend.algorithms.dimensionality_reduction import DimensionalityReductionAlgorithms
from backend.algorithms.svm import SVMAlgorithm
from backend.algorithms.decision_tree import DecisionTreeAlgorithm
from backend.algorithms.knn import KNNAlgorithm
from backend.algorithms.bayesian_classification import BayesianClassificationAlgorithm
from backend.algorithms.linear_regression import LinearRegressionAlgorithm
from backend.algorithms.sgd import SGDAlgorithm
from backend.algorithms.reinforcement_learning import ReinforcementLearningAlgorithm, build_environment

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__, 
            template_folder='frontend/templates',
            static_folder='frontend/static')
CORS(app)  # 启用CORS支持前后端分离

# 全局变量存储当前会话的模型和数据
session_data = {
    'models': {},
    'datasets': {},
    'scalers': {},
    'training_history': {}
}

MAX_EXCEL_FILE_SIZE = 5 * 1024 * 1024
EXCEL_PREVIEW_ROWS = 5

# 初始化算法模块
data_gen = DataGenerator()
clustering_alg = ClusteringAlgorithms()
neural_net_alg = NeuralNetworkAlgorithms()
ensemble_alg = EnsembleAlgorithms()
dim_reduction_alg = DimensionalityReductionAlgorithms()

# 初始化新的独立算法模块
svm_alg = SVMAlgorithm()
decision_tree_alg = DecisionTreeAlgorithm()
knn_alg = KNNAlgorithm()
bayesian_alg = BayesianClassificationAlgorithm()
linear_regression_alg = LinearRegressionAlgorithm()
sgd_alg = SGDAlgorithm()
rl_alg = ReinforcementLearningAlgorithm()


@app.route('/')
def index():
    """提供主页面"""
    return send_from_directory('frontend/templates', 'index.html')


@app.route('/html/<path:filename>')
def serve_html(filename):
    """提供HTML文件"""
    return send_from_directory('frontend/templates', filename)


# ================== 数据生成API ==================

@app.route('/api/generate_data', methods=['POST'])
def generate_data():
    """生成训练数据"""
    try:
        config = request.json
        dataset_type = config.get('type', 'classification')
        data_shape = config.get('shape', 'blobs')
        n_samples = config.get('n_samples', 200)
        n_classes = config.get('n_classes', 2)
        noise = config.get('noise', 0.1)
        random_state = config.get('random_state', 42)
        
        # 映射HTML传递的shape值到数据生成器期望的值
        shape_mapping = {
            # 分类数据映射
            'make_classification': 'random',
            'make_circles': 'circles', 
            'make_moons': 'moons',
            'make_blobs': 'blobs',
            'make_gaussian_quantiles': 'gaussian',
            'xor': 'xor',
            'spiral': 'spiral',
            'checkerboard': 'checkerboard',
            
            # 回归数据映射
            'linear': 'linear',
            'polynomial': 'polynomial', 
            'sinusoidal': 'sinusoidal',
            'sine': 'sinusoidal',  # ensemble.html 使用的值
            'exponential': 'exponential',
            'logarithmic': 'logarithmic',
            
            # 聚类数据映射
            'blobs': 'blobs',
            'circles': 'circles',
            'moons': 'moons',
            'anisotropic': 'anisotropic',
            'varied': 'varied',
            
            # 其他映射
            '线性分离': 'random',
            '同心圆': 'circles',
            '月牙形': 'moons', 
            '聚类数据': 'blobs',
            '分类数据': 'random',
            '随机分布': 'random',
            '高斯分布': 'gaussian',
            'XOR模式': 'xor',
            
            # sklearn 内置分类数据集名称保持原值
        }
        
        # 应用映射
        mapped_shape = shape_mapping.get(data_shape, data_shape)
        
        logger.info(f"生成数据: type={dataset_type}, shape={data_shape}->{mapped_shape}, samples={n_samples}")
        
        # 处理 sklearn 内置分类数据集
        if data_shape in ['iris', 'wine', 'breast_cancer', 'digits', 'digits_2d']:
            from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
            
            if data_shape == 'iris':
                dataset = load_iris()
            elif data_shape == 'wine':
                dataset = load_wine()
            elif data_shape == 'breast_cancer':
                dataset = load_breast_cancer()
            elif data_shape in ['digits', 'digits_2d']:
                dataset = load_digits()
            
            X = dataset.data
            y = dataset.target
            
            # 如果是多维数据，只取前两个特征用于可视化
            if X.shape[1] > 2:
                X = X[:, :2]
                
            # 确保数据类型可以JSON序列化
            X = X.astype(float)
            y = y.astype(int)
                
        elif dataset_type == 'classification':
            X, y = data_gen.generate_classification_data(
                shape=mapped_shape, 
                n_samples=n_samples, 
                n_classes=n_classes,
                noise=noise,
                random_state=random_state
            )
        elif dataset_type == 'regression':
            X, y = data_gen.generate_regression_data(
                shape=mapped_shape,
                n_samples=n_samples,
                noise=noise,
                random_state=random_state
            )
        elif dataset_type == 'clustering':
            # 检查是否为自定义手绘数据
            if config.get('custom', False) and 'data' in config:
                # 直接使用传入的手绘数据
                X = np.array(config['data'])
                logger.info(f"使用手绘数据: {X.shape}")
            else:
                X = data_gen.generate_clustering_data(
                    shape=mapped_shape,
                    n_samples=n_samples,
                    noise=noise,
                    random_state=random_state
                )
            y = None
        else:
            raise ValueError(f"不支持的数据类型: {dataset_type}")
        
        # 存储数据
        # 可选命名空间用于隔离同一页面内的多个实验状态，例如线性回归的
        # “引导学习”和“自由实验”。未传入时保持原有会话键格式兼容旧客户端。
        session_namespace = config.get('session_namespace')
        session_key = f"{dataset_type}_{data_shape}"
        if session_namespace:
            safe_namespace = ''.join(
                character for character in str(session_namespace)
                if character.isalnum() or character in ('_', '-')
            )[:40]
            if safe_namespace:
                session_key = f"{session_key}_{safe_namespace}"
        stored_data = {
            'X': X.tolist() if X is not None else None,
            'y': y.tolist() if y is not None else None,
            'config': config
        }
        
        session_data['datasets'][session_key] = stored_data
        
        # 构造返回结果
        result_data = {
            'X': X.tolist() if X is not None else None,
            'y': y.tolist() if y is not None else None,
            'n_samples': len(X) if X is not None else 0,
            'n_features': X.shape[1] if X is not None else 0,
            'n_classes': len(np.unique(y)) if y is not None else 0
        }
        
        result = {
            'status': 'success',
            'data': result_data,
            'session_key': session_key
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"数据生成错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


# ================== 聚类算法API ==================

@app.route('/api/clustering/train', methods=['POST'])
def train_clustering():
    """训练聚类模型"""
    try:
        config = request.json
        session_key = config.get('session_key')
        algorithm = config.get('algorithm', 'kmeans')
        params = config.get('params', {})
        
        # 获取数据
        if session_key not in session_data['datasets']:
            return jsonify({'status': 'error', 'message': '数据未找到，请先生成数据'}), 400
            
        dataset = session_data['datasets'][session_key]
        X = np.array(dataset['X'])
        
        logger.info(f"训练聚类模型: {algorithm}")
        
        # 训练模型
        model, labels, centers, training_info = clustering_alg.train_model(
            X, algorithm=algorithm, **params
        )
        
        # 存储模型
        model_key = f"clustering_{algorithm}_{session_key}"
        session_data['models'][model_key] = {
            'model': model,
            'algorithm': algorithm,
            'labels': labels.tolist(),
            'centers': centers.tolist() if centers is not None else None,
            'training_info': training_info
        }
        
        result = {
            'status': 'success',
            'model_key': model_key,
            'results': {
                'labels': labels.tolist(),
                'centers': centers.tolist() if centers is not None else None,
                'n_clusters': len(np.unique(labels[labels >= 0])),  # 排除噪声点(-1)
                'training_info': training_info
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"聚类训练错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/clustering/compare', methods=['POST'])
def compare_clustering_algorithms():
    """对比多个聚类算法"""
    try:
        config = request.json
        session_key = config.get('session_key')
        algorithms = config.get('algorithms', ['kmeans', 'dbscan', 'hierarchical', 'gmm'])
        algorithm_params = config.get('algorithm_params', {})
        
        # 获取数据
        if session_key not in session_data['datasets']:
            return jsonify({'status': 'error', 'message': '数据未找到，请先生成数据'}), 400
            
        dataset = session_data['datasets'][session_key]
        X = np.array(dataset['X'])
        
        logger.info(f"开始对比聚类算法: {algorithms}")
        
        # 执行算法对比
        comparison_results = clustering_alg.compare_algorithms(
            X, algorithms=algorithms, **algorithm_params
        )
        
        # 存储对比结果
        comparison_key = f"clustering_comparison_{session_key}_{int(time.time())}"
        session_data['models'][comparison_key] = {
            'type': 'clustering_comparison',
            'algorithms': algorithms,
            'results': comparison_results,
            'session_key': session_key
        }
        
        # 准备返回结果
        result_data = {
            'algorithms': {},
            'comparison_metrics': comparison_results.get('comparison_metrics', {}),
            'best_algorithm': comparison_results.get('best_algorithm'),
            'execution_times': comparison_results.get('execution_times', {})
        }
        
        # 整理算法结果数据
        for algorithm, algo_result in comparison_results['algorithms'].items():
            if 'error' not in algo_result:
                result_data['algorithms'][algorithm] = {
                    'labels': algo_result['labels'],
                    'centers': algo_result['centers'],
                    'training_info': algo_result['training_info'],
                    'execution_time': algo_result['execution_time'],
                    'n_clusters': len(np.unique(algo_result['labels'])) - (1 if -1 in algo_result['labels'] else 0)
                }
            else:
                result_data['algorithms'][algorithm] = {
                    'error': algo_result['error'],
                    'execution_time': 0
                }
        
        result = {
            'status': 'success',
            'comparison_key': comparison_key,
            'results': result_data
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"聚类算法对比错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


# ================== 聚类单步调试API ==================

@app.route('/api/clustering/kmeans/debug_init', methods=['POST'])
def kmeans_debug_init():
    """初始化K-means单步调试"""
    try:
        config = request.json
        session_key = config.get('session_key')
        n_clusters = config.get('n_clusters', 3)
        init_method = config.get('init', 'k-means++')
        max_iter = config.get('max_iter', 100)
        
        # 获取数据
        if session_key not in session_data['datasets']:
            return jsonify({'status': 'error', 'message': '数据未找到，请先生成数据'}), 400
            
        dataset = session_data['datasets'][session_key]
        X = np.array(dataset['X'])
        
        logger.info(f"初始化K-means单步调试: {n_clusters}聚类")
        
        # 初始化调试状态
        debug_state = clustering_alg.kmeans_debug_init(
            X, n_clusters=n_clusters, init=init_method, max_iter=max_iter
        )
        
        # 存储调试状态
        debug_key = f"kmeans_debug_{session_key}_{int(time.time())}"
        session_data['training_history'][debug_key] = debug_state
        
        result = {
            'status': 'success',
            'debug_key': debug_key,
            'state': debug_state
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"K-means调试初始化错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/clustering/kmeans/debug_step', methods=['POST'])
def kmeans_debug_step():
    """执行K-means调试步骤"""
    try:
        config = request.json
        debug_key = config.get('debug_key')
        
        # 获取调试状态
        if debug_key not in session_data['training_history']:
            return jsonify({'status': 'error', 'message': '调试状态未找到'}), 400
            
        debug_state = session_data['training_history'][debug_key]
        
        logger.info(f"执行K-means调试步骤: 第{debug_state['current_iter']}轮")
        
        # 执行调试步骤
        updated_state = clustering_alg.kmeans_debug_step(debug_state)
        
        # 更新存储的状态
        session_data['training_history'][debug_key] = updated_state
        
        result = {
            'status': 'success',
            'state': updated_state,
            'converged': updated_state['converged'],
            'finished': updated_state['converged'] or updated_state['current_iter'] >= updated_state['max_iter']
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"K-means调试步骤错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/clustering/dbscan/debug_init', methods=['POST'])
def dbscan_debug_init():
    """初始化DBSCAN单步调试"""
    try:
        config = request.json
        session_key = config.get('session_key')
        eps = config.get('eps', 0.5)
        min_samples = config.get('min_samples', 5)
        
        # 获取数据
        if session_key not in session_data['datasets']:
            return jsonify({'status': 'error', 'message': '数据未找到，请先生成数据'}), 400
            
        dataset = session_data['datasets'][session_key]
        X = np.array(dataset['X'])
        
        logger.info(f"初始化DBSCAN单步调试: eps={eps}, min_samples={min_samples}")
        
        # 初始化调试状态
        debug_state = clustering_alg.dbscan_debug_init(X, eps=eps, min_samples=min_samples)
        
        # 存储调试状态
        debug_key = f"dbscan_debug_{session_key}_{int(time.time())}"
        session_data['training_history'][debug_key] = debug_state
        
        result = {
            'status': 'success',
            'debug_key': debug_key,
            'state': debug_state
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"DBSCAN调试初始化错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/clustering/dbscan/debug_step', methods=['POST'])
def dbscan_debug_step():
    """执行DBSCAN调试步骤"""
    try:
        config = request.json
        debug_key = config.get('debug_key')
        
        # 获取调试状态
        if debug_key not in session_data['training_history']:
            return jsonify({'status': 'error', 'message': '调试状态未找到'}), 400
            
        debug_state = session_data['training_history'][debug_key]
        
        logger.info(f"执行DBSCAN调试步骤: 点{debug_state['current_point_idx']}")
        
        # 执行调试步骤
        updated_state = clustering_alg.dbscan_debug_step(debug_state)
        
        # 更新存储的状态
        session_data['training_history'][debug_key] = updated_state
        
        result = {
            'status': 'success',
            'state': updated_state,
            'finished': updated_state['finished']
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"DBSCAN调试步骤错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/clustering/debug_state/<debug_key>', methods=['GET'])
def get_clustering_debug_state(debug_key):
    """获取聚类调试状态"""
    try:
        if debug_key not in session_data['training_history']:
            return jsonify({'status': 'error', 'message': '调试状态未找到'}), 404
            
        debug_state = session_data['training_history'][debug_key]
        
        result = {
            'status': 'success',
            'state': debug_state
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"获取调试状态错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


# ================== SVM算法API ==================

@app.route('/api/svm/train', methods=['POST'])
def train_svm():
    """训练SVM模型"""
    try:
        config = request.json
        session_key = config.get('session_key')
        params = config.get('params', {})
        
        # 获取数据
        if session_key not in session_data['datasets']:
            return jsonify({'status': 'error', 'message': '数据未找到，请先生成数据'}), 400
            
        dataset = session_data['datasets'][session_key]
        X = np.array(dataset['X'])
        y = np.array(dataset['y'])
        
        logger.info("训练SVM模型")
        
        # 训练模型
        result = svm_alg.train_model(X, y, **params)
        
        # 存储模型
        model_key = f"svm_{session_key}"
        session_data['models'][model_key] = result
        
        # 存储scaler（如果使用了）
        if result.get('scaler') is not None:
            session_data['scalers'][session_key] = result['scaler']
        
        response_result = {
            'status': 'success',
            'model_key': model_key,
            'results': {
                'metrics': result['metrics'],
                'predictions': result['predictions'],
                'full_data_with_labels': result['full_data_with_labels'],
                'training_info': result['training_info'],
                'algorithm_params': result['algorithm_params'],
                'decision_boundary': result['decision_boundary'],
                'confusion_matrices': result['confusion_matrices'],
                'data_info': result['data_info']
            }
        }
        
        return jsonify(response_result)
        
    except Exception as e:
        logger.error(f"SVM训练错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/svm/predict', methods=['POST'])
def predict_svm():
    """使用训练好的SVM模型进行预测"""
    try:
        config = request.json
        model_key = config.get('model_key')
        X_pred = np.array(config.get('X'))
        
        if model_key not in session_data['models']:
            return jsonify({'status': 'error', 'message': '模型未找到'}), 400
            
        model_info = session_data['models'][model_key]
        model = model_info['model']
        scaler = model_info.get('scaler')
        
        result = svm_alg.predict_with_proba(model, X_pred, scaler)
        
        return jsonify({
            'status': 'success',
            **result
        })
        
    except Exception as e:
        logger.error(f"SVM预测错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


# ================== 决策树算法API ==================

@app.route('/api/decision_tree/train', methods=['POST'])
def train_decision_tree():
    """训练决策树模型"""
    try:
        config = request.json
        session_key = config.get('session_key')
        params = config.get('params', {})
        
        # 获取数据
        if session_key not in session_data['datasets']:
            return jsonify({'status': 'error', 'message': '数据未找到，请先生成数据'}), 400
            
        dataset = session_data['datasets'][session_key]
        X = np.array(dataset['X'])
        y = np.array(dataset['y'])
        feature_names = dataset.get('feature_names', None)
        
        logger.info("训练决策树模型")
        
        # 如果有特征名称，添加到参数中
        if feature_names:
            params['feature_names'] = feature_names
        
        # 训练模型
        result = decision_tree_alg.train_model(X, y, **params)
        
        # 存储模型
        model_key = f"decision_tree_{session_key}"
        result['algorithm'] = 'decision_tree'
        session_data['models'][model_key] = result
        
        response_result = {
            'status': 'success',
            'model_key': model_key,
            'results': {
                'metrics': result['metrics'],
                'predictions': result['predictions'],
                'full_data_with_labels': result['full_data_with_labels'],
                'training_info': result['training_info'],
                'algorithm_params': result['algorithm_params'],
                'decision_boundary': result['decision_boundary'],
                'confusion_matrices': result['confusion_matrices'],
                'data_info': result['data_info']
            }
        }
        
        return jsonify(response_result)
        
    except Exception as e:
        logger.error(f"决策树训练错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/decision_tree/predict', methods=['POST'])
def predict_decision_tree():
    """使用训练好的决策树模型进行预测"""
    try:
        config = request.json
        model_key = config.get('model_key')
        X_pred = np.array(config.get('X'))
        
        if model_key not in session_data['models']:
            return jsonify({'status': 'error', 'message': '模型未找到'}), 400
            
        model_info = session_data['models'][model_key]
        model = model_info['model']
        scaler = model_info.get('scaler')
        
        result = decision_tree_alg.predict_with_proba(model, X_pred, scaler)
        
        return jsonify({
            'status': 'success',
            **result
        })
        
    except Exception as e:
        logger.error(f"决策树预测错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/decision_tree/decision_path', methods=['POST'])
def get_decision_tree_path():
    """获取决策树的决策路径"""
    try:
        config = request.json
        model_key = config.get('model_key')
        X_sample = np.array(config.get('X'))
        
        if model_key not in session_data['models']:
            return jsonify({'status': 'error', 'message': '模型未找到'}), 400
            
        model_info = session_data['models'][model_key]
        
        # 检查是否为决策树模型
        if model_info['algorithm'] != 'decision_tree':
            return jsonify({'status': 'error', 'message': '该模型不是决策树模型'}), 400
        
        model = model_info['model']
        scaler = model_info.get('scaler')
        
        # 数据预处理
        if scaler is not None:
            X_sample = scaler.transform(X_sample)
        
        # 获取决策路径
        decision_paths = decision_tree_alg.get_decision_path(model, X_sample)
        
        # 获取预测结果
        predictions = model.predict(X_sample)
        probabilities = None
        
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(X_sample)
            except Exception as e:
                logger.warning(f"无法获取预测概率: {e}")
        
        result = {
            'status': 'success',
            'decision_paths': decision_paths,
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist() if probabilities is not None else None
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"获取决策路径错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/generate_full_dimensional_data', methods=['POST'])
def generate_full_dimensional_data():
    """生成完整维度数据（专门用于决策树）"""
    try:
        config = request.json
        shape = config.get('shape', 'iris')
        n_samples = config.get('n_samples', 200)
        random_state = config.get('random_state', 42)
        
        logger.info(f"生成完整维度数据: {shape}")
        
        # 生成数据
        X, y, feature_names = data_gen.generate_full_dimensional_data(
            shape=shape, n_samples=n_samples, random_state=random_state
        )
        
        # 生成session key
        session_key = f"full_{shape}_{n_samples}"
        
        # 存储数据
        session_data['datasets'][session_key] = {
            'X': X.tolist(),
            'y': y.tolist(),
            'feature_names': feature_names,
            'type': 'classification',
            'shape': shape,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y))
        }
        
        result = {
            'status': 'success',
            'session_key': session_key,
            'data': {
                'X': X.tolist(),
                'y': y.tolist(),
                'feature_names': feature_names,
                'n_samples': len(X),
                'n_features': X.shape[1],
                'n_classes': len(np.unique(y))
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"生成完整维度数据错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


# ================== KNN算法API ==================

@app.route('/api/knn/train', methods=['POST'])
def train_knn():
    """训练KNN模型"""
    try:
        config = request.json
        session_key = config.get('session_key')
        params = config.get('params', {})
        
        # 获取数据
        if session_key not in session_data['datasets']:
            return jsonify({'status': 'error', 'message': '数据未找到，请先生成数据'}), 400
            
        dataset = session_data['datasets'][session_key]
        X = np.array(dataset['X'])
        y = np.array(dataset['y'])
        
        logger.info("训练KNN模型")
        
        # 训练模型
        result = knn_alg.train_model(X, y, **params)
        
        # 存储模型
        model_key = f"knn_{session_key}"
        session_data['models'][model_key] = result
        
        # 存储scaler（如果使用了）
        if result.get('scaler') is not None:
            session_data['scalers'][session_key] = result['scaler']
        
        response_result = {
            'status': 'success',
            'model_key': model_key,
            'results': {
                'metrics': result['metrics'],
                'predictions': result['predictions'],
                'full_data_with_labels': result['full_data_with_labels'],
                'training_info': result['training_info'],
                'algorithm_params': result['algorithm_params'],
                'decision_boundary': result['decision_boundary'],
                'confusion_matrices': result['confusion_matrices'],
                'data_info': result['data_info']
            }
        }
        
        return jsonify(response_result)
        
    except Exception as e:
        logger.error(f"KNN训练错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/knn/predict', methods=['POST'])
def predict_knn():
    """使用训练好的KNN模型进行预测"""
    try:
        config = request.json
        model_key = config.get('model_key')
        X_pred = np.array(config.get('X'))
        
        if model_key not in session_data['models']:
            return jsonify({'status': 'error', 'message': '模型未找到'}), 400
            
        model_info = session_data['models'][model_key]
        model = model_info['model']
        scaler = model_info.get('scaler')
        
        result = knn_alg.predict_with_proba(model, X_pred, scaler)
        
        return jsonify({
            'status': 'success',
            **result
        })
        
    except Exception as e:
        logger.error(f"KNN预测错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


# ================== 朴素贝叶斯算法API ==================

@app.route('/api/bayesian_classification/train', methods=['POST'])
def train_bayesian_classification():
    """训练朴素贝叶斯模型"""
    try:
        config = request.json
        session_key = config.get('session_key')
        params = config.get('params', {})
        
        # 获取数据
        if session_key not in session_data['datasets']:
            return jsonify({'status': 'error', 'message': '数据未找到，请先生成数据'}), 400
            
        dataset = session_data['datasets'][session_key]
        X = np.array(dataset['X'])
        y = np.array(dataset['y'])
        
        logger.info("训练朴素贝叶斯模型")
        
        # 训练高斯贝叶斯模型
        result = bayesian_alg.train_model(X, y, **params)
        
        # 存储模型
        model_key = f"bayesian_classification_{session_key}"
        session_data['models'][model_key] = result
        
        response_result = {
            'status': 'success',
            'model_key': model_key,
            'results': {
                'metrics': result['metrics'],
                'predictions': result['predictions'],
                'full_data_with_labels': result['full_data_with_labels'],
                'training_info': result['training_info'],
                'algorithm_params': result['algorithm_params'],
                'decision_boundary': result['decision_boundary'],
                'confusion_matrices': result['confusion_matrices'],
                'data_info': result['data_info']
            }
        }
        
        return jsonify(response_result)
        
    except Exception as e:
        logger.error(f"朴素贝叶斯训练错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400



@app.route('/api/bayesian_classification/probabilities', methods=['POST'])
def get_bayesian_probabilities():
    """获取贝叶斯分类的详细概率信息"""
    try:
        config = request.json
        model_key = config.get('model_key')
        X_point = np.array(config.get('X_point'))
        
        if model_key not in session_data['models']:
            return jsonify({'status': 'error', 'message': '模型未找到'}), 400
            
        model_info = session_data['models'][model_key]
        model = model_info['model']
        scaler = model_info.get('scaler')
        
        # 获取详细概率信息
        detailed_probs = bayesian_alg.get_bayesian_probabilities(model, X_point, scaler)
        
        result = {
            'status': 'success',
            'detailed_probabilities': detailed_probs
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"获取贝叶斯概率错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


# ================== 线性回归算法API ==================


def _json_safe_excel_value(value):
    """Convert worksheet values into JSON-safe preview values."""
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _read_excel_upload(upload):
    """Validate and parse an uploaded .xlsx workbook without writing it to disk."""
    if upload is None or not upload.filename:
        raise ValueError('请选择要上传的 Excel 文件')
    if not upload.filename.lower().endswith('.xlsx'):
        raise ValueError('仅支持 .xlsx 格式的 Excel 文件')

    content = upload.stream.read(MAX_EXCEL_FILE_SIZE + 1)
    if len(content) > MAX_EXCEL_FILE_SIZE:
        raise ValueError('Excel 文件不能超过 5 MB')
    if not content:
        raise ValueError('上传的 Excel 文件为空')

    try:
        workbook = load_workbook(BytesIO(content), read_only=True, data_only=True)
    except Exception as error:
        raise ValueError('无法解析 Excel 文件，请确认文件未损坏且未加密') from error

    sheets = {}
    try:
        for worksheet in workbook.worksheets:
            rows = worksheet.iter_rows(values_only=True)
            raw_header = next(rows, None)
            if raw_header is None:
                sheets[worksheet.title] = pd.DataFrame()
                continue

            header = list(raw_header)
            while header and header[-1] is None:
                header.pop()
            columns = [str(value).strip() if value is not None else '' for value in header]
            if not columns:
                sheets[worksheet.title] = pd.DataFrame()
                continue
            if any(not column for column in columns):
                raise ValueError(f'工作表“{worksheet.title}”包含空列名')
            if len(set(columns)) != len(columns):
                raise ValueError(f'工作表“{worksheet.title}”包含重复列名')

            values = []
            for row in rows:
                normalized = list(row[:len(columns)])
                normalized.extend([None] * (len(columns) - len(normalized)))
                if any(value is not None and value != '' for value in normalized):
                    values.append(normalized)
            sheets[worksheet.title] = pd.DataFrame(values, columns=columns)
    finally:
        workbook.close()

    if not sheets or all(frame.empty for frame in sheets.values()):
        raise ValueError('Excel 文件中没有可用的数据行')
    return upload.filename, sheets


def _excel_sheet_metadata(name, frame):
    numeric_columns = []
    invalid_counts = {}
    for column in frame.columns:
        converted = pd.to_numeric(frame[column], errors='coerce')
        non_empty = frame[column].notna() & frame[column].astype(str).str.strip().ne('')
        convertible = converted.notna()
        if convertible.any():
            numeric_columns.append(column)
        invalid_counts[column] = int((non_empty & ~convertible).sum())

    preview = [
        {column: _json_safe_excel_value(value) for column, value in row.items()}
        for row in frame.head(EXCEL_PREVIEW_ROWS).to_dict(orient='records')
    ]
    return {
        'name': name,
        'columns': list(frame.columns),
        'numeric_columns': numeric_columns,
        'invalid_counts': invalid_counts,
        'total_rows': int(len(frame)),
        'preview': preview,
    }


@app.route('/api/linear_regression/excel/preview', methods=['POST'])
def preview_linear_regression_excel():
    """Return worksheet and column metadata for an uploaded Excel workbook."""
    try:
        filename, sheets = _read_excel_upload(request.files.get('file'))
        return jsonify({
            'status': 'success',
            'filename': filename,
            'sheets': [
                _excel_sheet_metadata(name, frame)
                for name, frame in sheets.items()
            ],
        })
    except ValueError as error:
        return jsonify({'status': 'error', 'message': str(error)}), 400
    except Exception as error:
        logger.exception('Excel 预览错误')
        return jsonify({'status': 'error', 'message': f'Excel 预览失败: {error}'}), 400


@app.route('/api/linear_regression/excel/import', methods=['POST'])
def import_linear_regression_excel():
    """Create a regression dataset from selected columns in an Excel workbook."""
    try:
        filename, sheets = _read_excel_upload(request.files.get('file'))
        sheet_name = request.form.get('sheet_name', '')
        target_column = request.form.get('target_column', '')
        feature_columns = request.form.getlist('feature_columns')

        if sheet_name not in sheets:
            raise ValueError('所选工作表不存在')
        frame = sheets[sheet_name]
        if frame.empty:
            raise ValueError('所选工作表没有可用的数据行')
        if not feature_columns:
            raise ValueError('请至少选择一个特征列')
        if not target_column:
            raise ValueError('请选择目标列')
        if len(set(feature_columns)) != len(feature_columns):
            raise ValueError('特征列不能重复')
        if target_column in feature_columns:
            raise ValueError('目标列不能同时作为特征列')

        selected_columns = feature_columns + [target_column]
        missing_columns = [column for column in selected_columns if column not in frame.columns]
        if missing_columns:
            raise ValueError(f'列不存在: {"、".join(missing_columns)}')

        numeric_frame = frame[selected_columns].apply(pd.to_numeric, errors='coerce')
        numeric_frame = numeric_frame.replace([np.inf, -np.inf], np.nan)
        valid_rows = numeric_frame.notna().all(axis=1)
        cleaned = numeric_frame.loc[valid_rows]
        dropped_rows = int(len(frame) - len(cleaned))
        if len(cleaned) < 10:
            raise ValueError('清洗后至少需要 10 行完整数值数据')

        X = cleaned[feature_columns].to_numpy(dtype=float)
        y = cleaned[target_column].to_numpy(dtype=float)
        session_key = f'regression_excel_{uuid.uuid4().hex}'
        feature_info = {
            'dataset_name': filename,
            'description': f'来自工作表“{sheet_name}”的 Excel 数据',
            'target': target_column,
            'features_used': feature_columns,
            'total_features': len(feature_columns),
            'source_type': 'excel',
        }
        session_data['datasets'][session_key] = {
            'X': X.tolist(),
            'y': y.tolist(),
            'feature_info': feature_info,
            'config': {
                'type': 'regression',
                'shape': 'excel',
                'filename': filename,
                'sheet_name': sheet_name,
                'feature_columns': feature_columns,
                'target_column': target_column,
            },
        }

        return jsonify({
            'status': 'success',
            'session_key': session_key,
            'data': {
                'X': X.tolist(),
                'y': y.tolist(),
                'n_samples': int(len(X)),
                'n_features': int(X.shape[1]),
                'feature_info': feature_info,
                'import_summary': {
                    'original_rows': int(len(frame)),
                    'valid_rows': int(len(cleaned)),
                    'dropped_rows': dropped_rows,
                },
            },
        })
    except ValueError as error:
        return jsonify({'status': 'error', 'message': str(error)}), 400
    except Exception as error:
        logger.exception('Excel 导入错误')
        return jsonify({'status': 'error', 'message': f'Excel 导入失败: {error}'}), 400

@app.route('/api/linear_regression/train', methods=['POST'])
def train_linear_regression():
    """训练线性回归模型"""
    try:
        config = request.json
        session_key = config.get('session_key')
        params = config.get('params', {})
        
        # 获取数据
        if session_key not in session_data['datasets']:
            return jsonify({'status': 'error', 'message': '数据未找到，请先生成数据'}), 400
            
        dataset = session_data['datasets'][session_key]
        X = np.array(dataset['X'])
        y = np.array(dataset['y'])
        feature_info = dataset.get('feature_info')  # 获取特征信息（如果有的话）
        
        # 获取算法和优化器参数
        algorithm = config.get('algorithm', 'linear')
        optimizer = params.pop('optimizer', 'auto')  # 使用pop避免重复传递
        
        logger.info(f"训练线性回归模型: algorithm={algorithm}, optimizer={optimizer}")
        
        # 添加优化器参数
        if optimizer == 'sgd':
            params.setdefault('max_iter', 1000)
            params.setdefault('tol', 1e-3)
            params.setdefault('learning_rate', 'invscaling')
            params.setdefault('eta0', 0.01)
        
        # 传递特征信息给算法（如果有的话）
        if feature_info:
            params['feature_info'] = feature_info
            
        # 训练模型
        result = linear_regression_alg.train_model(X, y, algorithm=algorithm, optimizer=optimizer, **params)
        
        # 存储模型
        model_key = f"linear_regression_{session_key}"
        session_data['models'][model_key] = result
        
        # 存储scaler和polynomial_transformer（如果使用了）
        if result.get('scaler') is not None:
            session_data['scalers'][session_key] = result['scaler']
        if result.get('polynomial_transformer') is not None:
            session_data['scalers'][f"{session_key}_poly"] = result['polynomial_transformer']
        
        response_result = {
            'status': 'success',
            'model_key': model_key,
            'results': {
                'metrics': result['metrics'],
                'predictions': result['predictions'],
                'training_info': result['training_info'],
                'prediction_curve': result['prediction_curve'],
                'residuals': result['residuals'],
                'point_results': result['point_results'],
                'loss_summary': result['loss_summary'],
                'model_equation': result['model_equation'],
                'data_info': result['data_info']
            }
        }
        
        return jsonify(response_result)
        
    except Exception as e:
        logger.error(f"线性回归训练错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/linear_regression/predict', methods=['POST'])
def predict_linear_regression():
    """使用训练好的线性回归模型进行预测"""
    try:
        config = request.json
        model_key = config.get('model_key')
        X_pred = np.array(config.get('X'))
        
        if model_key not in session_data['models']:
            return jsonify({'status': 'error', 'message': '模型未找到'}), 400
            
        model_info = session_data['models'][model_key]
        model = model_info['model']
        scaler = model_info.get('scaler')
        polynomial_transformer = model_info.get('polynomial_transformer')
        
        result = linear_regression_alg.predict_new_data(model, X_pred, scaler, polynomial_transformer)
        
        return jsonify({
            'status': 'success',
            **result
        })
        
    except Exception as e:
        logger.error(f"线性回归预测错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


# ================== SGD算法API ==================

@app.route('/api/sgd/train', methods=['POST'])
def train_sgd():
    """训练SGD模型"""
    try:
        config = request.json
        session_key = config.get('session_key')
        params = config.get('params', {})
        
        # 获取数据
        if session_key not in session_data['datasets']:
            return jsonify({'status': 'error', 'message': '数据未找到，请先生成数据'}), 400
            
        dataset = session_data['datasets'][session_key]
        X = np.array(dataset['X'])
        y = np.array(dataset['y'])
        
        logger.info("训练SGD模型")
        
        # 训练模型
        result = sgd_alg.train_model(X, y, **params)
        
        # 存储模型
        model_key = f"sgd_{session_key}"
        session_data['models'][model_key] = result
        
        # 存储scaler（如果使用了）
        if result.get('scaler') is not None:
            session_data['scalers'][session_key] = result['scaler']
        
        response_result = {
            'status': 'success',
            'model_key': model_key,
            'results': {
                'metrics': result['metrics'],
                'predictions': result['predictions'],
                'training_history': result['training_history'],
                'training_info': result['training_info'],
                'prediction_curve': result['prediction_curve'],
                'residuals': result['residuals'],
                'data_info': result['data_info']
            }
        }
        
        return jsonify(response_result)
        
    except Exception as e:
        logger.error(f"SGD训练错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/sgd/predict', methods=['POST'])
def predict_sgd():
    """使用训练好的SGD模型进行预测"""
    try:
        config = request.json
        model_key = config.get('model_key')
        X_pred = np.array(config.get('X'))
        
        if model_key not in session_data['models']:
            return jsonify({'status': 'error', 'message': '模型未找到'}), 400
            
        model_info = session_data['models'][model_key]
        model = model_info['model']
        scaler = model_info.get('scaler')
        
        result = sgd_alg.predict_new_data(model, X_pred, scaler)
        
        return jsonify({
            'status': 'success',
            **result
        })
        
    except Exception as e:
        logger.error(f"SGD预测错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/sgd/start_training_session', methods=['POST'])
def start_sgd_training_session():
    """启动SGD多优化器训练会话"""
    try:
        config = request.json
        session_key = config.get('session_key')
        optimizers_config = config.get('optimizers_config', {})
        params = config.get('params', {})
        
        # 获取数据
        if session_key not in session_data['datasets']:
            return jsonify({'status': 'error', 'message': '数据未找到，请先生成数据'}), 400
            
        dataset = session_data['datasets'][session_key]
        X = np.array(dataset['X'])
        y = np.array(dataset['y'])
        
        logger.info(f"启动SGD训练会话: {session_key}")
        
        # 启动训练会话
        session_info = sgd_alg.start_training_session(
            session_key, X, y, optimizers_config, **params
        )
        
        # 存储训练会话信息
        session_data['training_history'][session_key] = session_info
        
        return jsonify({
            'status': 'success',
            'session_info': session_info
        })
        
    except Exception as e:
        logger.error(f"启动SGD训练会话错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/sgd/train_step', methods=['POST'])
def sgd_train_step():
    """执行SGD训练的一步"""
    try:
        config = request.json
        session_key = config.get('session_key')
        batch_size = config.get('batch_size')
        
        logger.info(f"执行SGD训练步骤: {session_key}")
        
        # 执行训练步骤
        step_result = sgd_alg.train_step(session_key, batch_size)
        
        return jsonify({
            'status': 'success',
            'step_result': step_result
        })
        
    except Exception as e:
        logger.error(f"SGD训练步骤错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/sgd/get_training_state', methods=['POST'])
def get_sgd_training_state():
    """获取SGD训练状态"""
    try:
        config = request.json
        session_key = config.get('session_key')
        
        # 获取训练状态
        training_state = sgd_alg.get_training_state(session_key)
        
        return jsonify({
            'status': 'success',
            'training_state': training_state
        })
        
    except Exception as e:
        logger.error(f"获取SGD训练状态错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/sgd/compare_optimizers', methods=['POST'])
def compare_sgd_optimizers():
    """比较多个SGD优化器的性能"""
    try:
        config = request.json
        session_key = config.get('session_key')
        optimizers_config = config.get('optimizers_config', {})
        max_iter = config.get('max_iter', 50)
        params = config.get('params', {})
        
        # 获取数据
        if session_key not in session_data['datasets']:
            return jsonify({'status': 'error', 'message': '数据未找到，请先生成数据'}), 400
            
        dataset = session_data['datasets'][session_key]
        X = np.array(dataset['X'])
        y = np.array(dataset['y'])
        
        logger.info("比较SGD优化器性能")
        
        # 进行比较
        comparison_result = sgd_alg.compare_optimizers(
            X, y, optimizers_config, max_iter, **params
        )
        
        # 存储比较结果
        comparison_key = f"comparison_{session_key}"
        session_data['models'][comparison_key] = comparison_result
        
        return jsonify({
            'status': 'success',
            'comparison_result': comparison_result,
            'comparison_key': comparison_key
        })
        
    except Exception as e:
        logger.error(f"SGD优化器比较错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/sgd/get_available_models', methods=['GET'])
def get_available_sgd_models():
    """获取可用的SGD模型列表"""
    try:
        sgd_algorithm = SGDAlgorithm()
        models = sgd_algorithm.get_available_models()
        
        return jsonify({
            'status': 'success',
            'models': models
        })
    except Exception as e:
        logger.error(f"获取可用模型失败: {e}")
        return jsonify({
            'status': 'error', 
            'message': str(e)
        }), 500

@app.route('/api/sgd/get_available_optimizers', methods=['GET'])
def get_available_sgd_optimizers():
    """获取可用的SGD优化器列表"""
    try:
        optimizers_info = {
            'sgd': {
                'name': '标准SGD',
                'description': '标准随机梯度下降',
                'params': ['learning_rate'],
                'complexity': '简单'
            },
            'momentum': {
                'name': '动量SGD',
                'description': '带动量的随机梯度下降',
                'params': ['learning_rate', 'momentum'],
                'complexity': '简单'
            },
            'adagrad': {
                'name': 'AdaGrad',
                'description': '自适应梯度算法',
                'params': ['learning_rate', 'epsilon'],
                'complexity': '中等'
            },
            'rmsprop': {
                'name': 'RMSprop',
                'description': 'RMSprop优化器',
                'params': ['learning_rate', 'decay_rate', 'epsilon'],
                'complexity': '中等'
            },
            'adam': {
                'name': 'Adam',
                'description': 'Adam优化器',
                'params': ['learning_rate', 'beta1', 'beta2', 'epsilon'],
                'complexity': '复杂'
            },
            'nesterov': {
                'name': 'Nesterov',
                'description': 'Nesterov动量',
                'params': ['learning_rate', 'momentum'],
                'complexity': '中等'
            }
        }
        
        return jsonify({
            'status': 'success',
            'optimizers': optimizers_info
        })
        
    except Exception as e:
        logger.error(f"获取SGD优化器信息错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/sgd/reset_training_session', methods=['POST'])
def reset_sgd_training_session():
    """重置SGD训练会话"""
    try:
        config = request.json
        session_key = config.get('session_key')
        
        # 清除训练状态
        if session_key in sgd_alg.training_states:
            del sgd_alg.training_states[session_key]
            
        # 清除相关会话数据
        if session_key in session_data['training_history']:
            del session_data['training_history'][session_key]
            
        logger.info(f"重置SGD训练会话: {session_key}")
        
        return jsonify({
            'status': 'success',
            'message': '训练会话已重置'
        })
        
    except Exception as e:
        logger.error(f"重置SGD训练会话错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/data/get_available_types', methods=['GET'])
def get_available_data_types():
    """获取可用的数据类型"""
    try:
        data_types = data_gen.get_available_data_types()
        
        return jsonify({
            'status': 'success',
            'data_types': data_types
        })
        
    except Exception as e:
        logger.error(f"获取数据类型错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/data/get_description', methods=['POST'])
def get_data_description():
    """获取数据类型描述"""
    try:
        config = request.json
        data_type = config.get('data_type')
        shape = config.get('shape')
        
        description = data_gen.get_data_description(data_type, shape)
        
        return jsonify({
            'status': 'success',
            'description': description
        })
        
    except Exception as e:
        logger.error(f"获取数据描述错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/data/get_recommended_params', methods=['POST'])
def get_recommended_params():
    """获取推荐参数"""
    try:
        config = request.json
        data_type = config.get('data_type')
        shape = config.get('shape')
        
        params = data_gen.get_recommended_params(data_type, shape)
        
        return jsonify({
            'status': 'success',
            'recommended_params': params
        })
        
    except Exception as e:
        logger.error(f"获取推荐参数错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


# ================== 神经网络API ==================

@app.route('/api/neural_network/train', methods=['POST'])
def train_neural_network():
    """训练神经网络模型"""
    try:
        config = request.json
        session_key = config.get('session_key')
        params = config.get('params', {})
        
        # 获取数据
        if session_key not in session_data['datasets']:
            return jsonify({'status': 'error', 'message': '数据未找到，请先生成数据'}), 400
            
        dataset = session_data['datasets'][session_key]
        X = np.array(dataset['X'])
        y = np.array(dataset['y'])
        
        # 数据分割
        test_size = params.pop('test_size', 0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # 数据标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        session_data['scalers'][session_key] = scaler
        
        logger.info("训练神经网络模型")
        
        # 训练模型
        model, training_history, train_test_info = neural_net_alg.train_model(
            X_train, y_train, **params
        )
        
        # 预测和评估
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # 计算评估指标
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test)
        }
        
        # 存储模型
        model_key = f"neural_network_{session_key}"
        session_data['models'][model_key] = {
            'model': model,
            'algorithm': 'neural_network',
            'train_data': {'X': X_train.tolist(), 'y': y_train.tolist()},
            'test_data': {'X': X_test.tolist(), 'y': y_test.tolist()},
            'predictions': {'train': y_pred_train.tolist(), 'test': y_pred_test.tolist()},
            'metrics': metrics,
            'training_history': training_history,
            'train_test_info': train_test_info
        }
        
        result = {
            'status': 'success',
            'model_key': model_key,
            'results': {
                'metrics': metrics,
                'predictions': {
                    'train': y_pred_train.tolist(),
                    'test': y_pred_test.tolist()
                },
                'training_history': training_history,
                'train_test_info': train_test_info
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"神经网络训练错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/neural_network/decision_boundary', methods=['POST'])
def get_neural_network_decision_boundary():
    """获取神经网络决策边界"""
    try:
        config = request.json
        session_key = config.get('session_key')
        model_key = config.get('model_key')
        resolution = config.get('resolution', 100)
        
        # 获取模型和数据
        if model_key not in session_data['models']:
            return jsonify({'status': 'error', 'message': '模型未找到'}), 400
            
        if session_key not in session_data['datasets']:
            return jsonify({'status': 'error', 'message': '数据集未找到'}), 400
            
        model_info = session_data['models'][model_key]
        model = model_info['model']
        dataset = session_data['datasets'][session_key]
        
        X = np.array(dataset['X'])
        y = np.array(dataset['y'])
        
        # 获取scaler
        scaler = session_data['scalers'].get(session_key)
        if scaler is None:
            return jsonify({'status': 'error', 'message': '数据标准化器未找到'}), 400
        
        # 使用neural_net_alg生成决策边界
        boundary_data = neural_net_alg.visualize_decision_boundary(model, X, y, resolution)
        
        if boundary_data is None:
            return jsonify({'status': 'error', 'message': '无法生成决策边界，仅支持2D数据'}), 400
        
        return jsonify({
            'status': 'success',
            'boundary': boundary_data
        })
        
    except Exception as e:
        logger.error(f"获取神经网络决策边界错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/neural_network/train_step', methods=['POST'])
def train_neural_network_step():
    """分步训练神经网络模型（支持实时更新）"""
    try:
        config = request.json
        session_key = config.get('session_key')
        step = config.get('step', 1)  # 当前训练步骤
        params = config.get('params', {})
        
        # 获取数据
        if session_key not in session_data['datasets']:
            return jsonify({'status': 'error', 'message': '数据未找到，请先生成数据'}), 400
            
        dataset = session_data['datasets'][session_key]
        X = np.array(dataset['X'])
        y = np.array(dataset['y'])
        
        # 检查是否已经有训练中的模型
        model_key = f"neural_network_{session_key}"
        
        if step == 1:
            # 第一步：初始化训练
            # 数据分割
            test_size = params.pop('test_size', 0.2)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # 数据标准化
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            session_data['scalers'][session_key] = scaler
            
            # 创建模型但不完全训练
            network_type = params.get('network_type', 'mlp')
            
            if network_type == 'perceptron':
                from sklearn.linear_model import Perceptron
                model = Perceptron(
                    max_iter=10,  # 初始少量迭代
                    eta0=params.get('learning_rate_init', 0.001),
                    random_state=42
                )
            else:
                from sklearn.neural_network import MLPClassifier
                hidden_layer_sizes = params.get('hidden_layer_sizes', (100, 50))
                if isinstance(hidden_layer_sizes, str):
                    hidden_layer_sizes = tuple(map(int, hidden_layer_sizes.split(',')))
                
                model = MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation=params.get('activation', 'relu'),
                    solver=params.get('solver', 'adam'),
                    learning_rate_init=params.get('learning_rate_init', 0.001),
                    max_iter=10,  # 初始少量迭代
                    alpha=params.get('alpha', 0.0001),
                    random_state=42,
                    warm_start=True
                )
            
            # 初始训练
            model.fit(X_train, y_train)
            
            # 存储训练状态
            session_data['models'][model_key] = {
                'model': model,
                'algorithm': 'neural_network',
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test,
                'params': params,
                'current_iter': 10,
                'max_iter': params.get('max_iter', 500),
                'training_history': []
            }
            
        else:
            # 继续训练
            if model_key not in session_data['models']:
                return jsonify({'status': 'error', 'message': '训练会话未找到'}), 400
                
            model_info = session_data['models'][model_key]
            model = model_info['model']
            X_train = model_info['X_train']
            y_train = model_info['y_train']
            X_test = model_info['X_test']
            y_test = model_info['y_test']
            
            # 增加迭代次数并继续训练
            step_size = 20  # 每步增加20次迭代
            model_info['current_iter'] = min(model_info['current_iter'] + step_size, model_info['max_iter'])
            model.max_iter = model_info['current_iter']
            
            model.fit(X_train, y_train)
        
        # 计算当前性能
        model_info = session_data['models'][model_key]
        X_train = model_info['X_train']
        y_train = model_info['y_train']
        X_test = model_info['X_test']
        y_test = model_info['y_test']
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        # 计算损失
        try:
            if hasattr(model, 'predict_proba'):
                train_proba = model.predict_proba(X_train)
                test_proba = model.predict_proba(X_test)
                from sklearn.metrics import log_loss
                train_loss = log_loss(y_train, train_proba)
                test_loss = log_loss(y_test, test_proba)
            else:
                train_loss = 1 - train_accuracy
                test_loss = 1 - test_accuracy
        except:
            train_loss = 1 - train_accuracy
            test_loss = 1 - test_accuracy
        
        # 记录训练历史
        history_item = {
            'iteration': model_info['current_iter'],
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'train_loss': float(train_loss),
            'test_loss': float(test_loss)
        }
        
        model_info['training_history'].append(history_item)
        
        # 检查是否完成训练
        is_finished = model_info['current_iter'] >= model_info['max_iter']
        
        # 如果完成，生成决策边界
        boundary_data = None
        if X.shape[1] == 2:  # 只为2D数据生成边界
            scaler = session_data['scalers'][session_key]
            neural_net_alg.scaler = scaler  # 设置scaler
            boundary_data = neural_net_alg.visualize_decision_boundary(model, X, y, resolution=50)
        
        result = {
            'status': 'success',
            'step': step,
            'current_iter': model_info['current_iter'],
            'max_iter': model_info['max_iter'],
            'is_finished': is_finished,
            'metrics': {
                'train_accuracy': float(train_accuracy),
                'test_accuracy': float(test_accuracy),
                'train_loss': float(train_loss),
                'test_loss': float(test_loss)
            },
            'training_history': model_info['training_history'],
            'boundary_data': boundary_data
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"分步训练错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


# ================== 集成学习API ==================

@app.route('/api/ensemble/train', methods=['POST'])
def train_ensemble():
    """训练集成学习模型"""
    try:
        config = request.json
        session_key = config.get('session_key')
        algorithm = config.get('algorithm', 'random_forest')
        params = config.get('params', {})
        
        # 获取数据
        if session_key not in session_data['datasets']:
            return jsonify({'status': 'error', 'message': '数据未找到，请先生成数据'}), 400
            
        dataset = session_data['datasets'][session_key]
        X = np.array(dataset['X'])
        y = np.array(dataset['y'])
        
        # 数据分割
        test_size = params.pop('test_size', 0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        logger.info(f"训练集成学习模型: {algorithm}")
        
        # 训练模型
        model, feature_importance = ensemble_alg.train_model(
            X_train, y_train, algorithm=algorithm, **params
        )
        
        # 预测和评估
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # 计算评估指标
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test)
        }
        
        # 存储模型
        model_key = f"ensemble_{algorithm}_{session_key}"
        session_data['models'][model_key] = {
            'model': model,
            'algorithm': algorithm,
            'train_data': {'X': X_train.tolist(), 'y': y_train.tolist()},
            'test_data': {'X': X_test.tolist(), 'y': y_test.tolist()},
            'predictions': {'train': y_pred_train.tolist(), 'test': y_pred_test.tolist()},
            'metrics': metrics,
            'feature_importance': feature_importance.tolist() if feature_importance is not None else None
        }
        
        result = {
            'status': 'success',
            'model_key': model_key,
            'results': {
                'metrics': metrics,
                'predictions': {
                    'train': y_pred_train.tolist(),
                    'test': y_pred_test.tolist()
                },
                'feature_importance': feature_importance.tolist() if feature_importance is not None else None
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"集成学习训练错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/ensemble/decision_boundaries', methods=['POST'])
def get_ensemble_decision_boundaries():
    """获取集成学习模型的决策边界"""
    try:
        config = request.json
        model_key = config.get('model_key')
        
        if model_key not in session_data['models']:
            return jsonify({'status': 'error', 'message': '模型未找到'}), 400
            
        model_info = session_data['models'][model_key]
        model = model_info['model']
        algorithm = model_info['algorithm']
        
        # 计算数据范围
        train_data = model_info['train_data']
        X_train = np.array(train_data['X'])
        
        x_min, x_max = X_train[:, 0].min(), X_train[:, 0].max()
        y_min, y_max = X_train[:, 1].min(), X_train[:, 1].max()
        
        # 获取决策边界
        boundaries = ensemble_alg.get_base_estimators_decision_boundaries(
            model, [x_min, x_max], [y_min, y_max], algorithm
        )
        
        # 获取增强的集成结果
        enhanced_results = ensemble_alg.get_enhanced_ensemble_results(
            model, 
            X_train, 
            np.array(model_info['train_data']['y']),
            np.array(model_info['test_data']['X']),
            np.array(model_info['test_data']['y']),
            algorithm
        )
        
        return jsonify({
            'status': 'success',
            'boundaries': boundaries,
            'enhanced_results': enhanced_results,
            'data_range': {
                'x_min': x_min, 'x_max': x_max,
                'y_min': y_min, 'y_max': y_max
            }
        })
        
    except Exception as e:
        logger.error(f"获取集成决策边界错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/ensemble/base_learners', methods=['GET'])
def get_base_learners():
    """获取可用的基学习器配置"""
    try:
        task_type = request.args.get('task_type', 'classification')
        base_learners = ensemble_alg.get_available_base_learners(task_type)
        
        return jsonify({
            'status': 'success',
            'base_learners': base_learners
        })
        
    except Exception as e:
        logger.error(f"获取基学习器配置错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


# ================== 降维算法API ==================

@app.route('/api/dimensionality_reduction/datasets', methods=['GET'])
def get_dr_datasets():
    """获取可用的数据集列表"""
    try:
        datasets = {
            'builtin': {
                'iris': {'name': '鸢尾花数据集', 'features': 4, 'samples': 150, 'classes': 3},
                'wine': {'name': '红酒数据集', 'features': 13, 'samples': 178, 'classes': 3},
                'breast_cancer': {'name': '乳腺癌数据集', 'features': 30, 'samples': 569, 'classes': 2},
                'digits': {'name': '手写数字数据集', 'features': 64, 'samples': 1797, 'classes': 10},
                'digits_2d': {'name': '手写数字数据集(2D)', 'features': 2, 'samples': 1797, 'classes': 10}
            },
            'generated': {
                'description': '生成高维数据',
                'options': {
                    'n_samples': [100, 200, 300, 500],
                    'n_features': [5, 10, 20, 50, 100],
                    'n_classes': [2, 3, 4, 5],
                    'noise': [0.1, 0.2, 0.3, 0.5]
                }
            }
        }
        
        return jsonify({
            'status': 'success',
            'datasets': datasets,
            'supported_algorithms': dim_reduction_alg.get_supported_algorithms()
        })
    
    except Exception as e:
        logger.error(f"获取数据集列表错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/dimensionality_reduction/load_data', methods=['POST'])
def load_dr_data():
    """加载降维数据"""
    try:
        config = request.json
        data_type = config.get('type', 'builtin')
        
        if data_type == 'builtin':
            dataset_name = config.get('dataset', 'iris')
            n_samples = config.get('n_samples')
            
            X, y, feature_names, target_names = dim_reduction_alg.load_dataset(
                dataset_name=dataset_name, 
                n_samples=n_samples
            )
            
        elif data_type == 'generated':
            n_samples = config.get('n_samples', 300)
            n_features = config.get('n_features', 10)
            n_classes = config.get('n_classes', 3)
            noise = config.get('noise', 0.1)
            random_state = config.get('random_state', 42)
            
            X, y = dim_reduction_alg.generate_high_dimensional_data(
                n_samples=n_samples,
                n_features=n_features,
                n_classes=n_classes,
                noise=noise,
                random_state=random_state
            )
            
            feature_names = [f'Feature_{i+1}' for i in range(n_features)]
            target_names = [f'Class_{i}' for i in range(n_classes)]
            
        else:
            return jsonify({'status': 'error', 'message': '不支持的数据类型'}), 400
        
        # 存储数据
        session_key = f"dr_{data_type}_{config.get('dataset', 'generated')}"
        session_data['datasets'][session_key] = {
            'X': X.tolist(),
            'y': y.tolist() if y is not None else None,
            'feature_names': feature_names.tolist() if hasattr(feature_names, 'tolist') else list(feature_names),
            'target_names': target_names.tolist() if hasattr(target_names, 'tolist') else list(target_names),
            'config': config
        }
        
        result = {
            'status': 'success',
            'session_key': session_key,
            'data': {
                'n_samples': len(X),
                'n_features': X.shape[1],
                'n_classes': len(np.unique(y)) if y is not None else None,
                'y': y.tolist() if y is not None else None,
                'feature_names': feature_names.tolist() if hasattr(feature_names, 'tolist') else list(feature_names),
                'target_names': target_names.tolist() if hasattr(target_names, 'tolist') else list(target_names)
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"加载数据错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/dimensionality_reduction/apply', methods=['POST'])
def apply_dimensionality_reduction():
    
    """应用降维算法"""
    try:
        config = request.json
        session_key = config.get('session_key')
        algorithm = config.get('algorithm', 'pca')
        params = config.get('params', {})
        
        # 获取数据
        if session_key not in session_data['datasets']:
            return jsonify({'status': 'error', 'message': '数据未找到，请先加载数据'}), 400
        
        dataset = session_data['datasets'][session_key]
        X = np.array(dataset['X'])
        y = np.array(dataset['y']) if dataset['y'] is not None else None
        
        logger.info(f"应用 {algorithm.upper()} 降维算法")
        
        # 应用降维算法
        if algorithm == 'pca':
            result = dim_reduction_alg.apply_pca(X, **params)
        elif algorithm == 'tsne':
            result = dim_reduction_alg.apply_tsne(X, **params)
        elif algorithm == 'lda':
            if y is None:
                return jsonify({'status': 'error', 'message': 'LDA需要标签数据'}), 400
            result = dim_reduction_alg.apply_lda(X, y, **params)
        elif algorithm == 'umap':
            result = dim_reduction_alg.apply_umap(X, **params)
        else:
            return jsonify({'status': 'error', 'message': f'不支持的算法: {algorithm}'}), 400
        
        # 计算特征重要性（如果适用）
        feature_importance = None
        if algorithm in ['pca', 'lda']:
            feature_importance = dim_reduction_alg.get_feature_importance(
                result, dataset['feature_names']
            )
        
        # 计算重构误差（如果适用）
        reconstruction_error = None
        if algorithm == 'pca':
            reconstruction_error = dim_reduction_alg.evaluate_reconstruction_error(X, result)
        
        # 存储结果
        result_key = f"dr_{algorithm}_{session_key}"
        session_data['models'][result_key] = {
            'result': result,
            'algorithm': algorithm,
            'feature_importance': feature_importance,
            'reconstruction_error': reconstruction_error,
            'original_data': {'X': X.tolist(), 'y': y.tolist() if y is not None else None},
            'dataset_info': dataset
        }
        
        # 准备返回结果
        response_result = {
            'status': 'success',
            'result_key': result_key,
            'data': {
                'X_reduced': result['X_reduced'].tolist(),
                'algorithm': algorithm,
                'n_components': result.get('n_components', params.get('n_components', 2)),
                'feature_importance': feature_importance,
                'reconstruction_error': reconstruction_error
            }
        }
        
        # 添加算法特定的信息
        if algorithm == 'pca':
            response_result['data'].update({
                'explained_variance_ratio': result['explained_variance_ratio'],
                'cumulative_variance_ratio': result['cumulative_variance_ratio']
            })
        elif algorithm == 'tsne':
            response_result['data'].update({
                'kl_divergence': result['kl_divergence'],
                'n_iter_final': result['n_iter_final']
            })
        elif algorithm == 'lda':
            response_result['data'].update({
                'explained_variance_ratio': result['explained_variance_ratio']
            })
        
        return jsonify(response_result)
        
    except Exception as e:
        logger.error(f"降维算法应用错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/dimensionality_reduction/compare', methods=['POST'])
def compare_dimensionality_reduction():
    """比较多种降维算法"""
    try:
        config = request.json
        session_key = config.get('session_key')
        # 获取数据
        if session_key not in session_data['datasets']:
            return jsonify({'status': 'error', 'message': '数据未找到，请先加载数据'}), 400
        
        dataset = session_data['datasets'][session_key]
        X = np.array(dataset['X'])
        y = np.array(dataset['y']) if dataset['y'] is not None else None
        
        # 动态确定默认算法列表
        default_algorithms = ['pca', 'tsne']
        if y is not None:  # 如果有标签数据，添加LDA
            default_algorithms.insert(1, 'lda')
        algorithms = config.get('algorithms', default_algorithms)
        algorithm_params = config.get('algorithm_params', {})
        
        logger.info(f"比较降维算法: {', '.join(algorithms)}")
        
        # 比较算法
        results = dim_reduction_alg.compare_algorithms(
            X, y=y, algorithms=algorithms, **algorithm_params
        )
        
        # 处理结果
        comparison_results = {}
        for algorithm, result in results.items():
            if 'error' in result:
                comparison_results[algorithm] = {'error': result['error']}
                continue
            
            # 计算特征重要性
            feature_importance = None
            if algorithm in ['pca', 'lda']:
                feature_importance = dim_reduction_alg.get_feature_importance(
                    result, dataset['feature_names']
                )
            
            # 计算重构误差
            reconstruction_error = None
            if algorithm == 'pca':
                reconstruction_error = dim_reduction_alg.evaluate_reconstruction_error(X, result)
            
            comparison_results[algorithm] = {
                'X_reduced': result['X_reduced'].tolist(),
                'feature_importance': feature_importance,
                'reconstruction_error': reconstruction_error,
                'algorithm_info': {k: v for k, v in result.items() 
                                 if k not in ['X_reduced', 'scaler']}
            }
        
        # 存储比较结果
        comparison_key = f"dr_comparison_{session_key}"
        session_data['models'][comparison_key] = {
            'results': results,
            'comparison_results': comparison_results,
            'algorithms': algorithms,
            'original_data': {'X': X.tolist(), 'y': y.tolist() if y is not None else None},
            'dataset_info': dataset
        }
        
        result = {
            'status': 'success',
            'comparison_key': comparison_key,
            'results': comparison_results,
            'algorithms': algorithms
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"算法比较错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/dimensionality_reduction/info/<result_key>', methods=['GET'])
def get_dr_result_info(result_key):
    """获取降维结果详细信息"""
    try:
        if result_key not in session_data['models']:
            return jsonify({'status': 'error', 'message': '结果未找到'}), 404
        
        model_info = session_data['models'][result_key]
        
        # 准备返回信息（排除不能序列化的对象）
        info = {
            'algorithm': model_info.get('algorithm'),
            'feature_importance': model_info.get('feature_importance'),
            'reconstruction_error': model_info.get('reconstruction_error'),
            'dataset_info': {
                'feature_names': model_info['dataset_info']['feature_names'],
                'target_names': model_info['dataset_info']['target_names'],
                'n_samples': len(model_info['original_data']['X']),
                'n_features': len(model_info['original_data']['X'][0])
            }
        }
        
        # 添加算法特定信息
        if 'result' in model_info:
            result = model_info['result']
            if result.get('algorithm') == 'pca':
                info['pca_info'] = {
                    'explained_variance_ratio': result['explained_variance_ratio'],
                    'cumulative_variance_ratio': result['cumulative_variance_ratio'],
                    'n_components': result['n_components']
                }
            elif result.get('algorithm') == 'tsne':
                info['tsne_info'] = {
                    'kl_divergence': result['kl_divergence'],
                    'n_iter_final': result['n_iter_final']
                }
            elif result.get('algorithm') == 'lda':
                info['lda_info'] = {
                    'explained_variance_ratio': result['explained_variance_ratio'],
                    'n_components': result['n_components']
                }
        
        return jsonify({
            'status': 'success',
            'info': info
        })
        
    except Exception as e:
        logger.error(f"获取结果信息错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


# ================== 通用API ==================

@app.route('/api/model_info/<model_key>', methods=['GET'])
def get_model_info(model_key):
    """获取模型信息"""
    try:
        if model_key not in session_data['models']:
            return jsonify({'status': 'error', 'message': '模型未找到'}), 404
            
        model_info = session_data['models'][model_key]
        
        # 移除不能序列化的对象，只返回基本信息
        result = {
            'status': 'success',
            'model_info': {
                'algorithm': model_info.get('algorithm'),
                'metrics': model_info.get('metrics'),
                'training_info': model_info.get('training_info'),
                'has_model': True
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"获取模型信息错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/clear_session', methods=['POST'])
def clear_session():
    """清除会话数据"""
    try:
        session_data['models'].clear()
        session_data['datasets'].clear()
        session_data['scalers'].clear()
        session_data['training_history'].clear()
        
        return jsonify({'status': 'success', 'message': '会话已清除'})
        
    except Exception as e:
        logger.error(f"清除会话错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


# ================== 错误处理 ==================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'status': 'error', 'message': '接口不存在'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'status': 'error', 'message': '服务器内部错误'}), 500



# ================== 强化学习 API (Q-Learning 网格世界) ==================

@app.route('/api/rl/info', methods=['GET'])
def rl_info():
    """返回可用的强化学习环境列表"""
    try:
        environments = rl_alg.available_environments()
        for key, item in environments.items():
            item['grid'] = rl_alg.serialize_grid(build_environment(key))
        return jsonify({
            'status': 'success',
            'environments': environments
        })
    except Exception as e:
        logger.error(f"RL info error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/rl/train', methods=['POST'])
def rl_train():
    """训练 Q-Learning 智能体并返回可视化所需全部数据"""
    try:
        config = request.json or {}
        env_name = config.get('env', 'grid_world')
        params = config.get('params', {})

        episodes = int(params.get('episodes', 500))
        alpha = float(params.get('alpha', 0.1))
        gamma = float(params.get('gamma', 0.95))
        epsilon = float(params.get('epsilon', 1.0))
        epsilon_decay = float(params.get('epsilon_decay', 0.995))
        epsilon_min = float(params.get('epsilon_min', 0.01))
        random_state = int(params.get('random_state', 42))

        # 参数合法性约束
        episodes = max(50, min(episodes, 3000))
        alpha = max(0.01, min(alpha, 1.0))
        gamma = max(0.0, min(gamma, 0.99))
        epsilon = max(0.0, min(epsilon, 1.0))
        epsilon_decay = max(0.9, min(epsilon_decay, 1.0))
        epsilon_min = max(0.0, min(epsilon_min, 0.5))

        logger.info(f"RL training: env={env_name}, episodes={episodes}, alpha={alpha}, gamma={gamma}")
        result = rl_alg.train(
            env_name=env_name,
            episodes=episodes,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            random_state=random_state,
        )

        return jsonify({'status': 'success', 'results': result})

    except Exception as e:
        logger.error(f"RL training error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


if __name__ == '__main__':
    # 创建必要的目录
    os.makedirs('backend', exist_ok=True)
    os.makedirs('backend/algorithms', exist_ok=True)
    
    # 从环境变量获取配置
    port = int(os.environ.get('PORT', 5432))
    debug = os.environ.get('FLASK_ENV', 'production') == 'development'
    
    logger.info("启动机器学习算法可视化平台后端服务...")
    logger.info(f"访问地址: http://localhost:{port}")
    
    app.run(debug=debug, host='0.0.0.0', port=port)
