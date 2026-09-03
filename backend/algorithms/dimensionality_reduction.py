#!/usr/bin/env python3
"""
降维算法实现模块
支持 PCA、t-SNE、LDA、UMAP 等多种降维算法
提供可视化和教学功能
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logger.warning("UMAP not available. Install with: pip install umap-learn")


class DimensionalityReductionAlgorithms:
    """降维算法实现类"""
    
    def __init__(self):
        """初始化降维算法"""
        self.supported_algorithms = ['pca', 'tsne', 'lda']
        if UMAP_AVAILABLE:
            self.supported_algorithms.append('umap')
        
        self.algorithm_names = {
            'pca': '主成分分析 (PCA)',
            'tsne': 't-SNE',
            'lda': '线性判别分析 (LDA)',
            'umap': 'UMAP'
        }
    
    def get_supported_algorithms(self):
        """获取支持的算法列表"""
        return self.supported_algorithms
    
    def load_dataset(self, dataset_name='iris', n_samples=None):
        """
        加载预定义数据集
        
        Args:
            dataset_name: 数据集名称 ('iris', 'wine', 'breast_cancer', 'digits')
            n_samples: 限制样本数量
        
        Returns:
            tuple: (X, y, feature_names, target_names)
        """
        datasets = {
            'iris': load_iris,
            'wine': load_wine,
            'breast_cancer': load_breast_cancer,
            'digits': load_digits
        }
        
        if dataset_name not in datasets:
            raise ValueError(f"不支持的数据集: {dataset_name}")
        
        data = datasets[dataset_name]()
        X, y = data.data, data.target
        
        if n_samples and n_samples < len(X):
            # 随机采样指定数量的样本
            indices = np.random.choice(len(X), n_samples, replace=False)
            X = X[indices]
            y = y[indices]
        
        return X, y, data.feature_names, data.target_names
    
    def generate_high_dimensional_data(self, n_samples=300, n_features=10, n_classes=3, 
                                     noise=0.1, random_state=42):
        """
        生成高维测试数据
        
        Args:
            n_samples: 样本数量
            n_features: 特征数量
            n_classes: 类别数量
            noise: 噪声水平
            random_state: 随机种子
        
        Returns:
            tuple: (X, y)
        """
        np.random.seed(random_state)
        
        # 为每个类别生成不同的均值和协方差
        X_list = []
        y_list = []
        
        for i in range(n_classes):
            # 每个类别的样本数量
            n_class_samples = n_samples // n_classes
            if i < n_samples % n_classes:
                n_class_samples += 1
            
            # 随机生成均值
            mean = np.random.randn(n_features) * 2
            
            # 生成协方差矩阵
            A = np.random.randn(n_features, n_features)
            cov = np.dot(A, A.T) + noise * np.eye(n_features)
            
            # 生成数据
            X_class = np.random.multivariate_normal(mean, cov, n_class_samples)
            y_class = np.full(n_class_samples, i)
            
            X_list.append(X_class)
            y_list.append(y_class)
        
        X = np.vstack(X_list)
        y = np.hstack(y_list)
        
        # 打乱数据
        indices = np.random.permutation(len(X))
        return X[indices], y[indices]
    
    def apply_pca(self, X, n_components=2, standardize=True):
        """
        应用PCA降维
        
        Args:
            X: 输入数据
            n_components: 主成分数量
            standardize: 是否标准化数据
        
        Returns:
            dict: 包含降维结果和相关信息
        """
        if standardize:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X
            scaler = None
        
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X_scaled)
        
        # 计算累积方差解释率
        cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
        
        result = {
            'X_reduced': X_reduced,
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance_ratio': cumulative_variance_ratio.tolist(),
            'components': pca.components_.tolist(),
            'mean': pca.mean_.tolist(),
            'n_components': pca.n_components_,
            'n_features_in': getattr(pca, 'n_features_in_', X.shape[1]),  # 兼容不同sklearn版本
            'algorithm': 'pca',
            'scaler': scaler
        }
        
        return result
    
    def apply_tsne(self, X, n_components=2, perplexity=30, learning_rate=200, 
                   max_iter=1000, standardize=True, random_state=42):
        """
        应用t-SNE降维
        
        Args:
            X: 输入数据
            n_components: 降维后维度
            perplexity: 困惑度参数
            learning_rate: 学习率
            max_iter: 迭代次数 (使用max_iter而不是n_iter)
            standardize: 是否标准化数据
            random_state: 随机种子
        
        Returns:
            dict: 包含降维结果和相关信息
        """
        if standardize:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X
            scaler = None
        
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            max_iter=max_iter,  # 使用max_iter而不是n_iter
            random_state=random_state,
            verbose=1 if logger.level <= logging.INFO else 0
        )
        
        X_reduced = tsne.fit_transform(X_scaled)
        
        result = {
            'X_reduced': X_reduced,
            'kl_divergence': tsne.kl_divergence_,
            'n_iter_final': getattr(tsne, 'n_iter_', max_iter),  # 兼容不同sklearn版本
            'perplexity': perplexity,
            'learning_rate': learning_rate,
            'algorithm': 'tsne',
            'scaler': scaler
        }
        
        return result
    
    def apply_lda(self, X, y, n_components=None, standardize=True):
        """
        应用LDA降维
        
        Args:
            X: 输入数据
            y: 标签
            n_components: 降维后维度
            standardize: 是否标准化数据
        
        Returns:
            dict: 包含降维结果和相关信息
        """
        if standardize:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X
            scaler = None
        
        # LDA的最大维度是min(n_features, n_classes-1)
        n_classes = len(np.unique(y))
        max_components = min(X.shape[1], n_classes - 1)
        
        if n_components is None:
            n_components = min(2, max_components)
        else:
            n_components = min(n_components, max_components)
        
        lda = LDA(n_components=n_components)
        X_reduced = lda.fit_transform(X_scaled, y)
        
        result = {
            'X_reduced': X_reduced,
            'explained_variance_ratio': lda.explained_variance_ratio_.tolist(),
            'scalings': lda.scalings_.tolist(),
            'means': lda.means_.tolist(),
            'n_components': lda.n_components,
            'algorithm': 'lda',
            'scaler': scaler
        }
        
        return result
    
    def apply_umap(self, X, n_components=2, n_neighbors=15, min_dist=0.1, 
                   metric='euclidean', standardize=True, random_state=42):
        """
        应用UMAP降维
        
        Args:
            X: 输入数据
            n_components: 降维后维度
            n_neighbors: 邻居数量
            min_dist: 最小距离
            metric: 距离度量
            standardize: 是否标准化数据
            random_state: 随机种子
        
        Returns:
            dict: 包含降维结果和相关信息
        """
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP not available. Install with: pip install umap-learn")
        
        if standardize:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X
            scaler = None
        
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state
        )
        
        X_reduced = reducer.fit_transform(X_scaled)
        
        result = {
            'X_reduced': X_reduced,
            'n_neighbors': n_neighbors,
            'min_dist': min_dist,
            'metric': metric,
            'algorithm': 'umap',
            'scaler': scaler
        }
        
        return result
    
    def compare_algorithms(self, X, y=None, algorithms=None, **kwargs):
        """
        比较多种降维算法的效果
        
        Args:
            X: 输入数据
            y: 标签（可选，LDA需要）
            algorithms: 要比较的算法列表
            **kwargs: 各算法的参数
        
        Returns:
            dict: 各算法的降维结果
        """
        if algorithms is None:
            algorithms = ['pca', 'tsne']
            if y is not None:
                algorithms.insert(1, 'lda')  # 将LDA插入到tsne之前
            if UMAP_AVAILABLE:
                algorithms.append('umap')
        
        results = {}
        
        for algorithm in algorithms:
            try:
                if algorithm == 'pca':
                    result = self.apply_pca(X, **kwargs.get('pca', {}))
                elif algorithm == 'tsne':
                    result = self.apply_tsne(X, **kwargs.get('tsne', {}))
                elif algorithm == 'lda' and y is not None:
                    result = self.apply_lda(X, y, **kwargs.get('lda', {}))
                elif algorithm == 'umap':
                    result = self.apply_umap(X, **kwargs.get('umap', {}))
                else:
                    continue
                
                results[algorithm] = result
                logger.info(f"{algorithm.upper()} 降维完成")
                
            except Exception as e:
                logger.error(f"{algorithm.upper()} 降维失败: {str(e)}")
                results[algorithm] = {'error': str(e)}
        
        return results
    
    def get_feature_importance(self, result, feature_names=None):
        """
        获取特征重要性（仅适用于PCA和LDA）
        
        Args:
            result: 降维结果
            feature_names: 特征名称列表
        
        Returns:
            dict: 特征重要性信息
        """
        algorithm = result.get('algorithm')
        
        if algorithm == 'pca':
            components = np.array(result['components'])
            
            # 计算每个特征的总重要性（所有主成分的绝对值之和）
            feature_importance = np.sum(np.abs(components), axis=0)
            feature_importance = feature_importance / np.sum(feature_importance)
            
            importance_data = {
                'importance': feature_importance.tolist(),
                'components': components.tolist(),
                'explained_variance_ratio': result['explained_variance_ratio']
            }
            
        elif algorithm == 'lda':
            scalings = np.array(result['scalings'])
            
            # 计算每个特征的重要性
            feature_importance = np.sum(np.abs(scalings), axis=1)
            feature_importance = feature_importance / np.sum(feature_importance)
            
            importance_data = {
                'importance': feature_importance.tolist(),
                'scalings': scalings.tolist(),
                'explained_variance_ratio': result['explained_variance_ratio']
            }
            
        else:
            return {'error': f'{algorithm} 不支持特征重要性分析'}
        
        if feature_names:
            importance_data['feature_names'] = feature_names
            # 创建特征重要性排序
            indices = np.argsort(feature_importance)[::-1]
            importance_data['ranked_features'] = [
                {'name': feature_names[i], 'importance': feature_importance[i]}
                for i in indices
            ]
        
        return importance_data
    
    def evaluate_reconstruction_error(self, X_original, result):
        """
        评估重构误差（仅适用于PCA）
        
        Args:
            X_original: 原始数据
            result: 降维结果
        
        Returns:
            float: 重构误差
        """
        if result.get('algorithm') != 'pca':
            return None
        
        # 标准化原始数据
        if result.get('scaler'):
            X_scaled = result['scaler'].transform(X_original)
        else:
            X_scaled = X_original
        
        # 重构数据
        components = np.array(result['components'])
        mean = np.array(result['mean'])
        X_reduced = result['X_reduced']
        
        X_reconstructed = np.dot(X_reduced, components) + mean
        
        # 计算重构误差
        mse = np.mean((X_scaled - X_reconstructed) ** 2)
        return mse