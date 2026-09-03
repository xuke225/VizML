"""
数据生成器模块
使用 sklearn 和 numpy 生成各种类型的机器学习数据集
"""

import numpy as np
import pandas as pd
from sklearn.datasets import (
    make_blobs, make_circles, make_moons, make_classification,
    make_regression, make_gaussian_quantiles, make_s_curve,
    load_iris, load_wine, load_breast_cancer, load_digits, load_diabetes
)
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class DataGenerator:
    """数据生成器类"""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def generate_classification_data(self, shape='blobs', n_samples=200, noise=0.1, random_state=42, n_classes=None):
        """
        生成分类数据
        
        Args:
            shape: 数据形状 ('blobs', 'circles', 'moons', 'gaussian', 's_curve', 'xor', 'spiral', 'checkerboard', 'multi_cluster', 'imbalanced', 'iris', 'wine', 'breast_cancer', 'digits_2d')
            n_samples: 样本数量
            noise: 噪声水平
            random_state: 随机种子
            
        Returns:
            X: 特征矩阵
            y: 标签向量
        """
        logger.info(f"生成分类数据: {shape}, samples={n_samples}, noise={noise}")
        
        if shape == 'blobs':
            centers = n_classes if n_classes is not None else 3
            X, y = make_blobs(
                n_samples=n_samples, 
                centers=centers, 
                n_features=2,
                cluster_std=1.5 + noise * 2,
                random_state=random_state
            )
            
        elif shape == 'circles':
            if n_classes is not None and n_classes > 2:
                # 生成多个同心圆
                np.random.seed(random_state)
                X_list = []
                y_list = []
                
                for i in range(n_classes):
                    radius = 0.5 + i * 0.8  # 不同半径的圆
                    n_per_class = n_samples // n_classes
                    if i == n_classes - 1:  # 最后一类包含剩余样本
                        n_per_class = n_samples - (n_classes - 1) * (n_samples // n_classes)
                    
                    angles = np.random.uniform(0, 2*np.pi, n_per_class)
                    x = radius * np.cos(angles) + np.random.normal(0, noise, n_per_class)
                    y_coord = radius * np.sin(angles) + np.random.normal(0, noise, n_per_class)
                    
                    X_list.append(np.column_stack([x, y_coord]))
                    y_list.extend([i] * n_per_class)
                
                X = np.vstack(X_list)
                y = np.array(y_list)
                
                # 随机打乱
                indices = np.random.permutation(len(X))
                X = X[indices]
                y = y[indices]
            else:
                X, y = make_circles(
                    n_samples=n_samples, 
                    noise=noise, 
                    factor=0.6,
                    random_state=random_state
                )
            
        elif shape == 'moons':
            X, y = make_moons(
                n_samples=n_samples, 
                noise=noise,
                random_state=random_state
            )
            
        elif shape == 'gaussian':
            X, y = make_gaussian_quantiles(
                n_samples=n_samples,
                n_features=2,
                n_classes=2,
                cov=1 + noise,
                random_state=random_state
            )
            
        elif shape == 's_curve':
            X, y_continuous = make_s_curve(
                n_samples=n_samples,
                noise=noise,
                random_state=random_state
            )
            # 只取前两个维度并创建二分类标签
            X = X[:, [0, 2]]  # 取第1和第3维度
            y = (y_continuous > np.median(y_continuous)).astype(int)
            
        elif shape == 'xor':
            # 生成XOR模式的数据
            np.random.seed(random_state)
            n_per_class = n_samples // 4
            
            # 四个象限的中心点
            centers = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
            labels = [0, 1, 1, 0]  # XOR标签
            
            X_list = []
            y_list = []
            
            for center, label in zip(centers, labels):
                X_cluster = np.random.multivariate_normal(
                    center, 
                    [[noise, 0], [0, noise]], 
                    n_per_class
                )
                X_list.append(X_cluster)
                y_list.extend([label] * n_per_class)
            
            X = np.vstack(X_list)
            y = np.array(y_list)
            
            # 随机打乱
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]
            
        elif shape == 'spiral':
            # 生成双螺旋数据
            np.random.seed(random_state)
            n_per_spiral = n_samples // 2
            
            t = np.linspace(0, 4*np.pi, n_per_spiral)
            
            # 第一条螺旋
            x1 = t * np.cos(t) + np.random.normal(0, noise, n_per_spiral)
            y1 = t * np.sin(t) + np.random.normal(0, noise, n_per_spiral)
            
            # 第二条螺旋（旋转180度）
            x2 = -t * np.cos(t) + np.random.normal(0, noise, n_per_spiral)
            y2 = -t * np.sin(t) + np.random.normal(0, noise, n_per_spiral)
            
            X = np.column_stack([np.hstack([x1, x2]), np.hstack([y1, y2])])
            y = np.hstack([np.zeros(n_per_spiral), np.ones(n_per_spiral)])
            
            # 随机打乱
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]
            
        elif shape == 'checkerboard':
            # 生成棋盘模式数据
            np.random.seed(random_state)
            
            X = np.random.uniform(-3, 3, (n_samples, 2))
            # 棋盘模式：根据坐标的奇偶性决定类别
            y = ((np.floor(X[:, 0]) + np.floor(X[:, 1])) % 2).astype(int)
            
            # 添加噪声
            X += np.random.normal(0, noise, X.shape)
            
        elif shape == 'multi_cluster':
            # 多类别多簇团数据
            X, y = make_blobs(
                n_samples=n_samples,
                centers=8,
                n_features=2,
                cluster_std=0.8 + noise,
                random_state=random_state
            )
            # 将多个簇团合并为3个类别
            y = y % 3
            
        elif shape == 'imbalanced':
            # 不平衡数据集
            np.random.seed(random_state)
            
            # 多数类：80%的数据
            n_majority = int(n_samples * 0.8)
            X_maj = np.random.multivariate_normal(
                [0, 0], 
                [[1 + noise, 0.2], [0.2, 1 + noise]], 
                n_majority
            )
            y_maj = np.zeros(n_majority)
            
            # 少数类：20%的数据
            n_minority = n_samples - n_majority
            X_min = np.random.multivariate_normal(
                [3, 3], 
                [[0.5 + noise, 0], [0, 0.5 + noise]], 
                n_minority
            )
            y_min = np.ones(n_minority)
            
            X = np.vstack([X_maj, X_min])
            y = np.hstack([y_maj, y_min])
            
            # 随机打乱
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]
            
        elif shape == 'linear_separable':
            # 生成简单的线性可分数据
            np.random.seed(random_state)
            n_per_class = n_samples // 2
            
            # 第一类：左下角
            X1 = np.random.multivariate_normal(
                [-1.5, -1.5], 
                [[0.2 + noise, 0.1], [0.1, 0.2 + noise]], 
                n_per_class
            )
            
            # 第二类：右上角
            X2 = np.random.multivariate_normal(
                [1.5, 1.5], 
                [[0.2 + noise, 0.1], [0.2, 0.2 + noise]], 
                n_samples - n_per_class
            )
            
            X = np.vstack([X1, X2])
            y = np.hstack([np.zeros(n_per_class), np.ones(n_samples - n_per_class)])
            
            # 随机打乱
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]
            
        elif shape == 'classification':
            n_cls = n_classes if n_classes is not None else 3
            X, y = make_classification(
                n_samples=n_samples,
                n_features=2,
                n_redundant=0,
                n_informative=2,
                n_classes=n_cls,
                n_clusters_per_class=1,
                class_sep=max(0.1, 1.0 - noise),
                random_state=random_state
            )
            
        elif shape == 'random':
            X, y = make_classification(
                n_samples=n_samples,
                n_features=2,
                n_redundant=0,
                n_informative=2,
                n_classes=3,
                n_clusters_per_class=1,
                class_sep=1.0 - noise,
                random_state=random_state
            )
            
        # 真实数据集
        elif shape == 'iris':
            iris = load_iris()
            X = iris.data[:, :2]  # 只取前两个特征用于2D可视化
            y = iris.target
            # 转换为二分类问题
            y = (y > 0).astype(int)
            
        elif shape == 'wine':
            wine = load_wine()
            # 使用主成分分析降维到2D
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=random_state)
            X = pca.fit_transform(wine.data)
            y = wine.target
            # 转换为二分类问题
            y = (y > 0).astype(int)
            
        elif shape == 'breast_cancer':
            cancer = load_breast_cancer()
            # 使用主成分分析降维到2D
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=random_state)
            X = pca.fit_transform(cancer.data)
            y = cancer.target
            
        elif shape == 'digits_2d':
            digits = load_digits()
            # 使用主成分分析降维到2D
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=random_state)
            X = pca.fit_transform(digits.data)
            y = digits.target
            
        elif shape == 'diabetes':
            diabetes = load_diabetes()
            # 使用主成分分析降维到2D
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=random_state)
            X = pca.fit_transform(diabetes.data)
            # 将回归目标转换为二分类问题
            y = (diabetes.target > np.median(diabetes.target)).astype(int)
            
        else:
            raise ValueError(f"不支持的数据形状: {shape}")
        
        # 如果是真实数据集且需要子采样
        if shape in ['iris', 'wine', 'breast_cancer', 'digits_2d', 'diabetes'] and len(X) > n_samples:
            # 随机采样到指定数量
            np.random.seed(random_state)
            indices = np.random.choice(len(X), n_samples, replace=False)
            X = X[indices]
            y = y[indices]
        
        return X, y
    
    def generate_full_dimensional_data(self, shape='iris', n_samples=200, random_state=42):
        """
        生成完整维度的分类数据（专门用于决策树等需要多维特征的算法）
        
        Args:
            shape: 数据形状 ('iris', 'wine', 'breast_cancer', 'digits')
            n_samples: 样本数量
            random_state: 随机种子
            
        Returns:
            X: 完整维度特征矩阵
            y: 标签向量
            feature_names: 特征名称列表
        """
        logger.info(f"生成完整维度数据: {shape}, samples={n_samples}")
        
        if shape == 'iris':
            iris = load_iris()
            X = iris.data  # 保留全部4个特征
            y = iris.target
            feature_names = iris.feature_names
            
        elif shape == 'wine':
            wine = load_wine()
            X = wine.data  # 保留全部13个特征
            y = wine.target
            # 转换为二分类问题
            y = (y > 0).astype(int)
            feature_names = wine.feature_names
            
        elif shape == 'breast_cancer':
            cancer = load_breast_cancer()
            X = cancer.data  # 保留全部30个特征
            y = cancer.target
            feature_names = cancer.feature_names
            
        elif shape == 'digits':
            digits = load_digits()
            X = digits.data  # 保留全部64个像素特征
            y = digits.target
            # 转换为二分类问题（0-4 vs 5-9）
            y = (y >= 5).astype(int)
            feature_names = [f'pixel_{i}' for i in range(X.shape[1])]
            
        else:
            raise ValueError(f"不支持的完整维度数据形状: {shape}")
        
        # 如果需要子采样
        if len(X) > n_samples:
            np.random.seed(random_state)
            indices = np.random.choice(len(X), n_samples, replace=False)
            X = X[indices]
            y = y[indices]
        
        logger.info(f"生成完整维度数据完成: {X.shape[1]}个特征, {len(np.unique(y))}个类别")
        
        return X, y, feature_names
    
    def generate_regression_data(self, shape='linear', n_samples=200, noise=0.1, random_state=42):
        """
        生成回归数据
        
        Args:
            shape: 数据形状类型：
                - 基础函数: 'linear', 'quadratic', 'cubic', 'polynomial'
                - 三角函数: 'sinusoidal', 'cosine', 'damped_sine'
                - 复杂函数: 'composite', 'spiral', 'multimodal'
                - 不连续函数: 'step_function', 'sawtooth', 'piecewise_linear'
                - 特殊情况: 'heteroscedastic', 'outliers'
                - 其他: 'exponential', 'logarithmic'
            n_samples: 样本数量  
            noise: 噪声水平
            random_state: 随机种子
            
        Returns:
            X: 特征矩阵
            y: 目标值向量
        """
        logger.info(f"生成回归数据: {shape}, samples={n_samples}, noise={noise}")
        
        np.random.seed(random_state)
        # 合成数据集
        if shape == 'linear':
            X, y = make_regression(
                n_samples=n_samples,
                n_features=1,
                noise=noise * 10,
                random_state=random_state
            )
            
        elif shape == 'polynomial':
            X = np.linspace(-2, 2, n_samples).reshape(-1, 1)
            y = 0.5 * X.ravel()**3 - 2 * X.ravel()**2 + X.ravel() + np.random.normal(0, noise, n_samples)
            
        elif shape == 'quadratic':
            X = np.linspace(-3, 3, n_samples).reshape(-1, 1)
            y = 0.8 * X.ravel()**2 + 0.3 * X.ravel() + np.random.normal(0, noise, n_samples)
            
        elif shape == 'cubic':
            X = np.linspace(-2, 2, n_samples).reshape(-1, 1)
            y = X.ravel()**3 + 0.5 * X.ravel()**2 - X.ravel() + np.random.normal(0, noise, n_samples)
            
        elif shape == 'sinusoidal':
            X = np.linspace(0, 4*np.pi, n_samples).reshape(-1, 1)
            y = np.sin(X.ravel()) + 0.3 * np.sin(3*X.ravel()) + np.random.normal(0, noise, n_samples)
            
        elif shape == 'cosine':
            X = np.linspace(0, 3*np.pi, n_samples).reshape(-1, 1)
            y = 2 * np.cos(X.ravel()) + 0.5 * np.cos(2*X.ravel()) + np.random.normal(0, noise, n_samples)
            
        elif shape == 'damped_sine':
            X = np.linspace(0, 6*np.pi, n_samples).reshape(-1, 1)
            y = np.exp(-X.ravel()/10) * np.sin(X.ravel()) + np.random.normal(0, noise, n_samples)
            
        elif shape == 'composite':
            # 复合函数：多项式+三角函数
            X = np.linspace(-2, 2, n_samples).reshape(-1, 1)
            y = (0.3 * X.ravel()**2 + np.sin(2*np.pi*X.ravel()) + 
                 0.1 * X.ravel()**3 + np.random.normal(0, noise, n_samples))
            
        elif shape == 'step_function':
            # 阶跃函数（非连续）
            X = np.linspace(-3, 3, n_samples).reshape(-1, 1)
            y = np.where(X.ravel() < -2, -1,
                        np.where(X.ravel() < -1, 0,
                                np.where(X.ravel() < 0, 1,
                                        np.where(X.ravel() < 1, 2,
                                                np.where(X.ravel() < 2, 1, 0)))))
            y = y + np.random.normal(0, noise, n_samples)
            
        elif shape == 'spiral':
            # 螺旋形数据
            X = np.linspace(0, 4*np.pi, n_samples).reshape(-1, 1)
            t = X.ravel()
            y = t * np.cos(t) + t * np.sin(t) + np.random.normal(0, noise * 2, n_samples)
            
        elif shape == 'heteroscedastic':
            # 异方差数据（噪声随X变化）
            X = np.linspace(-2, 2, n_samples).reshape(-1, 1)
            base_y = 2 * X.ravel() + 1
            variable_noise = noise * (1 + np.abs(X.ravel()))
            y = base_y + np.random.normal(0, variable_noise, n_samples)
            
        elif shape == 'outliers':
            # 带异常值的线性数据
            X = np.linspace(-2, 2, n_samples).reshape(-1, 1)
            y = 1.5 * X.ravel() + 0.5 + np.random.normal(0, noise, n_samples)
            # 添加异常值
            n_outliers = max(1, n_samples // 20)
            outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
            y[outlier_indices] += np.random.normal(0, noise * 20, n_outliers)
            
        elif shape == 'multimodal':
            # 多峰分布数据
            X = np.linspace(-4, 4, n_samples).reshape(-1, 1)
            y = (np.exp(-(X.ravel() + 2)**2) + np.exp(-(X.ravel() - 2)**2) + 
                 0.5 * np.exp(-X.ravel()**2) + np.random.normal(0, noise, n_samples))
            
        elif shape == 'sawtooth':
            # 锯齿波
            X = np.linspace(0, 4*np.pi, n_samples).reshape(-1, 1)
            y = 2 * (X.ravel() % (2*np.pi)) / (2*np.pi) - 1
            y = y + np.random.normal(0, noise, n_samples)
            
        elif shape == 'piecewise_linear':
            # 分段线性函数
            X = np.linspace(-3, 3, n_samples).reshape(-1, 1)
            y = np.where(X.ravel() < -1, -X.ravel() - 1,
                        np.where(X.ravel() < 1, 2*X.ravel(),
                                -0.5*X.ravel() + 2.5))
            y = y + np.random.normal(0, noise, n_samples)
            
        elif shape == 'exponential':
            X = np.linspace(0, 2, n_samples).reshape(-1, 1)
            y = np.exp(X.ravel()) + np.random.normal(0, noise * 5, n_samples)
            
        elif shape == 'logarithmic':
            X = np.linspace(0.1, 5, n_samples).reshape(-1, 1)
            y = np.log(X.ravel()) + np.random.normal(0, noise, n_samples)
            
        elif shape == 'step':
            X = np.linspace(-2, 2, n_samples).reshape(-1, 1)
            y = np.where(X.ravel() < -1, -1,
                        np.where(X.ravel() < 0, 0,
                                np.where(X.ravel() < 1, 1, 2))) + np.random.normal(0, noise, n_samples)
        
        else:
            raise ValueError(f"不支持的数据形状: {shape}")

        return X, y
    
    def generate_clustering_data(self, shape='blobs', n_samples=200, noise=0.1, random_state=42):
        """
        生成聚类数据
        
        Args:
            shape: 数据形状 ('blobs', 'circles', 'moons', 'anisotropic', 'varied', 'smiley', 'petals')
            n_samples: 样本数量
            noise: 噪声水平
            random_state: 随机种子
            
        Returns:
            X: 特征矩阵（无标签）
        """
        logger.info(f"生成聚类数据: {shape}, samples={n_samples}, noise={noise}")
        
        if shape == 'blobs':
            X, _ = make_blobs(
                n_samples=n_samples,
                centers=4,
                n_features=2,
                cluster_std=1.0 + noise,
                random_state=random_state
            )
            
        elif shape == 'circles':
            # 生成同心圆数据
            X_inner, _ = make_circles(
                n_samples=n_samples//2,
                noise=noise,
                factor=0.3,
                random_state=random_state
            )
            X_outer, _ = make_circles(
                n_samples=n_samples//2,
                noise=noise,
                factor=0.8,
                random_state=random_state + 1
            )
            X = np.vstack([X_inner, X_outer])
            
        elif shape == 'moons':
            X, _ = make_moons(
                n_samples=n_samples,
                noise=noise,
                random_state=random_state
            )
            
        elif shape == 'anisotropic':
            # 生成各向异性的簇
            X, _ = make_blobs(
                n_samples=n_samples,
                centers=3,
                cluster_std=1.5,
                random_state=random_state
            )
            
            # 应用变换矩阵使簇变得各向异性
            transformation = np.array([[0.6, -0.6], [-0.4, 0.8]])
            X = X @ transformation
            
        elif shape == 'varied':
            # 生成不同密度的簇
            np.random.seed(random_state)
            
            # 高密度簇
            X1 = np.random.multivariate_normal([0, 0], [[0.3, 0], [0, 0.3]], n_samples//3)
            # 中密度簇  
            X2 = np.random.multivariate_normal([3, 3], [[0.8, 0], [0, 0.8]], n_samples//3)
            # 低密度簇
            X3 = np.random.multivariate_normal([-2, 3], [[1.5, 0], [0, 1.5]], n_samples//3)
            
            X = np.vstack([X1, X2, X3])
            
            # 添加额外的噪声点
            if n_samples % 3 != 0:
                n_extra = n_samples - len(X)
                X_noise = np.random.uniform(-4, 6, (n_extra, 2))
                X = np.vstack([X, X_noise])
        
        elif shape == 'smiley':
            # 生成笑脸状分布
            np.random.seed(random_state)
            
            # 笑脸参数
            face_radius = 3.0  # 脸部半径
            
            # 计算每个部分的样本数
            n_face = int(n_samples * 0.55)  # 脸部轮廓 55%
            n_left_eye = int(n_samples * 0.15)  # 左眼 15%  
            n_right_eye = int(n_samples * 0.15)  # 右眼 15%
            n_mouth = n_samples - n_face - n_left_eye - n_right_eye  # 嘴巴剩余部分
            
            X_list = []
            
            # 1. 脸部轮廓 - 圆形边界
            angles_face = np.random.uniform(0, 2*np.pi, n_face)
            # 在半径附近生成点，添加一些厚度
            radius_variation = np.random.normal(face_radius, 0.2 + noise * 0.5, n_face)
            face_x = radius_variation * np.cos(angles_face)
            face_y = radius_variation * np.sin(angles_face)
            X_list.append(np.column_stack([face_x, face_y]))
            
            # 2. 左眼 - 左上方的小圆形簇
            left_eye_center = [-face_radius * 0.35, face_radius * 0.25]
            left_eye_points = np.random.multivariate_normal(
                left_eye_center,
                [[0.15 + noise * 0.3, 0], [0, 0.15 + noise * 0.3]],
                n_left_eye
            )
            X_list.append(left_eye_points)
            
            # 3. 右眼 - 右上方的小圆形簇  
            right_eye_center = [face_radius * 0.35, face_radius * 0.25]
            right_eye_points = np.random.multivariate_normal(
                right_eye_center,
                [[0.15 + noise * 0.3, 0], [0, 0.15 + noise * 0.3]],
                n_right_eye
            )
            X_list.append(right_eye_points)
            
            # 4. 嘴巴 - 下方的弧形分布（微笑）
            mouth_center_y = -face_radius * 0.3
            mouth_width = face_radius * 0.8
            mouth_height = face_radius * 0.25
            
            # 生成弧形的x坐标
            mouth_x = np.random.uniform(-mouth_width/2, mouth_width/2, n_mouth)
            # 计算对应的y坐标，形成向上的弧形（微笑）
            mouth_y = []
            for x in mouth_x:
                # 使用椭圆的下半部分公式，但翻转为向上的弧形
                normalized_x = x / (mouth_width/2)  # 标准化到[-1, 1]
                if abs(normalized_x) <= 1:
                    # 椭圆方程: y = -sqrt(1 - (x/a)²) * b + offset
                    arc_y = -np.sqrt(max(0, 1 - normalized_x**2)) * mouth_height + mouth_center_y
                    # 添加噪声
                    arc_y += np.random.normal(0, noise * 0.3)
                    mouth_y.append(arc_y)
                else:
                    mouth_y.append(mouth_center_y + np.random.normal(0, noise * 0.3))
            
            mouth_points = np.column_stack([mouth_x, mouth_y])
            X_list.append(mouth_points)
            
            X = np.vstack(X_list)
 
        
        elif shape == 'petals':
            # 生成花瓣状分布
            np.random.seed(random_state)
            
            n_petals = 6  # 增加到6个花瓣
            n_per_petal = n_samples // n_petals
            
            X_list = []
            
            for i in range(n_petals):
                # 每个花瓣的角度
                base_angle = i * 2 * np.pi / n_petals
                
                # 生成当前花瓣的点
                n_current = n_per_petal if i < n_petals - 1 else n_samples - i * n_per_petal
                
                # 使用更细长的花瓣形状
                # 花瓣从中心向外延伸，形成放射状
                petal_length = np.random.uniform(1.0, 4.0, n_current)  # 花瓣长度变化
                petal_width = np.random.normal(0, 0.3 + noise * 0.5, n_current)  # 花瓣宽度（垂直于长度方向）
                
                # 在局部坐标系中生成细长花瓣点
                # x方向是花瓣长度，y方向是花瓣宽度
                local_x = petal_length + np.random.normal(0, noise * 0.3, n_current)
                local_y = petal_width
                
                local_points = np.column_stack([local_x, local_y])
                
                # 旋转到对应角度
                cos_a, sin_a = np.cos(base_angle), np.sin(base_angle)
                rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                rotated_points = local_points @ rotation_matrix.T
                
                X_list.append(rotated_points)
            
            # 添加花心（中央的小簇）
            n_used = sum(len(points) for points in X_list)
            n_center = n_samples - n_used
            if n_center > 0:
                center_points = np.random.multivariate_normal(
                    [0.0, 0.0], 
                    [[0.2 + noise, 0], [0, 0.2 + noise]], 
                    n_center
                )
                X_list.append(center_points)
            
            X = np.vstack(X_list)
                
        else:
            raise ValueError(f"不支持的数据形状: {shape}")
        
        return X
    
    def add_outliers(self, X, y=None, outlier_fraction=0.1, random_state=42):
        """
        向数据中添加异常值
        
        Args:
            X: 特征矩阵
            y: 标签向量（可选）
            outlier_fraction: 异常值比例
            random_state: 随机种子
            
        Returns:
            X_with_outliers: 包含异常值的特征矩阵
            y_with_outliers: 包含异常值的标签向量（如果提供了y）
        """
        np.random.seed(random_state)
        
        n_outliers = int(len(X) * outlier_fraction)
        
        if n_outliers > 0:
            # 生成异常值
            x_min, x_max = X[:, 0].min(), X[:, 0].max()
            y_min, y_max = X[:, 1].min(), X[:, 1].max()
            
            # 在数据范围外生成异常值
            range_x = x_max - x_min
            range_y = y_max - y_min
            
            outliers_x = np.random.uniform(
                x_min - 0.5 * range_x, 
                x_max + 0.5 * range_x, 
                n_outliers
            )
            outliers_y = np.random.uniform(
                y_min - 0.5 * range_y, 
                y_max + 0.5 * range_y, 
                n_outliers
            )
            
            outliers = np.column_stack([outliers_x, outliers_y])
            X_with_outliers = np.vstack([X, outliers])
            
            if y is not None:
                # 为异常值分配随机标签
                outlier_labels = np.random.choice(np.unique(y), n_outliers)
                y_with_outliers = np.hstack([y, outlier_labels])
                return X_with_outliers, y_with_outliers
            else:
                return X_with_outliers
        
        return X if y is None else (X, y)
    
    def normalize_data(self, X, fit=True):
        """
        标准化数据
        
        Args:
            X: 特征矩阵
            fit: 是否拟合scaler
            
        Returns:
            X_normalized: 标准化后的特征矩阵
        """
        if fit:
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)
    
    def get_available_data_types(self):
        """
        获取所有可用的数据类型
        
        Returns:
            data_types: 数据类型字典
        """
        return {
            'regression': {
                'basic': ['linear', 'quadratic', 'cubic', 'polynomial'],
                'trigonometric': ['sinusoidal', 'cosine', 'damped_sine'],
                'complex': ['composite', 'spiral', 'multimodal'],
                'discontinuous': ['step_function', 'sawtooth', 'piecewise_linear'],
                'special': ['heteroscedastic', 'outliers'],
                'other': ['exponential', 'logarithmic']
            },
            'classification': {
                'basic': ['blobs', 'circles', 'moons'],
                'complex': ['gaussian', 'xor', 'spiral'],
                'real_datasets': ['iris', 'wine', 'breast_cancer']
            },
            'clustering': {
                'basic': ['blobs', 'circles', 'moons'],
                'complex': ['anisotropic', 'varied', 'smiley', 'petals']
            }
        }
    
    def get_data_description(self, data_type, shape):
        """
        获取数据类型的描述信息
        
        Args:
            data_type: 数据类型 ('regression', 'classification', 'clustering')
            shape: 具体形状
            
        Returns:
            description: 描述字典
        """
        descriptions = {
            'regression': {
                'linear': {'name': '线性回归', 'description': '简单的线性关系数据', 'complexity': '简单'},
                'quadratic': {'name': '二次函数', 'description': '抛物线形状的二次关系', 'complexity': '简单'},
                'cubic': {'name': '三次函数', 'description': 'S型曲线的三次关系', 'complexity': '中等'},
                'polynomial': {'name': '多项式', 'description': '复杂的高次多项式关系', 'complexity': '中等'},
                'sinusoidal': {'name': '正弦波', 'description': '周期性正弦函数关系', 'complexity': '中等'},
                'cosine': {'name': '余弦波', 'description': '周期性余弦函数关系', 'complexity': '中等'},
                'damped_sine': {'name': '衰减正弦', 'description': '振幅逐渐减小的正弦波', 'complexity': '复杂'},
                'composite': {'name': '复合函数', 'description': '多项式与三角函数的组合', 'complexity': '复杂'},
                'spiral': {'name': '螺旋型', 'description': '螺旋形状的非线性关系', 'complexity': '复杂'},
                'multimodal': {'name': '多峰分布', 'description': '具有多个峰值的复杂分布', 'complexity': '复杂'},
                'step_function': {'name': '阶跃函数', 'description': '不连续的阶梯状函数', 'complexity': '复杂'},
                'sawtooth': {'name': '锯齿波', 'description': '锯齿状的周期函数', 'complexity': '复杂'},
                'piecewise_linear': {'name': '分段线性', 'description': '由多段直线组成的函数', 'complexity': '中等'},
                'heteroscedastic': {'name': '异方差', 'description': '噪声大小随输入变化', 'complexity': '中等'},
                'outliers': {'name': '带异常值', 'description': '包含异常点的线性数据', 'complexity': '中等'},
                'exponential': {'name': '指数函数', 'description': '指数增长关系', 'complexity': '简单'},
                'logarithmic': {'name': '对数函数', 'description': '对数增长关系', 'complexity': '简单'}
            }
        }
        
        if data_type in descriptions and shape in descriptions[data_type]:
            return descriptions[data_type][shape]
        else:
            return {'name': shape, 'description': '未知数据类型', 'complexity': '未知'}
    
    def get_recommended_params(self, data_type, shape):
        """
        获取特定数据类型的推荐参数
        
        Args:
            data_type: 数据类型
            shape: 数据形状
            
        Returns:
            params: 推荐参数字典
        """
        recommendations = {
            'regression': {
                'linear': {'learning_rate': 0.01, 'max_iter': 50, 'optimizer': 'sgd'},
                'quadratic': {'learning_rate': 0.005, 'max_iter': 100, 'optimizer': 'adam'},
                'cubic': {'learning_rate': 0.003, 'max_iter': 150, 'optimizer': 'adam'},
                'polynomial': {'learning_rate': 0.001, 'max_iter': 200, 'optimizer': 'adam'},
                'sinusoidal': {'learning_rate': 0.01, 'max_iter': 150, 'optimizer': 'momentum'},
                'cosine': {'learning_rate': 0.01, 'max_iter': 150, 'optimizer': 'momentum'},
                'damped_sine': {'learning_rate': 0.008, 'max_iter': 200, 'optimizer': 'rmsprop'},
                'composite': {'learning_rate': 0.005, 'max_iter': 250, 'optimizer': 'adam'},
                'spiral': {'learning_rate': 0.003, 'max_iter': 300, 'optimizer': 'adam'},
                'multimodal': {'learning_rate': 0.01, 'max_iter': 200, 'optimizer': 'rmsprop'},
                'step_function': {'learning_rate': 0.02, 'max_iter': 100, 'optimizer': 'sgd'},
                'sawtooth': {'learning_rate': 0.015, 'max_iter': 150, 'optimizer': 'momentum'},
                'piecewise_linear': {'learning_rate': 0.01, 'max_iter': 100, 'optimizer': 'sgd'},
                'heteroscedastic': {'learning_rate': 0.01, 'max_iter': 150, 'optimizer': 'momentum'},
                'outliers': {'learning_rate': 0.005, 'max_iter': 200, 'optimizer': 'adagrad'},
                'exponential': {'learning_rate': 0.001, 'max_iter': 150, 'optimizer': 'adam'},
                'logarithmic': {'learning_rate': 0.01, 'max_iter': 100, 'optimizer': 'sgd'}
            }
        }
        
        default_params = {'learning_rate': 0.01, 'max_iter': 100, 'optimizer': 'sgd'}
        
        if data_type in recommendations and shape in recommendations[data_type]:
            return recommendations[data_type][shape]
        else:
            return default_params
