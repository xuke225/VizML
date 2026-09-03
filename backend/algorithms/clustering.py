"""
聚类算法模块
基于 sklearn 实现各种聚类算法
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import logging
import time
import copy

logger = logging.getLogger(__name__)


class ClusteringAlgorithms:
    """聚类算法类"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        np.random.seed(42)
    
    def train_model(self, X, algorithm='kmeans', **kwargs):
        """
        训练聚类模型
        
        Args:
            X: 特征矩阵
            algorithm: 算法类型 ('kmeans', 'dbscan', 'hierarchical', 'gmm')
            **kwargs: 算法参数
            
        Returns:
            model: 训练好的模型
            labels: 聚类标签
            centers: 聚类中心（如果有）
            training_info: 训练信息
        """
        logger.info(f"训练聚类模型: {algorithm}")
        
        # 数据预处理
        if kwargs.get('normalize', False):
            X = self.scaler.fit_transform(X)
        
        training_info = {}
        centers = None
        
        if algorithm == 'kmeans':
            model = self._train_kmeans(X, **kwargs)
            labels = model.labels_
            centers = model.cluster_centers_
            training_info = {
                'n_iter': model.n_iter_,
                'inertia': model.inertia_
            }
            
        elif algorithm == 'dbscan':
            model = self._train_dbscan(X, **kwargs)
            labels = model.labels_
            training_info = {
                'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
                'n_noise': list(labels).count(-1)
            }
            
        elif algorithm == 'hierarchical':
            model = self._train_hierarchical(X, **kwargs)
            labels = model.labels_
            training_info = {
                'n_clusters': model.n_clusters_,
                'n_connected_components': model.n_connected_components_
            }
            
        elif algorithm == 'gmm':
            model = self._train_gmm(X, **kwargs)
            labels = model.predict(X)
            centers = model.means_
            training_info = {
                'n_iter': model.n_iter_,
                'lower_bound': model.lower_bound_,
                'aic': model.aic(X),
                'bic': model.bic(X),
                'weights': model.weights_.tolist(),
                'covariances': model.covariances_.tolist(),
                'covariance_type': model.covariance_type
            }
            
        else:
            raise ValueError(f"不支持的聚类算法: {algorithm}")
        
        # 计算聚类评估指标
        if len(set(labels)) > 1:  # 至少有两个聚类
            try:
                # 排除噪声点计算silhouette_score
                valid_indices = labels != -1
                if np.sum(valid_indices) > 1 and len(set(labels[valid_indices])) > 1:
                    training_info['silhouette_score'] = silhouette_score(
                        X[valid_indices], labels[valid_indices]
                    )
                    training_info['calinski_harabasz_score'] = calinski_harabasz_score(
                        X[valid_indices], labels[valid_indices]
                    )
            except Exception as e:
                logger.warning(f"无法计算聚类评估指标: {e}")
        
        return model, labels, centers, training_info
    
    def _train_kmeans(self, X, n_clusters=3, max_iter=100, init='k-means++', **kwargs):
        """训练K-Means模型"""
        model = KMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            init=init,
            random_state=42,
            n_init=10
        )
        model.fit(X)
        return model
    
    def _train_dbscan(self, X, eps=0.05, min_samples=5, **kwargs):
        """训练DBSCAN模型"""
        model = DBSCAN(
            eps=eps,
            min_samples=min_samples
        )
        model.fit(X)
        return model
    
    def _train_hierarchical(self, X, n_clusters=3, linkage='ward', **kwargs):
        """训练层次聚类模型"""
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage
        )
        model.fit(X)
        return model
    
    def _train_gmm(self, X, n_components=3, covariance_type='full', init_params='kmeans', 
                   max_iter=100, tol=1e-3, **kwargs):
        """训练高斯混合模型"""
        model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            init_params=init_params,
            max_iter=max_iter,
            tol=tol,
            random_state=42
        )
        model.fit(X)
        return model
    
    def compare_algorithms(self, X, algorithms=['kmeans', 'dbscan', 'hierarchical', 'gmm'], **algorithm_params):
        """
        对比多个聚类算法的性能
        
        Args:
            X: 特征矩阵
            algorithms: 要对比的算法列表
            **algorithm_params: 各算法的参数字典
            
        Returns:
            comparison_results: 包含各算法结果和对比指标的字典
        """
        logger.info(f"开始对比聚类算法: {algorithms}")
        
        comparison_results = {
            'algorithms': {},
            'comparison_metrics': {},
            'best_algorithm': None,
            'execution_times': {}
        }
        
        all_labels = {}
        valid_algorithms = []
        
        # 运行每个算法
        for algorithm in algorithms:
            try:
                start_time = time.time()
                
                # 获取算法特定参数
                params = algorithm_params.get(algorithm, {})
                
                # 训练模型
                model, labels, centers, training_info = self.train_model(X, algorithm=algorithm, **params)
                
                execution_time = time.time() - start_time
                comparison_results['execution_times'][algorithm] = execution_time
                
                # 存储结果
                comparison_results['algorithms'][algorithm] = {
                    'model': model,
                    'labels': labels.tolist(),
                    'centers': centers.tolist() if centers is not None else None,
                    'training_info': training_info,
                    'execution_time': execution_time
                }
                
                all_labels[algorithm] = labels
                valid_algorithms.append(algorithm)
                
                logger.info(f"算法 {algorithm} 完成，用时 {execution_time:.3f}秒")
                
            except Exception as e:
                logger.error(f"算法 {algorithm} 执行失败: {e}")
                comparison_results['algorithms'][algorithm] = {
                    'error': str(e),
                    'execution_time': 0
                }
        
        # 计算对比指标
        if len(valid_algorithms) > 1:
            comparison_results['comparison_metrics'] = self._calculate_comparison_metrics(
                X, all_labels, valid_algorithms
            )
            
            # 选择最佳算法
            comparison_results['best_algorithm'] = self._select_best_algorithm(
                comparison_results['algorithms'], comparison_results['comparison_metrics']
            )
        
        return comparison_results
    
    def _calculate_comparison_metrics(self, X, all_labels, algorithms):
        """计算算法间的对比指标"""
        metrics = {}
        
        for algorithm in algorithms:
            labels = all_labels[algorithm]
            algo_metrics = {}
            
            # 计算内部评估指标
            unique_labels = set(labels)
            if len(unique_labels) > 1:
                # 排除噪声点
                valid_indices = labels != -1
                if np.sum(valid_indices) > 1 and len(set(labels[valid_indices])) > 1:
                    try:
                        algo_metrics['silhouette_score'] = silhouette_score(X[valid_indices], labels[valid_indices])
                        algo_metrics['calinski_harabasz_score'] = calinski_harabasz_score(X[valid_indices], labels[valid_indices])
                    except:
                        algo_metrics['silhouette_score'] = -1
                        algo_metrics['calinski_harabasz_score'] = -1
                else:
                    algo_metrics['silhouette_score'] = -1
                    algo_metrics['calinski_harabasz_score'] = -1
            else:
                algo_metrics['silhouette_score'] = -1
                algo_metrics['calinski_harabasz_score'] = -1
            
            # 计算聚类统计信息
            algo_metrics['n_clusters'] = len(unique_labels) - (1 if -1 in unique_labels else 0)
            algo_metrics['n_noise'] = int(np.sum(labels == -1))
            algo_metrics['n_points_clustered'] = int(np.sum(labels != -1))
            
            metrics[algorithm] = algo_metrics
        
        # 计算算法间的一致性指标
        algorithm_pairs = [(algorithms[i], algorithms[j]) 
                          for i in range(len(algorithms)) 
                          for j in range(i+1, len(algorithms))]
        
        pairwise_metrics = {}
        for algo1, algo2 in algorithm_pairs:
            labels1 = all_labels[algo1]
            labels2 = all_labels[algo2]
            
            # 调整兰德指数
            try:
                ari = adjusted_rand_score(labels1, labels2)
                nmi = normalized_mutual_info_score(labels1, labels2)
            except:
                ari = 0
                nmi = 0
            
            pairwise_metrics[f"{algo1}_vs_{algo2}"] = {
                'adjusted_rand_score': ari,
                'normalized_mutual_info': nmi
            }
        
        metrics['pairwise_comparison'] = pairwise_metrics
        return metrics
    
    def _select_best_algorithm(self, algorithms_results, comparison_metrics):
        """根据多个指标选择最佳算法"""
        scores = {}
        
        for algorithm in algorithms_results.keys():
            if 'error' in algorithms_results[algorithm]:
                continue
                
            metrics = comparison_metrics.get(algorithm, {})
            score = 0
            
            # 轮廓系数 (越高越好)
            silhouette = metrics.get('silhouette_score', -1)
            if silhouette > 0:
                score += silhouette * 0.4
            
            # Calinski-Harabasz指数 (越高越好, 标准化处理)
            ch_score = metrics.get('calinski_harabasz_score', -1)
            if ch_score > 0:
                score += min(ch_score / 1000, 1) * 0.3  # 标准化到[0,1]
            
            # 聚类点比例 (越高越好)
            n_clustered = metrics.get('n_points_clustered', 0)
            total_points = n_clustered + metrics.get('n_noise', 0)
            if total_points > 0:
                clustered_ratio = n_clustered / total_points
                score += clustered_ratio * 0.2
            
            # 执行时间惩罚 (越快越好)
            exec_time = algorithms_results[algorithm].get('execution_time', float('inf'))
            if exec_time > 0:
                time_penalty = max(0, 1 - exec_time / 10)  # 10秒以上严重扣分
                score += time_penalty * 0.1
            
            scores[algorithm] = score
        
        # 返回得分最高的算法
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        else:
            return None
    
    # ================== 单步调试功能 ==================
    
    def kmeans_debug_init(self, X, n_clusters=3, init='k-means++', max_iter=100):
        """
        初始化K-means单步调试
        
        Args:
            X: 特征矩阵
            n_clusters: 聚类数量
            init: 初始化方法
            max_iter: 最大迭代次数
            
        Returns:
            debug_state: 调试状态字典
        """
        logger.info(f"初始化K-means单步调试，聚类数量: {n_clusters}")
        
        X = np.array(X)
        n_samples, n_features = X.shape
        
        # 初始化质心
        if init == 'k-means++':
            centroids = self._kmeans_plus_plus_init(X, n_clusters)
        elif init == 'random':
            centroids = X[np.random.choice(n_samples, n_clusters, replace=False)]
        else:
            # 如果是数组，直接使用
            centroids = np.array(init)
        
        # 初始化调试状态
        debug_state = {
            'X': X.tolist(),
            'n_clusters': n_clusters,
            'max_iter': max_iter,
            'current_iter': 0,
            'converged': False,
            'centroids_history': [centroids.tolist()],
            'labels_history': [],
            'distances_history': [],
            'inertia_history': [],
            'step_info': [],
            'current_centroids': centroids.tolist(),
            'previous_centroids': None,
            'step_type': 'init'  # init, assign, update, converged
        }
        
        # 计算初始距离和标签
        distances = euclidean_distances(X, centroids)
        labels = np.argmin(distances, axis=1)
        inertia = np.sum(np.min(distances**2, axis=1))
        
        debug_state['labels_history'].append(labels.tolist())
        debug_state['distances_history'].append(distances.tolist())
        debug_state['inertia_history'].append(float(inertia))
        debug_state['step_info'].append({
            'step': 0,
            'type': 'init',
            'description': f'初始化 {n_clusters} 个质心',
            'details': {
                'centroids': centroids.tolist(),
                'inertia': float(inertia)
            }
        })
        
        return debug_state
    
    def kmeans_debug_step(self, debug_state):
        """
        执行K-means的一个调试步骤
        
        Args:
            debug_state: 当前调试状态
            
        Returns:
            updated_debug_state: 更新后的调试状态
        """
        if debug_state['converged'] or debug_state['current_iter'] >= debug_state['max_iter']:
            return debug_state
        
        X = np.array(debug_state['X'])
        current_centroids = np.array(debug_state['current_centroids'])
        current_iter = debug_state['current_iter']
        
        # 分配步骤：计算每个点到质心的距离并分配标签
        distances = euclidean_distances(X, current_centroids)
        labels = np.argmin(distances, axis=1)
        
        debug_state['step_type'] = 'assign'
        debug_state['labels_history'].append(labels.tolist())
        debug_state['distances_history'].append(distances.tolist())
        
        step_info = {
            'step': len(debug_state['step_info']),
            'type': 'assign',
            'iteration': current_iter + 1,
            'description': f'第 {current_iter + 1} 轮：分配数据点到最近质心',
            'details': {
                'labels': labels.tolist(),
                'distances': distances.tolist()
            }
        }
        
        # 更新步骤：重新计算质心
        new_centroids = np.array([X[labels == i].mean(axis=0) if np.sum(labels == i) > 0 
                                 else current_centroids[i] for i in range(debug_state['n_clusters'])])
        
        # 检查收敛性
        centroids_shift = np.sum((new_centroids - current_centroids) ** 2)
        converged = bool(centroids_shift < 1e-6)
        
        # 计算惯性
        inertia = np.sum(np.min(distances**2, axis=1))
        
        # 更新调试状态
        debug_state['previous_centroids'] = current_centroids.tolist()
        debug_state['current_centroids'] = new_centroids.tolist()
        debug_state['centroids_history'].append(new_centroids.tolist())
        debug_state['inertia_history'].append(float(inertia))
        debug_state['current_iter'] += 1
        debug_state['converged'] = converged
        
        step_info['details'].update({
            'old_centroids': current_centroids.tolist(),
            'new_centroids': new_centroids.tolist(),
            'centroids_shift': float(centroids_shift),
            'inertia': float(inertia),
            'converged': converged
        })
        
        if converged:
            debug_state['step_type'] = 'converged'
            step_info['type'] = 'converged'
            step_info['description'] = f'算法收敛！质心位移: {centroids_shift:.6f}'
        else:
            debug_state['step_type'] = 'update'
            step_info['type'] = 'update'
            step_info['description'] = f'第 {current_iter + 1} 轮：更新质心位置'
        
        debug_state['step_info'].append(step_info)
        
        return debug_state
    
    def _kmeans_plus_plus_init(self, X, n_clusters):
        """K-means++初始化算法"""
        n_samples, n_features = X.shape
        centroids = np.empty((n_clusters, n_features))
        
        # 随机选择第一个质心
        centroids[0] = X[np.random.randint(n_samples)]
        
        # 选择其余质心
        for c_id in range(1, n_clusters):
            # 计算每个点到最近质心的距离
            distances = np.array([min([euclidean_distances([x], [c])[0][0]**2 for c in centroids[:c_id]]) for x in X])
            
            # 按距离概率选择下一个质心
            probabilities = distances / distances.sum()
            cumulative_probabilities = probabilities.cumsum()
            r = np.random.rand()
            
            for j, p in enumerate(cumulative_probabilities):
                if r < p:
                    i = j
                    break
            
            centroids[c_id] = X[i]
        
        return centroids
    
    def dbscan_debug_init(self, X, eps=0.5, min_samples=5):
        """
        初始化DBSCAN单步调试
        
        Args:
            X: 特征矩阵
            eps: 邻域半径
            min_samples: 核心点最小样本数
            
        Returns:
            debug_state: 调试状态字典
        """
        logger.info(f"初始化DBSCAN单步调试，eps: {eps}, min_samples: {min_samples}")
        
        X = np.array(X)
        n_samples = X.shape[0]
        
        debug_state = {
            'X': X.tolist(),
            'eps': eps,
            'min_samples': min_samples,
            'n_samples': n_samples,
            'current_point_idx': 0,
            'labels': [-1] * n_samples,  # -1表示未处理，-2表示噪声点
            'visited': [False] * n_samples,
            'current_cluster': 0,
            'step_info': [],
            'neighbors_cache': {},  # 缓存邻域信息
            'core_points': [],
            'border_points': [],
            'noise_points': [],
            'finished': False,
            'step_type': 'init'  # init, scan_point, expand_cluster, finished
        }
        
        # 预计算距离矩阵
        distances = euclidean_distances(X)
        debug_state['distances'] = distances.tolist()
        
        # 初始化步骤信息
        debug_state['step_info'].append({
            'step': 0,
            'type': 'init',
            'description': f'初始化DBSCAN算法，eps={eps}, min_samples={min_samples}',
            'details': {
                'total_points': n_samples,
                'eps': eps,
                'min_samples': min_samples
            }
        })
        
        return debug_state
    
    def dbscan_debug_step(self, debug_state):
        """
        执行DBSCAN的一个调试步骤
        
        Args:
            debug_state: 当前调试状态
            
        Returns:
            updated_debug_state: 更新后的调试状态
        """
        if debug_state['finished']:
            return debug_state
        
        X = np.array(debug_state['X'])
        distances = np.array(debug_state['distances'])
        current_idx = debug_state['current_point_idx']
        
        # 如果所有点都已处理，算法结束
        if current_idx >= debug_state['n_samples']:
            debug_state['finished'] = True
            debug_state['step_type'] = 'finished'
            
            # 最终统计
            final_stats = self._dbscan_final_stats(debug_state)
            debug_state['step_info'].append({
                'step': len(debug_state['step_info']),
                'type': 'finished',
                'description': '算法完成！',
                'details': final_stats
            })
            
            return debug_state
        
        # 跳过已访问的点
        while current_idx < debug_state['n_samples'] and debug_state['visited'][current_idx]:
            current_idx += 1
        
        if current_idx >= debug_state['n_samples']:
            debug_state['finished'] = True
            return debug_state
        
        # 标记当前点为已访问
        debug_state['visited'][current_idx] = True
        
        # 找到邻域
        neighbors = self._get_neighbors(distances, current_idx, debug_state['eps'])
        debug_state['neighbors_cache'][current_idx] = neighbors
        
        step_info = {
            'step': len(debug_state['step_info']),
            'type': 'scan_point',
            'current_point': current_idx,
            'description': f'扫描点 {current_idx}，找到 {len(neighbors)} 个邻居',
            'details': {
                'point': X[current_idx].tolist(),
                'neighbors': neighbors,
                'neighbor_count': len(neighbors)
            }
        }
        
        # 检查是否是核心点
        if len(neighbors) >= debug_state['min_samples']:
            # 是核心点，开始扩展聚类
            debug_state['core_points'].append(current_idx)
            debug_state['labels'][current_idx] = debug_state['current_cluster']
            
            # 扩展聚类
            self._expand_cluster_debug(debug_state, current_idx, neighbors, distances)
            
            step_info['type'] = 'expand_cluster'
            step_info['description'] = f'点 {current_idx} 是核心点，创建聚类 {debug_state["current_cluster"]}'
            step_info['details'].update({
                'is_core': True,
                'cluster_id': debug_state['current_cluster']
            })
            
            debug_state['current_cluster'] += 1
        else:
            # 不是核心点，暂时标记为噪声点
            debug_state['labels'][current_idx] = -1
            step_info['details'].update({
                'is_core': False,
                'marked_as_noise': True
            })
        
        debug_state['step_info'].append(step_info)
        debug_state['current_point_idx'] = current_idx + 1
        
        return debug_state
    
    def _get_neighbors(self, distances, point_idx, eps):
        """获取点的邻域"""
        return np.where(distances[point_idx] <= eps)[0].tolist()
    
    def _expand_cluster_debug(self, debug_state, core_point, neighbors, distances):
        """扩展聚类（用于调试）"""
        cluster_id = debug_state['current_cluster']
        seed_set = list(neighbors)
        i = 0
        
        while i < len(seed_set):
            neighbor_idx = seed_set[i]
            
            # 如果邻居未访问过
            if not debug_state['visited'][neighbor_idx]:
                debug_state['visited'][neighbor_idx] = True
                
                # 获取邻居的邻域
                neighbor_neighbors = self._get_neighbors(distances, neighbor_idx, debug_state['eps'])
                debug_state['neighbors_cache'][neighbor_idx] = neighbor_neighbors
                
                # 如果邻居也是核心点，将其邻居加入种子集
                if len(neighbor_neighbors) >= debug_state['min_samples']:
                    debug_state['core_points'].append(neighbor_idx)
                    for nn in neighbor_neighbors:
                        if nn not in seed_set:
                            seed_set.append(nn)
            
            # 如果邻居还没有被分配到聚类，将其分配到当前聚类
            if debug_state['labels'][neighbor_idx] == -1:
                debug_state['labels'][neighbor_idx] = cluster_id
                if neighbor_idx not in debug_state['core_points']:
                    debug_state['border_points'].append(neighbor_idx)
            
            i += 1
    
    def _dbscan_final_stats(self, debug_state):
        """计算DBSCAN最终统计信息"""
        labels = debug_state['labels']
        
        # 重新计算噪声点（最终未被分配到任何聚类的点）
        noise_points = [i for i, label in enumerate(labels) if label == -1]
        debug_state['noise_points'] = noise_points
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = len(noise_points)
        
        return {
            'n_clusters': n_clusters,
            'n_core_points': len(debug_state['core_points']),
            'n_border_points': len(debug_state['border_points']),
            'n_noise_points': n_noise,
            'core_points': debug_state['core_points'],
            'border_points': debug_state['border_points'],
            'noise_points': noise_points,
            'final_labels': labels
        }