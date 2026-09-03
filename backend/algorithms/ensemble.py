"""
集成学习算法模块
基于 sklearn 实现各种集成学习算法
"""

import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    VotingClassifier, VotingRegressor,
    StackingClassifier, StackingRegressor,
    BaggingClassifier, BaggingRegressor
)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import logging

logger = logging.getLogger(__name__)


class EnsembleAlgorithms:
    """集成学习算法类"""
    
    def __init__(self):
        pass
    
    def train_model(self, X, y, algorithm='random_forest', task_type='classification', return_enhanced=False, **kwargs):
        """
        训练集成学习模型
        
        Args:
            X: 特征矩阵
            y: 目标向量
            algorithm: 算法类型 ('random_forest', 'adaboost', 'gradient_boosting', 'voting', 'stacking', 'bagging', 'xgboost')
            task_type: 任务类型 ('classification' 或 'regression')
            **kwargs: 算法参数
            
        Returns:
            model: 训练好的模型
            feature_importance: 特征重要性（如果支持）
        """
        logger.info(f"训练集成学习模型: {algorithm} ({task_type})")
        
        feature_importance = None
        
        if algorithm == 'random_forest':
            model = self._train_random_forest(X, y, task_type, **kwargs)
            feature_importance = model.feature_importances_
            
        elif algorithm == 'adaboost':
            model = self._train_adaboost(X, y, task_type, **kwargs)
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
                
        elif algorithm == 'gradient_boosting':
            model = self._train_gradient_boosting(X, y, task_type, **kwargs)
            feature_importance = model.feature_importances_
            
            
        elif algorithm == 'voting':
            model = self._train_voting(X, y, task_type, **kwargs)
            # Voting集成的特征重要性是各基学习器的平均
            feature_importance = self._calculate_voting_feature_importance(model, X)
            
        elif algorithm == 'stacking':
            model = self._train_stacking(X, y, task_type, **kwargs)
            # Stacking集成的特征重要性是各基学习器的平均
            feature_importance = self._calculate_stacking_feature_importance(model, X)
            
        elif algorithm == 'bagging':
            model = self._train_bagging(X, y, task_type, **kwargs)
            # Bagging集成的特征重要性是各基学习器的平均
            feature_importance = self._calculate_bagging_feature_importance(model, X)
            
        elif algorithm == 'xgboost':
            model = self._train_xgboost(X, y, task_type, **kwargs)
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
            
        else:
            raise ValueError(f"不支持的集成算法: {algorithm}")
        
        return model, feature_importance
    
    def get_enhanced_ensemble_results(self, model, X_train, y_train, X_test, y_test, algorithm):
        """
        获取增强的集成学习结果，包含基学习器详细信息
        
        Args:
            model: 训练好的集成模型
            X_train, y_train: 训练数据
            X_test, y_test: 测试数据
            algorithm: 算法类型
            
        Returns:
            results: 增强结果字典
        """
        try:
            results = {
                'base_learners': [],
                'individual_predictions': {},
                'decision_boundaries': None
            }
            
            # 获取基学习器性能
            base_performances = self.get_base_estimator_performance(model, X_test, y_test, algorithm)
            
            # 计算基学习器统计信息
            if base_performances:
                performances = [p['score'] for p in base_performances]
                results['base_learners_stats'] = {
                    'count': len(performances),
                    'avg_performance': np.mean(performances),
                    'best_performance': max(performances),
                    'worst_performance': min(performances),
                    'std_performance': np.std(performances)
                }
            
            # 获取个体预测
            individual_preds_train = self.predict_with_individual_estimators(model, X_train, algorithm)
            individual_preds_test = self.predict_with_individual_estimators(model, X_test, algorithm)
            
            results['individual_predictions'] = {
                'train': individual_preds_train,
                'test': individual_preds_test
            }
            
            # 获取基学习器信息
            if hasattr(model, 'estimators_'):
                max_estimators = len(model.estimators_)
                for i in range(max_estimators):
                    estimator = model.estimators_[i]
                    
                    # 找到对应的性能数据
                    perf_data = None
                    for perf in base_performances:
                        if perf.get('estimator_index') == i:
                            perf_data = perf
                            break
                    
                    base_info = {
                        'id': i,
                        'type': type(estimator).__name__,
                        'performance': perf_data.get('score', 0) if perf_data else 0,
                        'metric': perf_data.get('metric', 'accuracy') if perf_data else 'accuracy'
                    }
                    
                    # 添加特定算法的信息
                    if hasattr(estimator, 'tree_') and hasattr(estimator.tree_, 'node_count'):
                        base_info['node_count'] = estimator.tree_.node_count
                        base_info['max_depth'] = estimator.tree_.max_depth
                    
                    results['base_learners'].append(base_info)
                    
            elif hasattr(model, 'named_estimators_'):
                for name, estimator in model.named_estimators_.items():
                    # 找到对应的性能数据
                    perf_data = None
                    for perf in base_performances:
                        if perf.get('estimator_name') == name:
                            perf_data = perf
                            break
                    
                    base_info = {
                        'id': name,
                        'type': type(estimator).__name__,
                        'performance': perf_data.get('score', 0) if perf_data else 0,
                        'metric': perf_data.get('metric', 'accuracy') if perf_data else 'accuracy'
                    }
                    
                    results['base_learners'].append(base_info)
            
            return results
            
        except Exception as e:
            logger.error(f"获取增强集成结果失败: {e}")
            return {
                'base_learners': [],
                'individual_predictions': {'train': [], 'test': []},
                'decision_boundaries': None
            }

    def _create_base_learners(self, task_type, base_learner_types=None, base_learner_params=None):
        """
        创建基学习器列表
        
        Args:
            task_type: 任务类型 ('classification' 或 'regression')
            base_learner_types: 基学习器类型列表
            base_learner_params: 基学习器参数字典
            
        Returns:
            estimators: 基学习器列表
        """
        if base_learner_types is None:
            if task_type == 'classification':
                base_learner_types = ['decision_tree', 'svm', 'knn']
            else:
                base_learner_types = ['decision_tree', 'svr', 'knn']
        
        if base_learner_params is None:
            base_learner_params = {}
        
        estimators = []
        
        for learner_type in base_learner_types:
            params = base_learner_params.get(learner_type, {})
            
            if learner_type == 'decision_tree':
                if task_type == 'classification':
                    estimator = DecisionTreeClassifier(random_state=42, **params)
                else:
                    estimator = DecisionTreeRegressor(random_state=42, **params)
                    
            elif learner_type == 'svm' or learner_type == 'svr':
                if task_type == 'classification':
                    estimator = SVC(probability=True, random_state=42, **params)
                else:
                    estimator = SVR(**params)
                    
            elif learner_type == 'knn':
                if task_type == 'classification':
                    estimator = KNeighborsClassifier(**params)
                else:
                    estimator = KNeighborsRegressor(**params)
                    
            elif learner_type == 'naive_bayes':
                if task_type == 'classification':
                    estimator = GaussianNB(**params)
                else:
                    continue  # 朴素贝叶斯不支持回归
                    
            elif learner_type == 'logistic_regression' or learner_type == 'linear_regression':
                if task_type == 'classification':
                    estimator = LogisticRegression(random_state=42, max_iter=1000, **params)
                else:
                    estimator = LinearRegression(**params)
                    
            elif learner_type == 'neural_network':
                if task_type == 'classification':
                    estimator = MLPClassifier(random_state=42, max_iter=500, **params)
                else:
                    estimator = MLPRegressor(random_state=42, max_iter=500, **params)
                    
            else:
                logger.warning(f"不支持的基学习器类型: {learner_type}")
                continue
            
            estimators.append((learner_type, estimator))
        
        return estimators
    
    def _train_random_forest(self, X, y, task_type, **kwargs):
        """训练随机森林"""
        n_estimators = kwargs.get('n_estimators', 100)
        max_depth = kwargs.get('max_depth', None)
        min_samples_split = kwargs.get('min_samples_split', 2)
        min_samples_leaf = kwargs.get('min_samples_leaf', 1)
        max_features = kwargs.get('max_features', 'sqrt' if task_type == 'classification' else 'auto')
        
        if task_type == 'classification':
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=42
            )
        else:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=42
            )
        
        model.fit(X, y)
        return model
    
    def _train_adaboost(self, X, y, task_type, **kwargs):
        """训练AdaBoost"""
        n_estimators = kwargs.get('n_estimators', 50)
        learning_rate = kwargs.get('learning_rate', 1.0)
        
        if task_type == 'classification':
            base_estimator = DecisionTreeClassifier(max_depth=1, random_state=42)
            model = AdaBoostClassifier(
                estimator=base_estimator,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=42
            )
        else:
            base_estimator = DecisionTreeRegressor(max_depth=1, random_state=42)
            model = AdaBoostRegressor(
                estimator=base_estimator,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=42
            )
        
        model.fit(X, y)
        return model
    
    def _train_gradient_boosting(self, X, y, task_type, **kwargs):
        """训练梯度提升"""
        n_estimators = kwargs.get('n_estimators', 100)
        learning_rate = kwargs.get('learning_rate', 0.1)
        max_depth = kwargs.get('max_depth', 3)
        subsample = kwargs.get('subsample', 1.0)
        
        if task_type == 'classification':
            model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                random_state=42
            )
        else:
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                random_state=42
            )
        
        model.fit(X, y)
        return model
    
    def _train_voting(self, X, y, task_type, **kwargs):
        """训练投票集成"""
        voting = kwargs.get('voting', 'hard' if task_type == 'classification' else None)
        base_learner_types = kwargs.get('base_learner_types')
        base_learner_params = kwargs.get('base_learner_params')
        
        # 如果没有指定基学习器，使用默认配置
        if base_learner_types is None:
            if task_type == 'classification':
                estimators = [
                    ('dt', DecisionTreeClassifier(random_state=42)),
                    ('svm', SVC(probability=True, random_state=42)),
                    ('lr', LogisticRegression(random_state=42, max_iter=1000))
                ]
            else:
                estimators = [
                    ('dt', DecisionTreeRegressor(random_state=42)),
                    ('svr', SVR()),
                    ('lr', LinearRegression())
                ]
        else:
            # 使用可配置的基学习器
            estimators = self._create_base_learners(task_type, base_learner_types, base_learner_params)
        
        if task_type == 'classification':
            model = VotingClassifier(
                estimators=estimators,
                voting=voting
            )
        else:
            model = VotingRegressor(estimators=estimators)
        
        model.fit(X, y)
        return model

    def _train_stacking(self, X, y, task_type, **kwargs):
        """训练Stacking集成"""
        base_learner_types = kwargs.get('base_learner_types')
        base_learner_params = kwargs.get('base_learner_params')
        meta_learner_type = kwargs.get('meta_learner_type', 'logistic_regression')
        meta_learner_params = kwargs.get('meta_learner_params', {})
        cv = kwargs.get('cv', 5)
        stack_method = kwargs.get('stack_method', 'auto')
        
        # 创建基学习器
        estimators = self._create_base_learners(task_type, base_learner_types, base_learner_params)
        
        # 创建元学习器
        if meta_learner_type == 'decision_tree':
            if task_type == 'classification':
                meta_learner = DecisionTreeClassifier(random_state=42, **meta_learner_params)
            else:
                meta_learner = DecisionTreeRegressor(random_state=42, **meta_learner_params)
        elif meta_learner_type == 'svm':
            if task_type == 'classification':
                meta_learner = SVC(probability=True, random_state=42, **meta_learner_params)
            else:
                meta_learner = SVR(**meta_learner_params)
        elif meta_learner_type == 'logistic_regression' or meta_learner_type == 'linear_regression':
            if task_type == 'classification':
                meta_learner = LogisticRegression(random_state=42, max_iter=1000, **meta_learner_params)
            else:
                meta_learner = LinearRegression(**meta_learner_params)
        else:
            # 默认元学习器
            if task_type == 'classification':
                meta_learner = LogisticRegression(random_state=42, max_iter=1000)
            else:
                meta_learner = LinearRegression()
        
        # 创建Stacking模型
        if task_type == 'classification':
            model = StackingClassifier(
                estimators=estimators,
                final_estimator=meta_learner,
                cv=cv,
                stack_method=stack_method,
                n_jobs=-1
            )
        else:
            model = StackingRegressor(
                estimators=estimators,
                final_estimator=meta_learner,
                cv=cv,
                n_jobs=-1
            )
        
        model.fit(X, y)
        return model
    
    def _train_bagging(self, X, y, task_type, **kwargs):
        """训练Bagging集成"""
        n_estimators = kwargs.get('n_estimators', 10)
        max_samples = kwargs.get('max_samples', 1.0)
        max_features = kwargs.get('max_features', 1.0)
        bootstrap = kwargs.get('bootstrap', True)
        bootstrap_features = kwargs.get('bootstrap_features', False)
        base_estimator_type = kwargs.get('base_estimator_type', 'decision_tree')
        base_estimator_params = kwargs.get('base_estimator_params', {})
        
        # 创建基学习器
        if base_estimator_type == 'decision_tree':
            if task_type == 'classification':
                base_estimator = DecisionTreeClassifier(random_state=42, **base_estimator_params)
            else:
                base_estimator = DecisionTreeRegressor(random_state=42, **base_estimator_params)
        elif base_estimator_type == 'svm':
            if task_type == 'classification':
                base_estimator = SVC(probability=True, random_state=42, **base_estimator_params)
            else:
                base_estimator = SVR(**base_estimator_params)
        elif base_estimator_type == 'knn':
            if task_type == 'classification':
                base_estimator = KNeighborsClassifier(**base_estimator_params)
            else:
                base_estimator = KNeighborsRegressor(**base_estimator_params)
        elif base_estimator_type == 'linear':
            if task_type == 'classification':
                base_estimator = LogisticRegression(random_state=42, max_iter=1000, **base_estimator_params)
            else:
                base_estimator = LinearRegression(**base_estimator_params)
        else:
            # 默认使用决策树
            if task_type == 'classification':
                base_estimator = DecisionTreeClassifier(random_state=42, **base_estimator_params)
            else:
                base_estimator = DecisionTreeRegressor(random_state=42, **base_estimator_params)
        
        # 创建Bagging模型
        if task_type == 'classification':
            model = BaggingClassifier(
                estimator=base_estimator,
                n_estimators=n_estimators,
                max_samples=max_samples,
                max_features=max_features,
                bootstrap=bootstrap,
                bootstrap_features=bootstrap_features,
                random_state=42,
                n_jobs=-1
            )
        else:
            model = BaggingRegressor(
                estimator=base_estimator,
                n_estimators=n_estimators,
                max_samples=max_samples,
                max_features=max_features,
                bootstrap=bootstrap,
                bootstrap_features=bootstrap_features,
                random_state=42,
                n_jobs=-1
            )
        
        model.fit(X, y)
        return model
    
    def _train_xgboost(self, X, y, task_type, **kwargs):
        """训练XGBoost"""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost未安装。请运行 'pip install xgboost' 安装。")
        
        n_estimators = kwargs.get('n_estimators', 100)
        max_depth = kwargs.get('max_depth', 6)
        learning_rate = kwargs.get('learning_rate', 0.3)
        subsample = kwargs.get('subsample', 1.0)
        colsample_bytree = kwargs.get('colsample_bytree', 1.0)
        reg_alpha = kwargs.get('reg_alpha', 0)
        reg_lambda = kwargs.get('reg_lambda', 1)
        
        if task_type == 'classification':
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        else:
            model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=42
            )
        
        model.fit(X, y)
        return model
   
    def _calculate_voting_feature_importance(self, model, X):
        """计算投票集成模型的特征重要性"""
        try:
            importances = []
            for name, estimator in model.named_estimators_.items():
                if hasattr(estimator, 'feature_importances_'):
                    importances.append(estimator.feature_importances_)
                elif hasattr(estimator, 'coef_'):
                    # 对于线性模型，使用系数的绝对值作为重要性
                    coef = estimator.coef_
                    if coef.ndim > 1:
                        coef = np.mean(np.abs(coef), axis=0)
                    else:
                        coef = np.abs(coef)
                    importances.append(coef)
            
            if importances:
                return np.mean(importances, axis=0)
            else:
                return None
        except Exception as e:
            logger.warning(f"无法计算Voting特征重要性: {e}")
            return None
    
    def _calculate_stacking_feature_importance(self, model, X):
        """计算Stacking集成模型的特征重要性"""
        try:
            importances = []
            
            # 从基学习器获取特征重要性
            for name, estimator in model.named_estimators_.items():
                if hasattr(estimator, 'feature_importances_'):
                    importances.append(estimator.feature_importances_)
                elif hasattr(estimator, 'coef_'):
                    # 对于线性模型，使用系数的绝对值作为重要性
                    coef = estimator.coef_
                    if coef.ndim > 1:
                        coef = np.mean(np.abs(coef), axis=0)
                    else:
                        coef = np.abs(coef)
                    importances.append(coef)
            
            if importances:
                return np.mean(importances, axis=0)
            else:
                return None
        except Exception as e:
            logger.warning(f"无法计算Stacking特征重要性: {e}")
            return None
    
    def _calculate_bagging_feature_importance(self, model, X):
        """计算Bagging集成模型的特征重要性"""
        try:
            importances = []
            
            # 从基学习器获取特征重要性
            for estimator in model.estimators_:
                if hasattr(estimator, 'feature_importances_'):
                    importances.append(estimator.feature_importances_)
                elif hasattr(estimator, 'coef_'):
                    # 对于线性模型，使用系数的绝对值作为重要性
                    coef = estimator.coef_
                    if coef.ndim > 1:
                        coef = np.mean(np.abs(coef), axis=0)
                    else:
                        coef = np.abs(coef)
                    importances.append(coef)
            
            if importances:
                return np.mean(importances, axis=0)
            else:
                return None
        except Exception as e:
            logger.warning(f"无法计算Bagging特征重要性: {e}")
            return None
    
    def get_ensemble_info(self, model, algorithm):
        """
        获取集成模型信息
        
        Args:
            model: 训练好的集成模型
            algorithm: 算法类型
            
        Returns:
            info: 模型信息字典
        """
        info = {
            'algorithm': algorithm,
            'n_estimators': getattr(model, 'n_estimators', None)
        }
        
        if algorithm == 'random_forest':
            info.update({
                'max_depth': model.max_depth,
                'min_samples_split': model.min_samples_split,
                'min_samples_leaf': model.min_samples_leaf,
                'max_features': model.max_features,
                'oob_score': getattr(model, 'oob_score_', None) if hasattr(model, 'oob_score_') else None
            })
            
        elif algorithm == 'adaboost':
            info.update({
                'learning_rate': model.learning_rate,
                'estimator_weights': getattr(model, 'estimator_weights_', None),
                'estimator_errors': getattr(model, 'estimator_errors_', None)
            })
            
        elif algorithm == 'gradient_boosting':
            info.update({
                'learning_rate': model.learning_rate,
                'max_depth': model.max_depth,
                'subsample': model.subsample,
                'train_score': getattr(model, 'train_score_', None)
            })
            
        elif algorithm == 'voting':
            estimator_names = list(model.named_estimators_.keys())
            info.update({
                'voting': getattr(model, 'voting', None),
                'estimators': estimator_names
            })
            
        elif algorithm == 'stacking':
            estimator_names = list(model.named_estimators_.keys())
            info.update({
                'estimators': estimator_names,
                'final_estimator': type(model.final_estimator_).__name__,
                'stack_method': getattr(model, 'stack_method_', None),
                'cv': model.cv if hasattr(model, 'cv') else None
            })
            
        elif algorithm == 'bagging':
            info.update({
                'base_estimator': type(model.estimator).__name__ if model.estimator else 'DecisionTree',
                'max_samples': model.max_samples,
                'max_features': model.max_features,
                'bootstrap': model.bootstrap,
                'bootstrap_features': model.bootstrap_features,
                'oob_score': getattr(model, 'oob_score_', None) if hasattr(model, 'oob_score_') else None
            })
            
        elif algorithm == 'xgboost':
            info.update({
                'learning_rate': getattr(model, 'learning_rate', None),
                'max_depth': getattr(model, 'max_depth', None),
                'subsample': getattr(model, 'subsample', None),
                'colsample_bytree': getattr(model, 'colsample_bytree', None),
                'reg_alpha': getattr(model, 'reg_alpha', None),
                'reg_lambda': getattr(model, 'reg_lambda', None),
                'best_iteration': getattr(model, 'best_iteration', None),
                'best_score': getattr(model, 'best_score', None)
            })
        
        return info
    
    def get_base_estimator_performance(self, model, X, y, algorithm):
        """
        获取基学习器的性能
        
        Args:
            model: 训练好的集成模型
            X: 特征矩阵
            y: 目标向量
            algorithm: 算法类型
            
        Returns:
            performances: 基学习器性能列表
        """
        performances = []
        
        try:
            # XGBoost特殊处理 - 没有estimators_属性，但可以获得整体性能
            if algorithm == 'xgboost':
                y_pred = model.predict(X)
                if hasattr(model, 'classes_'):  # 分类任务
                    score = accuracy_score(y, y_pred)
                    metric_name = 'accuracy'
                else:  # 回归任务
                    score = mean_squared_error(y, y_pred)
                    metric_name = 'mse'
                
                performances.append({
                    'estimator_name': 'xgboost_ensemble',
                    'score': score,
                    'metric': metric_name
                })
            elif hasattr(model, 'estimators_'):
                for i, estimator in enumerate(model.estimators_):
                    y_pred = estimator.predict(X)
                    if hasattr(model, 'classes_'):  # 分类任务
                        score = accuracy_score(y, y_pred)
                        metric_name = 'accuracy'
                    else:  # 回归任务
                        score = mean_squared_error(y, y_pred)
                        metric_name = 'mse'
                    
                    performances.append({
                        'estimator_index': i,
                        'score': score,
                        'metric': metric_name
                    })
                    
            elif hasattr(model, 'named_estimators_'):  # Voting 或 Stacking
                for name, estimator in model.named_estimators_.items():
                    y_pred = estimator.predict(X)
                    
                    if hasattr(model, 'classes_'):  # 分类任务
                        score = accuracy_score(y, y_pred)
                        metric_name = 'accuracy'
                    else:  # 回归任务
                        score = mean_squared_error(y, y_pred)
                        metric_name = 'mse'
                    
                    performances.append({
                        'estimator_name': name,
                        'score': score,
                        'metric': metric_name
                    })
                    
        except Exception as e:
            logger.warning(f"无法计算基学习器性能: {e}")
        
        return performances
    
    def predict_with_individual_estimators(self, model, X, algorithm):
        """
        获取每个基学习器的预测结果
        
        Args:
            model: 训练好的集成模型
            X: 输入特征
            algorithm: 算法类型
            
        Returns:
            predictions: 各基学习器的预测结果
        """
        individual_predictions = []
        
        try:
            # XGBoost特殊处理 - 没有estimators_属性
            if algorithm == 'xgboost':
                y_pred = model.predict(X)
                individual_predictions.append({
                    'estimator_name': 'xgboost_ensemble',
                    'predictions': y_pred.tolist()
                })
            elif hasattr(model, 'estimators_'):
                for i, estimator in enumerate(model.estimators_):
                    y_pred = estimator.predict(X)
                    individual_predictions.append({
                        'estimator_index': i,
                        'predictions': y_pred.tolist()
                    })
                    
            elif hasattr(model, 'named_estimators_'):  # Voting 或 Stacking
                for name, estimator in model.named_estimators_.items():
                    y_pred = estimator.predict(X)
                    individual_predictions.append({
                        'estimator_name': name,
                        'predictions': y_pred.tolist()
                    })
                    
        except Exception as e:
            logger.warning(f"无法获取基学习器预测: {e}")
        
        # 添加集成预测
        ensemble_pred = model.predict(X)
        individual_predictions.append({
            'estimator_name': 'ensemble',
            'predictions': ensemble_pred.tolist()
        })
        
        return individual_predictions
    
    def get_learning_curve(self, model, X, y, algorithm):
        """
        获取学习曲线数据（对于支持的算法）
        
        Args:
            model: 训练好的集成模型
            X: 特征矩阵
            y: 目标向量
            algorithm: 算法类型
            
        Returns:
            learning_curve: 学习曲线数据
        """
        if algorithm == 'gradient_boosting' and hasattr(model, 'train_score_'):
            # 梯度提升有训练分数历史
            return {
                'train_scores': model.train_score_.tolist(),
                'n_estimators': list(range(1, len(model.train_score_) + 1))
            }
        elif algorithm == 'adaboost' and hasattr(model, 'estimator_errors_'):
            # AdaBoost有估计器错误历史
            return {
                'estimator_errors': model.estimator_errors_.tolist(),
                'estimator_weights': model.estimator_weights_.tolist() if hasattr(model, 'estimator_weights_') else None,
                'n_estimators': list(range(1, len(model.estimator_errors_) + 1))
            }
        else:
            logger.info(f"算法 {algorithm} 不支持学习曲线")
            return None
    
    def get_base_estimators_decision_boundaries(self, model, X_range, y_range, algorithm, n_points=50):
        """
        获取基学习器的决策边界
        
        Args:
            model: 训练好的集成模型
            X_range: X轴范围 [min, max]
            y_range: Y轴范围 [min, max]
            algorithm: 算法类型
            n_points: 网格点数量
            
        Returns:
            boundaries: 基学习器决策边界数据
        """
        try:
            # 验证输入参数
            if not hasattr(model, 'predict'):
                logger.error("模型没有predict方法")
                return []
            
            if len(X_range) != 2 or len(y_range) != 2:
                logger.error("X_range和y_range必须是长度为2的列表")
                return []
            # 创建网格点
            x_min, x_max = X_range
            y_min, y_max = y_range
            
            # 扩展范围以包含边界区域
            x_range = x_max - x_min
            y_range = y_max - y_min
            x_min -= 0.1 * x_range
            x_max += 0.1 * x_range
            y_min -= 0.1 * y_range
            y_max += 0.1 * y_range
            
            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, n_points),
                np.linspace(y_min, y_max, n_points)
            )
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            
            boundaries = []
            
            if hasattr(model, 'estimators_'):
                # 对于有estimators_属性的模型（如RandomForest, AdaBoost, GradientBoosting, Bagging）
                max_estimators = len(model.estimators_)
                for i in range(max_estimators):
                    estimator = model.estimators_[i]
                    try:
                        # 检查基学习器的特征数量需求
                        estimator_grid_points = grid_points
                        
                        # 检查是否需要特征选择
                        if hasattr(model, 'estimators_features_') and i < len(model.estimators_features_):
                            feature_indices = model.estimators_features_[i]
                            if len(feature_indices) != grid_points.shape[1]:
                                # 需要特征选择
                                estimator_grid_points = grid_points[:, feature_indices]
                        
                        # 检查特征数量是否匹配
                        expected_features = getattr(estimator, 'n_features_in_', grid_points.shape[1])
                        if estimator_grid_points.shape[1] != expected_features:
                            logger.warning(f"基学习器{i}特征数量不匹配: 期望{expected_features}, 实际{estimator_grid_points.shape[1]}")
                            continue
                        
                        # 预测网格点
                        if hasattr(estimator, 'predict'):
                            Z = estimator.predict(estimator_grid_points)
                            Z = Z.reshape(xx.shape)
                            
                            # 计算该学习器的置信度（如果可用）
                            confidence = None
                            if hasattr(estimator, 'predict_proba'):
                                try:
                                    proba = estimator.predict_proba(estimator_grid_points)
                                    confidence = np.max(proba, axis=1).reshape(xx.shape)
                                except Exception as e:
                                    logger.warning(f"计算基学习器{i}置信度失败: {e}")
                                    pass
                            
                            boundaries.append({
                                'estimator_id': i,
                                'estimator_type': type(estimator).__name__,
                                'xx': xx.tolist(),
                                'yy': yy.tolist(),
                                'Z': Z.tolist(),
                                'confidence': confidence.tolist() if confidence is not None else None,
                                'x_range': [x_min, x_max],
                                'y_range': [y_min, y_max]
                            })
                    except Exception as e:
                        logger.warning(f"无法获取基学习器{i}的决策边界: {e}")
                        continue
                        
            elif hasattr(model, 'named_estimators_'):
                # 对于投票集成
                for name, estimator in model.named_estimators_.items():
                    try:
                        if hasattr(estimator, 'predict'):
                            Z = estimator.predict(grid_points)
                            Z = Z.reshape(xx.shape)
                            
                            # 计算置信度
                            confidence = None
                            if hasattr(estimator, 'predict_proba'):
                                try:
                                    proba = estimator.predict_proba(grid_points)
                                    confidence = np.max(proba, axis=1).reshape(xx.shape)
                                except:
                                    pass
                            
                            boundaries.append({
                                'estimator_id': name,
                                'estimator_type': type(estimator).__name__,
                                'xx': xx.tolist(),
                                'yy': yy.tolist(),
                                'Z': Z.tolist(),
                                'confidence': confidence.tolist() if confidence is not None else None,
                                'x_range': [x_min, x_max],
                                'y_range': [y_min, y_max]
                            })
                    except Exception as e:
                        logger.warning(f"无法获取基学习器{name}的决策边界: {e}")
                        continue
            
            # 添加集成模型的决策边界
            try:
                # 集成模型应该能够处理原始特征数量
                Z_ensemble = model.predict(grid_points)
                Z_ensemble = Z_ensemble.reshape(xx.shape)
                
                # 集成模型置信度
                ensemble_confidence = None
                if hasattr(model, 'predict_proba'):
                    try:
                        ensemble_proba = model.predict_proba(grid_points)
                        ensemble_confidence = np.max(ensemble_proba, axis=1).reshape(xx.shape)
                    except Exception as e:
                        logger.warning(f"计算集成模型置信度失败: {e}")
                        pass
                
                boundaries.append({
                    'estimator_id': 'ensemble',
                    'estimator_type': 'Ensemble',
                    'xx': xx.tolist(),
                    'yy': yy.tolist(),
                    'Z': Z_ensemble.tolist(),
                    'confidence': ensemble_confidence.tolist() if ensemble_confidence is not None else None,
                    'x_range': [x_min, x_max],
                    'y_range': [y_min, y_max]
                })
            except Exception as e:
                logger.warning(f"无法获取集成模型的决策边界: {e}")
            
            return boundaries
            
        except Exception as e:
            logger.error(f"获取基学习器决策边界失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            return []
    
    def get_available_base_learners(self, task_type='classification'):
        """
        获取可用的基学习器类型和默认参数
        
        Args:
            task_type: 任务类型
            
        Returns:
            dict: 基学习器配置信息
        """
        if task_type == 'classification':
            return {
                'decision_tree': {
                    'name': '决策树',
                    'class': 'DecisionTreeClassifier',
                    'params': {
                        'max_depth': {'type': 'int', 'default': None, 'min': 1, 'max': 20},
                        'min_samples_split': {'type': 'int', 'default': 2, 'min': 2, 'max': 20},
                        'min_samples_leaf': {'type': 'int', 'default': 1, 'min': 1, 'max': 10},
                        'criterion': {'type': 'select', 'default': 'gini', 'options': ['gini', 'entropy']}
                    }
                },
                'svm': {
                    'name': '支持向量机',
                    'class': 'SVC',
                    'params': {
                        'C': {'type': 'float', 'default': 1.0, 'min': 0.01, 'max': 100},
                        'kernel': {'type': 'select', 'default': 'rbf', 'options': ['linear', 'poly', 'rbf', 'sigmoid']},
                        'gamma': {'type': 'select', 'default': 'scale', 'options': ['scale', 'auto']}
                    }
                },
                'knn': {
                    'name': 'K近邻',
                    'class': 'KNeighborsClassifier',
                    'params': {
                        'n_neighbors': {'type': 'int', 'default': 5, 'min': 1, 'max': 20},
                        'weights': {'type': 'select', 'default': 'uniform', 'options': ['uniform', 'distance']},
                        'metric': {'type': 'select', 'default': 'minkowski', 'options': ['euclidean', 'manhattan', 'minkowski']}
                    }
                },
                'naive_bayes': {
                    'name': '朴素贝叶斯',
                    'class': 'GaussianNB',
                    'params': {
                        'var_smoothing': {'type': 'float', 'default': 1e-9, 'min': 1e-10, 'max': 1e-5}
                    }
                },
                'logistic_regression': {
                    'name': '逻辑回归',
                    'class': 'LogisticRegression',
                    'params': {
                        'C': {'type': 'float', 'default': 1.0, 'min': 0.01, 'max': 100},
                        'penalty': {'type': 'select', 'default': 'l2', 'options': ['l1', 'l2', 'elasticnet', 'none']},
                        'solver': {'type': 'select', 'default': 'lbfgs', 'options': ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']}
                    }
                },
                'neural_network': {
                    'name': '神经网络',
                    'class': 'MLPClassifier',
                    'params': {
                        'hidden_layer_sizes': {'type': 'tuple', 'default': (100,), 'description': '隐藏层神经元数量'},
                        'activation': {'type': 'select', 'default': 'relu', 'options': ['identity', 'logistic', 'tanh', 'relu']},
                        'alpha': {'type': 'float', 'default': 0.0001, 'min': 1e-6, 'max': 1.0}
                    }
                }
            }
        else:  # regression
            return {
                'decision_tree': {
                    'name': '决策树回归',
                    'class': 'DecisionTreeRegressor',
                    'params': {
                        'max_depth': {'type': 'int', 'default': None, 'min': 1, 'max': 20},
                        'min_samples_split': {'type': 'int', 'default': 2, 'min': 2, 'max': 20},
                        'min_samples_leaf': {'type': 'int', 'default': 1, 'min': 1, 'max': 10}
                    }
                },
                'svr': {
                    'name': '支持向量回归',
                    'class': 'SVR',
                    'params': {
                        'C': {'type': 'float', 'default': 1.0, 'min': 0.01, 'max': 100},
                        'kernel': {'type': 'select', 'default': 'rbf', 'options': ['linear', 'poly', 'rbf', 'sigmoid']},
                        'epsilon': {'type': 'float', 'default': 0.1, 'min': 0.01, 'max': 1.0}
                    }
                },
                'knn': {
                    'name': 'K近邻回归',
                    'class': 'KNeighborsRegressor',
                    'params': {
                        'n_neighbors': {'type': 'int', 'default': 5, 'min': 1, 'max': 20},
                        'weights': {'type': 'select', 'default': 'uniform', 'options': ['uniform', 'distance']}
                    }
                },
                'linear_regression': {
                    'name': '线性回归',
                    'class': 'LinearRegression',
                    'params': {
                        'fit_intercept': {'type': 'bool', 'default': True}
                    }
                },
                'neural_network': {
                    'name': '神经网络回归',
                    'class': 'MLPRegressor',
                    'params': {
                        'hidden_layer_sizes': {'type': 'tuple', 'default': (100,), 'description': '隐藏层神经元数量'},
                        'activation': {'type': 'select', 'default': 'relu', 'options': ['identity', 'logistic', 'tanh', 'relu']},
                        'alpha': {'type': 'float', 'default': 0.0001, 'min': 1e-6, 'max': 1.0}
                    }
                }
            }