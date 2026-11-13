"""Deterministic, stateless checkpoints; never interpolate a training boundary."""
import math
import warnings
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

from .engine import (BOUNDS, GRID_SIZE, describe_model, estimator_input,
                     models_for, number, prepare, selected_model_keys)


class TreePrefixClassifier:
    """Stop at unexpanded nodes of an already fitted tree (not tree training)."""

    def __init__(self, estimator, expanded_nodes):
        self.estimator = estimator
        self.expanded_nodes = set(expanded_nodes)
        self.tree_ = estimator.tree_
        self.classes_ = estimator.classes_

    def get_params(self, deep=True):
        return self.estimator.get_params(deep=deep)

    def paths(self, X):
        # sklearn's tree predictor casts input to float32 before comparing.
        for sample in np.asarray(X, dtype=np.float32):
            node, path = 0, [0]
            while node in self.expanded_nodes:
                feature = self.tree_.feature[node]
                node = (self.tree_.children_left[node] if sample[feature] <= self.tree_.threshold[node]
                        else self.tree_.children_right[node])
                path.append(int(node))
            yield path

    def apply(self, X):
        return np.array([p[-1] for p in self.paths(X)])

    def decision_path(self, X):
        paths = list(self.paths(X))
        indices = [node for path in paths for node in path]
        indptr = np.cumsum([0] + [len(p) for p in paths])
        return csr_matrix((np.ones(len(indices)), indices, indptr), shape=(len(X), self.tree_.node_count))

    def predict_proba(self, X):
        counts = self.tree_.value[self.apply(X), 0, :]
        return counts / counts.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def split_segments(raw, scaler):
    """True thresholds, clipped to each node's ancestor region in world space."""
    result = []

    def visit(node, bounds):
        if node not in raw.expanded_nodes:
            return
        axis = int(raw.tree_.feature[node])
        threshold = raw.tree_.threshold[node]
        if scaler is not None:
            threshold = threshold * scaler.scale_[axis] + scaler.mean_[axis]
        x0, x1, y0, y1 = bounds
        a, b = ([threshold, y0], [threshold, y1]) if axis == 0 else ([x0, threshold], [x1, threshold])
        result.append({"node": int(node), "axis": axis, "threshold": float(threshold),
                       "a": a, "b": b})
        left, right = list(bounds), list(bounds)
        left[axis * 2 + 1] = min(left[axis * 2 + 1], threshold)
        right[axis * 2] = max(right[axis * 2], threshold)
        visit(int(raw.tree_.children_left[node]), left)
        visit(int(raw.tree_.children_right[node]), right)

    visit(0, BOUNDS)
    return result


def describe_stage(key, model, stage, data, X, y, train, test, mesh, elapsed, convergence_warning=False):
    """Shared factual annotations for manual checkpoints and timed trajectories."""
    encoded = describe_model(key, model, data, X, y, train, test, mesh, elapsed, convergence_warning)
    raw, _, scaler = estimator_input(model, X)
    count, configured = stage["current"], model._arena_parameters
    if stage["mode"] == "incremental_epochs":
        encoded["fit_info"]["iterations"] = count
        encoded["fit_info"]["iteration_limit_reached"] = False
        stage["loss_curve"] = list(encoded["fit_info"]["loss_curve"])
    if key == "svm":
        stage["support_ids"] = [data["points"][train[i]]["id"] for i in raw.support_]
        stage["support_count"] = len(raw.support_)
    elif key == "tree":
        stage["segments"] = split_segments(raw, scaler)
        frontier = np.unique(raw.apply(X[train]))
        stage["frontier_impurity"] = float(sum(raw.tree_.impurity[n] * raw.tree_.n_node_samples[n]
                                               for n in frontier) / len(train))
    elif key == "knn":
        labels = {p["id"]: p["label"] for p in data["points"]}
        for explanation in encoded["explanations"].values():
            neighbors = explanation["neighbors"][:count]
            distances = np.array([n["distance"] for n in neighbors])
            weights = np.ones(count)
            if configured["weights"] == "distance":
                weights = ((distances == 0).astype(float) if (distances == 0).any() else 1 / distances)
            votes = np.bincount([labels[n["id"]] for n in neighbors], weights=weights, minlength=2)
            explanation.update(revealed_neighbors=neighbors, partial_probabilities=(votes / votes.sum()).tolist(),
                               partial_prediction=int(votes.argmax()), vote_weights=weights.tolist())
    elif key == "bayes":
        means, deviations = raw.theta_, np.sqrt(raw.var_)
        if scaler is not None:
            means = means * scaler.scale_ + scaler.mean_
            deviations = deviations * scaler.scale_
        stage["gaussians"] = [{"label": int(c), "mean": m.tolist(), "std": s.tolist()}
                              for c, m, s in zip(raw.classes_, means, deviations)]
    if hasattr(model, "predict_proba"):
        stage["training_pool_log_loss"] = float(log_loss(y[train], model.predict_proba(X[train]), labels=[0, 1]))
    encoded["observation"] = stage
    return encoded


def fit_observation(payload, keep_models=False):
    step = number(payload.get("step", 1), "观察步数", 1, 10000, True)
    data, X, y, train, test = prepare(payload)
    model_keys = selected_model_keys(payload)
    if "bayes" in model_keys and np.max(np.var(X[train], axis=0)) == 0:
        raise ValueError("训练坐标完全重合，请先分散样本。")
    models = models_for(payload.get("params", {}), data["seed"], len(train), model_keys)
    xx, yy = np.meshgrid(np.linspace(BOUNDS[0], BOUNDS[1], GRID_SIZE),
                         np.linspace(BOUNDS[3], BOUNDS[2], GRID_SIZE))
    mesh = np.column_stack([xx.ravel(), yy.ravel()])
    result = dict(data, version=payload.get("version"), models={}, model_keys=list(model_keys),
                  grid={"size": GRID_SIZE, "bounds": BOUNDS, "row_order": "top_to_bottom"},
                  ranks={}, observation={"step": step, "replay": "deterministic_from_start",
                                        "ranked": False, "metrics_scope": "training_pool_and_fixed_holdout"})
    with threadpool_limits(limits=1):
        for key, model in models.items():
            configured = model._arena_parameters
            raw = model.steps[-1][1] if hasattr(model, "steps") else model
            scaler = model.steps[0][1] if hasattr(model, "steps") else None
            start = perf_counter()
            stage = {}
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                if key == "mlp" and (configured["network_type"] == "perceptron" or configured["solver"] != "lbfgs"):
                    # Fit preprocessing once on the full training pool, never holdout.
                    scaler = model.steps[0][1]
                    Xt = scaler.fit_transform(X[train])
                    count = min(step, configured["max_iter"])
                    for _ in range(count):
                        raw.partial_fit(Xt, y[train], classes=np.array([0, 1]))
                    stage = dict(mode="incremental_epochs", current=count, total=configured["max_iter"],
                                 unit="轮", title="逐轮更新权重", complete=count >= configured["max_iter"],
                                 note="每轮完整遍历训练集，保留优化器状态；各快照从同一种子重放。固定轮数的 partial_fit 模式，不等同于完整对决的 fit 停止策略。",
                                 weight_norm=float(sum(np.linalg.norm(w) for w in raw.coefs_)) if hasattr(raw, "coefs_") else float(np.linalg.norm(raw.coef_)))
                elif key in ("svm", "mlp"):
                    count = min(step, configured["max_iter"])
                    raw.set_params(max_iter=count)
                    model.fit(X[train], y[train])
                    actual = int(np.asarray(raw.n_iter_).sum())
                    stage = dict(mode="independent_budget", current=count, total=configured["max_iter"],
                                 unit="次预算", actual_iterations=actual,
                                 title="SMO 优化预算" if key == "svm" else "L-BFGS 优化预算",
                                 complete=count >= configured["max_iter"],
                                 note="同一种子、独立重新拟合到指定迭代预算；求解器未开放断点续训接口，不是接着上一次训练。实际迭代次数可能小于预算。")
                elif key == "forest":
                    count = min(step, configured["n_estimators"])
                    raw.set_params(n_estimators=count)
                    model.fit(X[train], y[train])
                    stage = dict(mode="forest_prefix", current=count, total=configured["n_estimators"], unit="棵树",
                                 title="累积独立的决策树", complete=count >= configured["n_estimators"],
                                 note="固定种子下前 N 棵真实树的概率平均；增加的是集成成员，不是梯度迭代。")
                elif key == "tree":
                    model.fit(X[train], y[train])
                    nodes = np.flatnonzero(raw.tree_.feature >= 0)
                    count = min(step, len(nodes))
                    prefix = TreePrefixClassifier(raw, nodes[:count])
                    if hasattr(model, "steps"):
                        model.steps[-1] = (model.steps[-1][0], prefix)
                    else:
                        prefix._arena_parameters = configured
                        model = models[key] = prefix
                    raw = prefix
                    stage = dict(mode="tree_structure", current=count, total=len(nodes), unit="处分裂",
                                 title="展开已训练树", complete=count >= len(nodes),
                                 note="这是已训练树的前序结构展开，不是训练迭代。尚未展开的节点按该节点真实类别分布预测。")
                elif key == "bayes":
                    # Prefer two per class (prepare guarantees both, not two each).
                    shuffled = np.random.RandomState(data["seed"] + 4081).permutation(train)
                    first = np.concatenate([shuffled[y[shuffled] == c][:2] for c in (0, 1)])
                    initial = set(first.tolist())
                    order = np.concatenate([first, [i for i in shuffled if i not in initial]])
                    total = math.ceil(len(train) / 8)
                    count = min(step, total)
                    seen = order[:min(count * 8, len(train))]
                    if hasattr(model, "steps"):
                        scaler = model.steps[0][1]
                        scaler.fit(X[train])
                        raw.fit(scaler.transform(X[seen]), y[seen])
                    else:
                        raw.fit(X[seen], y[seen])
                    stage = dict(mode="cumulative_statistics", current=count, total=total, unit="批数据",
                                 title="累积样本估计分布", complete=count >= total,
                                 seen_ids=[data["points"][i]["id"] for i in seen], seen_count=len(seen),
                                 note="每批 8 个训练样本，累积重新估计均值、方差及先验；不是梯度优化。淡点尚未参与估计，椭圆表示各类均值 ±1 标准差（非置信区间）。" +
                                      ("标准化尺度固定由整个训练池拟合。" if scaler is not None else ""))
                else:
                    model.fit(X[train], y[train])
                    count = min(step, configured["n_neighbors"])
                    stage = dict(mode="neighbor_vote", current=count, total=configured["n_neighbors"], unit="个邻居",
                                 title="逐个邻居投票", complete=count >= configured["n_neighbors"],
                                 note="KNN 没有训练迭代：背景始终为完整 K 的边界。点击样本，观察前 N 个真实邻居的距离与部分投票；不会把部分投票伪装成新模型。")
            elapsed = (perf_counter() - start) * 1000
            result["models"][key] = describe_stage(key, model, stage, data, X, y, train, test, mesh, elapsed,
                any(issubclass(w.category, ConvergenceWarning) for w in caught))
    return (result, models, X, mesh) if keep_models else result
