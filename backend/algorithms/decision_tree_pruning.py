"""决策树剪枝实验引擎。

该模块把预剪枝、代价复杂度后剪枝和对照实验封装在独立文件中，
避免让原有决策树训练流程承担过多教学分析逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class PruningConfig:
    """经过校验的剪枝参数。"""

    method: str = "pre"
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_leaf_nodes: int | None = None
    min_impurity_decrease: float = 0.0
    ccp_alpha: float = 0.0
    cv_folds: int = 5


class DecisionTreePruningEngine:
    """训练并比较未剪枝、预剪枝和后剪枝模型。"""

    METHODS = {"none", "pre", "post_auto", "post_manual"}
    METHOD_LABELS = {
        "none": "未剪枝基线",
        "pre": "预剪枝",
        "post_auto": "自动后剪枝",
        "post_manual": "手动后剪枝",
    }
    MAX_PATH_POINTS = 24

    def __init__(self, criterion="gini", max_features=None, random_state=42):
        self.common_params = {
            "criterion": criterion,
            "max_features": max_features,
            "random_state": random_state,
        }

    def fit(self, X_train, y_train, X_test, y_test, config: PruningConfig):
        """返回选中的模型及可直接 JSON 序列化的教学分析数据。"""
        self._validate(config)

        unpruned_model = self._new_model().fit(X_train, y_train)
        pre_model = self._new_model(**self._pre_params(config)).fit(X_train, y_train)

        pruning_path, auto_alpha, selection_rule, effective_folds = self._build_pruning_path(
            X_train, y_train, X_test, y_test, config.cv_folds,
            manual_alpha=config.ccp_alpha if config.method == "post_manual" else None,
        )
        selected_alpha = config.ccp_alpha if config.method == "post_manual" else auto_alpha
        if config.method == "post_manual" and pruning_path:
            for point in pruning_path:
                point.pop("selected", None)
            selected_point = next(
                point for point in pruning_path if point["alpha"] == selected_alpha
            )
            selected_point["selected"] = True
            selected_point["selection_label"] = "手动设置"
        post_model = self._new_model(ccp_alpha=selected_alpha).fit(X_train, y_train)

        models = {
            "none": unpruned_model,
            "pre": pre_model,
            "post_auto": post_model,
            "post_manual": post_model,
        }
        selected_model = models[config.method]

        comparison = [
            self._summarize("none", unpruned_model, X_train, y_train, X_test, y_test),
            self._summarize("pre", pre_model, X_train, y_train, X_test, y_test),
            self._summarize(
                "post_manual" if config.method == "post_manual" else "post_auto",
                post_model,
                X_train,
                y_train,
                X_test,
                y_test,
                ccp_alpha=selected_alpha,
            ),
        ]

        for item in comparison:
            item["selected"] = item["method"] == config.method or (
                config.method == "post_manual" and item["method"] == "post_manual"
            )

        baseline = comparison[0]
        selected_summary = self._summarize(
            config.method,
            selected_model,
            X_train,
            y_train,
            X_test,
            y_test,
            ccp_alpha=selected_alpha if config.method.startswith("post") else 0.0,
        )
        reduction = {
            "nodes_removed": int(baseline["node_count"] - selected_summary["node_count"]),
            "node_reduction_ratio": self._ratio_reduction(
                baseline["node_count"], selected_summary["node_count"]
            ),
            "leaves_removed": int(baseline["n_leaves"] - selected_summary["n_leaves"]),
            "depth_reduction": int(baseline["depth"] - selected_summary["depth"]),
            "test_accuracy_change": float(
                selected_summary["test_accuracy"] - baseline["test_accuracy"]
            ),
            "generalization_gap_change": float(
                selected_summary["generalization_gap"] - baseline["generalization_gap"]
            ),
        }

        return selected_model, {
            "method": config.method,
            "method_label": self.METHOD_LABELS[config.method],
            "selected_alpha": float(selected_alpha) if config.method.startswith("post") else 0.0,
            "recommended_alpha": float(auto_alpha),
            "selection_rule": (
                "使用手动设置的 ccp_alpha"
                if config.method == "post_manual"
                else selection_rule if config.method == "post_auto"
                else "按预剪枝参数在生长过程中限制分裂；后剪枝仅作为对照模型。"
                if config.method == "pre" else "当前显示未剪枝基线；其他策略仅作为对照模型。"
            ),
            "cv_folds": int(effective_folds),
            "comparison": comparison,
            "selected_model": selected_summary,
            "reduction": reduction,
            "path": pruning_path,
            "parameter_notes": self._parameter_notes(config, selected_alpha),
            "tree_comparison": self._tree_comparison(unpruned_model, selected_model, config.method),
        }

    @staticmethod
    def _tree_comparison(baseline, selected, method):
        """Compare shared decision regions, never sklearn's renumbered node IDs.

        Pre-pruning may change a split rather than just truncate a subtree.
        Such regions are explicitly unmatched, not falsely labelled as removed.
        The flat baseline graph is also safe to serialize for very deep trees.
        """
        original, final = baseline.tree_, selected.tree_
        nodes = []
        stack = [(0, 0, None, None, 0, None)]
        while stack:
            old_id, new_id, parent, side, depth, inherited = stack.pop()
            old_leaf = original.children_left[old_id] == -1
            new_leaf = new_id is not None and final.children_left[new_id] == -1
            status = inherited
            if status is None:
                if new_leaf and not old_leaf:
                    status = "cut"
                elif old_leaf and new_leaf:
                    status = "retained"
                elif (not old_leaf and not new_leaf
                      and original.feature[old_id] == final.feature[new_id]
                      and np.isclose(original.threshold[old_id], final.threshold[new_id],
                                     rtol=0, atol=1e-12)):
                    status = "retained"
                else:
                    status = "changed"

            node = {
                "id": int(old_id), "parent": parent, "side": side, "depth": depth,
                "status": status, "is_leaf": bool(old_leaf),
                "feature": None if old_leaf else int(original.feature[old_id]),
                "threshold": None if old_leaf else float(original.threshold[old_id]),
                "samples": int(original.n_node_samples[old_id]),
                "impurity": float(original.impurity[old_id]),
                "prediction": baseline.classes_.tolist()[int(np.argmax(original.value[old_id][0]))],
                "selected_id": int(new_id) if status in {"retained", "cut"} else None,
            }
            if node["selected_id"] is not None:
                node["selected_prediction"] = selected.classes_.tolist()[
                    int(np.argmax(final.value[new_id][0]))]
            nodes.append(node)
            if not old_leaf:
                child_status = ("removed" if status in {"cut", "removed"} else
                                "unmatched" if status in {"changed", "unmatched"} else None)
                for direction, old_child, new_children in (
                    ("right", original.children_right[old_id], final.children_right),
                    ("left", original.children_left[old_id], final.children_left),
                ):
                    next_id = int(new_children[new_id]) if child_status is None else None
                    stack.append((int(old_child), next_id, int(old_id), direction,
                                  depth + 1, child_status))

        counts = {status: sum(n["status"] == status for n in nodes)
                  for status in ("retained", "cut", "removed", "changed", "unmatched")}
        # Always export the COMPLETE selected tree independently of matching.
        # A different pre-pruning split does not mean its selected descendants
        # disappear: they still participate in prediction and must be drawn.
        mapped = {n["selected_id"]: n for n in nodes if n["selected_id"] is not None}
        selected_nodes = []
        stack = [(0, None, None, 0)]
        while stack:
            node_id, parent, side, depth = stack.pop()
            is_leaf = final.children_left[node_id] == -1
            original_node = mapped.get(node_id)
            selected_nodes.append({
                "id": int(node_id), "parent": parent, "side": side, "depth": depth,
                "selected_id": int(node_id),
                "baseline_id": original_node["id"] if original_node else None,
                "status": "cut" if original_node and original_node["status"] == "cut" else "retained",
                "is_leaf": bool(is_leaf),
                "feature": None if is_leaf else int(final.feature[node_id]),
                "threshold": None if is_leaf else float(final.threshold[node_id]),
                "samples": int(final.n_node_samples[node_id]),
                "impurity": float(final.impurity[node_id]),
                "prediction": selected.classes_.tolist()[int(np.argmax(final.value[node_id][0]))],
            })
            if not is_leaf:
                stack.append((int(final.children_right[node_id]), int(node_id), "right", depth + 1))
                stack.append((int(final.children_left[node_id]), int(node_id), "left", depth + 1))
        return {
            "nodes": nodes, "counts": counts,
            "selected_nodes": selected_nodes,
            "baseline_node_count": int(original.node_count),
            "selected_node_count": int(final.node_count),
            "exact_subtree": not (counts["changed"] or counts["unmatched"]),
            "removed_label": "未继续生长" if method == "pre" else "已剪除",
            "cut_label": "停止分裂" if method == "pre" else "剪枝位置",
        }

    def _new_model(self, **overrides):
        params = {**self.common_params, **overrides}
        return DecisionTreeClassifier(**params)

    @staticmethod
    def _pre_params(config):
        return {
            "max_depth": config.max_depth,
            "min_samples_split": config.min_samples_split,
            "min_samples_leaf": config.min_samples_leaf,
            "max_leaf_nodes": config.max_leaf_nodes,
            "min_impurity_decrease": config.min_impurity_decrease,
            "ccp_alpha": 0.0,
        }

    def _build_pruning_path(self, X_train, y_train, X_test, y_test, requested_folds,
                            manual_alpha=None):
        path_model = self._new_model()
        path = path_model.cost_complexity_pruning_path(X_train, y_train)
        alphas = np.unique(np.maximum(path.ccp_alphas, 0.0))
        if len(alphas) > self.MAX_PATH_POINTS:
            indices = np.unique(
                np.linspace(0, len(alphas) - 1, self.MAX_PATH_POINTS, dtype=int)
            )
            alphas = alphas[indices]
        if manual_alpha is not None:
            alphas = np.unique(np.append(alphas, manual_alpha))

        class_counts = np.unique(y_train, return_counts=True)[1]
        effective_folds = min(int(requested_folds), int(class_counts.min()))
        # 某一类不足两个样本时无法进行可靠的分层验证，不强行增加折数。
        effective_folds = effective_folds if effective_folds >= 2 else 0
        splitter = (
            StratifiedKFold(n_splits=effective_folds, shuffle=True,
                            random_state=self.common_params["random_state"])
            if effective_folds else None
        )

        points = []
        for alpha in alphas:
            candidate = self._new_model(ccp_alpha=float(alpha))
            scores = (
                cross_val_score(candidate, X_train, y_train, cv=splitter,
                                scoring="accuracy", error_score="raise")
                if splitter is not None else None
            )
            candidate.fit(X_train, y_train)
            leaf_mask = candidate.tree_.children_left == -1
            impurity = np.sum(
                candidate.tree_.impurity[leaf_mask]
                * candidate.tree_.weighted_n_node_samples[leaf_mask]
            ) / candidate.tree_.weighted_n_node_samples[0]
            points.append(
                {
                    "alpha": float(alpha),
                    "impurity": float(impurity),
                    "cv_mean": float(scores.mean()) if scores is not None else None,
                    "cv_std": float(scores.std(ddof=1)) if scores is not None else None,
                    "cv_se": float(scores.std(ddof=1) / np.sqrt(len(scores)))
                    if scores is not None else None,
                    "test_accuracy": float(candidate.score(X_test, y_test)),
                    "depth": int(candidate.get_depth()),
                    "n_leaves": int(candidate.get_n_leaves()),
                    "node_count": int(candidate.tree_.node_count),
                }
            )

        if splitter is None:
            points[0]["selected"] = True
            points[0]["selection_label"] = "样本不足，保留基线"
            return points, 0.0, (
                "某类别训练样本不足 2 个，无法分层交叉验证；自动后剪枝保留 alpha=0。"
            ), 0

        best_index = int(np.argmax([point["cv_mean"] for point in points]))
        best = points[best_index]
        threshold = best["cv_mean"] - best["cv_se"]
        eligible = [point for point in points if point["cv_mean"] >= threshold]
        # one-standard-error 规则：在统计上表现接近最优的模型中选最简单的一棵。
        selected = max(eligible, key=lambda point: (point["alpha"], -point["node_count"]))
        selected["selected"] = True
        selected["selection_label"] = "自动选择"

        rule = (
            "交叉验证 one-standard-error 规则：在接近最高准确率的候选中，"
            "选择结构最简单的树"
        )
        return points, selected["alpha"], rule, effective_folds

    def _summarize(
        self, method, model, X_train, y_train, X_test, y_test, ccp_alpha=0.0
    ):
        train_accuracy = float(model.score(X_train, y_train))
        test_accuracy = float(model.score(X_test, y_test))
        return {
            "method": method,
            "label": self.METHOD_LABELS[method],
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "generalization_gap": float(abs(train_accuracy - test_accuracy)),
            "depth": int(model.get_depth()),
            "n_leaves": int(model.get_n_leaves()),
            "node_count": int(model.tree_.node_count),
            "ccp_alpha": float(ccp_alpha),
        }

    @staticmethod
    def _ratio_reduction(original, selected):
        if original <= 0:
            return 0.0
        return float((original - selected) / original)

    @staticmethod
    def _parameter_notes(config, selected_alpha):
        if config.method == "none":
            return ["不限制树的生长，用作观察过拟合和复杂度的基线。"]
        if config.method == "pre":
            return [
                f"最大深度：{config.max_depth if config.max_depth is not None else '不限'}",
                f"内部节点最少样本：{config.min_samples_split}",
                f"叶节点最少样本：{config.min_samples_leaf}",
                f"最大叶节点：{config.max_leaf_nodes if config.max_leaf_nodes is not None else '不限'}",
                f"最小不纯度下降：{config.min_impurity_decrease:g}",
            ]
        return [
            "先让树充分生长，再用最小代价复杂度剪枝删除收益较低的分支。",
            f"最终 ccp_alpha：{selected_alpha:.6g}",
        ]

    def _validate(self, config):
        if config.method not in self.METHODS:
            raise ValueError("pruning_method 必须是 none、pre、post_auto 或 post_manual")
        integer_fields = {
            "max_depth": config.max_depth,
            "min_samples_split": config.min_samples_split,
            "min_samples_leaf": config.min_samples_leaf,
            "max_leaf_nodes": config.max_leaf_nodes,
            "cv_folds": config.cv_folds,
        }
        for name, value in integer_fields.items():
            if value is None and name in {"max_depth", "max_leaf_nodes"}:
                continue
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} 必须是整数")
        for name in ("min_impurity_decrease", "ccp_alpha"):
            value = getattr(config, name)
            if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
                raise ValueError(f"{name} 必须是有限非负数")
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} 必须是有限非负数")
        if config.max_depth is not None and config.max_depth < 1:
            raise ValueError("max_depth 必须大于等于 1")
        if config.min_samples_split < 2:
            raise ValueError("min_samples_split 必须大于等于 2")
        if config.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf 必须大于等于 1")
        if config.max_leaf_nodes is not None and config.max_leaf_nodes < 2:
            raise ValueError("max_leaf_nodes 必须大于等于 2")
        if config.min_impurity_decrease < 0:
            raise ValueError("min_impurity_decrease 不能为负数")
        if config.ccp_alpha < 0:
            raise ValueError("ccp_alpha 不能为负数")
        if not 2 <= config.cv_folds <= 10:
            raise ValueError("cv_folds 必须在 2 到 10 之间")
