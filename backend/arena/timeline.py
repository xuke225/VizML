"""Record real operation durations; prediction/encoding is outside the clock."""
import base64
import math
import warnings
from time import perf_counter

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from .engine import BOUNDS, GRID_SIZE, models_for, number, prepare, selected_model_keys
from .observation import TreePrefixClassifier, describe_stage

MAX_TIMELINE_STEPS = 1000


def record_timeline(payload):
    limit = number(payload.get("steps", 50), "连续演示步数", 1, MAX_TIMELINE_STEPS, True)
    recording_start = perf_counter()
    data, X, y, train, test = prepare(payload)
    model_keys = selected_model_keys(payload)
    if "bayes" in model_keys and np.max(np.var(X[train], axis=0)) == 0:
        raise ValueError("训练坐标完全重合，请先分散样本。")
    models = models_for(payload.get("params", {}), data["seed"], len(train), model_keys)
    xx, yy = np.meshgrid(np.linspace(BOUNDS[0], BOUNDS[1], GRID_SIZE),
                         np.linspace(BOUNDS[3], BOUNDS[2], GRID_SIZE))
    mesh = np.column_stack([xx.ravel(), yy.ravel()])
    result = dict(data, version=payload.get("version"), model_keys=list(model_keys), tracks={},
                  grid={"size": GRID_SIZE, "bounds": BOUNDS, "row_order": "top_to_bottom"},
                  timeline={"steps": limit, "max_steps": MAX_TIMELINE_STEPS,
                            "clock": "per_model_measured_operations_ms", "grid_encoding": "packbits_base64_big",
                            "excludes": ["mesh_prediction", "metrics", "explanation_encoding", "network", "rendering"],
                            "independent_tracks": True, "ranked": False})
    with threadpool_limits(limits=1):
        for key, model in models.items():
            track_start = perf_counter()
            p = model._arena_parameters
            raw = model.steps[-1][1] if hasattr(model, "steps") else model
            scaler = model.steps[0][1] if hasattr(model, "steps") else None
            setup_start = perf_counter()
            Xt = scaler.fit_transform(X[train]) if scaler is not None else X[train]
            all_Xt = scaler.transform(X) if scaler is not None else X
            nodes, neighbors, distances, order = None, None, None, None
            if key == "tree":
                raw.fit(Xt, y[train])
                nodes = np.flatnonzero(raw.tree_.feature >= 0)
                total, mode, unit, title = len(nodes), "tree_structure", "处分裂", "展开已训练树"
                note = "完整树训练耗时单独计入准备阶段；各步计时是已训练节点的结构展开，不是树训练迭代。"
            elif key == "knn":
                raw.fit(Xt, y[train])
                distances, neighbors = raw.kneighbors(all_Xt)
                total, mode, unit, title = p["n_neighbors"], "neighbor_vote", "个邻居", "逐个邻居投票"
                note = "准备耗时包含建立索引和查询全部展示样本的邻居；逐步计时为全部样本的部分投票。背景始终为完整 K 边界，不是训练迭代。"
            elif key == "bayes":
                shuffled = np.random.RandomState(data["seed"] + 4081).permutation(train)
                first = np.concatenate([shuffled[y[shuffled] == c][:2] for c in (0, 1)])
                initial = set(first.tolist())
                order = np.concatenate([first, [i for i in shuffled if i not in initial]])
                total, mode, unit, title = math.ceil(len(train)/8), "cumulative_statistics", "批数据", "累积估计分布"
                note = "每步累积 8 个训练样本重新估计分布，不是梯度迭代；淡点未参与估计，椭圆为均值 ±1 标准差。标准化（若启用）固定由整个训练池拟合。"
            elif key == "forest":
                total, mode, unit, title = p["n_estimators"], "forest_prefix", "棵树", "逐棵增长随机森林"
                note = "warm_start 保留已有树，每步实际增加一棵树；各树的类别概率平均。计时含 sklearn 管理既有树的开销。"
                raw.set_params(warm_start=True)
            elif key == "mlp" and (p["network_type"] == "perceptron" or p["solver"] != "lbfgs"):
                total, mode, unit, title = p["max_iter"], "incremental_epochs", "轮", "逐轮更新权重"
                note = "每步一次真实 partial_fit，完整遍历训练集并保留优化器状态；固定轮数增量模式，不等于完整 fit 的停止策略。"
            else:
                total, mode, unit = p["max_iter"], "independent_budget", "次预算"
                title = "SMO 独立预算求解" if key == "svm" else "L-BFGS 独立预算求解"
                note = "第 N 步是 max_iter=N 的独立重新拟合：耗时是该次完整求解耗时，不是第 N 次内部迭代耗时，也不是续训。"
            setup_ms = (perf_counter() - setup_start) * 1000
            track = dict(setup_ms=setup_ms, mode=mode, timing_note=note, frames=[])
            elapsed = setup_ms
            # A tree that cannot split has one truthful root-only checkpoint.
            for step in range(1, min(limit, max(1, total)) + 1):
                count = min(step, total)
                stage = dict(mode=mode, current=count, total=total, unit=unit, title=title,
                             complete=count >= total, note=note)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    started = perf_counter()
                    if mode == "incremental_epochs":
                        raw.partial_fit(Xt, y[train], classes=np.array([0, 1]))
                    elif mode == "independent_budget":
                        raw.set_params(max_iter=count)
                        raw.fit(Xt, y[train])
                    elif key == "forest":
                        raw.set_params(n_estimators=count)
                        raw.fit(Xt, y[train])
                    elif key == "tree":
                        prefix = TreePrefixClassifier(raw, nodes[:count])
                        if scaler is not None:
                            model.steps[-1] = (model.steps[-1][0], prefix)
                        else:
                            prefix._arena_parameters = p
                            model = prefix
                    elif key == "bayes":
                        seen = order[:min(count*8, len(train))]
                        seen_X = scaler.transform(X[seen]) if scaler is not None else X[seen]
                        raw.fit(seen_X, y[seen])
                    else:
                        # This is the actual partial-vote operation, not a timer
                        # around delay or drawing. Distances were queried once.
                        d = distances[:, :count]
                        weights = np.ones(d.shape)
                        if p["weights"] == "distance":
                            zero = d == 0
                            with np.errstate(divide="ignore"):
                                weights = 1/d
                            rows = zero.any(axis=1)
                            weights[rows] = zero[rows]
                        labels = y[train][neighbors[:, :count]]
                        vote_b = ((labels == 1) * weights).sum(axis=1) / weights.sum(axis=1)
                    duration = (perf_counter() - started) * 1000
                elapsed += duration
                if mode == "independent_budget":
                    stage["actual_iterations"] = int(np.asarray(raw.n_iter_).sum())
                if mode == "incremental_epochs":
                    stage["weight_norm"] = float(sum(np.linalg.norm(w) for w in raw.coefs_)) if hasattr(raw, "coefs_") else float(np.linalg.norm(raw.coef_))
                if key == "bayes":
                    stage.update(seen_ids=[data["points"][i]["id"] for i in seen], seen_count=len(seen))
                # Everything below is intentionally EXCLUDED from the replay clock.
                stage["timing"] = dict(step_ms=duration, cumulative_ms=elapsed, setup_ms=setup_ms)
                encoded = describe_stage(key, model, stage, data, X, y, train, test, mesh, elapsed,
                    any(issubclass(w.category, ConvergenceWarning) for w in caught))
                if key == "knn":
                    # Use probabilities from the actual timed voting operation.
                    for point, probability in zip(data["points"], vote_b):
                        encoded["explanations"][point["id"]].update(
                            partial_probabilities=[float(1-probability), float(probability)],
                            partial_prediction=int(probability > 0.5))
                grid = encoded.pop("grid")
                encoded["grid_bits"] = base64.b64encode(np.packbits(np.asarray(grid, dtype=np.uint8)).tobytes()).decode("ascii")
                encoded.pop("train_ids")
                encoded.pop("test_ids")
                track["frames"].append(dict(step=count, duration_ms=duration, elapsed_ms=elapsed, model=encoded))
            track["duration_ms"] = elapsed
            track["complete"] = track["frames"][-1]["model"]["observation"]["complete"]
            # Wall-clock recording includes predictions/encoding, unlike replay.
            track["recording_ms"] = (perf_counter() - track_start) * 1000
            result["tracks"][key] = track
    result["timeline"]["duration_ms"] = max(t["duration_ms"] for t in result["tracks"].values())
    result["timeline"]["recording_ms"] = (perf_counter() - recording_start) * 1000
    return result
