"""Real predictions on a common mesh, with a single leakage-free holdout."""
import hashlib
import json
import math
import warnings
from time import perf_counter

import numpy as np
from sklearn.datasets import make_circles, make_moons
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from .parameters import normalize_parameters, create_classifier, active, PARAMETERS

BOUNDS = [-3.0, 3.0, -3.0, 3.0]
GRID_SIZE = 120
MODEL_KEYS = ("svm", "knn", "tree", "forest", "mlp", "bayes")


def number(value, name, low, high, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须为数字。")
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} 必须在 {low}～{high} 之间。")
    if integer and int(value) != value:
        raise ValueError(f"{name} 必须为整数。")
    return int(value) if integer else float(value)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]


def generate(payload):
    n = number(payload.get("count", 180), "样本数量", 24, 600, True)
    noise = number(payload.get("noise", 0.13), "生成噪声", 0, 0.5)
    seed = number(payload.get("seed", 42), "随机种子", 0, 1000000, True)
    kind = payload.get("kind", "moons")
    if kind == "moons":
        X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
        X = (X - [0.5, 0.25]) * 1.45
    elif kind == "circles":
        X, y = make_circles(n_samples=n, noise=noise, factor=0.45, random_state=seed)
        X *= 1.9
    elif kind == "xor":
        rng = np.random.RandomState(seed)
        X = rng.uniform(-2, 2, (n, 2))
        y = (X[:, 0] * X[:, 1] > 0).astype(int)
        X += rng.normal(0, noise, X.shape)
    else:
        raise ValueError("请选择月牙、圆环或 XOR 数据。自由绘制请直接添加样本。")
    X = np.clip(X, -2.95, 2.95)
    return {"points": [{"id": str(i), "x": float(p[0]), "y": float(p[1]), "label": int(y[i])}
                       for i, p in enumerate(X)], "bounds": BOUNDS, "seed": seed}


def prepare(payload):
    points = payload.get("points")
    if not isinstance(points, list) or not 24 <= len(points) <= 600:
        raise ValueError("需要 24～600 个样本才能进行独立留出评估；每类至少 6 个。")
    clean, seen = [], set()
    for p in points:
        if not isinstance(p, dict):
            raise ValueError("样本格式错误。")
        pid = p.get("id")
        if not isinstance(pid, str) or not 1 <= len(pid) <= 40 or pid in seen:
            raise ValueError("每个样本需要唯一的字符串 id（1～40 字符）。")
        seen.add(pid)
        label = number(p.get("label"), "样本类别", 0, 1, True)
        clean.append({"id": pid, "x": number(p.get("x"), "x 坐标", -3, 3),
                      "y": number(p.get("y"), "y 坐标", -3, 3), "label": label})
    # Canonical order makes client ordering irrelevant to the shared split.
    clean.sort(key=lambda p: p["id"])
    X = np.array([[p["x"], p["y"]] for p in clean])
    y = np.array([p["label"] for p in clean])
    if min(np.bincount(y, minlength=2)) < 6:
        raise ValueError("A、B 两类各需至少 6 个样本，避免输出误导性排名。")
    seed = number(payload.get("seed", 42), "随机种子", 0, 1000000, True)
    ratio = number(payload.get("test_size", 0.25), "测试集比例", 0.1, 0.5)
    train, test = train_test_split(np.arange(len(y)), test_size=ratio, random_state=seed, stratify=y)
    if min(np.bincount(y[test], minlength=2)) < 2:
        raise ValueError("当前划分的某类测试样本不足 2 个，请增加该类样本或提高测试集比例。")
    challenge = payload.get("challenge", {})
    if not isinstance(challenge, dict):
        raise ValueError("挑战参数格式错误。")
    noise = number(challenge.get("noise", 0), "训练坐标噪声", 0, 0.5)
    corruption = number(challenge.get("corruption", 0), "训练标签污染比例", 0, 0.4)
    effective_X, effective_y = X.copy(), y.copy()
    rng = np.random.RandomState(seed + 1701)
    effective_X[train] += rng.normal(size=(len(train), 2)) * noise
    clipped = int(np.any(np.abs(effective_X[train]) > 2.95, axis=1).sum()) if noise else 0
    if noise:
        effective_X[train] = np.clip(effective_X[train], -2.95, 2.95)
    selected = np.random.RandomState(seed + 2903).permutation(train)[:int(len(train) * corruption)]
    effective_y[selected] = 1 - effective_y[selected]
    if len(np.unique(effective_y[train])) < 2:
        raise ValueError("污染后训练集只剩一个类别，请降低污染比例或增加少数类样本。")
    train_set, selected_set = set(train.tolist()), set(selected.tolist())
    samples = [dict(p, x=float(effective_X[i, 0]), y=float(effective_X[i, 1]),
                    clean_label=p["label"], label=int(effective_y[i]),
                    role="train" if i in train_set else "test", corrupted=i in selected_set)
               for i, p in enumerate(clean)]
    split = {"train_ids": [clean[i]["id"] for i in train], "test_ids": [clean[i]["id"] for i in test]}
    return {"points": samples, "bounds": BOUNDS, "split": split, "seed": seed,
            "test_size": ratio, "challenge": {"noise": noise, "corruption": corruption,
            "corrupted_count": len(selected), "clipped_count": clipped},
            "fingerprint": digest(samples),
            "holdout_fingerprint": digest([clean[i] for i in test])}, effective_X, effective_y, train, test


def metrics(y, pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(y, pred)), "precision": float(precision),
            "recall": float(recall), "f1": float(f1)}


def selected_model_keys(payload):
    selected = payload.get("selected_models", list(MODEL_KEYS))
    if not isinstance(selected, list) or not 2 <= len(selected) <= len(MODEL_KEYS):
        raise ValueError("请选择 2～6 个不同的参赛算法。")
    if any(not isinstance(key, str) or key not in MODEL_KEYS for key in selected):
        raise ValueError("参赛算法包含未知模型。")
    if len(set(selected)) != len(selected):
        raise ValueError("参赛算法不能重复。")
    return tuple(key for key in MODEL_KEYS if key in selected)


def models_for(params, seed, n_train, model_keys=MODEL_KEYS):
    if not isinstance(params, dict):
        raise ValueError("模型参数格式错误。")
    if set(params) - set(MODEL_KEYS):
        raise ValueError("包含未知模型参数。")
    models = {}
    for key in model_keys:
        p = normalize_parameters(key, params.get(key, {}))
        if key == "knn" and p["n_neighbors"] > n_train:
            raise ValueError("K 不能大于训练样本数。")
        if key == "mlp" and p["network_type"] == "mlp" and p["solver"] != "lbfgs" and isinstance(p["batch_size"], int) and p["batch_size"] > n_train:
            raise ValueError("批大小不能超过本轮训练样本数；请选择 auto 或减小批大小，不会静默裁剪。")
        estimator = create_classifier(key, p, seed)
        scale = p.get("scale_features", key == "mlp")
        model = make_pipeline(StandardScaler(), estimator) if scale else estimator
        model._arena_parameters = p
        models[key] = model
    return models


def estimator_input(model, X):
    if hasattr(model, "steps"):
        return model.steps[-1][1], model.steps[0][1].transform(X), model.steps[0][1]
    return model, X, None


def describe_model(key, model, data, X, y, train, test, mesh, elapsed, convergence_warning=False):
    pred = model.predict(X)
    grid = model.predict(mesh)
    explanations = {}
    estimator, transformed_X, scaler = estimator_input(model, X)
    fit_info = {"convergence_warning": convergence_warning, "scale_features": scaler is not None}
    if key == "svm" or (key == "mlp" and model._arena_parameters["network_type"] == "perceptron"):
        for i, score in enumerate(model.decision_function(X)):
            explanations[data["points"][i]["id"]] = {"decision_score": float(score)}
        if key == "svm":
            fit_info.update(effective_gamma=float(estimator._gamma), fit_status=int(estimator.fit_status_))
        else:
            fit_info.update(network=[2, 1], network_type="perceptron", solver="perceptron",
                            iterations=int(estimator.n_iter_), final_loss=None, loss_curve=[],
                            iteration_limit_reached=convergence_warning)
    elif key == "knn":
        distances, indices = estimator.kneighbors(transformed_X)
        for i, pid in enumerate(p["id"] for p in data["points"]):
            explanations[pid] = {"neighbors": [
                {"id": data["points"][train[j]]["id"], "distance": float(d)}
                for j, d in zip(indices[i], distances[i])]}
    elif key == "tree":
        path = estimator.decision_path(transformed_X)
        for i, p in enumerate(data["points"]):
            steps = []
            for node in path.indices[path.indptr[i]:path.indptr[i+1]]:
                feature = int(estimator.tree_.feature[node])
                if feature >= 0 and (not hasattr(estimator, "expanded_nodes") or node in estimator.expanded_nodes):
                    threshold = float(estimator.tree_.threshold[node])
                    world_threshold = threshold if scaler is None else threshold * scaler.scale_[feature] + scaler.mean_[feature]
                    steps.append({"feature": feature, "threshold": float(world_threshold),
                                  "model_threshold": threshold,
                                  "direction": "≤" if transformed_X[i, feature] <= threshold else ">"})
            explanations[p["id"]] = {"path": steps, "leaf": int(estimator.apply(transformed_X[i:i+1])[0])}
    else:
        probabilities = model.predict_proba(X)
        if not np.isfinite(probabilities).all():
            raise ValueError(f"{key} 未能计算有效概率，请检查数据或调整参数。")
        if key == "forest":
            # sklearn RF predicts by mean leaf probabilities, NOT hard-vote majority.
            votes_b = np.sum([tree.predict(X) == 1 for tree in model.estimators_], axis=0)
            fit_info.update(tree_count=len(model.estimators_), aggregation="mean_probability")
        elif key == "mlp":
            network = estimator
            fit_info.update(network=[2, *network.hidden_layer_sizes, int(network.n_outputs_)],
                            network_type="mlp", solver=network.solver, activation=network.activation,
                            iterations=int(network.n_iter_), final_loss=float(network.loss_),
                            loss_curve=[float(v) for v in getattr(network, "loss_curve_", [])],
                            effective_batch_size=(None if network.solver == "lbfgs" else min(200, len(train)) if network.batch_size == "auto" else network.batch_size),
                            iteration_limit_reached=convergence_warning)
        elif key == "bayes":
            fit_info.update(means=estimator.theta_.tolist(), variances=estimator.var_.tolist(),
                            priors=estimator.class_prior_.tolist())
        for i, p in enumerate(data["points"]):
            explanation = {"probabilities": probabilities[i].tolist()}
            if key == "forest":
                explanation["tree_votes"] = [len(model.estimators_) - int(votes_b[i]), int(votes_b[i])]
            explanations[p["id"]] = explanation
    return {
        "predictions": {p["id"]: int(pred[i]) for i, p in enumerate(data["points"])},
        "grid": grid.astype(int).tolist(), "train": metrics(y[train], pred[train]),
        "test": metrics(y[test], pred[test]), "train_ms": elapsed, "explanations": explanations,
        "fit_info": fit_info,
        "configured_params": {f["name"]: model._arena_parameters[f["name"]]
                              for f in PARAMETERS[key]["fields"] if active(f, model._arena_parameters)},
        "effective_params": estimator.get_params(deep=False),
        "preprocessing": {"standardize": scaler is not None, "fit_scope": "train_only",
                          "mean": scaler.mean_.tolist() if scaler is not None else None,
                          "scale": scaler.scale_.tolist() if scaler is not None else None},
        "train_ids": data["split"]["train_ids"], "test_ids": data["split"]["test_ids"]
    }


def fit_experiment(payload, keep_models=False):
    data, X, y, train, test = prepare(payload)
    model_keys = selected_model_keys(payload)
    models = models_for(payload.get("params", {}), data["seed"], len(train), model_keys)
    if "bayes" in models and np.max(np.var(X[train], axis=0)) == 0:
        raise ValueError("高斯朴素贝叶斯无法从完全重合的训练坐标估计分布，请先分散训练样本。")
    xs = np.linspace(BOUNDS[0], BOUNDS[1], GRID_SIZE)
    # Row zero is the top of the shared data canvas and all six model canvases.
    ys = np.linspace(BOUNDS[3], BOUNDS[2], GRID_SIZE)
    xx, yy = np.meshgrid(xs, ys)
    mesh = np.column_stack([xx.ravel(), yy.ravel()])
    result = dict(data, grid={"size": GRID_SIZE, "bounds": BOUNDS, "row_order": "top_to_bottom"},
                  models={}, model_keys=list(model_keys), version=payload.get("version"))
    # Also bounds sklearn's native CPU pools in concurrent local Flask requests.
    with threadpool_limits(limits=1):
        for key, model in models.items():
            start = perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(X[train], y[train])
            elapsed = (perf_counter() - start) * 1000
            result["models"][key] = describe_model(
                key, model, data, X, y, train, test, mesh, elapsed,
                any(issubclass(w.category, ConvergenceWarning) for w in caught))
    ordered = sorted(result["models"], key=lambda k: (-result["models"][k]["test"]["f1"],
                                                      -result["models"][k]["test"]["accuracy"]))
    ranks = {}
    last_score = None
    for index, key in enumerate(ordered):
        score = tuple(round(result["models"][key]["test"][m], 12) for m in ("f1", "accuracy"))
        ranks[key] = index + 1 if score != last_score else ranks[ordered[index-1]]
        last_score = score
    result["ranks"] = ranks
    return (result, models, X, mesh) if keep_models else result
