"""Arena-only classifiers and parameter adapters matching the native lab controls.

Legacy labs are deliberately untouched. Arena owns its classifier construction,
single split and training-only preprocessing; no legacy endpoint depends on it.
"""
import math

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def field(name, label, dom_id, default, kind="number", **options):
    return dict(name=name, label=label, id=dom_id, default=default, type=kind, **options)


PARAMETERS = {
    "svm": dict(title="SVM", lab="/html/svm.html", fields=[
        field("kernel", "核函数", "kernel", "rbf", "select",
              options=[["linear", "线性"], ["rbf", "RBF"], ["poly", "多项式"], ["sigmoid", "Sigmoid"]]),
        field("C", "惩罚参数 C", "c", 1.0, min=0.01, max=1000, step=0.01),
        field("gamma", "Gamma", "gamma", 0.1, "gamma", min=0.001, max=10, step=0.001,
              when={"kernel": ["rbf", "poly", "sigmoid"]}),
        field("degree", "多项式次数", "svm-degree", 3, min=1, max=10, integer=True,
              when={"kernel": ["poly"]}),
        field("coef0", "独立项 coef0", "svm-coef0", 0.0, min=-5, max=5, step=0.1,
              when={"kernel": ["poly", "sigmoid"]}),
        field("max_iter", "最大迭代", "svm-iterations", 1000, min=100, max=10000, integer=True, step=100),
        field("tol", "容忍度", "svm-tol", 0.001, min=0.0001, max=0.01, step=0.0001),
        field("scale_features", "标准化特征", "svm-scale", True, "boolean"),
    ]),
    "knn": dict(title="KNN", lab="/html/knn.html", fields=[
        field("n_neighbors", "邻居 K", "k", 5, min=1, max=31, integer=True),
        field("metric", "距离度量", "knn-metric", "euclidean", "select",
              options=[["euclidean", "欧氏"], ["manhattan", "曼哈顿"], ["minkowski", "闵可夫斯基"], ["chebyshev", "切比雪夫"]]),
        field("p", "闵可夫斯基 p", "knn-p", 2, min=1, max=10, step=0.1, when={"metric": ["minkowski"]}),
        field("weights", "投票权重", "weights", "uniform", "select",
              options=[["uniform", "等权"], ["distance", "距离"]]),
        field("scale_features", "标准化特征", "knn-scale", True, "boolean"),
    ]),
    "tree": dict(title="决策树", lab="/html/decision_tree.html", fields=[
        field("max_depth", "最大深度（留空不限）", "depth", 5, min=1, max=15, integer=True, nullable=True),
        field("min_samples_split", "最少分裂样本", "tree-split", 2, min=2, max=50, integer=True),
        field("min_samples_leaf", "叶节点最少样本", "leaf", 1, min=1, max=20, integer=True),
        field("max_features", "候选特征", "tree-features", "sqrt", "select",
              options=[["all", "全部"], ["sqrt", "sqrt"], ["log2", "log2"]]),
        field("criterion", "分裂准则", "tree-criterion", "gini", "select",
              options=[["gini", "Gini"], ["entropy", "信息熵"]]),
        field("scale_features", "标准化特征", "tree-scale", False, "boolean"),
    ]),
    "forest": dict(title="随机森林", lab="/html/ensemble.html", fields=[
        field("n_estimators", "树数量", "forest-trees", 10, min=3, max=120, integer=True),
        field("max_depth", "最大深度（留空不限）", "forest-depth", 5, min=1, max=20, integer=True, nullable=True),
        field("min_samples_split", "最少分裂样本", "forest-split", 2, min=2, max=20, integer=True),
        field("min_samples_leaf", "叶节点最少样本", "forest-leaf", 1, min=1, max=20, integer=True),
        field("max_features", "候选特征", "forest-features", "sqrt", "select",
              options=[["sqrt", "sqrt"], ["log2", "log2"], ["auto", "auto（分类等价 sqrt）"], ["all", "全部"]]),
    ]),
    "mlp": dict(title="神经网络", lab="/html/neural_network.html", fields=[
        field("network_type", "网络类型", "mlp-type", "mlp", "select",
              options=[["mlp", "多层感知机"], ["perceptron", "单层感知机"]]),
        field("hidden_layer_sizes", "隐藏层结构", "mlp-hidden", "100,50", "layers",
              options=[["50", "单层 50"], ["100,50", "双层 100,50"], ["100,100,50", "三层 100,100,50"],
                       ["100,100,100,100,50", "五层 100,100,100,100,50"]],
              when={"network_type": ["mlp"]}),
        field("activation", "激活函数", "mlp-activation", "relu", "select",
              options=[["relu", "ReLU"], ["tanh", "Tanh"], ["logistic", "Sigmoid"]],
              when={"network_type": ["mlp"]}),
        field("solver", "求解器", "mlp-solver", "adam", "select",
              options=[["adam", "Adam"], ["sgd", "SGD"], ["lbfgs", "L-BFGS"]],
              when={"network_type": ["mlp"]}),
        field("learning_rate_init", "初始学习率", "mlp-rate", 0.001, min=0.0001, max=0.1, step=0.0001,
              inactive_when={"network_type": ["mlp"], "solver": ["lbfgs"]}),
        field("max_iter", "最大迭代", "mlp-iterations", 500, min=100, max=2000, integer=True, step=100),
        field("alpha", "L2 正则化", "mlp-alpha", 0.0001, min=0, max=1, step=0.0001,
              when={"network_type": ["mlp"]}),
        field("batch_size", "批大小（auto 自动）", "mlp-batch", "auto", "batch",
              when={"network_type": ["mlp"], "solver": ["adam", "sgd"]}),
    ], note="与原实验一致，神经网络始终标准化。自定义最多 8 层、每层 256、合计 1024 单元；超限报错，不截断。"),
    "bayes": dict(title="高斯贝叶斯", lab="/html/bayesian_classification.html", fields=[
        field("var_smoothing", "方差平滑", "bayes-smoothing", 1e-9, min=1e-12, max=0.1, step="any"),
        field("priors", "类别先验", "bayes-priors", None, "priors"),
        field("scale_features", "标准化特征", "bayes-scale", False, "boolean"),
    ]),
}


def defaults(key):
    return {f["name"]: f["default"] for f in PARAMETERS[key]["fields"]}


def active(field_info, config):
    def matches(conditions):
        return all(config.get(name) in choices for name, choices in conditions.items())
    return matches(field_info.get("when", {})) and not (
        field_info.get("inactive_when") and matches(field_info["inactive_when"]))


def layer_sizes(value, bounded=True):
    if isinstance(value, str):
        try:
            value = [int(n.strip()) for n in value.split(",")]
        except ValueError:
            raise ValueError("隐藏层结构应为逗号分隔的正整数，如 64,32,16。")
    if (not isinstance(value, (list, tuple)) or not value
            or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in value)):
        raise ValueError("隐藏层结构必须为正整数列表。")
    if bounded and (len(value) > 8 or max(value) > 256 or sum(value) > 1024):
        raise ValueError("自定义网络最多 8 层、每层 1～256 单元、总计不超过 1024；不会自动截断。")
    return tuple(value)


def normalize_parameters(key, supplied):
    if not isinstance(supplied, dict):
        raise ValueError(f"{key} 模型参数必须为对象。")
    p = dict(supplied)
    # Compatibility with the initial Arena API; all new UI fields use lab names.
    aliases = {"knn": {"k": "n_neighbors"}, "tree": {"depth": "max_depth", "leaf": "min_samples_leaf"},
               "forest": {"trees": "n_estimators", "depth": "max_depth", "leaf": "min_samples_leaf", "features": "max_features"},
               "mlp": {"iterations": "max_iter"}, "bayes": {"smoothing": "var_smoothing"}}
    if key == "mlp" and ("width" in p or "layers" in p):
        if "hidden_layer_sizes" in p:
            raise ValueError("不要同时指定旧版宽度/层数和 hidden_layer_sizes。")
        width, count = p.pop("width", 32), p.pop("layers", 2)
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (width, count)) or not 1 <= count <= 8:
            raise ValueError("MLP 宽度和层数应为有效整数。")
        p.setdefault("hidden_layer_sizes", [width] * count)
    for old, new in aliases.get(key, {}).items():
        if old in p:
            if new in p:
                raise ValueError(f"不要同时指定 {old} 和 {new}。")
            p[new] = p.pop(old)
    if "max_features" in p and p["max_features"] in (None, "None", "none", "1.0", 1.0):
        p["max_features"] = "all"
    unknown = set(p) - set(defaults(key))
    if unknown:
        raise ValueError(f"{key} 不支持这些参数：{', '.join(sorted(unknown))}。")
    config = dict(defaults(key), **p)
    for f in PARAMETERS[key]["fields"]:
        name, value = f["name"], config[f["name"]]
        if not active(f, config):
            config[name] = f["default"]
            continue
        kind = f["type"]
        if value is None and f.get("nullable"):
            continue
        if kind == "boolean":
            if not isinstance(value, bool):
                raise ValueError(f'{f["label"]} 必须为布尔值。')
        elif kind == "select":
            if value not in [o[0] for o in f["options"]]:
                raise ValueError(f'{f["label"]} 不是有效选项。')
        elif kind == "layers":
            config[name] = layer_sizes(value)
        elif kind == "priors":
            if value is not None:
                if isinstance(value, str):
                    try:
                        value = [float(n.strip()) for n in value.split(",")]
                    except ValueError:
                        raise ValueError("先验应为两个概率，如 0.7,0.3。")
                if (not isinstance(value, (list, tuple)) or len(value) != 2
                        or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 < v < 1 for v in value)
                        or not math.isclose(sum(value), 1.0, abs_tol=1e-8)):
                    raise ValueError("A、B 先验概率必须为正数且和为 1。")
                config[name] = list(value)
        elif kind == "batch":
            if value != "auto" and (isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 600):
                raise ValueError("批大小应为 auto 或 1～600 的整数。")
        elif kind == "gamma" and isinstance(value, str) and value in ("scale", "auto"):
            continue
        else:
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    or not f["min"] <= value <= f["max"] or (f.get("integer") and int(value) != value)):
                raise ValueError(f'{f["label"]} 必须在 {f["min"]}～{f["max"]} 范围内' +
                                 ("且为整数。" if f.get("integer") else "。"))
            if f.get("integer"):
                config[name] = int(value)
    return config


def feature_choice(value, forest=False):
    if value in (None, "all", "None", "none", "1.0", 1.0):
        return None
    # Old sklearn accepted 'auto'; modern sklearn removed this spelling.
    return "sqrt" if forest and value == "auto" else value


def create_classifier(key, config, seed=42, warm_start=False):
    """Create an Arena estimator without changing native lab implementations."""
    p = dict(defaults(key), **config)
    if key == "svm":
        return SVC(kernel=p["kernel"], C=p["C"], gamma=p["gamma"], degree=int(p["degree"]),
                   coef0=p["coef0"], max_iter=p["max_iter"], tol=p["tol"],
                   probability=True, random_state=seed)
    if key == "knn":
        return KNeighborsClassifier(n_neighbors=p["n_neighbors"], weights=p["weights"], metric=p["metric"], p=p["p"])
    if key in ("tree", "forest"):
        common = dict(max_depth=p["max_depth"], min_samples_split=p["min_samples_split"],
                      min_samples_leaf=p["min_samples_leaf"], max_features=feature_choice(p["max_features"], key == "forest"),
                      random_state=seed)
        return (DecisionTreeClassifier(criterion=p["criterion"], **common) if key == "tree" else
                RandomForestClassifier(n_estimators=p["n_estimators"], n_jobs=1, **common))
    if key == "bayes":
        return GaussianNB(var_smoothing=p["var_smoothing"], priors=p["priors"])
    if p["network_type"] == "perceptron":
        return Perceptron(max_iter=p["max_iter"], eta0=p["learning_rate_init"], random_state=seed)
    # Public Arena input is bounded by normalize_parameters before construction.
    hidden = layer_sizes(p["hidden_layer_sizes"], bounded=False)
    return MLPClassifier(hidden_layer_sizes=hidden, activation=p["activation"], solver=p["solver"],
                         learning_rate_init=p["learning_rate_init"], max_iter=p["max_iter"],
                         batch_size=p["batch_size"], alpha=p["alpha"], random_state=seed, warm_start=warm_start)
