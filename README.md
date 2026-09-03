# VizML：机器学习算法可视化平台

VizML 是一个面向机器学习教学与交互演示的 Web 应用。项目使用 Flask 提供页面和 REST API，使用 scikit-learn 等库完成数据生成、模型训练和评估，并通过原生 HTML、CSS、JavaScript 与 Canvas/图表库展示决策边界、训练过程、聚类结果、回归曲线和降维效果。

本项目适合用于：

- 直观理解常见机器学习算法的工作方式；
- 调整超参数并观察模型行为变化；
- 比较多种算法或优化器的效果；
- 学习 Flask 与 scikit-learn 的轻量级集成方式。

> 当前仓库是 **Flask + 原生前端** 的单体应用，不需要 Node.js，也没有 Vue/Vite 构建步骤。

## 功能概览

- 生成分类、回归、聚类和高维演示数据；
- 使用交互式控件配置模型与训练参数；
- 展示训练集/测试集指标、预测结果和混淆矩阵；
- 线性回归提供“引导学习 / 自由实验”双模式，可分步探索斜率与截距、残差、损失地形、梯度下降和泛化；
- 线性回归自由实验支持上传 `.xlsx`、预览工作表、自定义选择多个特征和目标列；
- 可视化二维分类决策边界、支持向量、近邻和决策路径；
- 单步观察 K-Means、DBSCAN、神经网络与 SGD 的训练过程；
- 比较聚类、降维和 SGD 优化器；
- 展示集成模型的基学习器、特征重要性和决策边界；
- 提供统一 JSON API，便于扩展页面或接入其他客户端。

## 支持的算法

| 类别 | 算法与能力 |
| --- | --- |
| 分类 | 支持向量机（SVM）、K 近邻（KNN）、决策树、高斯朴素贝叶斯 |
| 回归 | 线性回归、岭回归、Lasso、多项式回归、正规方程、SGD 回归 |
| 聚类 | K-Means、DBSCAN、层次聚类、高斯混合模型（GMM） |
| 神经网络 | 单层感知机、MLP 分类、训练历史和分步训练 |
| 集成学习 | 随机森林、AdaBoost、梯度提升、Voting、Stacking、Bagging、XGBoost |
| 降维 | PCA、t-SNE、LDA、UMAP |
| 优化器 | SGD、Momentum、AdaGrad、RMSprop、Adam、Nesterov |

UMAP 和 XGBoost 分别依赖 `umap-learn` 与 `xgboost`，二者已列入 `requirements.txt`。UMAP 未成功安装时，其余降维算法仍可使用。

## 技术栈

- Python 3.8+
- Flask 2.3.3、Flask-CORS
- scikit-learn 1.3.0
- NumPy、Pandas、SciPy
- Matplotlib、Seaborn
- UMAP、XGBoost
- 原生 HTML5、CSS3、JavaScript、Fetch API、Canvas
- Chart.js、Plotly.js（固定版本并由 Flask 本地托管）

## 快速开始

### 1. 获取代码

```bash
git clone <repository-url>
cd VizML
```

如果已经取得项目目录，直接进入目录即可。

### 2. 创建环境并安装依赖

推荐使用项目自带启动器：

```bash
python3 start.py --setup
```

该命令会在项目根目录创建 `.venv`，并根据 `requirements.txt` 安装依赖。

也可以手动创建虚拟环境。

macOS / Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell：

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. 启动应用

```bash
python start.py
```

打开 <http://localhost:5432> 即可进入首页。

> **端口说明：** 后端、前端 API 客户端和启动脚本统一使用 `5432` 端口。

### 4. 其他启动命令

```bash
# 调试模式
python start.py --debug

# 仅安装 requirements.txt 中的依赖（要求 .venv 已存在）
python start.py --install

# 查看 Python、操作系统和虚拟环境信息
python start.py --info
```

如果已经激活虚拟环境，也可以直接运行 Flask 应用：

macOS / Linux：

```bash
PORT=5432 python app.py
```

Windows PowerShell：

```powershell
$env:PORT = "5432"
python app.py
```

## 使用流程

1. 从首页选择一个算法模块；
2. 选择数据类型、样本数量、噪声和随机种子并生成数据；
3. 调整算法参数并开始训练；
4. 查看指标、决策边界、曲线或训练历史；
5. 在支持交互预测的页面点击画布或输入新样本；
6. 使用重置功能重新生成数据并比较其他参数。

线性回归页面默认进入“引导学习”，包含四个连续实验和即时理解检查；需要直接配置数据集、回归算法和优化器时，可切换到“自由实验”。自由实验支持不超过 5 MB 的 `.xlsx` 文件：第一行作为列名，上传后选择工作表、一个或多个数值特征列以及一个目标列即可导入。页面同时提供可下载的房价示例文件。

### Excel 多元回归示例

示例文件位于 [`frontend/static/examples/linear_regression_example.xlsx`](frontend/static/examples/linear_regression_example.xlsx)，包含“房屋面积、房龄、距市中心距离、房价”四列。在自由实验页面点击“使用示例数据”，确认前三列为特征、房价为目标，导入后点击“开始训练”，即可查看真实值与预测值、残差和模型系数。

所有算法页面也可以直接访问：

| 页面 | 地址 |
| --- | --- |
| 线性回归 | `/html/linear_regression.html` |
| SGD | `/html/sgd.html` |
| SVM | `/html/svm.html` |
| KNN | `/html/knn.html` |
| 决策树 | `/html/decision_tree.html` |
| 贝叶斯分类 | `/html/bayesian_classification.html` |
| 聚类 | `/html/clustering.html` |
| 神经网络 | `/html/neural_network.html` |
| 集成学习 | `/html/ensemble.html` |
| 降维 | `/html/dimensionality_reduction.html` |

例如：<http://localhost:5432/html/svm.html>。

## 可用数据

数据由 `backend/data_generator.py` 生成或从 scikit-learn 内置数据集加载。

- 分类数据：`blobs`、`circles`、`moons`、`gaussian`、`s_curve`、`xor`、`spiral`、`checkerboard`、`multi_cluster`、`imbalanced`、`linear_separable`、`classification`、`random`；
- 分类内置数据：Iris、Wine、Breast Cancer、Digits、Diabetes（根据页面需要截取、降维或转换标签）；
- 回归数据：线性、二次、三次、多项式、正弦/余弦、衰减正弦、复合函数、螺旋、多峰、阶跃、锯齿、分段线性、异方差、异常值、指数和对数数据；
- 回归自定义数据：可上传 `.xlsx`，无效数值行会在导入时过滤并提示；
- 聚类数据：`blobs`、`circles`、`moons`、`anisotropic`、`varied`、`smiley`、`petals`，也支持前端手绘点；
- 高维数据：Iris、Wine、Breast Cancer、Digits，以及可配置的合成高维分类数据。

## 项目结构

```text
VizML/
├── app.py                         # Flask 应用、页面路由和全部 API
├── start.py                       # 环境检查、依赖安装和服务启动器
├── run.sh                         # macOS/Linux 快捷启动脚本
├── requirements.txt               # Python 依赖
├── README.md                      # 项目使用说明
├── version.md                     # 当前版本与更新记录
├── .gitignore                     # Git 忽略规则
├── backend/
│   ├── data_generator.py          # 合成/内置数据集生成与参数推荐
│   └── algorithms/
│       ├── base.py                # 算法基类、评估与可视化公共能力
│       ├── svm.py
│       ├── knn.py
│       ├── decision_tree.py
│       ├── bayesian_classification.py
│       ├── linear_regression.py
│       ├── sgd.py
│       ├── clustering.py
│       ├── neural_network.py
│       ├── ensemble.py
│       └── dimensionality_reduction.py
└── frontend/
    ├── templates/                  # 首页与各算法的独立 HTML 页面
    └── static/
        ├── css/design-system.css   # 全站设计系统与响应式主题
        ├── js/api-client.js        # API 客户端和通用可视化工具
        ├── js/ui-shell.js          # 共享品牌栏和页面增强
        ├── js/visualization-theme.js # 统一图表色板与绘图工具
        └── vendor/                 # 本地 Chart.js、Plotly 及许可证
```

## 系统工作方式

典型请求链路如下：

```text
浏览器页面
   │
   ├─ POST /api/generate_data ──> 生成数据并返回 session_key
   │
   ├─ POST /api/<algorithm>/train ──> 训练模型并返回 model_key
   │
   └─ POST /api/<algorithm>/predict ──> 使用 model_key 预测新样本
                                      │
                                      └─ JSON 结果交给页面绘制
```

`app.py` 使用全局 `session_data` 字典保存：

- `datasets`：由 `session_key` 标识的数据集；
- `models`：由 `model_key` 标识的已训练模型及结果；
- `scalers`：训练时使用的特征缩放器；
- `training_history`：需要分步展示的训练状态。

这些数据只存在于当前 Python 进程中，服务重启后会全部丢失。

## API 使用

本文示例统一使用 `http://localhost:5432`。请求和响应均使用 JSON。成功响应通常包含 `status: "success"`；参数、数据或模型无效时通常返回 HTTP 400；资源不存在时返回 HTTP 404。

### 生成数据

```http
POST /api/generate_data
Content-Type: application/json
```

示例：

```bash
curl -X POST http://localhost:5432/api/generate_data \
  -H 'Content-Type: application/json' \
  -d '{
    "type": "classification",
    "shape": "moons",
    "n_samples": 200,
    "n_classes": 2,
    "noise": 0.15,
    "random_state": 42
  }'
```

响应中的 `session_key` 用于后续训练请求。当前键由数据类型和形状组合而成；再次生成同类型、同形状的数据会覆盖原数据。

### 训练模型

大部分监督学习接口遵循以下结构：

```bash
curl -X POST http://localhost:5432/api/svm/train \
  -H 'Content-Type: application/json' \
  -d '{
    "session_key": "classification_moons",
    "params": {
      "kernel": "rbf",
      "C": 1.0,
      "gamma": "scale",
      "scale_features": true
    }
  }'
```

聚类和集成学习接口还需要顶层 `algorithm` 字段。成功训练后返回的 `model_key` 用于预测、决策路径或详情查询。

### 预测新样本

```bash
curl -X POST http://localhost:5432/api/svm/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "model_key": "svm_classification_moons",
    "X": [[0.25, 0.50], [1.20, -0.10]]
  }'
```

`X` 必须是二维数组，并且特征数量应与训练数据一致。

### API 一览

| 模块 | 方法与路径 | 用途 |
| --- | --- | --- |
| 页面 | `GET /`、`GET /html/<path:filename>` | 首页与算法页面 |
| 数据 | `POST /api/generate_data` | 生成分类、回归或聚类数据 |
| 数据 | `POST /api/generate_full_dimensional_data` | 生成决策树所用完整维度数据 |
| 数据 | `GET /api/data/get_available_types` | 查询可用数据类型 |
| 数据 | `POST /api/data/get_description` | 查询数据说明 |
| 数据 | `POST /api/data/get_recommended_params` | 查询推荐参数 |
| 聚类 | `POST /api/clustering/train` | 训练单个聚类模型 |
| 聚类 | `POST /api/clustering/compare` | 比较多个聚类算法 |
| 聚类 | `POST /api/clustering/kmeans/debug_init` | 初始化 K-Means 单步演示 |
| 聚类 | `POST /api/clustering/kmeans/debug_step` | 执行 K-Means 单步 |
| 聚类 | `POST /api/clustering/dbscan/debug_init` | 初始化 DBSCAN 单步演示 |
| 聚类 | `POST /api/clustering/dbscan/debug_step` | 执行 DBSCAN 单步 |
| 聚类 | `GET /api/clustering/debug_state/<debug_key>` | 查询聚类调试状态 |
| 分类 | `POST /api/svm/train`、`POST /api/svm/predict` | SVM 训练与预测 |
| 分类 | `POST /api/knn/train`、`POST /api/knn/predict` | KNN 训练与预测 |
| 分类 | `POST /api/decision_tree/train`、`POST /api/decision_tree/predict` | 决策树训练与预测 |
| 分类 | `POST /api/decision_tree/decision_path` | 查询样本决策路径 |
| 分类 | `POST /api/bayesian_classification/train` | 高斯朴素贝叶斯训练 |
| 分类 | `POST /api/bayesian_classification/probabilities` | 查询详细后验概率 |
| 回归 | `POST /api/linear_regression/train`、`POST /api/linear_regression/predict` | 回归训练与预测 |
| 回归 | `POST /api/linear_regression/excel/preview` | 预览 Excel 工作表、列与样例数据 |
| 回归 | `POST /api/linear_regression/excel/import` | 从 Excel 选择特征列和目标列并创建数据会话 |
| SGD | `POST /api/sgd/train`、`POST /api/sgd/predict` | SGD 模型训练与预测 |
| SGD | `POST /api/sgd/start_training_session` | 初始化分步训练 |
| SGD | `POST /api/sgd/train_step`、`POST /api/sgd/get_training_state` | 单步训练与状态查询 |
| SGD | `POST /api/sgd/compare_optimizers` | 比较优化器 |
| SGD | `GET /api/sgd/get_available_models`、`GET /api/sgd/get_available_optimizers` | 查询可用模型和优化器 |
| SGD | `POST /api/sgd/reset_training_session` | 重置训练会话 |
| 神经网络 | `POST /api/neural_network/train` | 训练感知机或 MLP |
| 神经网络 | `POST /api/neural_network/decision_boundary` | 生成二维决策边界 |
| 神经网络 | `POST /api/neural_network/train_step` | 分步训练 |
| 集成学习 | `POST /api/ensemble/train` | 训练集成模型 |
| 集成学习 | `POST /api/ensemble/decision_boundaries` | 获取基学习器决策边界 |
| 集成学习 | `GET /api/ensemble/base_learners` | 查询可用基学习器 |
| 降维 | `GET /api/dimensionality_reduction/datasets` | 查询数据集和可用算法 |
| 降维 | `POST /api/dimensionality_reduction/load_data` | 加载内置或合成高维数据 |
| 降维 | `POST /api/dimensionality_reduction/apply` | 执行一种降维算法 |
| 降维 | `POST /api/dimensionality_reduction/compare` | 比较多种降维算法 |
| 降维 | `GET /api/dimensionality_reduction/info/<result_key>` | 查询降维结果详情 |
| 通用 | `GET /api/model_info/<model_key>` | 查询模型摘要 |
| 通用 | `POST /api/clear_session` | 清空当前进程中的全部数据和模型 |

更具体的算法参数及返回字段可查看 `app.py` 中对应路由和 `backend/algorithms/` 中的实现。

## 添加新算法

1. 在 `backend/algorithms/` 新建算法模块，并按任务类型复用 `BaseAlgorithm`、`BaseClassificationAlgorithm` 或 `BaseRegressionAlgorithm`；
2. 在 `app.py` 初始化算法对象并注册 `/api/<algorithm>/<action>` 路由；
3. 在 `frontend/templates/` 创建同名页面，通过 `/static/js/api-client.js` 或 Fetch API 调用后端；
4. 在 `frontend/templates/index.html` 添加导航入口；
5. 使用 Flask 测试客户端或浏览器验证数据生成、训练、预测及异常响应。

分类算法的最小结构示例：

```python
from .base import BaseClassificationAlgorithm


class NewAlgorithm(BaseClassificationAlgorithm):
    def train_model(self, X, y, **params):
        # 构造模型、训练并返回页面所需结果
        ...
```

新增接口时建议保持以下约定：

- 数据生成返回 `session_key`；
- 模型训练接收 `session_key` 并返回 `model_key`；
- 参数统一放在 `params` 字段中，算法选择放在 `algorithm` 字段中；
- 响应包含 `status`，错误响应同时包含可读的 `message`；
- 传给 `jsonify` 的 NumPy 数组和标量应先转换为 Python 列表和数值。

## 开发与验证

仓库目前没有配置自动化测试、代码格式化或 lint 命令。修改后至少建议执行：

```bash
# 检查 Python 文件是否存在语法错误
python -m compileall app.py backend

# 启动后验证首页和基础数据接口
curl http://localhost:5432/
curl http://localhost:5432/api/data/get_available_types
```

涉及具体算法时，还应按“生成数据 → 训练模型 → 预测/可视化”的完整链路进行验证。

## 已知限制

- 会话、数据集、模型和训练历史全部保存在单进程内存中，不会持久化；
- 不同访问者共享同一个全局状态，`/api/clear_session` 会清除所有当前状态；
- 当前实现适合本地教学演示，不适合直接部署为多用户生产服务；
- 前端 API 地址使用 `localhost:5432`，通过其他端口或远程主机访问前需要调整前端配置；
- Chart.js 与 Plotly 已在项目内本地托管，算法页面无需公共 CDN 即可加载图表；
- 部分二维可视化只适用于恰好包含两个特征的数据；
- `run.sh` 与 `python start.py` 均默认监听 5432 端口；
- `requirements.txt` 使用固定的较旧科学计算栈；如在较新的 Python 版本上安装失败，优先使用 Python 3.8～3.11 的独立虚拟环境。

## 常见问题

### 页面能打开，但训练请求失败

确认服务由 `python start.py` 启动并监听 5432 端口，然后在浏览器开发者工具的 Network/Console 中检查请求地址和后端返回的 `message`。

### 提示“数据未找到，请先生成数据”

训练接口依赖数据生成接口返回的 `session_key`。服务重启、调用 `/api/clear_session` 或传错键值后，需要重新生成数据。

### 提示“模型未找到”

预测和详情接口依赖训练响应中的 `model_key`。模型只保存在当前服务进程内，重启后需要重新训练。

### 安装 `xgboost`、`umap-learn` 或科学计算依赖失败

先升级 pip，并确认正在使用受依赖支持的 Python 版本：

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

### California Housing 数据无法加载

该数据集可能需要首次在线下载。检查网络连接，或改用项目内无需下载的合成数据和其他 scikit-learn 内置数据集。

## 安全提示

Flask 应用监听 `0.0.0.0`，同一局域网中的设备可能访问该服务。不要在不可信网络中暴露调试模式，也不要直接将当前应用作为公网生产服务运行。

## 许可证

仓库当前未提供 `LICENSE` 文件。在公开分发、复用或接受外部贡献前，请由项目维护者补充明确的开源许可证。
