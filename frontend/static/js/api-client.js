/**
 * 机器学习算法可视化平台 - API客户端
 * 封装与后端Flask服务的通信
 */

/**
 * 用户身份管理：为每个浏览器自动分配唯一 ID，可自愿填写昵称/学号
 */
const VizHistory = (function () {
    const USER_KEY = 'vizml_user_id';
    const NAME_KEY = 'vizml_username';

    function getUserId() {
        let id = null;
        try { id = localStorage.getItem(USER_KEY); } catch (e) {}
        if (!id) {
            id = (window.crypto && crypto.randomUUID)
                ? crypto.randomUUID()
                : 'u-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2);
            try { localStorage.setItem(USER_KEY, id); } catch (e) {}
        }
        return id;
    }

    function getUsername() {
        try { return localStorage.getItem(NAME_KEY) || ''; } catch (e) { return ''; }
    }

    function setUsername(name) {
        try { localStorage.setItem(NAME_KEY, name || ''); } catch (e) {}
    }

    function headers() {
        return { 'X-User-Id': getUserId(), 'X-Username': getUsername() };
    }

    return { getUserId, getUsername, setUsername, headers };
})();
window.VizHistory = VizHistory;

class MLApiClient {
    constructor(baseUrl = 'http://localhost:5432') {
        this.baseUrl = baseUrl;
        this.sessionData = new Map(); // 存储会话数据
        this._lastDataConfig = null;  // 最近一次数据生成配置（用于历史复现）
    }

    /**
     * 发送HTTP请求的通用方法
     */
    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        
        const config = {
            method: 'GET',
            ...options,
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                ...VizHistory.headers(),
                ...(options.headers || {})
            }
        };

        try {
            const response = await fetch(url, config);
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.message || `HTTP ${response.status}`);
            }
            
            // 缓存最近一次数据生成配置（用于历史复现）
            if (config.method === 'POST' && (endpoint === '/api/generate_data' || endpoint === '/api/generate_full_dimensional_data')) {
                try { this._lastDataConfig = config.body ? JSON.parse(config.body) : {}; } catch (e) {}
            }
            
            // 训练类请求成功后自动记录历史
            this._maybeRecordHistory(endpoint, config, data);
            
            return data;
        } catch (error) {
            console.error(`API请求失败 (${endpoint}):`, error);
            throw error;
        }
    }

    /**
     * 识别训练/对比请求，成功后自动保存一条历史记录（失败不抛错）
     */
    _maybeRecordHistory(endpoint, config, data) {
        if (config.method !== 'POST') return;
        if (endpoint.indexOf('/api/history/') !== -1) return;
        if (!data || data.status !== 'success') return;

        const match = endpoint.match(/^\/api\/([a-z_]+)\/(train|compare)$/);
        if (!match) return;

        const module = match[1];
        const action = match[2];

        let body = {};
        try { body = config.body ? JSON.parse(config.body) : {}; } catch (e) { body = {}; }

        const metrics = (data.results && data.results.metrics) || data.metrics || {};
        const params = body.params || {};
        const datasetKey = body.session_key || data.session_key || '';

        this.saveHistory({
            module,
            label: action === 'compare' ? '算法对比' : '模型训练',
            dataset: datasetKey,
            params,
            metrics,
            replay: {
                data: this._lastDataConfig || {},
                train: body
            }
        }).catch(() => {});
    }

    /**
     * 生成数据
     */
    async generateData(config) {
        try {
            const result = await this.request('/api/generate_data', {
                method: 'POST',
                body: JSON.stringify(config)
            });
            
            if (result.status === 'success') {
                // 缓存会话数据
                this.sessionData.set(result.session_key, result.data);
                return result;
            } else {
                throw new Error(result.message);
            }
        } catch (error) {
            console.error('数据生成失败:', error);
            throw error;
        }
    }

    /**
     * 训练聚类模型
     */
    async trainClustering(config) {
        try {
            const result = await this.request('/api/clustering/train', {
                method: 'POST',
                body: JSON.stringify(config)
            });
            
            return result;
        } catch (error) {
            console.error('聚类训练失败:', error);
            throw error;
        }
    }

    /**
     * 对比多个聚类算法
     */
    async compareClusteringAlgorithms(config) {
        try {
            const result = await this.request('/api/clustering/compare', {
                method: 'POST',
                body: JSON.stringify(config)
            });
            
            return result;
        } catch (error) {
            console.error('聚类算法对比失败:', error);
            throw error;
        }
    }




    /**
     * 训练SVM模型
     */
    async trainSVM(config) {
        try {
            const result = await this.request('/api/svm/train', {
                method: 'POST',
                body: JSON.stringify(config)
            });
            
            return result;
        } catch (error) {
            console.error('SVM训练失败:', error);
            throw error;
        }
    }

    /**
     * SVM预测
     */
    async predictSVM(config) {
        try {
            const result = await this.request('/api/svm/predict', {
                method: 'POST',
                body: JSON.stringify(config)
            });
            
            return result;
        } catch (error) {
            console.error('SVM预测失败:', error);
            throw error;
        }
    }

    /**
     * 训练决策树模型
     */
    async trainDecisionTree(config) {
        try {
            const result = await this.request('/api/decision_tree/train', {
                method: 'POST',
                body: JSON.stringify(config)
            });
            
            return result;
        } catch (error) {
            console.error('决策树训练失败:', error);
            throw error;
        }
    }

    /**
     * 决策树预测
     */
    async predictDecisionTree(config) {
        try {
            const result = await this.request('/api/decision_tree/predict', {
                method: 'POST',
                body: JSON.stringify(config)
            });
            
            return result;
        } catch (error) {
            console.error('决策树预测失败:', error);
            throw error;
        }
    }

    /**
     * 获取决策树的决策路径
     */
    async getDecisionTreePath(config) {
        try {
            const result = await this.request('/api/decision_tree/decision_path', {
                method: 'POST',
                body: JSON.stringify(config)
            });
            return result;
        } catch (error) {
            console.error('获取决策路径失败:', error);
            throw error;
        }
    }

    /**
     * 生成完整维度数据
     */
    async generateFullDimensionalData(config) {
        try {
            const result = await this.request('/api/generate_full_dimensional_data', {
                method: 'POST',
                body: JSON.stringify(config)
            });
            return result;
        } catch (error) {
            console.error('生成完整维度数据失败:', error);
            throw error;
        }
    }

    /**
     * 训练KNN模型
     */
    async trainKNN(config) {
        try {
            const result = await this.request('/api/knn/train', {
                method: 'POST',
                body: JSON.stringify(config)
            });
            
            return result;
        } catch (error) {
            console.error('KNN训练失败:', error);
            throw error;
        }
    }

    /**
     * KNN预测
     */
    async predictKNN(config) {
        try {
            const result = await this.request('/api/knn/predict', {
                method: 'POST',
                body: JSON.stringify(config)
            });
            
            return result;
        } catch (error) {
            console.error('KNN预测失败:', error);
            throw error;
        }
    }

    /**
     * 训练朴素贝叶斯模型
     */
    async trainBayesianClassification(config) {
        try {
            const result = await this.request('/api/bayesian_classification/train', {
                method: 'POST',
                body: JSON.stringify(config)
            });
            
            return result;
        } catch (error) {
            console.error('朴素贝叶斯训练失败:', error);
            throw error;
        }
    }

    /**
     * 朴素贝叶斯预测
     */
    async predictBayesianClassification(config) {
        try {
            const result = await this.request('/api/bayesian_classification/predict', {
                method: 'POST',
                body: JSON.stringify(config)
            });
            
            return result;
        } catch (error) {
            console.error('朴素贝叶斯预测失败:', error);
            throw error;
        }
    }

    /**
     * 获取贝叶斯分类的详细概率信息
     */
    async getBayesianProbabilities(config) {
        try {
            const result = await this.request('/api/bayesian_classification/probabilities', {
                method: 'POST',
                body: JSON.stringify(config)
            });
            
            return result;
        } catch (error) {
            console.error('获取贝叶斯概率失败:', error);
            throw error;
        }
    }

    /**
     * 预览线性回归 Excel 文件
     */
    async previewLinearRegressionExcel(file) {
        const formData = new FormData();
        formData.append('file', file);
        return this.request('/api/linear_regression/excel/preview', {
            method: 'POST',
            headers: { 'Accept': 'application/json' },
            body: formData
        });
    }

    /**
     * 将 Excel 中选择的列导入为线性回归数据集
     */
    async importLinearRegressionExcel(file, sheetName, featureColumns, targetColumn) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('sheet_name', sheetName);
        featureColumns.forEach((column) => formData.append('feature_columns', column));
        formData.append('target_column', targetColumn);

        const result = await this.request('/api/linear_regression/excel/import', {
            method: 'POST',
            headers: { 'Accept': 'application/json' },
            body: formData
        });
        if (result.status === 'success') {
            this.sessionData.set(result.session_key, result.data);
        }
        return result;
    }

    /**
     * 训练线性回归模型
     */
    async trainLinearRegression(config) {
        try {
            const result = await this.request('/api/linear_regression/train', {
                method: 'POST',
                body: JSON.stringify(config)
            });
            
            return result;
        } catch (error) {
            console.error('线性回归训练失败:', error);
            throw error;
        }
    }

    /**
     * 线性回归预测
     */
    async predictLinearRegression(config) {
        try {
            const result = await this.request('/api/linear_regression/predict', {
                method: 'POST',
                body: JSON.stringify(config)
            });
            
            return result;
        } catch (error) {
            console.error('线性回归预测失败:', error);
            throw error;
        }
    }

    /**
     * 训练SGD模型
     */
    async trainSGD(config) {
        try {
            const result = await this.request('/api/sgd/train', {
                method: 'POST',
                body: JSON.stringify(config)
            });
            
            return result;
        } catch (error) {
            console.error('SGD训练失败:', error);
            throw error;
        }
    }

    /**
     * SGD预测
     */
    async predictSGD(config) {
        try {
            const result = await this.request('/api/sgd/predict', {
                method: 'POST',
                body: JSON.stringify(config)
            });
            
            return result;
        } catch (error) {
            console.error('SGD预测失败:', error);
            throw error;
        }
    }

    /**
     * 训练神经网络模型
     */
    async trainNeuralNetwork(config) {
        try {
            const result = await this.request('/api/neural_network/train', {
                method: 'POST',
                body: JSON.stringify(config)
            });
            
            return result;
        } catch (error) {
            console.error('神经网络训练失败:', error);
            throw error;
        }
    }

    /**
     * 获取神经网络决策边界
     */
    async getDecisionBoundary(config) {
        try {
            const result = await this.request('/api/neural_network/decision_boundary', {
                method: 'POST',
                body: JSON.stringify(config)
            });
            
            return result;
        } catch (error) {
            console.error('获取决策边界失败:', error);
            throw error;
        }
    }

    /**
     * 分步训练神经网络模型
     */
    async trainNeuralNetworkStep(config) {
        try {
            const result = await this.request('/api/neural_network/train_step', {
                method: 'POST',
                body: JSON.stringify(config)
            });
            
            return result;
        } catch (error) {
            console.error('分步训练失败:', error);
            throw error;
        }
    }

    /**
     * 训练集成学习模型
     */
    async trainEnsemble(config) {
        try {
            const result = await this.request('/api/ensemble/train', {
                method: 'POST',
                body: JSON.stringify(config)
            });
            
            return result;
        } catch (error) {
            console.error('集成学习训练失败:', error);
            throw error;
        }
    }

    /**
     * 获取集成学习模型的决策边界
     */
    async getEnsembleDecisionBoundaries(config) {
        try {
            const result = await this.request('/api/ensemble/decision_boundaries', {
                method: 'POST',
                body: JSON.stringify(config)
            });
            
            return result;
        } catch (error) {
            console.error('获取集成决策边界失败:', error);
            throw error;
        }
    }

    /**
     * 获取降维数据集列表
     */
    async getDimensionalityReductionDatasets() {
        try {
            const result = await this.request('/api/dimensionality_reduction/datasets');
            return result;
        } catch (error) {
            console.error('获取降维数据集失败:', error);
            throw error;
        }
    }

    /**
     * 加载降维数据
     */
    async loadDimensionalityReductionData(config) {
        try {
            const result = await this.request('/api/dimensionality_reduction/load_data', {
                method: 'POST',
                body: JSON.stringify(config)
            });
            return result;
        } catch (error) {
            console.error('加载降维数据失败:', error);
            throw error;
        }
    }

    /**
     * 应用降维算法
     */
    async applyDimensionalityReduction(config) {
        try {
            const result = await this.request('/api/dimensionality_reduction/apply', {
                method: 'POST',
                body: JSON.stringify(config)
            });
            return result;
        } catch (error) {
            console.error('应用降维算法失败:', error);
            throw error;
        }
    }

    /**
     * 比较降维算法
     */
    async compareDimensionalityReduction(config) {
        try {
            const result = await this.request('/api/dimensionality_reduction/compare', {
                method: 'POST',
                body: JSON.stringify(config)
            });
            return result;
        } catch (error) {
            console.error('比较降维算法失败:', error);
            throw error;
        }
    }

    /**
     * 获取降维结果信息
     */
    async getDimensionalityReductionInfo(resultKey) {
        try {
            const result = await this.request(`/api/dimensionality_reduction/info/${resultKey}`);
            return result;
        } catch (error) {
            console.error('获取降维结果信息失败:', error);
            throw error;
        }
    }

    /**
     * 获取模型信息
     */
    async getModelInfo(modelKey) {
        try {
            const result = await this.request(`/api/model_info/${modelKey}`);
            return result;
        } catch (error) {
            console.error('获取模型信息失败:', error);
            throw error;
        }
    }

    /**
     * 清除会话
     */
    async clearSession() {
        try {
            const result = await this.request('/api/clear_session', {
                method: 'POST'
            });
            
            // 清除本地缓存
            this.sessionData.clear();
            
            return result;
        } catch (error) {
            console.error('清除会话失败:', error);
            throw error;
        }
    }

    /**
     * 获取缓存的会话数据
     */
    getSessionData(sessionKey) {
        return this.sessionData.get(sessionKey);
    }

    /**
     * 保存一条历史运行记录
     */
    async saveHistory(record) {
        return this.request('/api/history/save', {
            method: 'POST',
            body: JSON.stringify({
                ...record,
                user_id: VizHistory.getUserId(),
                username: VizHistory.getUsername()
            })
        });
    }

    /**
     * 查询当前用户的历史记录
     */
    async listHistory() {
        return this.request(`/api/history/list?user_id=${encodeURIComponent(VizHistory.getUserId())}`);
    }

    /**
     * 删除单条历史记录
     */
    async deleteHistory(recordId) {
        return this.request('/api/history/delete', {
            method: 'POST',
            body: JSON.stringify({ id: recordId, user_id: VizHistory.getUserId() })
        });
    }

    /**
     * 清空当前用户的全部历史记录
     */
    async clearHistory() {
        return this.request('/api/history/clear', {
            method: 'POST',
            body: JSON.stringify({ user_id: VizHistory.getUserId() })
        });
    }

    /**
     * 更新昵称并同步到已有记录
     */
    async updateUsername(username) {
        VizHistory.setUsername(username);
        return this.request('/api/history/username', {
            method: 'POST',
            body: JSON.stringify({ user_id: VizHistory.getUserId(), username })
        });
    }

    /**
     * 检查服务器连接状态
     */
    async checkConnection() {
        try {
            const url = `${this.baseUrl}/`;
            const response = await fetch(url);
            // 只要能获得响应就认为连接成功（不管是HTML还是JSON）
            return response.ok;
        } catch (error) {
            console.warn('后端服务连接失败，可能需要启动Flask服务器');
            return false;
        }
    }

    /**
     * 通用GET请求
     */
    async get(endpoint) {
        return await this.request(endpoint);
    }

    /**
     * 通用POST请求
     */
    async post(endpoint, data) {
        return await this.request(endpoint, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    }
}

/**
 * 数据可视化辅助类
 */
class MLVisualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.width = this.canvas.width;
        this.height = this.canvas.height;
    }

    /**
     * 清空画布
     */
    clear() {
        if (window.VizTheme) {
            VizTheme.clear(this.ctx, this.width, this.height);
        } else {
            this.ctx.fillStyle = '#FBFCFE';
            this.ctx.fillRect(0, 0, this.width, this.height);
        }
        this.drawGrid();
    }

    /**
     * 绘制网格
     */
    drawGrid() {
        if (window.VizTheme) {
            VizTheme.drawGrid(this.ctx, this.width, this.height);
            return;
        }
        this.ctx.strokeStyle = '#E7EDF5';
        this.ctx.lineWidth = 1;
        for (let i = 0; i <= this.width; i += 24) {
            this.ctx.beginPath(); this.ctx.moveTo(i, 0); this.ctx.lineTo(i, this.height); this.ctx.stroke();
        }
        for (let i = 0; i <= this.height; i += 24) {
            this.ctx.beginPath(); this.ctx.moveTo(0, i); this.ctx.lineTo(this.width, i); this.ctx.stroke();
        }
    }

    /**
     * 绘制数据点
     */
    drawDataPoints(X, y = null, colors = null) {
        const defaultColors = window.VizTheme ? VizTheme.palette : ['#4F46E5', '#0F9D8A', '#E07A3F', '#D64C7F', '#2B8AC6'];
        
        for (let i = 0; i < X.length; i++) {
            const x = X[i][0] * this.width;
            const yPos = (1 - X[i][1]) * this.height;
            
            let color = '#4F46E5';
            if (y && y[i] !== undefined) {
                color = defaultColors[y[i] % defaultColors.length];
            } else if (colors && colors[i]) {
                color = colors[i];
            }
            
            if (window.VizTheme) VizTheme.drawPoint(this.ctx, x, yPos, color, 4);
            else {
                this.ctx.beginPath(); this.ctx.arc(x, yPos, 4, 0, 2 * Math.PI);
                this.ctx.fillStyle = color; this.ctx.fill();
                this.ctx.strokeStyle = '#ffffff'; this.ctx.lineWidth = 1; this.ctx.stroke();
            }
        }
    }

    /**
     * 绘制回归线
     */
    drawRegressionLine(coefficients, intercept, color = '#4F46E5') {
        if (coefficients.length !== 1) {
            console.warn('只支持单变量线性回归的可视化');
            return;
        }
        
        const slope = coefficients[0];
        
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 3;
        this.ctx.beginPath();
        
        const x1 = 0;
        const y1 = (1 - (slope * 0 + intercept)) * this.height;
        const x2 = this.width;
        const y2 = (1 - (slope * 1 + intercept)) * this.height;
        
        this.ctx.moveTo(x1, y1);
        this.ctx.lineTo(x2, y2);
        this.ctx.stroke();
    }

    /**
     * 绘制聚类中心
     */
    drawClusterCenters(centers, colors = null) {
        const defaultColors = window.VizTheme ? VizTheme.palette : ['#4F46E5', '#0F9D8A', '#E07A3F', '#D64C7F', '#2B8AC6'];
        
        centers.forEach((center, i) => {
            const x = center[0] * this.width;
            const y = (1 - center[1]) * this.height;
            
            const color = colors ? colors[i] : defaultColors[i % defaultColors.length];
            
            // 绘制十字标记
            this.ctx.strokeStyle = color;
            this.ctx.lineWidth = 3;
            this.ctx.beginPath();
            this.ctx.moveTo(x - 8, y);
            this.ctx.lineTo(x + 8, y);
            this.ctx.moveTo(x, y - 8);
            this.ctx.lineTo(x, y + 8);
            this.ctx.stroke();
            
            // 绘制中心圆
            this.ctx.beginPath();
            this.ctx.arc(x, y, 6, 0, Math.PI * 2);
            this.ctx.fillStyle = color;
            this.ctx.fill();
            this.ctx.strokeStyle = '#ffffff';
            this.ctx.lineWidth = 2;
            this.ctx.stroke();
        });
    }

    /**
     * 坐标转换：数据坐标到画布坐标
     */
    dataToCanvas(dataX, dataY) {
        return {
            x: dataX * this.width,
            y: (1 - dataY) * this.height
        };
    }

    /**
     * 坐标转换：画布坐标到数据坐标
     */
    canvasToData(canvasX, canvasY) {
        return {
            x: canvasX / this.width,
            y: 1 - (canvasY / this.height)
        };
    }
}

/**
 * 工具函数
 */
class MLUtils {
    /**
     * 显示加载状态
     */
    static showLoading(elementId, message = '处理中...') {
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = `
                <div class="loading-spinner">
                    <div class="spinner"></div>
                    <span>${message}</span>
                </div>
            `;
        }
    }

    /**
     * 隐藏加载状态
     */
    static hideLoading(elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = '';
        }
    }

    /**
     * 显示错误消息
     */
    static showError(elementId, error) {
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = `
                <div class="error-message">
                    <strong>错误:</strong> ${error.message || error}
                </div>
            `;
        }
    }

    /**
     * 更新指标显示
     */
    static updateMetrics(metrics, containerSelector) {
        const container = document.querySelector(containerSelector);
        if (!container) return;

        let html = '';
        for (const [key, value] of Object.entries(metrics)) {
            const displayName = this.getMetricDisplayName(key);
            const displayValue = typeof value === 'number' ? value.toFixed(4) : value;
            html += `
                <div class="metric-item">
                    <span class="metric-label">${displayName}:</span>
                    <span class="metric-value">${displayValue}</span>
                </div>
            `;
        }
        
        container.innerHTML = html;
    }

    /**
     * 获取指标的显示名称
     */
    static getMetricDisplayName(metricKey) {
        const displayNames = {
            'train_accuracy': '训练准确率',
            'test_accuracy': '测试准确率',
            'train_rmse': '训练RMSE',
            'test_rmse': '测试RMSE',
            'train_r2': '训练R²',
            'test_r2': '测试R²',
            'silhouette_score': '轮廓系数',
            'n_clusters': '聚类数量',
            'inertia': '惯性',
            'mse': 'MSE',
            'mae': 'MAE',
            'r2': 'R²分数'
        };
        
        return displayNames[metricKey] || metricKey;
    }

    /**
     * 数据预处理：标准化
     */
    static normalizeData(X) {
        if (X.length === 0) return X;
        
        const features = X[0].length;
        const means = new Array(features).fill(0);
        const stds = new Array(features).fill(0);
        
        // 计算均值
        for (let i = 0; i < X.length; i++) {
            for (let j = 0; j < features; j++) {
                means[j] += X[i][j];
            }
        }
        for (let j = 0; j < features; j++) {
            means[j] /= X.length;
        }
        
        // 计算标准差
        for (let i = 0; i < X.length; i++) {
            for (let j = 0; j < features; j++) {
                stds[j] += Math.pow(X[i][j] - means[j], 2);
            }
        }
        for (let j = 0; j < features; j++) {
            stds[j] = Math.sqrt(stds[j] / X.length);
        }
        
        // 标准化
        const normalizedX = X.map(row => 
            row.map((val, j) => 
                stds[j] === 0 ? 0 : (val - means[j]) / stds[j]
            )
        );
        
        return {
            X: normalizedX,
            means: means,
            stds: stds
        };
    }
}

// 添加样式
const style = document.createElement('style');
style.textContent = `
    .loading-spinner {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 20px;
        color: #6B7890;
    }
    
    .spinner {
        width: 20px;
        height: 20px;
        border: 2px solid #f3f3f3;
        border-top: 2px solid #2B8AC6;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-right: 10px;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .error-message {
        padding: 15px;
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
        border-radius: 4px;
        margin: 10px 0;
    }
    
    .metric-item {
        display: flex;
        justify-content: space-between;
        margin-bottom: 8px;
        padding: 5px 10px;
        background-color: #F8FAFD;
        border-radius: 4px;
    }
    
    .metric-label {
        font-weight: 500;
        color: #6B7890;
    }
    
    .metric-value {
        font-weight: 600;
        color: #172033;
    }
`;
document.head.appendChild(style);

// 导出类（如果使用模块系统）
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { MLApiClient, MLVisualizer, MLUtils };
}

// 为向后兼容提供别名
window.APIClient = MLApiClient;
