(function () {
    'use strict';

    const state = {
        client: null,
        excelFile: null,
        excelPreview: null,
        currentData: null,
        sessionKey: null,
        modelKey: null,
        results: null
    };

    document.addEventListener('DOMContentLoaded', init);

    function init() {
        if (typeof MLApiClient === 'undefined') return;
        state.client = new MLApiClient();
        bindEvents();
        updateControlVisibility();
        updateSliderLabels();
        drawEmptyChart(document.getElementById('regressionCanvas'), '请先准备数据');
        checkConnection();
    }

    function bindEvents() {
        document.getElementById('generateDataBtn').addEventListener('click', generateSyntheticData);
        document.getElementById('previewExcelBtn').addEventListener('click', previewSelectedExcel);
        document.getElementById('useExampleBtn').addEventListener('click', useExampleData);
        document.getElementById('excelSheet').addEventListener('change', () => renderSelectedSheet(false));
        document.getElementById('targetColumn').addEventListener('change', syncFeatureChoices);
        document.getElementById('importExcelBtn').addEventListener('click', importExcelData);
        document.getElementById('algorithmType').addEventListener('change', updateControlVisibility);
        document.getElementById('optimizer').addEventListener('change', updateControlVisibility);
        document.getElementById('noiseLevel').addEventListener('input', updateSliderLabels);
        document.getElementById('testSize').addEventListener('input', () => {
            updateSliderLabels();
            updateDataStats();
        });
        document.getElementById('trainModelBtn').addEventListener('click', trainModel);
        document.getElementById('resetModelBtn').addEventListener('click', resetModel);
        document.getElementById('clearDataBtn').addEventListener('click', clearData);
        document.getElementById('clearSessionBtn').addEventListener('click', clearSession);
        document.getElementById('excelFile').addEventListener('change', (event) => {
            state.excelFile = event.target.files[0] || null;
            state.excelPreview = null;
            document.getElementById('excelConfigPanel').hidden = true;
        });
        window.addEventListener('resize', () => {
            if (state.currentData) renderVisualizations();
        });
    }

    async function checkConnection() {
        const banner = document.getElementById('statusBanner');
        try {
            if (!await state.client.checkConnection()) throw new Error('连接失败');
            banner.textContent = '后端服务已连接';
            banner.className = 'status-banner connected';
        } catch (error) {
            banner.textContent = '后端服务未连接，请先启动 Flask 服务';
            banner.className = 'status-banner disconnected';
        }
    }

    function updateSliderLabels() {
        document.getElementById('noiseLevelValue').textContent =
            Number(document.getElementById('noiseLevel').value).toFixed(2);
        document.getElementById('testSizeValue').textContent =
            Math.round(Number(document.getElementById('testSize').value) * 100) + '%';
    }

    function updateControlVisibility() {
        const algorithm = document.getElementById('algorithmType').value;
        const optimizer = document.getElementById('optimizer').value;
        document.getElementById('polynomialControls').hidden = algorithm !== 'polynomial';
        document.getElementById('regularizationControls').hidden =
            algorithm !== 'ridge' && algorithm !== 'lasso';
        document.getElementById('sgdControls').hidden = optimizer !== 'sgd';
    }

    async function generateSyntheticData() {
        setBusy('generateDataBtn', true, '正在生成...');
        try {
            const result = await state.client.generateData({
                type: 'regression',
                shape: document.getElementById('dataShape').value,
                n_samples: Number(document.getElementById('numSamples').value),
                noise: Number(document.getElementById('noiseLevel').value),
                random_state: 42
            });
            applyDataset(result, '已生成合成数据。');
        } catch (error) {
            showNotice(error.message || '生成数据失败', true);
        } finally {
            setBusy('generateDataBtn', false);
        }
    }

    async function previewSelectedExcel() {
        const inputFile = document.getElementById('excelFile').files[0];
        if (inputFile) state.excelFile = inputFile;
        if (!state.excelFile) {
            showNotice('请先选择一个 .xlsx 文件。', true);
            return;
        }
        await previewExcel(false);
    }

    async function useExampleData() {
        setBusy('useExampleBtn', true, '正在载入...');
        try {
            const response = await fetch('/static/examples/linear_regression_example.xlsx');
            if (!response.ok) throw new Error('示例文件下载失败');
            const blob = await response.blob();
            state.excelFile = new File([blob], 'linear_regression_example.xlsx', {
                type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            });
            await previewExcel(true);
        } catch (error) {
            showNotice(error.message || '无法载入示例数据', true);
        } finally {
            setBusy('useExampleBtn', false);
        }
    }

    async function previewExcel(isExample) {
        if (!state.excelFile.name.toLowerCase().endsWith('.xlsx')) {
            showNotice('仅支持 .xlsx 格式的 Excel 文件。', true);
            return;
        }
        if (state.excelFile.size > 5 * 1024 * 1024) {
            showNotice('Excel 文件不能超过 5 MB。', true);
            return;
        }

        setBusy('previewExcelBtn', true, '正在解析...');
        try {
            state.excelPreview = await state.client.previewLinearRegressionExcel(state.excelFile);
            const sheetSelect = document.getElementById('excelSheet');
            sheetSelect.replaceChildren(...state.excelPreview.sheets.map((sheet) => {
                const option = document.createElement('option');
                option.value = sheet.name;
                option.textContent = sheet.name;
                return option;
            }));
            document.getElementById('excelConfigPanel').hidden = false;
            document.getElementById('excelFileSummary').textContent =
                `${state.excelPreview.filename} · ${state.excelPreview.sheets.length} 个工作表`;
            renderSelectedSheet(isExample);
            showNotice('文件解析完成，请确认工作表、特征列和目标列。');
            document.getElementById('excelConfigPanel').scrollIntoView({ behavior: 'smooth', block: 'start' });
        } catch (error) {
            showNotice(error.message || 'Excel 预览失败', true);
        } finally {
            setBusy('previewExcelBtn', false);
        }
    }

    function selectedSheet() {
        if (!state.excelPreview) return null;
        const name = document.getElementById('excelSheet').value;
        return state.excelPreview.sheets.find((sheet) => sheet.name === name) || null;
    }

    function renderSelectedSheet(preferExampleColumns) {
        const sheet = selectedSheet();
        if (!sheet) return;

        const tableWrap = document.getElementById('excelPreviewTable');
        const head = sheet.columns.map((column) => `<th>${escapeHtml(column)}</th>`).join('');
        const body = sheet.preview.map((row) => {
            const cells = sheet.columns.map((column) => {
                const value = row[column] === null || row[column] === undefined ? '' : String(row[column]);
                return `<td>${escapeHtml(value)}</td>`;
            }).join('');
            return `<tr>${cells}</tr>`;
        }).join('');
        tableWrap.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;

        const numericColumns = sheet.numeric_columns || [];
        const target = preferExampleColumns && numericColumns.includes('房价')
            ? '房价'
            : numericColumns[numericColumns.length - 1];
        const targetSelect = document.getElementById('targetColumn');
        targetSelect.replaceChildren(...numericColumns.map((column) => {
            const option = document.createElement('option');
            option.value = column;
            option.textContent = column;
            option.selected = column === target;
            return option;
        }));

        const featureOptions = document.getElementById('featureColumnOptions');
        featureOptions.replaceChildren(...numericColumns.map((column) => {
            const label = document.createElement('label');
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.name = 'featureColumn';
            checkbox.value = column;
            checkbox.checked = column !== target;
            const invalidCount = sheet.invalid_counts && sheet.invalid_counts[column] || 0;
            const text = document.createElement('span');
            text.textContent = invalidCount ? `${column}（${invalidCount} 个非数值）` : column;
            label.append(checkbox, text);
            return label;
        }));
        syncFeatureChoices();
        document.getElementById('excelFileSummary').textContent =
            `${state.excelPreview.filename} · 工作表“${sheet.name}” · ${sheet.total_rows} 行`;
    }

    function syncFeatureChoices() {
        const target = document.getElementById('targetColumn').value;
        document.querySelectorAll('input[name="featureColumn"]').forEach((checkbox) => {
            checkbox.disabled = checkbox.value === target;
            if (checkbox.disabled) checkbox.checked = false;
        });
    }

    async function importExcelData() {
        const sheet = selectedSheet();
        const target = document.getElementById('targetColumn').value;
        const features = Array.from(document.querySelectorAll('input[name="featureColumn"]:checked'))
            .map((checkbox) => checkbox.value);
        if (!sheet || !target || features.length === 0) {
            showNotice('请选择至少一个特征列和一个目标列。', true);
            return;
        }

        setBusy('importExcelBtn', true, '正在导入...');
        try {
            const result = await state.client.importLinearRegressionExcel(
                state.excelFile, sheet.name, features, target
            );
            const summary = result.data.import_summary;
            const message = summary.dropped_rows
                ? `导入 ${summary.valid_rows} 行，已过滤 ${summary.dropped_rows} 行无效数据。`
                : `已成功导入 ${summary.valid_rows} 行数据。`;
            applyDataset(result, message);
        } catch (error) {
            showNotice(error.message || 'Excel 导入失败', true);
        } finally {
            setBusy('importExcelBtn', false);
        }
    }

    function applyDataset(result, message) {
        state.currentData = result.data;
        state.sessionKey = result.session_key;
        state.modelKey = null;
        state.results = null;
        document.getElementById('sessionKey').textContent = state.sessionKey;
        document.getElementById('trainingStatus').textContent = '数据已准备';
        document.getElementById('trainModelBtn').disabled = false;
        resetResultDisplay();
        updateDataStats();
        renderVisualizations();
        showNotice(message);
    }

    function updateDataStats() {
        const count = state.currentData ? state.currentData.X.length : 0;
        const featureCount = count ? state.currentData.X[0].length : 0;
        const testSize = Number(document.getElementById('testSize').value);
        document.getElementById('dataCount').textContent = count;
        document.getElementById('featureCount').textContent = featureCount;
        document.getElementById('trainSize').textContent = Math.floor(count * (1 - testSize));
    }

    async function trainModel() {
        if (!state.sessionKey) {
            showNotice('请先准备数据。', true);
            return;
        }
        const algorithm = document.getElementById('algorithmType').value;
        const optimizer = document.getElementById('optimizer').value;
        const params = {
            test_size: Number(document.getElementById('testSize').value),
            fit_intercept: true,
            optimizer
        };
        if (algorithm === 'polynomial') {
            params.polynomial_degree = Number(document.getElementById('polynomialDegree').value);
        }
        if (algorithm === 'ridge' || algorithm === 'lasso') {
            params.alpha = Number(document.getElementById('alpha').value);
        }
        if (optimizer === 'sgd') {
            params.eta0 = Number(document.getElementById('learningRate').value);
            params.max_iter = Number(document.getElementById('maxIter').value);
            params.learning_rate = 'invscaling';
            params.tol = 1e-3;
        }

        setBusy('trainModelBtn', true, '正在训练...');
        document.getElementById('trainingStatus').textContent = '训练中';
        try {
            const result = await state.client.trainLinearRegression({
                session_key: state.sessionKey,
                algorithm,
                params
            });
            state.modelKey = result.model_key;
            state.results = result.results;
            document.getElementById('trainingStatus').textContent = '训练完成';
            document.getElementById('modelAlgorithm').textContent =
                `${algorithmName(algorithm)} · ${optimizerName(optimizer)}`;
            document.getElementById('regressionEquation').textContent =
                result.results.model_equation || 'y = ?';
            renderMetrics(result.results.metrics);
            renderVisualizations();
            showNotice('模型训练完成。');
        } catch (error) {
            document.getElementById('trainingStatus').textContent = '训练失败';
            showNotice(error.message || '模型训练失败', true);
        } finally {
            setBusy('trainModelBtn', false);
        }
    }

    function renderMetrics(metrics) {
        const labels = {
            train_mse: '训练 MSE',
            test_mse: '测试 MSE',
            train_rmse: '训练 RMSE',
            test_rmse: '测试 RMSE',
            train_mae: '训练 MAE',
            test_mae: '测试 MAE',
            train_r2: '训练 R²',
            test_r2: '测试 R²',
            train_adjusted_r2: '训练调整 R²',
            test_adjusted_r2: '测试调整 R²'
        };
        const preferred = Object.keys(labels).filter((key) => key in metrics);
        document.getElementById('metricsDisplay').innerHTML = preferred.map((key) => {
            const value = Number(metrics[key]);
            return `<div class="lr-metric"><span>${labels[key]}</span><strong>${formatNumber(value)}</strong></div>`;
        }).join('');
    }

    function renderVisualizations() {
        const canvas = document.getElementById('regressionCanvas');
        if (!state.currentData || !state.currentData.X.length) {
            drawEmptyChart(canvas, '请先准备数据');
            return;
        }
        const featureCount = state.currentData.X[0].length;
        const targetName = state.currentData.feature_info && state.currentData.feature_info.target || '目标值';
        const featureName = state.currentData.feature_info &&
            state.currentData.feature_info.features_used &&
            state.currentData.feature_info.features_used[0] || '特征值';

        if (!state.results) {
            document.getElementById('diagnosticCharts').hidden = true;
            if (featureCount === 1) {
                document.getElementById('primaryChartTitle').textContent = '数据分布';
                document.getElementById('canvasInfo').textContent = `${featureName} 与 ${targetName}`;
                drawScatter(canvas, state.currentData.X.map((row, index) => ({
                    x: row[0], y: state.currentData.y[index], split: 'data'
                })), { xLabel: featureName, yLabel: targetName });
            } else {
                document.getElementById('primaryChartTitle').textContent = '多特征数据已准备';
                document.getElementById('canvasInfo').textContent =
                    `已选择 ${featureCount} 个特征，训练后显示预测诊断图。`;
                drawEmptyChart(canvas, '训练后显示真实值与预测值');
            }
            return;
        }

        const points = state.results.point_results || [];
        if (featureCount === 1) {
            document.getElementById('primaryChartTitle').textContent = '数据与拟合曲线';
            document.getElementById('canvasInfo').textContent = '蓝色为观测值，紫色为模型拟合曲线。';
            drawSingleFeatureFit(canvas, points, state.results.prediction_curve, featureName, targetName);
        } else {
            document.getElementById('primaryChartTitle').textContent = '真实值与预测值';
            document.getElementById('canvasInfo').textContent =
                '点越接近虚线，预测越接近真实值；颜色区分训练集和测试集。';
            drawScatter(canvas, points.map((point) => ({
                x: point.actual,
                y: point.predicted,
                split: point.split
            })), { xLabel: '真实值', yLabel: '预测值', idealLine: true });
        }

        document.getElementById('diagnosticCharts').hidden = false;
        drawScatter(document.getElementById('residualCanvas'), points.map((point) => ({
            x: point.predicted,
            y: point.residual,
            split: point.split
        })), { xLabel: '预测值', yLabel: '残差', zeroLine: true });
        drawCoefficients(state.results.training_info);
    }

    function drawSingleFeatureFit(canvas, points, curve, xLabel, yLabel) {
        const scatter = points.map((point) => ({
            x: point.features[0], y: point.actual, split: point.split
        }));
        const line = curve && curve.x ? curve.x.map((x, index) => ({ x, y: curve.y[index] })) : [];
        drawScatter(canvas, scatter, { xLabel, yLabel, line });
    }

    function drawScatter(canvas, points, options) {
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        const pad = { left: 72, right: 24, top: 28, bottom: 58 };
        const linePoints = options.line || [];
        const xs = points.map((point) => point.x).concat(linePoints.map((point) => point.x));
        const ys = points.map((point) => point.y).concat(linePoints.map((point) => point.y));
        if (options.idealLine) {
            const all = xs.concat(ys);
            xs.push(...all);
            ys.push(...all);
        }
        const xRange = paddedRange(xs);
        const yRange = paddedRange(ys);
        const sx = (value) => pad.left + (value - xRange.min) / (xRange.max - xRange.min) * (width - pad.left - pad.right);
        const sy = (value) => height - pad.bottom - (value - yRange.min) / (yRange.max - yRange.min) * (height - pad.top - pad.bottom);

        ctx.clearRect(0, 0, width, height);
        drawGrid(ctx, width, height, pad, xRange, yRange, options.xLabel, options.yLabel);
        if (options.idealLine) {
            const low = Math.max(xRange.min, yRange.min);
            const high = Math.min(xRange.max, yRange.max);
            drawReferenceLine(ctx, sx(low), sy(low), sx(high), sy(high), '#94a3b8', [7, 6]);
        }
        if (options.zeroLine && yRange.min <= 0 && yRange.max >= 0) {
            drawReferenceLine(ctx, pad.left, sy(0), width - pad.right, sy(0), '#94a3b8', [7, 6]);
        }
        if (linePoints.length) {
            ctx.save();
            ctx.strokeStyle = '#4f46e5';
            ctx.lineWidth = 3;
            ctx.beginPath();
            linePoints.forEach((point, index) => {
                if (index === 0) ctx.moveTo(sx(point.x), sy(point.y));
                else ctx.lineTo(sx(point.x), sy(point.y));
            });
            ctx.stroke();
            ctx.restore();
        }
        points.forEach((point) => {
            ctx.beginPath();
            ctx.arc(sx(point.x), sy(point.y), point.split === 'test' ? 5 : 4, 0, Math.PI * 2);
            ctx.fillStyle = point.split === 'test' ? '#c2415d' : point.split === 'train' ? '#0f9d8a' : '#4f46e5';
            ctx.globalAlpha = .78;
            ctx.fill();
        });
        ctx.globalAlpha = 1;
        if (points.some((point) => point.split === 'train')) drawLegend(ctx, width);
    }

    function drawGrid(ctx, width, height, pad, xRange, yRange, xLabel, yLabel) {
        ctx.save();
        ctx.font = '12px sans-serif';
        ctx.fillStyle = '#64748b';
        ctx.strokeStyle = '#e7edf5';
        ctx.lineWidth = 1;
        for (let index = 0; index <= 5; index += 1) {
            const x = pad.left + index / 5 * (width - pad.left - pad.right);
            const y = pad.top + index / 5 * (height - pad.top - pad.bottom);
            ctx.beginPath();
            ctx.moveTo(x, pad.top);
            ctx.lineTo(x, height - pad.bottom);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(pad.left, y);
            ctx.lineTo(width - pad.right, y);
            ctx.stroke();
            const xValue = xRange.min + index / 5 * (xRange.max - xRange.min);
            const yValue = yRange.max - index / 5 * (yRange.max - yRange.min);
            ctx.textAlign = 'center';
            ctx.fillText(shortNumber(xValue), x, height - pad.bottom + 20);
            ctx.textAlign = 'right';
            ctx.fillText(shortNumber(yValue), pad.left - 10, y + 4);
        }
        ctx.strokeStyle = '#94a3b8';
        ctx.strokeRect(pad.left, pad.top, width - pad.left - pad.right, height - pad.top - pad.bottom);
        ctx.fillStyle = '#44516a';
        ctx.font = '600 13px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(xLabel, pad.left + (width - pad.left - pad.right) / 2, height - 13);
        ctx.save();
        ctx.translate(18, pad.top + (height - pad.top - pad.bottom) / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText(yLabel, 0, 0);
        ctx.restore();
        ctx.restore();
    }

    function drawReferenceLine(ctx, x1, y1, x2, y2, color, dash) {
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.setLineDash(dash);
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
        ctx.restore();
    }

    function drawLegend(ctx, width) {
        ctx.save();
        ctx.font = '12px sans-serif';
        [['训练集', '#0f9d8a'], ['测试集', '#c2415d']].forEach(([label, color], index) => {
            const x = width - 150 + index * 70;
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(x, 15, 4, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillStyle = '#64748b';
            ctx.fillText(label, x + 8, 19);
        });
        ctx.restore();
    }

    function drawCoefficients(trainingInfo) {
        const canvas = document.getElementById('coefficientCanvas');
        const coefficients = (trainingInfo.coefficients || []).map(Number);
        const names = trainingInfo.feature_names || coefficients.map((_, index) => `x${index + 1}`);
        const ranked = coefficients.map((value, index) => ({ value, name: names[index] || `x${index + 1}` }))
            .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
        const visible = ranked.slice(0, 20);
        canvas.height = Math.max(360, visible.length * 28 + 70);
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        ctx.clearRect(0, 0, width, height);
        if (!visible.length) {
            drawEmptyChart(canvas, '没有可展示的系数');
            return;
        }
        const labelWidth = Math.min(210, Math.max(100, ...visible.map((item) => item.name.length * 12)));
        const left = labelWidth + 20;
        const right = 30;
        const maxAbs = Math.max(...visible.map((item) => Math.abs(item.value)), 1e-12);
        const zeroX = left + (width - left - right) / 2;
        ctx.strokeStyle = '#94a3b8';
        ctx.beginPath();
        ctx.moveTo(zeroX, 25);
        ctx.lineTo(zeroX, height - 25);
        ctx.stroke();
        ctx.font = '12px sans-serif';
        visible.forEach((item, index) => {
            const y = 42 + index * 28;
            const barWidth = Math.abs(item.value) / maxAbs * (width - left - right) / 2;
            ctx.textAlign = 'right';
            ctx.fillStyle = '#44516a';
            ctx.fillText(truncate(item.name, 26), left - 10, y + 4);
            ctx.fillStyle = item.value >= 0 ? '#4f46e5' : '#c2415d';
            ctx.fillRect(item.value >= 0 ? zeroX : zeroX - barWidth, y - 8, barWidth, 16);
        });
        document.getElementById('coefficientNote').textContent = ranked.length > 20
            ? `共 ${ranked.length} 个系数，当前显示绝对值最大的 20 项。系数大小受特征单位影响。`
            : '系数方向表示正负关系；系数大小受特征单位影响。';
    }

    function drawEmptyChart(canvas, message) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#f8fafd';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#94a3b8';
        ctx.font = '15px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(message, canvas.width / 2, canvas.height / 2);
    }

    function paddedRange(values) {
        const finite = values.map(Number).filter(Number.isFinite);
        if (!finite.length) return { min: 0, max: 1 };
        let min = Math.min(...finite);
        let max = Math.max(...finite);
        const span = max - min || Math.max(Math.abs(max), 1);
        min -= span * .08;
        max += span * .08;
        return { min, max };
    }

    function resetModel() {
        state.modelKey = null;
        state.results = null;
        document.getElementById('trainingStatus').textContent = state.currentData ? '数据已准备' : '未开始';
        resetResultDisplay();
        renderVisualizations();
    }

    function clearData() {
        state.currentData = null;
        state.sessionKey = null;
        state.modelKey = null;
        state.results = null;
        document.getElementById('sessionKey').textContent = '-';
        document.getElementById('trainingStatus').textContent = '未开始';
        document.getElementById('trainModelBtn').disabled = true;
        document.getElementById('diagnosticCharts').hidden = true;
        resetResultDisplay();
        updateDataStats();
        drawEmptyChart(document.getElementById('regressionCanvas'), '请先准备数据');
        showNotice('页面数据已清空。');
    }

    async function clearSession() {
        try {
            await state.client.clearSession();
            clearData();
            showNotice('全部服务端会话已清除。');
        } catch (error) {
            showNotice(error.message || '清除会话失败', true);
        }
    }

    function resetResultDisplay() {
        document.getElementById('regressionEquation').textContent = 'y = ?';
        document.getElementById('metricsDisplay').innerHTML = '<span class="lr-muted">等待训练...</span>';
        document.getElementById('modelAlgorithm').textContent = '-';
    }

    function setBusy(buttonId, busy, busyText) {
        const button = document.getElementById(buttonId);
        if (!button) return;
        if (busy) {
            button.dataset.originalText = button.textContent;
            button.textContent = busyText;
            button.disabled = true;
        } else {
            button.textContent = button.dataset.originalText || button.textContent;
            button.disabled = false;
        }
    }

    function showNotice(message, isError) {
        const notice = document.getElementById('dataNotice');
        notice.textContent = message;
        notice.classList.toggle('error', Boolean(isError));
        notice.hidden = false;
    }

    function algorithmName(value) {
        return { linear: '线性回归', ridge: '岭回归', lasso: 'Lasso 回归', polynomial: '多项式回归' }[value] || value;
    }

    function optimizerName(value) {
        return { auto: '自动', ols: '普通最小二乘', sgd: '随机梯度下降', normal_equation: '正则方程' }[value] || value;
    }

    function formatNumber(value) {
        if (!Number.isFinite(value)) return '-';
        if (Math.abs(value) >= 10000 || (Math.abs(value) > 0 && Math.abs(value) < .001)) return value.toExponential(3);
        return value.toFixed(4);
    }

    function shortNumber(value) {
        if (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < .01)) return value.toExponential(1);
        return Number(value.toFixed(2)).toString();
    }

    function truncate(value, length) {
        return value.length > length ? value.slice(0, length - 1) + '…' : value;
    }

    function escapeHtml(value) {
        return String(value)
            .replaceAll('&', '&amp;')
            .replaceAll('<', '&lt;')
            .replaceAll('>', '&gt;')
            .replaceAll('"', '&quot;')
            .replaceAll("'", '&#039;');
    }
})();
