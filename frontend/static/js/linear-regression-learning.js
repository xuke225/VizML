(function () {
    'use strict';

    const STAGES = [
        { title: VizMLI18n.t('调整直线'), short: VizMLI18n.t('猜一条线') },
        { title: VizMLI18n.t('理解残差'), short: VizMLI18n.t('误差有多大') },
        { title: VizMLI18n.t('寻找最低点'), short: VizMLI18n.t('模型怎样学习') },
        { title: VizMLI18n.t('检验泛化'), short: VizMLI18n.t('面对新数据') }
    ];

    const state = {
        stage: Number(sessionStorage.getItem('vizml-lr-stage') || 1),
        client: null,
        loading: false,
        error: '',
        sessionKey: null,
        data: [],
        results: null,
        w: 0,
        b: 0,
        optimumW: 0,
        optimumB: 0,
        wMin: -5,
        wMax: 5,
        bMin: -5,
        bMax: 5,
        showBest: false,
        errorMode: 'squared',
        revealTest: false,
        scenario: 'linear',
        seed: 42,
        gradientPath: [],
        gradientTimer: null,
        gradientSteps: 0,
        answers: {},
        resizeTimer: null
    };

    document.addEventListener('DOMContentLoaded', init);

    function init() {
        const panel = document.getElementById('guidedModePanel');
        if (!panel || typeof MLApiClient === 'undefined') return;

        state.stage = Math.max(1, Math.min(4, state.stage));
        state.client = new MLApiClient();
        bindModeSwitch();
        window.addEventListener('resize', onResize);
        loadDataset('linear');
    }

    function bindModeSwitch() {
        const buttons = document.querySelectorAll('[data-learning-mode]');
        buttons.forEach((button) => {
            button.addEventListener('click', () => {
                const mode = button.dataset.learningMode;
                buttons.forEach((candidate) => {
                    const active = candidate === button;
                    candidate.classList.toggle('active', active);
                    candidate.setAttribute('aria-selected', String(active));
                });
                document.getElementById('guidedModePanel').hidden = mode !== 'guided';
                document.getElementById('freeModePanel').hidden = mode !== 'free';
                if (mode === 'guided') requestAnimationFrame(drawActiveStage);
            });
        });
    }

    async function loadDataset(scenario, useNewSeed) {
        stopGradient();
        state.loading = true;
        state.error = '';
        state.scenario = scenario;
        state.revealTest = false;
        state.showBest = false;
        state.gradientPath = [];
        state.gradientSteps = 0;
        if (useNewSeed) state.seed += 1;
        render();

        const scenarioConfig = {
            linear: { shape: 'linear', noise: 0.12 },
            noisy: { shape: 'linear', noise: 0.5 },
            outliers: { shape: 'outliers', noise: 0.18 },
            nonlinear: { shape: 'sinusoidal', noise: 0.12 }
        }[scenario];

        try {
            const generated = await state.client.generateData({
                type: 'regression',
                shape: scenarioConfig.shape,
                n_samples: 30,
                noise: scenarioConfig.noise,
                random_state: state.seed,
                session_namespace: 'linear_learning'
            });
            const trained = await state.client.trainLinearRegression({
                session_key: generated.session_key,
                algorithm: 'linear',
                params: { test_size: 0.25, fit_intercept: true, optimizer: 'ols' }
            });

            state.sessionKey = generated.session_key;
            state.results = trained.results;
            state.data = trained.results.point_results || fallbackPoints(generated.data, trained.results);
            const coefficients = trained.results.training_info.coefficients;
            state.optimumW = Number(coefficients[0] || 0);
            state.optimumB = Number(trained.results.training_info.intercept || 0);
            setParameterRanges();
            state.w = clamp(0, state.wMin, state.wMax);
            state.b = clamp(mean(trainingPoints().map((point) => point.actual)), state.bMin, state.bMax);
            state.loading = false;
            render();
        } catch (error) {
            state.loading = false;
            state.error = error.message || VizMLI18n.t('无法加载教学数据');
            render();
        }
    }

    function fallbackPoints(data, results) {
        const predictions = data.X.map((features) => {
            return results.training_info.intercept + features[0] * results.training_info.coefficients[0];
        });
        return data.X.map((features, index) => ({
            index,
            features,
            actual: data.y[index],
            predicted: predictions[index],
            residual: data.y[index] - predictions[index],
            squared_error: Math.pow(data.y[index] - predictions[index], 2),
            split: index % 4 === 0 ? 'test' : 'train'
        }));
    }

    function setParameterRanges() {
        const points = trainingPoints();
        const xs = points.map((point) => point.features[0]);
        const ys = points.map((point) => point.actual);
        const xRange = Math.max(max(xs) - min(xs), 0.001);
        const yRange = Math.max(max(ys) - min(ys), 0.001);
        const slopeSpan = Math.max(yRange / xRange * 1.8, Math.abs(state.optimumW) * 1.2, 1);
        const interceptSpan = Math.max(yRange * 0.85, Math.abs(state.optimumB) * 1.2, 1);
        state.wMin = state.optimumW - slopeSpan;
        state.wMax = state.optimumW + slopeSpan;
        state.bMin = state.optimumB - interceptSpan;
        state.bMax = state.optimumB + interceptSpan;
    }

    function render() {
        const panel = document.getElementById('guidedModePanel');
        if (!panel) return;

        if (state.loading) {
            panel.innerHTML = VizMLI18n.t('<div class="learning-complete"><h2>正在准备教学数据…</h2><p>生成数据并计算作为参照的最佳直线。</p></div>');
            return;
        }

        if (state.error) {
            panel.innerHTML = '<div class="learning-complete">' + '<h2>' + VizMLI18n.t('教学实验暂时无法加载') + '</h2>' + '<p>' + (escapeHtml(state.error)) + '</p>' + '<button type="button" id="learningRetry">' + VizMLI18n.t('重试') + '</button>' + '</div>';
            document.getElementById('learningRetry').addEventListener('click', () => loadDataset(state.scenario));
            return;
        }

        panel.innerHTML = '\n            ' + '<div class="lr-learning-lab">' + '\n                ' + '<div class="learning-intro">' + '\n                    ' + '<div>' + '\n                        ' + '<h2>' + VizMLI18n.t('用四个实验看懂线性回归') + '</h2>' + '\n                        ' + '<p>' + VizMLI18n.t('先猜一条直线，再亲眼看看模型为什么会选择“误差最小”的那一条。') + '</p>' + '\n                    ' + '</div>' + '\n                    ' + '<button type="button" class="learning-new-data" id="learningNewData">' + VizMLI18n.t('换一组数据') + '</button>' + '\n                ' + '</div>' + '\n                ' + (progressHtml()) + '\n                ' + (stageHtml()) + '\n            ' + '</div>';

        bindCommonEvents();
        bindStageEvents();
        requestAnimationFrame(drawActiveStage);
    }

    function progressHtml() {
        return '<div class="learning-progress" aria-label="学习进度">' + (STAGES.map((item, index) => {
            const stage = index + 1;
            const classes = ['learning-step'];
            if (stage === state.stage) classes.push('active');
            if (stage < state.stage || state.answers[stage]) classes.push('complete');
            return '<button type="button" class="' + (classes.join(' ')) + '" data-go-stage="' + (stage) + '" aria-current="' + (stage === state.stage ? 'step' : 'false') + '">' + '<span>' + (stage) + '</span>' + (item.short) + '</button>';
        }).join('')) + '</div>';
    }

    function stageHtml() {
        if (state.stage === 1) return stageOneHtml();
        if (state.stage === 2) return stageTwoHtml();
        if (state.stage === 3) return stageThreeHtml();
        return stageFourHtml();
    }

    function stageOneHtml() {
        return '\n            ' + '<div class="learning-stage">' + '\n                ' + (visualCard(VizMLI18n.t('实验一：亲手调整一条直线'), VizMLI18n.t('移动斜率和截距，试着让直线靠近尽可能多的数据点。'), true)) + '\n                ' + '<aside class="learning-side-card">' + '\n                    ' + (equationHtml()) + '\n                    ' + (parameterControlsHtml()) + '\n                    ' + (lossStatsHtml(currentLineStats(trainingPoints()))) + '\n                    ' + '<div class="learning-callout">' + VizMLI18n.t('斜率决定直线倾斜的方向和程度；截距决定直线在纵轴上的起点。你也可以直接上下拖动图中的直线。') + '</div>' + '\n                    ' + '<div class="learning-actions">' + '\n                        ' + '<button type="button" class="learning-secondary" id="resetGuess">' + VizMLI18n.t('重置猜测') + '</button>' + '\n                        ' + '<button type="button" id="showBest">' + (state.showBest ? VizMLI18n.t('继续自己调整') : VizMLI18n.t('显示最佳直线')) + '</button>' + '\n                    ' + '</div>' + '\n                    ' + (formulaHtml(VizMLI18n.t('ŷ = wx + b。给定一个 x，直线就会产生预测值 ŷ。w 每增加 1，x 每变化 1 时，预测值的变化量也随之改变。'))) + '\n                ' + '</aside>' + '\n                ' + (checkpointHtml(1, VizMLI18n.t('如果斜率 w 从正数变成负数，直线会怎样？'), [
                    [VizMLI18n.t('向右上升'), 'up'], [VizMLI18n.t('向右下降'), 'down'], [VizMLI18n.t('只会上下平移'), 'move']
                ], 'down', VizMLI18n.t('负斜率表示 x 增大时，预测值反而减小，所以直线向右下降。'))) + '\n            ' + '</div>';
    }

    function stageTwoHtml() {
        const stats = currentLineStats(trainingPoints());
        return '\n            ' + '<div class="learning-stage">' + '\n                ' + (visualCard(VizMLI18n.t('实验二：误差是怎样被计算的'), VizMLI18n.t('每条竖线都是一个残差。悬停数据点，查看这个点为总误差贡献了多少。'), true)) + '\n                ' + '<aside class="learning-side-card">' + '\n                    ' + (equationHtml()) + '\n                    ' + (parameterControlsHtml()) + '\n                    ' + '<div class="learning-control">' + '\n                        ' + '<label>' + VizMLI18n.t('误差计算方式') + '</label>' + '\n                        ' + '<div class="learning-actions">' + '\n                            <button type="button" data-error-mode="squared" class="' + (state.errorMode === 'squared' ? '' : 'learning-secondary') + VizMLI18n.t('">平方误差') + '</button>' + '\n                            <button type="button" data-error-mode="absolute" class="' + (state.errorMode === 'absolute' ? '' : 'learning-secondary') + VizMLI18n.t('">绝对误差') + '</button>' + '\n                        ' + '</div>' + '\n                    ' + '</div>' + '\n                    ' + (lossStatsHtml(stats, state.errorMode)) + '\n                    ' + '<div class="learning-callout">' + VizMLI18n.t('残差 = 真实值 − 预测值。正残差在直线上方，负残差在直线下方；平方后它们都成为正数。') + '</div>' + '\n                    ' + (formulaHtml(VizMLI18n.t('残差 eᵢ = yᵢ − ŷᵢ。平方误差和 SSE = Σeᵢ²，均方误差 MSE = SSE / n。平方会放大特别大的错误。'))) + '\n                ' + '</aside>' + '\n                ' + (checkpointHtml(2, VizMLI18n.t('为什么正残差和负残差不能直接相加衡量模型好坏？'), [
                    [VizMLI18n.t('它们可能互相抵消'), 'cancel'], [VizMLI18n.t('计算机会报错'), 'error'], [VizMLI18n.t('残差没有单位'), 'unit']
                ], 'cancel', VizMLI18n.t('正负残差直接相加可能接近 0，即使每个点都离直线很远；平方可以避免抵消。'))) + '\n            ' + '</div>';
    }

    function stageThreeHtml() {
        const stats = currentLineStats(trainingPoints());
        return '\n            ' + '<div class="learning-stage">' + '\n                ' + (visualCard(VizMLI18n.t('实验三：在损失地图上寻找最低点'), VizMLI18n.t('地图上的每个位置代表一组斜率和截距。越偏绿色，MSE 越小。'), false, 'lossLandscape')) + '\n                ' + '<aside class="learning-side-card">' + '\n                    ' + (equationHtml()) + '\n                    ' + (lossStatsHtml(stats)) + '\n                    ' + '<canvas id="learningMiniCanvas" class="learning-canvas learning-mini-canvas" height="170" aria-label="当前参数对应的回归直线">' + '</canvas>' + '\n                    ' + '<div class="loss-legend">' + '</div>' + '\n                    ' + '<div class="loss-legend-labels">' + '<span>' + VizMLI18n.t('误差低') + '</span>' + '<span>' + VizMLI18n.t('误差高') + '</span>' + '</div>' + '\n                    ' + '<div class="learning-actions" style="margin-top: 14px">' + '\n                        ' + '<button type="button" class="learning-secondary" id="gradientReset">' + VizMLI18n.t('重置路径') + '</button>' + '\n                        ' + '<button type="button" id="gradientStep">' + VizMLI18n.t('单步下降') + '</button>' + '\n                        ' + '<button type="button" id="gradientPlay">' + VizMLI18n.t('自动播放') + '</button>' + '\n                    ' + '</div>' + '\n                    ' + '<div class="learning-callout">' + VizMLI18n.t('梯度指出损失上升最快的方向，所以模型向相反方向移动。OLS 最优点与地图最低点应当重合。') + '</div>' + '\n                    ' + (formulaHtml(VizMLI18n.t('MSE 对参数的梯度为 ∂MSE/∂w = −(2/n)Σxᵢ(yᵢ−ŷᵢ)，∂MSE/∂b = −(2/n)Σ(yᵢ−ŷᵢ)。每一步都用“当前参数 − 步长 × 梯度”更新。'))) + '\n                ' + '</aside>' + '\n                ' + (checkpointHtml(3, VizMLI18n.t('梯度下降每一步主要想实现什么？'), [
                    [VizMLI18n.t('让 MSE 变小'), 'lower'], [VizMLI18n.t('让斜率永远变大'), 'slope'], [VizMLI18n.t('删除误差最大的点'), 'delete']
                ], 'lower', VizMLI18n.t('梯度下降改变参数的目标是让损失逐步变小，而不是固定地增大或减小某个参数。'))) + '\n            ' + '</div>';
    }

    function stageFourHtml() {
        const splitStats = getSplitStats();
        return '\n            ' + '<div class="learning-stage">' + '\n                ' + (visualCard(VizMLI18n.t('实验四：用没见过的数据检验模型'), state.revealTest ? VizMLI18n.t('橙色三角形是训练时没有使用过的测试点。') : VizMLI18n.t('先观察蓝色训练点，再揭晓模型从未见过的测试点。'), false)) + '\n                ' + '<aside class="learning-side-card">' + '\n                    ' + (equationHtml(state.optimumW, state.optimumB)) + '\n                    ' + '<div class="learning-actions">' + '\n                        ' + '<button type="button" id="revealTest">' + (state.revealTest ? VizMLI18n.t('隐藏测试集') : VizMLI18n.t('揭晓测试集')) + '</button>' + '\n                    ' + '</div>' + '\n                    ' + '<div class="learning-stats">' + '\n                        ' + '<div class="learning-stat">' + '<span>' + VizMLI18n.t('训练集 MSE') + '</span>' + '<strong>' + (formatNumber(splitStats.train)) + '</strong>' + '</div>' + '\n                        ' + '<div class="learning-stat">' + '<span>' + VizMLI18n.t('测试集 MSE') + '</span>' + '<strong>' + (state.revealTest ? formatNumber(splitStats.test) : VizMLI18n.t('待揭晓')) + '</strong>' + '</div>' + '\n                    ' + '</div>' + '\n                    ' + '<div class="learning-control">' + '\n                        ' + '<label for="learningScenario">' + VizMLI18n.t('换一种数据情况') + '</label>' + '\n                        ' + '<select id="learningScenario">' + '\n                            <option value="linear" ' + (selected('linear')) + VizMLI18n.t('>清晰的线性关系') + '</option>' + '\n                            <option value="noisy" ' + (selected('noisy')) + VizMLI18n.t('>更多随机噪声') + '</option>' + '\n                            <option value="outliers" ' + (selected('outliers')) + VizMLI18n.t('>包含异常点') + '</option>' + '\n                            <option value="nonlinear" ' + (selected('nonlinear')) + VizMLI18n.t('>非线性关系') + '</option>' + '\n                        ' + '</select>' + '\n                    ' + '</div>' + '\n                    ' + '<div class="learning-callout">' + VizMLI18n.t('训练误差回答“记住得怎样”，测试误差回答“面对新数据怎样”。残差若呈现明显弯曲模式，通常说明直线并不适合这组数据。') + '</div>' + '\n                    ' + (formulaHtml(VizMLI18n.t('R² = 1 − SSE/TSS，用模型的误差与“永远预测平均值”的误差比较。R² 越接近 1，表示直线解释的数据变化越多；测试集指标更能反映泛化能力。'))) + '\n                ' + '</aside>' + '\n                ' + (checkpointHtml(4, VizMLI18n.t('哪一个指标更能反映模型面对新数据的表现？'), [
                    [VizMLI18n.t('训练集误差'), 'train'], [VizMLI18n.t('测试集误差'), 'test'], [VizMLI18n.t('样本编号'), 'index']
                ], 'test', VizMLI18n.t('测试集没有参与拟合，因此测试误差更能反映模型对未知数据的预测能力。'))) + '\n                ' + (state.answers[4] ? summaryHtml() : '') + '\n            ' + '</div>';
    }

    function selected(value) {
        return state.scenario === value ? 'selected' : '';
    }

    function visualCard(title, description, draggable, canvasId) {
        return `
            <section class="learning-visual-card">
                <div class="learning-stage-copy"><h2>${title}</h2><p>${description}</p></div>
                <div class="learning-canvas-wrap">
                    <canvas id="${canvasId || 'learningRegressionCanvas'}" class="learning-canvas${draggable ? ' is-draggable' : ''}" height="390" role="img" aria-label="${title}"></canvas>
                    <div id="learningTooltip" class="learning-tooltip" hidden></div>
                </div>
            </section>`;
    }

    function equationHtml(w, b) {
        const slope = w === undefined ? state.w : w;
        const intercept = b === undefined ? state.b : b;
        return `<div class="learning-equation">ŷ = ${formatNumber(slope)}x ${intercept >= 0 ? '+' : '−'} ${formatNumber(Math.abs(intercept))}</div>`;
    }

    function parameterControlsHtml() {
        return '\n            ' + '<div class="learning-control">' + '\n                ' + '<label for="learningSlope">' + '<span>' + VizMLI18n.t('斜率 w') + '</span>' + '<span>' + (formatNumber(state.w)) + '</span>' + '</label>' + '\n                <input id="learningSlope" type="range" min="' + (state.wMin) + '" max="' + (state.wMax) + '" value="' + (state.w) + '" step="' + ((state.wMax - state.wMin) / 240) + VizMLI18n.t('" aria-label="调整斜率">\n            ') + '</div>' + '\n            ' + '<div class="learning-control">' + '\n                ' + '<label for="learningIntercept">' + '<span>' + VizMLI18n.t('截距 b') + '</span>' + '<span>' + (formatNumber(state.b)) + '</span>' + '</label>' + '\n                <input id="learningIntercept" type="range" min="' + (state.bMin) + '" max="' + (state.bMax) + '" value="' + (state.b) + '" step="' + ((state.bMax - state.bMin) / 240) + VizMLI18n.t('" aria-label="调整截距">\n            ') + '</div>';
    }

    function lossStatsHtml(stats, mode) {
        const absolute = mode === 'absolute';
        return '\n            ' + '<div class="learning-stats">' + '\n                ' + '<div class="learning-stat">' + '<span>' + (absolute ? VizMLI18n.t('绝对误差和 SAE') : VizMLI18n.t('平方误差和 SSE')) + '</span>' + '<strong id="learningTotalLoss">' + (formatNumber(absolute ? stats.sae : stats.sse)) + '</strong>' + '</div>' + '\n                ' + '<div class="learning-stat">' + '<span>' + (absolute ? VizMLI18n.t('平均绝对误差 MAE') : VizMLI18n.t('均方误差 MSE')) + '</span>' + '<strong id="learningMeanLoss">' + (formatNumber(absolute ? stats.mae : stats.mse)) + '</strong>' + '</div>' + '\n            ' + '</div>';
    }

    function formulaHtml(text) {
        return '<details class="learning-formula">' + '<summary>' + VizMLI18n.t('展开数学解释') + '</summary>' + '<p>' + (text) + '</p>' + '</details>';
    }

    function checkpointHtml(stage, question, options, correct, explanation) {
        const answer = state.answers[stage];
        const feedback = answer ? '<p class="checkpoint-feedback ' + (answer.correct ? 'correct' : 'incorrect') + '">' + (answer.correct ? VizMLI18n.t('答对了。') : VizMLI18n.t('再想一步。')) + (explanation) + '</p>' : '<p class="checkpoint-feedback" aria-live="polite"></p>';
        const nav = '<div class="learning-actions" style="margin-top: 12px">' + (stage > 1 ? VizMLI18n.t('<button type="button" class="learning-secondary" data-stage-nav="prev">上一步</button>') : '') + (stage < 4 ? VizMLI18n.t('<button type="button" data-stage-nav="next">进入下一个实验</button>') : '') + '</div>';
        return '<section class="learning-checkpoint">' + '<h3>' + VizMLI18n.t('想一想：') + (question) + '</h3>' + '<div class="checkpoint-options">' + (options.map(([label, value]) => '<button type="button" data-check-stage="' + (stage) + '" data-check-value="' + (value) + '" data-check-correct="' + (correct) + '">' + (label) + '</button>').join('')) + '</div>' + (feedback) + (nav) + '</section>';
    }

    function summaryHtml() {
        return '<section class="learning-complete" style="grid-column: 1 / -1">' + '\n            ' + '<h2>' + VizMLI18n.t('你已经走完线性回归的核心链路') + '</h2>' + '\n            ' + '<p>' + VizMLI18n.t('一条直线并不是凭空出现的：模型定义预测、残差衡量错误、损失汇总错误，优化过程寻找损失更小的参数，最后再用新数据检验它。') + '</p>' + '\n            ' + '<div class="learning-summary-grid">' + '\n                ' + '<div>' + '<strong>' + VizMLI18n.t('直线模型') + '</strong>' + VizMLI18n.t('用斜率和截距产生预测') + '</div>' + '\n                ' + '<div>' + '<strong>' + VizMLI18n.t('残差') + '</strong>' + VizMLI18n.t('是真实值与预测值的差') + '</div>' + '\n                ' + '<div>' + '<strong>' + VizMLI18n.t('最小二乘') + '</strong>' + VizMLI18n.t('选择平方误差最小的参数') + '</div>' + '\n                ' + '<div>' + '<strong>' + VizMLI18n.t('泛化') + '</strong>' + VizMLI18n.t('要看模型未见过的数据') + '</div>' + '\n            ' + '</div>' + '\n            ' + '<button type="button" id="restartLearning">' + VizMLI18n.t('从头再做一次') + '</button>' + '\n        ' + '</section>';
    }

    function bindCommonEvents() {
        document.getElementById('learningNewData').addEventListener('click', () => loadDataset(state.scenario, true));
        document.querySelectorAll('[data-go-stage]').forEach((button) => {
            button.addEventListener('click', () => goToStage(Number(button.dataset.goStage)));
        });
        document.querySelectorAll('[data-stage-nav]').forEach((button) => {
            button.addEventListener('click', () => goToStage(state.stage + (button.dataset.stageNav === 'next' ? 1 : -1)));
        });
        document.querySelectorAll('[data-check-stage]').forEach((button) => {
            button.addEventListener('click', () => {
                const stage = Number(button.dataset.checkStage);
                state.answers[stage] = {
                    value: button.dataset.checkValue,
                    correct: button.dataset.checkValue === button.dataset.checkCorrect
                };
                render();
            });
        });
        const restart = document.getElementById('restartLearning');
        if (restart) restart.addEventListener('click', () => {
            state.answers = {};
            goToStage(1);
        });
    }

    function bindStageEvents() {
        bindParameterControls();

        const resetGuess = document.getElementById('resetGuess');
        if (resetGuess) resetGuess.addEventListener('click', () => {
            state.w = clamp(0, state.wMin, state.wMax);
            state.b = clamp(mean(trainingPoints().map((point) => point.actual)), state.bMin, state.bMax);
            state.showBest = false;
            render();
        });

        const showBest = document.getElementById('showBest');
        if (showBest) showBest.addEventListener('click', () => {
            state.showBest = !state.showBest;
            if (state.showBest) {
                state.w = state.optimumW;
                state.b = state.optimumB;
            }
            render();
        });

        document.querySelectorAll('[data-error-mode]').forEach((button) => {
            button.addEventListener('click', () => {
                state.errorMode = button.dataset.errorMode;
                render();
            });
        });

        const gradientReset = document.getElementById('gradientReset');
        if (gradientReset) gradientReset.addEventListener('click', resetGradient);
        const gradientStep = document.getElementById('gradientStep');
        if (gradientStep) gradientStep.addEventListener('click', () => {
            stopGradient();
            stepGradient();
            render();
        });
        const gradientPlay = document.getElementById('gradientPlay');
        if (gradientPlay) gradientPlay.addEventListener('click', toggleGradient);

        const revealTest = document.getElementById('revealTest');
        if (revealTest) revealTest.addEventListener('click', () => {
            state.revealTest = !state.revealTest;
            render();
        });

        const scenario = document.getElementById('learningScenario');
        if (scenario) scenario.addEventListener('change', () => loadDataset(scenario.value));
    }

    function bindParameterControls() {
        const slope = document.getElementById('learningSlope');
        const intercept = document.getElementById('learningIntercept');
        if (slope) slope.addEventListener('input', () => updateParameters(Number(slope.value), state.b));
        if (intercept) intercept.addEventListener('input', () => updateParameters(state.w, Number(intercept.value)));

        const canvas = document.getElementById('learningRegressionCanvas');
        if (!canvas || state.stage > 2) return;
        let dragging = false;
        canvas.addEventListener('pointerdown', (event) => {
            const map = canvas._learningMap;
            if (!map) return;
            const point = eventPosition(event, canvas);
            const dataX = map.toDataX(point.x);
            const predicted = state.w * dataX + state.b;
            if (Math.abs(map.y(predicted) - point.y) < 18) {
                dragging = true;
                canvas.classList.add('is-dragging');
                canvas.setPointerCapture(event.pointerId);
            }
        });
        canvas.addEventListener('pointermove', (event) => {
            if (dragging) {
                const map = canvas._learningMap;
                const point = eventPosition(event, canvas);
                const dataX = map.toDataX(point.x);
                const dataY = map.toDataY(point.y);
                updateParameters(state.w, clamp(dataY - state.w * dataX, state.bMin, state.bMax));
            } else if (state.stage === 2) {
                showPointTooltip(event, canvas);
            }
        });
        canvas.addEventListener('pointerup', () => {
            dragging = false;
            canvas.classList.remove('is-dragging');
        });
        canvas.addEventListener('pointerleave', () => {
            if (!dragging) hideTooltip();
        });
    }

    function updateParameters(w, b) {
        state.w = w;
        state.b = b;
        state.showBest = false;
        const equation = document.querySelector('.learning-equation');
        if (equation) equation.innerHTML = `ŷ = ${formatNumber(w)}x ${b >= 0 ? '+' : '−'} ${formatNumber(Math.abs(b))}`;
        const labels = document.querySelectorAll('.learning-control label span:last-child');
        if (labels[0]) labels[0].textContent = formatNumber(w);
        if (labels[1]) labels[1].textContent = formatNumber(b);
        const slope = document.getElementById('learningSlope');
        const intercept = document.getElementById('learningIntercept');
        if (slope) slope.value = String(w);
        if (intercept) intercept.value = String(b);
        updateLossText();
        drawActiveStage();
    }

    function updateLossText() {
        const stats = currentLineStats(trainingPoints());
        const total = document.getElementById('learningTotalLoss');
        const average = document.getElementById('learningMeanLoss');
        if (total) total.textContent = formatNumber(state.errorMode === 'absolute' ? stats.sae : stats.sse);
        if (average) average.textContent = formatNumber(state.errorMode === 'absolute' ? stats.mae : stats.mse);
    }

    function goToStage(stage) {
        stopGradient();
        state.stage = Math.max(1, Math.min(4, stage));
        sessionStorage.setItem('vizml-lr-stage', String(state.stage));
        if (state.stage === 3 && !state.gradientPath.length) {
            state.gradientPath = [{ w: state.w, b: state.b, mse: currentLineStats(trainingPoints()).mse }];
        }
        if (state.stage === 4) {
            state.w = state.optimumW;
            state.b = state.optimumB;
        }
        render();
    }

    function resetGradient() {
        stopGradient();
        state.w = clamp(0, state.wMin, state.wMax);
        state.b = clamp(mean(trainingPoints().map((point) => point.actual)), state.bMin, state.bMax);
        state.gradientSteps = 0;
        state.gradientPath = [{ w: state.w, b: state.b, mse: currentLineStats(trainingPoints()).mse }];
        render();
    }

    function toggleGradient() {
        if (state.gradientTimer) {
            stopGradient();
            render();
            return;
        }
        const button = document.getElementById('gradientPlay');
        if (button) button.textContent = VizMLI18n.t('暂停');
        state.gradientTimer = window.setInterval(() => {
            const done = stepGradient();
            updateLossText();
            drawActiveStage();
            const equation = document.querySelector('.learning-equation');
            if (equation) equation.innerHTML = `ŷ = ${formatNumber(state.w)}x ${state.b >= 0 ? '+' : '−'} ${formatNumber(Math.abs(state.b))}`;
            if (done || state.gradientSteps >= 80) {
                stopGradient();
                if (button) button.textContent = VizMLI18n.t('已到达最低点');
            }
        }, 230);
    }

    function stopGradient() {
        if (state.gradientTimer) window.clearInterval(state.gradientTimer);
        state.gradientTimer = null;
    }

    function stepGradient() {
        const points = trainingPoints();
        const n = points.length;
        let gradW = 0;
        let gradB = 0;
        points.forEach((point) => {
            const x = point.features[0];
            const residual = point.actual - (state.w * x + state.b);
            gradW += -2 * x * residual / n;
            gradB += -2 * residual / n;
        });
        const rate = stableLearningRate(points);
        state.w = clamp(state.w - rate * gradW, state.wMin, state.wMax);
        state.b = clamp(state.b - rate * gradB, state.bMin, state.bMax);
        state.gradientSteps += 1;
        const stats = currentLineStats(points);
        state.gradientPath.push({ w: state.w, b: state.b, mse: stats.mse });
        return Math.hypot(gradW, gradB) < 0.0005 || Math.hypot(state.w - state.optimumW, state.b - state.optimumB) < 0.001;
    }

    function stableLearningRate(points) {
        const meanX2 = mean(points.map((point) => point.features[0] ** 2));
        const meanX = mean(points.map((point) => point.features[0]));
        const trace = 2 * (meanX2 + 1);
        const determinant = 4 * Math.max(meanX2 - meanX * meanX, 0);
        const largestEigenvalue = (trace + Math.sqrt(Math.max(trace * trace - 4 * determinant, 0))) / 2;
        return 0.72 / Math.max(largestEigenvalue, 0.001);
    }

    function drawActiveStage() {
        if (!state.data.length || document.getElementById('guidedModePanel').hidden) return;
        if (state.stage === 3) {
            drawLossLandscape();
            const mini = document.getElementById('learningMiniCanvas');
            if (mini) drawRegression(mini, trainingPoints(), { residuals: false, mini: true });
            return;
        }
        const canvas = document.getElementById('learningRegressionCanvas');
        if (!canvas) return;
        if (state.stage === 4) {
            const visible = state.revealTest ? state.data : trainingPoints();
            drawRegression(canvas, visible, { w: state.optimumW, b: state.optimumB, showSplit: state.revealTest, residuals: state.revealTest });
        } else {
            drawRegression(canvas, trainingPoints(), { residuals: state.stage === 2, errorSquares: state.stage === 2 && state.errorMode === 'squared' });
        }
    }

    function drawRegression(canvas, points, options) {
        options = options || {};
        const w = options.w === undefined ? state.w : options.w;
        const b = options.b === undefined ? state.b : options.b;
        const prepared = prepareCanvas(canvas);
        if (!prepared || !points.length) return;
        const { ctx, width, height } = prepared;
        const padding = options.mini ? { left: 38, right: 14, top: 14, bottom: 30 } : { left: 54, right: 22, top: 20, bottom: 43 };
        const xs = points.map((point) => point.features[0]);
        const rawXMin = min(xs);
        const rawXMax = max(xs);
        const xPad = Math.max((rawXMax - rawXMin) * 0.08, 0.1);
        const xMin = rawXMin - xPad;
        const xMax = rawXMax + xPad;
        // 坐标轴固定在数据和最优直线范围内，避免拖动参数时坐标系不断缩放。
        const yCandidates = points.map((point) => point.actual).concat([
            state.optimumW * xMin + state.optimumB,
            state.optimumW * xMax + state.optimumB
        ]);
        const rawYMin = min(yCandidates);
        const rawYMax = max(yCandidates);
        const yPad = Math.max((rawYMax - rawYMin) * 0.1, 0.1);
        const yMin = rawYMin - yPad;
        const yMax = rawYMax + yPad;
        const plotWidth = width - padding.left - padding.right;
        const plotHeight = height - padding.top - padding.bottom;
        const xMap = (value) => padding.left + (value - xMin) / (xMax - xMin) * plotWidth;
        const yMap = (value) => padding.top + (yMax - value) / (yMax - yMin) * plotHeight;
        const toDataX = (value) => xMin + (value - padding.left) / plotWidth * (xMax - xMin);
        const toDataY = (value) => yMax - (value - padding.top) / plotHeight * (yMax - yMin);
        canvas._learningMap = { x: xMap, y: yMap, toDataX, toDataY };

        drawGrid(ctx, width, height, padding, xMin, xMax, yMin, yMax, options.mini);

        ctx.save();
        ctx.beginPath();
        ctx.rect(padding.left, padding.top, plotWidth, plotHeight);
        ctx.clip();

        if (options.residuals) {
            points.forEach((point) => {
                const x = xMap(point.features[0]);
                const actualY = yMap(point.actual);
                const predictedY = yMap(w * point.features[0] + b);
                ctx.strokeStyle = point.split === 'test' && options.showSplit ? '#e58a36' : '#d05b65';
                ctx.lineWidth = 1.5;
                ctx.setLineDash([4, 3]);
                ctx.beginPath();
                ctx.moveTo(x, actualY);
                ctx.lineTo(x, predictedY);
                ctx.stroke();
                ctx.setLineDash([]);
                if (options.errorSquares) {
                    const side = Math.min(Math.abs(actualY - predictedY), 42);
                    ctx.fillStyle = 'rgba(227, 166, 47, 0.12)';
                    ctx.strokeStyle = 'rgba(227, 166, 47, 0.55)';
                    ctx.fillRect(x + 3, Math.min(actualY, predictedY), side, side);
                    ctx.strokeRect(x + 3, Math.min(actualY, predictedY), side, side);
                }
            });
        }

        ctx.strokeStyle = '#c2415d';
        ctx.lineWidth = options.mini ? 2 : 3;
        ctx.beginPath();
        ctx.moveTo(xMap(xMin), yMap(w * xMin + b));
        ctx.lineTo(xMap(xMax), yMap(w * xMax + b));
        ctx.stroke();

        canvas._learningPoints = [];
        points.forEach((point) => {
            const x = xMap(point.features[0]);
            const y = yMap(point.actual);
            const isTest = point.split === 'test' && options.showSplit;
            ctx.fillStyle = isTest ? '#e58a36' : '#4f46e5';
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            if (isTest) {
                ctx.moveTo(x, y - 7);
                ctx.lineTo(x - 7, y + 6);
                ctx.lineTo(x + 7, y + 6);
                ctx.closePath();
            } else {
                ctx.arc(x, y, options.mini ? 3 : 5, 0, Math.PI * 2);
            }
            ctx.fill();
            ctx.stroke();
            canvas._learningPoints.push({ x, y, point, predicted: w * point.features[0] + b });
        });

        ctx.restore();

        if (options.showSplit && !options.mini) {
            ctx.font = '12px Arial';
            ctx.fillStyle = '#4f46e5';
            ctx.fillText(VizMLI18n.t('● 训练点'), padding.left + 5, padding.top + 15);
            ctx.fillStyle = '#d97706';
            ctx.fillText(VizMLI18n.t('▲ 测试点'), padding.left + 78, padding.top + 15);
        }
    }

    function drawGrid(ctx, width, height, padding, xMin, xMax, yMin, yMax, mini) {
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, width, height);
        ctx.font = mini ? '10px Arial' : '11px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        for (let i = 0; i <= 5; i += 1) {
            const x = padding.left + (width - padding.left - padding.right) * i / 5;
            const y = padding.top + (height - padding.top - padding.bottom) * i / 5;
            ctx.strokeStyle = '#e7ecf2';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(x, padding.top);
            ctx.lineTo(x, height - padding.bottom);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(padding.left, y);
            ctx.lineTo(width - padding.right, y);
            ctx.stroke();
            ctx.fillStyle = '#7b8798';
            ctx.fillText(formatNumber(xMin + (xMax - xMin) * i / 5), x, height - padding.bottom + 7);
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            ctx.fillText(formatNumber(yMax - (yMax - yMin) * i / 5), padding.left - 7, y);
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
        }
        ctx.strokeStyle = '#94a3b8';
        ctx.strokeRect(padding.left, padding.top, width - padding.left - padding.right, height - padding.top - padding.bottom);
        if (!mini) {
            ctx.fillStyle = '#526078';
            ctx.font = '12px Arial';
            ctx.fillText(VizMLI18n.t('特征 x'), padding.left + (width - padding.left - padding.right) / 2, height - 18);
            ctx.save();
            ctx.translate(15, padding.top + (height - padding.top - padding.bottom) / 2);
            ctx.rotate(-Math.PI / 2);
            ctx.fillText(VizMLI18n.t('目标 y'), 0, 0);
            ctx.restore();
        }
    }

    function drawLossLandscape() {
        const canvas = document.getElementById('lossLandscape');
        if (!canvas) return;
        const prepared = prepareCanvas(canvas);
        if (!prepared) return;
        const { ctx, width, height } = prepared;
        const padding = { left: 58, right: 24, top: 20, bottom: 48 };
        const plotW = width - padding.left - padding.right;
        const plotH = height - padding.top - padding.bottom;
        const columns = 52;
        const rows = 42;
        const losses = [];
        for (let row = 0; row < rows; row += 1) {
            for (let column = 0; column < columns; column += 1) {
                const w = state.wMin + (state.wMax - state.wMin) * column / (columns - 1);
                const b = state.bMax - (state.bMax - state.bMin) * row / (rows - 1);
                losses.push(currentLineStats(trainingPoints(), w, b).mse);
            }
        }
        const logLosses = losses.map((value) => Math.log1p(value));
        const low = min(logLosses);
        const high = max(logLosses);
        let index = 0;
        for (let row = 0; row < rows; row += 1) {
            for (let column = 0; column < columns; column += 1) {
                const amount = (logLosses[index++] - low) / Math.max(high - low, 0.00001);
                ctx.fillStyle = lossColor(amount);
                ctx.fillRect(padding.left + plotW * column / columns, padding.top + plotH * row / rows, plotW / columns + 1, plotH / rows + 1);
            }
        }
        ctx.strokeStyle = '#64748b';
        ctx.strokeRect(padding.left, padding.top, plotW, plotH);
        ctx.fillStyle = '#526078';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(VizMLI18n.t('斜率 w'), padding.left + plotW / 2, height - 16);
        ctx.save();
        ctx.translate(15, padding.top + plotH / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText(VizMLI18n.t('截距 b'), 0, 0);
        ctx.restore();
        ctx.fillText(formatNumber(state.wMin), padding.left, height - padding.bottom + 9);
        ctx.fillText(formatNumber(state.wMax), width - padding.right, height - padding.bottom + 9);
        ctx.textAlign = 'right';
        ctx.fillText(formatNumber(state.bMax), padding.left - 7, padding.top - 3);
        ctx.fillText(formatNumber(state.bMin), padding.left - 7, height - padding.bottom - 5);

        const xMap = (w) => padding.left + (w - state.wMin) / (state.wMax - state.wMin) * plotW;
        const yMap = (b) => padding.top + (state.bMax - b) / (state.bMax - state.bMin) * plotH;
        if (state.gradientPath.length > 1) {
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            ctx.beginPath();
            state.gradientPath.forEach((point, pathIndex) => {
                const x = xMap(point.w);
                const y = yMap(point.b);
                if (pathIndex === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();
        }
        drawMarker(ctx, xMap(state.optimumW), yMap(state.optimumB), '#ffffff', '#08735f', 8);
        ctx.fillStyle = '#08735f';
        ctx.textAlign = 'left';
        ctx.fillText(VizMLI18n.t('OLS 最优点'), xMap(state.optimumW) + 10, yMap(state.optimumB) - 7);
        drawMarker(ctx, xMap(state.w), yMap(state.b), '#c2415d', '#ffffff', 7);
        ctx.fillStyle = '#7f1d3a';
        ctx.fillText(VizMLI18n.t('当前参数（第 ') + (state.gradientSteps) + VizMLI18n.t(' 步）'), xMap(state.w) + 10, yMap(state.b) + 12);
    }

    function drawMarker(ctx, x, y, fill, stroke, radius) {
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fillStyle = fill;
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = stroke;
        ctx.stroke();
    }

    function lossColor(value) {
        const stops = value < 0.5
            ? interpolate([223, 247, 241], [246, 207, 101], value * 2)
            : interpolate([246, 207, 101], [214, 92, 92], (value - 0.5) * 2);
        return `rgb(${stops[0]}, ${stops[1]}, ${stops[2]})`;
    }

    function interpolate(start, end, amount) {
        return start.map((value, index) => Math.round(value + (end[index] - value) * amount));
    }

    function prepareCanvas(canvas) {
        const rect = canvas.getBoundingClientRect();
        if (!rect.width || !rect.height) return null;
        const ratio = Math.min(window.devicePixelRatio || 1, 2);
        canvas.width = Math.round(rect.width * ratio);
        canvas.height = Math.round(rect.height * ratio);
        const ctx = canvas.getContext('2d');
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
        return { ctx, width: rect.width, height: rect.height };
    }

    function showPointTooltip(event, canvas) {
        const tooltip = document.getElementById('learningTooltip');
        if (!tooltip || !canvas._learningPoints) return;
        const position = eventPosition(event, canvas);
        let nearest = null;
        let distance = 14;
        canvas._learningPoints.forEach((candidate) => {
            const nextDistance = Math.hypot(candidate.x - position.x, candidate.y - position.y);
            if (nextDistance < distance) {
                nearest = candidate;
                distance = nextDistance;
            }
        });
        if (!nearest) {
            hideTooltip();
            return;
        }
        const residual = nearest.point.actual - nearest.predicted;
        tooltip.innerHTML = VizMLI18n.t('真实值 y = ') + (formatNumber(nearest.point.actual)) + '<br>' + VizMLI18n.t('预测值 ŷ = ') + (formatNumber(nearest.predicted)) + '<br>' + VizMLI18n.t('残差 e = ') + (formatNumber(residual)) + '<br>' + 'e² = ' + (formatNumber(residual ** 2));
        tooltip.style.left = `${nearest.x}px`;
        tooltip.style.top = `${nearest.y}px`;
        tooltip.hidden = false;
    }

    function hideTooltip() {
        const tooltip = document.getElementById('learningTooltip');
        if (tooltip) tooltip.hidden = true;
    }

    function eventPosition(event, canvas) {
        const rect = canvas.getBoundingClientRect();
        return { x: event.clientX - rect.left, y: event.clientY - rect.top };
    }

    function currentLineStats(points, w, b) {
        const slope = w === undefined ? state.w : w;
        const intercept = b === undefined ? state.b : b;
        let sse = 0;
        let sae = 0;
        points.forEach((point) => {
            const residual = point.actual - (slope * point.features[0] + intercept);
            sse += residual * residual;
            sae += Math.abs(residual);
        });
        return {
            sse,
            mse: sse / Math.max(points.length, 1),
            sae,
            mae: sae / Math.max(points.length, 1)
        };
    }

    function getSplitStats() {
        const train = state.data.filter((point) => point.split === 'train');
        const test = state.data.filter((point) => point.split === 'test');
        return {
            train: currentLineStats(train, state.optimumW, state.optimumB).mse,
            test: currentLineStats(test, state.optimumW, state.optimumB).mse
        };
    }

    function trainingPoints() {
        return state.data.filter((point) => point.split === 'train');
    }

    function onResize() {
        window.clearTimeout(state.resizeTimer);
        state.resizeTimer = window.setTimeout(drawActiveStage, 120);
    }

    function mean(values) {
        return values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1);
    }

    function min(values) {
        return Math.min.apply(null, values);
    }

    function max(values) {
        return Math.max.apply(null, values);
    }

    function clamp(value, low, high) {
        return Math.max(low, Math.min(high, value));
    }

    function formatNumber(value) {
        if (!Number.isFinite(value)) return '—';
        const absolute = Math.abs(value);
        if (absolute > 0 && (absolute >= 10000 || absolute < 0.01)) return value.toExponential(2);
        if (absolute >= 100) return value.toFixed(1);
        return value.toFixed(2);
    }

    function escapeHtml(value) {
        const div = document.createElement('div');
        div.textContent = value;
        return div.innerHTML;
    }
}());
