/** Shared page shell and progressive UI enhancements for VizML. */
(function () {
    'use strict';

    const pages = {
        linear_regression: ['监督学习', '线性回归'],
        sgd: ['优化算法', 'SGD 优化器'],
        svm: ['监督学习', '支持向量机'],
        knn: ['监督学习', 'K 近邻'],
        decision_tree: ['监督学习', '决策树'],
        bayesian_classification: ['监督学习', '贝叶斯分类'],
        clustering: ['无监督学习', '聚类算法'],
        neural_network: ['模型训练', '神经网络'],
        ensemble: ['集成学习', '集成模型'],
        dimensionality_reduction: ['无监督学习', '降维算法']
    };

    const logo = `
        <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path d="M4 17.5 9 12l3.2 2.8L19.5 6" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/>
            <circle cx="4" cy="17.5" r="1.8" fill="currentColor"/><circle cx="9" cy="12" r="1.8" fill="currentColor"/>
            <circle cx="12.2" cy="14.8" r="1.8" fill="currentColor"/><circle cx="19.5" cy="6" r="1.8" fill="currentColor"/>
        </svg>`;

    const homeIcon = `
        <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path d="m4 10 8-6 8 6v9a1 1 0 0 1-1 1h-5v-6h-4v6H5a1 1 0 0 1-1-1v-9Z" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/>
        </svg>`;

    const historyIcon = `
        <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path d="M3 12a9 9 0 1 0 2.6-6.3" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
            <path d="M3 4v5h5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M12 7v5l3.5 2" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>`;

    const decorativeEmojiPrefix = /^\s*[\p{Extended_Pictographic}\uFE0F\u200D]+\s*/u;

    function cleanLabel(text) {
        return (text || '')
            .replace(decorativeEmojiPrefix, '')
            .replace(/\s*[:：]\s*$/, '')
            .trim();
    }

    function pageKey() {
        const name = location.pathname.split('/').pop() || '';
        return name.replace(/\.html$/, '');
    }

    function installTopbar(meta) {
        if (document.querySelector('.viz-topbar')) return;
        const bar = document.createElement('header');
        bar.className = 'viz-topbar';
        bar.innerHTML = `
            <div class="viz-topbar__inner">
                <a class="viz-brand" href="/" aria-label="返回 VizML 首页">
                    <span class="viz-brand__mark">${logo}</span><span>VizML</span>
                </a>
                <span class="viz-topbar__divider" aria-hidden="true"></span>
                <div class="viz-breadcrumb"><span>${meta[0]}</span><span aria-hidden="true">/</span><strong>${meta[1]}</strong></div>
                <span class="viz-topbar__spacer"></span>
                <span class="viz-workspace-badge">本地工作台</span>
                <button type="button" class="viz-history-btn" aria-label="历史记录">${historyIcon}<span>历史记录</span></button>
                <a class="viz-home-link" href="/">${homeIcon}<span>算法首页</span></a>
            </div>`;
        document.body.prepend(bar);
        const btn = bar.querySelector('.viz-history-btn');
        if (btn) btn.addEventListener('click', () => VizHistoryUI.open());
    }

    function enhanceHelp() {
        const candidates = document.querySelectorAll('.instructions, .algorithm-explanation');
        candidates.forEach((panel, index) => {
            if (panel.classList.contains('viz-help')) return;
            const text = panel.textContent || '';
            if (!text.includes('使用说明')) return;

            const title = Array.from(panel.children).find((element) =>
                element.matches('h1, h2, h3, h4, h5, h6, .explanation-title, strong') &&
                (element.textContent || '').includes('使用说明')
            );
            if (!title) return;

            let contentId = `viz-help-content-${index + 1}`;
            let suffix = index + 1;
            while (document.getElementById(contentId)) {
                suffix += 1;
                contentId = `viz-help-content-${suffix}`;
            }

            const heading = document.createElement('h4');
            heading.className = 'viz-help-heading';

            panel.classList.add('viz-help', 'viz-help--collapsed');
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'viz-help-toggle';
            button.setAttribute('aria-expanded', 'false');
            button.setAttribute('aria-controls', contentId);
            button.setAttribute('aria-label', '展开使用说明');

            const label = document.createElement('span');
            label.className = 'viz-help-toggle__label';
            label.textContent = cleanLabel(title.textContent) || '使用说明';

            const icon = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            icon.setAttribute('class', 'viz-help-toggle__icon');
            icon.setAttribute('viewBox', '0 0 20 20');
            icon.setAttribute('aria-hidden', 'true');
            icon.innerHTML = '<path d="m5.5 7.5 4.5 4.5 4.5-4.5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>';
            button.append(label, icon);
            heading.appendChild(button);

            const content = document.createElement('div');
            content.className = 'viz-help-content';
            content.id = contentId;
            Array.from(panel.childNodes).forEach((node) => {
                if (node !== title) content.appendChild(node);
            });
            content.hidden = true;
            panel.replaceChildren(heading, content);

            button.addEventListener('click', () => {
                const collapsed = panel.classList.toggle('viz-help--collapsed');
                content.hidden = collapsed;
                button.setAttribute('aria-expanded', String(!collapsed));
                button.setAttribute('aria-label', collapsed ? '展开使用说明' : '收起使用说明');
            });
        });
    }

    function annotatePage(key) {
        document.body.dataset.vizPage = key || 'home';
        document.querySelectorAll('canvas').forEach((canvas) => {
            canvas.setAttribute('role', canvas.getAttribute('role') || 'img');
            if (!canvas.getAttribute('aria-label')) {
                const heading = canvas.closest('.panel, .canvas-panel, .canvas-container')?.querySelector('h2, h3, h4');
                canvas.setAttribute('aria-label', heading?.textContent.trim() || '机器学习数据可视化');
            }
        });
    }

    function removeDecorativeEmoji() {
        document.querySelectorAll('button, h3, h4, .explanation-title').forEach((element) => {
            element.childNodes.forEach((node) => {
                if (node.nodeType === Node.TEXT_NODE && decorativeEmojiPrefix.test(node.textContent || '')) {
                    node.textContent = (node.textContent || '').replace(decorativeEmojiPrefix, '');
                }
            });
        });
    }

    const moduleNames = {
        svm: '支持向量机', knn: 'K 近邻', decision_tree: '决策树',
        bayesian_classification: '贝叶斯分类', linear_regression: '线性回归',
        sgd: 'SGD 优化器', clustering: '聚类', neural_network: '神经网络',
        ensemble: '集成学习', dimensionality_reduction: '降维'
    };

    function injectHistoryStyles() {
        if (document.getElementById('viz-history-style')) return;
        const style = document.createElement('style');
        style.id = 'viz-history-style';
        style.textContent = `
            .viz-history-btn {
                display: inline-flex; align-items: center; gap: 6px;
                background: transparent; border: 1px solid #d7deea; color: #42506b;
                border-radius: 8px; padding: 6px 12px; font-size: 13px; cursor: pointer;
            }
            .viz-history-btn svg { width: 16px; height: 16px; }
            .viz-history-btn:hover { background: #f2f6fc; border-color: #2B8AC6; color: #2B8AC6; }
            .viz-history-overlay {
                position: fixed; inset: 0; background: rgba(23, 32, 51, .45);
                z-index: 1000; display: none;
            }
            .viz-history-overlay.viz-history--open { display: flex; justify-content: flex-end; }
            .viz-history-panel {
                width: 440px; max-width: 94vw; height: 100%; background: #fff;
                display: flex; flex-direction: column; box-shadow: -10px 0 30px rgba(0,0,0,.12);
            }
            .viz-history-head {
                display: flex; align-items: center; justify-content: space-between;
                padding: 16px 20px; border-bottom: 1px solid #eef1f6;
            }
            .viz-history-head strong { font-size: 16px; color: #172033; }
            .viz-history-close {
                border: none; background: transparent; font-size: 22px; line-height: 1;
                color: #6B7890; cursor: pointer; padding: 4px 8px;
            }
            .viz-history-nick { padding: 14px 20px; border-bottom: 1px solid #eef1f6; }
            .viz-history-nick label { display: block; font-size: 12px; color: #6B7890; margin-bottom: 8px; }
            .viz-history-nick-row { display: flex; gap: 8px; }
            .viz-history-nick-row input {
                flex: 1; border: 1px solid #d7deea; border-radius: 8px; padding: 8px 12px; font-size: 14px;
            }
            .viz-history-nick-save {
                border: none; background: #2B8AC6; color: #fff; border-radius: 8px;
                padding: 0 14px; font-size: 13px; cursor: pointer;
            }
            .viz-history-toolbar {
                display: flex; align-items: center; justify-content: space-between;
                padding: 12px 20px;
            }
            .viz-history-hint { font-size: 12px; color: #8a94a8; }
            .viz-history-clear {
                border: 1px solid #f0c7cd; background: #fff5f5; color: #c0392b;
                border-radius: 8px; padding: 6px 12px; font-size: 12px; cursor: pointer;
            }
            .viz-history-list { list-style: none; margin: 0; padding: 0 20px 20px; overflow-y: auto; flex: 1; }
            .viz-history-item {
                border: 1px solid #eef1f6; border-radius: 10px; padding: 12px 14px; margin-bottom: 10px;
                background: #fbfcfe;
            }
            .viz-history-item-head { display: flex; align-items: center; gap: 8px; }
            .viz-history-item-head strong { font-size: 14px; color: #172033; }
            .viz-history-time { font-size: 12px; color: #8a94a8; flex: 1; }
            .viz-history-del {
                border: none; background: transparent; color: #c0392b; font-size: 12px; cursor: pointer;
            }
            .viz-history-replay {
                border: 1px solid #b8d8ef; background: #eef7fd; color: #2B8AC6;
                border-radius: 6px; padding: 3px 9px; font-size: 12px; cursor: pointer;
            }
            .viz-history-replay:hover { background: #dceefb; }
            .viz-history-item-meta { margin-top: 8px; display: flex; flex-wrap: wrap; gap: 6px; }
            .viz-history-chip {
                font-size: 12px; color: #42506b; background: #eef4fb; border-radius: 6px; padding: 3px 8px;
            }
            .viz-history-params { margin-top: 8px; }
            .viz-history-params summary { font-size: 12px; color: #2B8AC6; cursor: pointer; }
            .viz-history-params pre {
                font-size: 12px; color: #42506b; background: #f2f4f8; padding: 8px 10px;
                border-radius: 6px; overflow-x: auto; margin-top: 4px; white-space: pre-wrap; word-break: break-all;
            }
            .viz-history-empty { font-size: 14px; color: #8a94a8; text-align: center; padding: 24px 0; }
        `;
        document.head.appendChild(style);
    }

    const VizHistoryUI = {
        open() {
            injectHistoryStyles();
            this.buildPanel();
            this.overlay.classList.add('viz-history--open');
            this.refresh();
        },
        close() {
            if (this.overlay) this.overlay.classList.remove('viz-history--open');
        },
        buildPanel() {
            if (this.overlay) return;
            const overlay = document.createElement('div');
            overlay.className = 'viz-history-overlay';
            overlay.innerHTML = `
                <div class="viz-history-panel" role="dialog" aria-modal="true" aria-label="历史记录">
                    <div class="viz-history-head">
                        <strong>我的历史记录</strong>
                        <button type="button" class="viz-history-close" aria-label="关闭">×</button>
                    </div>
                    <div class="viz-history-nick">
                        <label for="viz-history-nick-input">昵称 / 学号（可选，用于区分历史记录）</label>
                        <div class="viz-history-nick-row">
                            <input id="viz-history-nick-input" type="text" maxlength="30" placeholder="留空表示匿名">
                            <button type="button" class="viz-history-nick-save">保存昵称</button>
                        </div>
                    </div>
                    <div class="viz-history-toolbar">
                        <span class="viz-history-hint">训练成功后会自动记录</span>
                        <button type="button" class="viz-history-clear">清空全部</button>
                    </div>
                    <ul class="viz-history-list"></ul>
                </div>`;
            document.body.appendChild(overlay);
            this.overlay = overlay;
            this.listEl = overlay.querySelector('.viz-history-list');
            this.nickInput = overlay.querySelector('#viz-history-nick-input');
            this.nickInput.value = VizHistory.getUsername();

            overlay.addEventListener('click', (e) => { if (e.target === overlay) this.close(); });
            overlay.querySelector('.viz-history-close').addEventListener('click', () => this.close());
            overlay.querySelector('.viz-history-nick-save').addEventListener('click', () => this.saveNick());
            overlay.querySelector('.viz-history-clear').addEventListener('click', () => this.clearAll());
        },
        async refresh() {
            if (!this.listEl) return;
            this.listEl.innerHTML = '<li class="viz-history-empty">加载中…</li>';
            try {
                const client = new MLApiClient();
                const data = await client.listHistory();
                this.renderRecords((data && data.records) || []);
            } catch (e) {
                this.listEl.innerHTML = '<li class="viz-history-empty">加载失败：' + (e.message || e) + '</li>';
            }
        },
        renderRecords(records) {
            this.listEl.innerHTML = '';
            if (!records.length) {
                this.listEl.innerHTML = '<li class="viz-history-empty">还没有记录，训练一次模型后会自动出现在这里。</li>';
                return;
            }
            records.forEach((r) => {
                const li = document.createElement('li');
                li.className = 'viz-history-item';
                const name = moduleNames[r.module] || r.module || '未知';
                const canReplay = !!(r.replay && r.replay.train && Object.keys(r.replay.train).length);
                li.innerHTML = `
                    <div class="viz-history-item-head">
                        <strong>${name}</strong>
                        <span class="viz-history-time">${r.created_at || ''}</span>
                        ${canReplay ? '<button type="button" class="viz-history-replay">复现</button>' : ''}
                        <button type="button" class="viz-history-del">删除</button>
                    </div>
                    <div class="viz-history-item-meta">
                        ${r.dataset ? `<span class="viz-history-chip">数据：${this.escapeHtml(r.dataset)}</span>` : ''}
                        ${this.metricText(r.metrics)}
                    </div>
                    ${this.paramText(r.params).trim() ? `<details class="viz-history-params"><summary>参数</summary><pre>${this.escapeHtml(this.paramText(r.params))}</pre></details>` : ''}`;
                const delBtn = li.querySelector('.viz-history-del');
                const replayBtn = li.querySelector('.viz-history-replay');
                if (delBtn) {
                    delBtn.addEventListener('click', async () => {
                        try {
                            const client = new MLApiClient();
                            await client.deleteHistory(r.id);
                            this.refresh();
                        } catch (e) { alert('删除失败：' + (e.message || e)); }
                    });
                }
                if (replayBtn) {
                    replayBtn.addEventListener('click', () => this.replay(r));
                }
                this.listEl.appendChild(li);
            });
        },
        replay(record) {
            const rp = record.replay || {};
            if (!rp.train || !Object.keys(rp.train).length) {
                alert('该记录缺少复现信息（可能是旧版本保存的）。');
                return;
            }
            if (!replayConfig[record.module]) {
                alert('该模块暂未支持一键复现，敬请期待。');
                return;
            }
            sessionStorage.setItem('vizml_replay', JSON.stringify({ module: record.module, ...rp }));
            this.close();
            window.location.href = '/html/' + record.module + '.html?replay=1';
        },
        metricText(metrics = {}) {
            const keys = Object.keys(metrics);
            if (!keys.length) return '';
            return keys.map((k) => {
                const v = metrics[k];
                const val = typeof v === 'number' ? v.toFixed(4) : v;
                return `<span class="viz-history-chip">${this.escapeHtml(k)}：${this.escapeHtml(String(val))}</span>`;
            }).join('');
        },
        paramText(params = {}) {
            const keys = Object.keys(params);
            if (!keys.length) return '';
            return keys.map((k) => `${k} = ${JSON.stringify(params[k])}`).join('\n');
        },
        escapeHtml(value) {
            return String(value)
                .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
        },
        async saveNick() {
            const name = (this.nickInput.value || '').trim();
            try {
                const client = new MLApiClient();
                await client.updateUsername(name);
                this.refresh();
            } catch (e) { alert('保存昵称失败：' + (e.message || e)); }
        },
        async clearAll() {
            if (!confirm('确定清空全部历史记录吗？此操作不可恢复。')) return;
            try {
                const client = new MLApiClient();
                await client.clearHistory();
                this.refresh();
            } catch (e) { alert('清空失败：' + (e.message || e)); }
        }
    };

    // 复现映射表：记录各算法页面「数据字段/算法字段/按钮」与后端参数的对应关系
    const replayConfig = {
        svm: {
            generateBtn: 'generateDataBtn', trainBtn: 'trainSvmBtn',
            dataFields: { shape: 'datasetType', n_samples: 'sampleSize', noise: 'noise' },
            paramFields: { C: 'cValue', kernel: 'kernel', gamma: 'gamma', degree: 'degree', coef0: 'coef0', max_iter: 'maxIter', tol: 'tolerance' }
        },
        knn: {
            generateBtn: 'generateDataBtn', trainBtn: 'trainKnnBtn',
            dataFields: { shape: 'datasetType', n_samples: 'sampleSize' },
            paramFields: { n_neighbors: 'kValue', metric: 'distanceMetric', weights: 'weights' }
        },
        decision_tree: {
            generateBtn: 'generateDataBtn', trainBtn: 'buildTreeBtn',
            dataFields: { shape: 'dataType', n_samples: 'sampleCount' },
            paramFields: { max_depth: 'maxDepth', min_samples_split: 'minSamplesSplit', min_samples_leaf: 'minSamplesLeaf', max_features: 'maxFeatures', criterion: 'criterion', use_full_features: 'useFullFeatures' }
        },
        ensemble: {
            generateBtn: 'generateDataBtn', trainBtn: 'trainEnsembleBtn',
            dataFields: { shape: 'datasetType', n_samples: 'sampleSize', noise: 'noiseLevel' },
            paramFields: { n_estimators: 'nEstimators', max_depth: 'maxDepth', learning_rate: 'learningRate', min_samples_split: 'minSamplesSplit', max_features: 'maxFeatures', subsample: 'subsample', voting: 'votingType', base_learner_types: 'learnerCheckboxes', cv: 'cv', stack_method: 'stackMethod', meta_learner_type: 'metaLearner', max_samples: 'maxSamples', bootstrap: 'bootstrap', base_estimator_type: 'baseEstimatorType', colsample_bytree: 'colsampleBytree', reg_alpha: 'regAlpha', reg_lambda: 'regLambda' }
        },
        linear_regression: {
            generateBtn: 'generateDataBtn', trainBtn: 'trainModelBtn',
            dataFields: { shape: 'dataShape', n_samples: 'numSamples', noise: 'noiseLevel' },
            paramFields: { test_size: 'testSize', optimizer: 'optimizer', polynomial_degree: 'polynomialDegree', alpha: 'alpha', eta0: 'learningRate', max_iter: 'maxIter' },
            extraFields: { algorithm: 'algorithmType' }
        },
        bayesian_classification: {
            generateBtn: '', trainBtn: 'trainBtn',
            dataFields: { shape: 'dataType', n_samples: 'nSamples', noise: 'noise', random_state: 'randomState' },
            paramFields: { test_size: 'testSize', var_smoothing: { id: 'varSmoothing', transform: 'log10' } }
        },
        clustering: {
            customApply: async function (rp) {
                const paramMaps = {
                    kmeans: { n_clusters: 'kClusters', max_iter: 'kMaxIter', init: 'kInit' },
                    dbscan: { eps: 'dbscanEps', min_samples: 'dbscanMinPts' },
                    hierarchical: { n_clusters: 'hClusters', linkage: 'linkage' },
                    gmm: { n_components: 'gmmComponents', covariance_type: 'covarianceType', init_params: 'initParams', max_iter: 'gmmMaxIter', tol: 'tolerance' }
                };
                const applyParams = (params, algo, prefix) => {
                    const map = paramMaps[algo] || {};
                    Object.keys(map).forEach((k) => {
                        if (params[k] === undefined) return;
                        let val = params[k];
                        if (algo === 'kmeans' && k === 'init') val = val === 'k-means++' ? 'kmeans++' : 'random';
                        setFieldValue(getById(prefix + map[k] + (prefix ? '-' + algo : '')), val);
                    });
                };

                await waitFor(() => getById('generateDataBtn') && getById('showFinalResultsBtn') && getById('singleModeBtn'), 5000);
                const data = rp.data || {};
                const train = rp.train || {};
                setFieldValue(getById('dataShape'), data.shape);
                setFieldValue(getById('numPoints'), data.n_samples);
                setFieldValue(getById('noise'), data.noise);

                if (train.algorithm) {
                    clickById('singleModeBtn');
                    setFieldValue(getById('algorithm'), train.algorithm);
                    applyParams(train.params || {}, train.algorithm, '');
                    clickById('generateDataBtn');
                    await waitFor(() => { const c = getById('dataCount'); return c && c.textContent !== '0'; }, 15000);
                    clickById('showFinalResultsBtn');
                } else if (train.algorithms && train.algorithms.length) {
                    clickById('compareModeBtn');
                    await waitFor(() => getById('compareAlgorithmsBtn'), 3000);
                    clickById('clearSelectionBtn');
                    train.algorithms.forEach((val) => {
                        const option = document.querySelector('.algorithm-option[data-algorithm="' + val + '"]');
                        if (option && !option.querySelector('input[type="checkbox"]').checked) option.click();
                    });
                    await waitFor(() => getById('comparisonParamPanels') && getById('comparisonParamPanels').children.length > 0, 3000);
                    const algoParams = train.algorithm_params || {};
                    train.algorithms.forEach((algo) => applyParams(algoParams[algo] || {}, algo, 'compare-'));
                    clickById('generateDataBtn');
                    await waitFor(() => { const c = getById('dataCount'); return c && c.textContent !== '0'; }, 15000);
                    await waitFor(() => { const b = getById('compareAlgorithmsBtn'); return b && !b.disabled; }, 5000);
                    clickById('compareAlgorithmsBtn');
                }
            }
        },
        neural_network: {
            customApply: async function (rp) {
                await waitFor(() => getById('generateDataBtn') && getById('realtimeTrainBtn'), 5000);
                const data = rp.data || {};
                const train = rp.train || {};
                const params = train.params || {};

                setFieldValue(getById('datasetType'), data.type);
                setFieldValue(getById('sampleSize'), data.n_samples);
                setFieldValue(getById('noise'), data.noise);
                if (data.type === 'custom') {
                    setFieldValue(getById('customShape'), data.shape);
                    setFieldValue(getById('customClassCount'), data.n_classes);
                }

                setFieldValue(getById('networkType'), params.network_type);
                setFieldValue(getById('activation'), params.activation);
                setFieldValue(getById('solver'), params.solver);
                setFieldValue(getById('learningRate'), params.learning_rate_init);
                setFieldValue(getById('maxIter'), params.max_iter);
                setFieldValue(getById('alpha'), params.alpha);

                if (Array.isArray(params.hidden_layer_sizes)) {
                    const str = params.hidden_layer_sizes.join(',');
                    const hiddenSel = getById('hiddenLayers');
                    if (['50', '100,50', '100,100,50', '100,100,100,100,50'].indexOf(str) !== -1) {
                        setFieldValue(hiddenSel, str);
                    } else {
                        setFieldValue(hiddenSel, 'custom');
                        setFieldValue(getById('customLayers'), str);
                    }
                }

                clickById('generateDataBtn');
                waitUntilEnabledAndClick('realtimeTrainBtn', 15000);
            }
        },
        sgd: {
            customApply: async function (rp) {
                await waitFor(() => getById('generateDataBtn') && getById('startTrainingBtn') && document.querySelectorAll('.optimizer-card').length > 0, 10000);
                const data = rp.data || {};
                const train = rp.train || {};

                setFieldValue(getById('dataType'), data.shape);
                setFieldValue(getById('sampleSize'), data.n_samples);
                setFieldValue(getById('noiseLevel'), data.noise);

                const model = train.model || {};
                if (model.model_type) {
                    setFieldValue(getById('modelType'), model.model_type);
                    if (model.polynomial_degree !== undefined) setFieldValue(getById('polynomialDegree'), model.polynomial_degree);
                }

                const opts = (train.optimizers && train.optimizers.length) ? train.optimizers : Object.keys(train.optimizers_config || {});
                const wanted = opts.filter((k) => ['sgd', 'momentum', 'nesterov', 'adagrad', 'rmsprop', 'adam'].indexOf(k) !== -1);
                const firstCfg = train.optimizers_config && wanted[0] ? train.optimizers_config[wanted[0]] : null;
                if (firstCfg && firstCfg.learning_rate !== undefined) setFieldValue(getById('learningRate'), firstCfg.learning_rate);
                if (train.max_iter !== undefined) setFieldValue(getById('maxIterations'), train.max_iter);

                wanted.forEach((key) => {
                    const card = document.querySelector('.optimizer-card[data-optimizer="' + key + '"]');
                    if (card && !card.classList.contains('active')) card.click();
                });
                document.querySelectorAll('.optimizer-card.active').forEach((card) => {
                    if (wanted.indexOf(card.getAttribute('data-optimizer')) === -1) card.click();
                });
                await waitFor(() => getById('optimizerParams') && getById('optimizerParams').children.length > 0, 3000);

                const paramDefs = {
                    momentum: ['momentum'], nesterov: ['momentum'],
                    adagrad: ['epsilon'], rmsprop: ['decay_rate', 'epsilon'],
                    adam: ['beta1', 'beta2', 'epsilon']
                };
                wanted.forEach((key) => {
                    const cfg = train.optimizers_config ? train.optimizers_config[key] : null;
                    if (!cfg) return;
                    (paramDefs[key] || []).forEach((pname) => {
                        if (cfg[pname] !== undefined) setFieldValue(getById(key + '_' + pname), cfg[pname]);
                    });
                });

                clickById('generateDataBtn');
                waitUntilEnabledAndClick('startTrainingBtn', 20000);
            }
        },
        dimensionality_reduction: {
            customApply: async function (rp) {
                await waitFor(() => getById('loadDataBtn') && getById('applyAlgorithmBtn'), 5000);
                const data = rp.data || {};
                const train = rp.train || {};
                setFieldValue(getById('dataType'), data.type);
                if (data.type === 'builtin') {
                    setFieldValue(getById('dataset'), data.dataset);
                } else {
                    setFieldValue(getById('nSamples'), data.n_samples);
                    setFieldValue(getById('nFeatures'), data.n_features);
                    setFieldValue(getById('nClasses'), data.n_classes);
                }

                if (train.algorithm) {
                    const tab = document.querySelector('.tab[data-algorithm="' + train.algorithm + '"]');
                    if (tab) tab.click();
                }
                const p = train.algorithm || 'pca';
                const maps = {
                    pca: { n_components: 'pcaComponents', standardize: 'pcaStandardize' },
                    tsne: { n_components: 'tsneComponents', perplexity: 'tsnePerplexity', learning_rate: 'tsneLearningRate', max_iter: 'tsneIterations' },
                    lda: { n_components: 'ldaComponents', standardize: 'ldaStandardize' },
                    umap: { n_components: 'umapComponents', n_neighbors: 'umapNeighbors', min_dist: 'umapMinDist', metric: 'umapMetric', standardize: 'umapStandardize' }
                };
                const params = train.params || {};
                const map = maps[p] || {};
                Object.keys(map).forEach((k) => {
                    if (params[k] !== undefined) setFieldValue(getById(map[k]), params[k]);
                });

                clickById('loadDataBtn');
                waitUntilEnabledAndClick('applyAlgorithmBtn', 20000);
            }
        }
    };

    function setFieldValue(el, value) {
        if (!el) return;
        if (value === undefined || value === null) return;
        if (el.tagName === 'SELECT') {
            const str = String(value);
            if (Array.from(el.options).some((o) => o.value === str)) el.value = str;
        } else if (el.type === 'checkbox') {
            el.checked = !!value;
        } else {
            el.value = value;
        }
        el.dispatchEvent(new Event('input', { bubbles: true }));
        el.dispatchEvent(new Event('change', { bubbles: true }));
    }

    function getById(id) { return document.getElementById(id); }
    function clickById(id) { const el = getById(id); if (el) el.click(); }
    function waitFor(predicate, timeout) {
        return new Promise((resolve) => {
            const started = Date.now();
            const iv = setInterval(() => {
                let ok = false;
                try { ok = !!predicate(); } catch (e) { ok = false; }
                if (ok) { clearInterval(iv); resolve(true); }
                else if (Date.now() - started > timeout) { clearInterval(iv); resolve(false); }
            }, 200);
        });
    }

    function waitUntilEnabledAndClick(btnId, timeout) {
        const started = Date.now();
        const iv = setInterval(() => {
            const btn = document.getElementById(btnId);
            if (btn && !btn.disabled) {
                clearInterval(iv);
                btn.click();
            } else if (Date.now() - started > timeout) {
                clearInterval(iv);
            }
        }, 300);
    }

    function applyReplay() {
        const raw = sessionStorage.getItem('vizml_replay');
        if (!raw) return;
        let rp = null;
        try { rp = JSON.parse(raw); } catch (e) { sessionStorage.removeItem('vizml_replay'); return; }
        const mod = replayConfig[rp.module];
        if (!mod) { sessionStorage.removeItem('vizml_replay'); return; }
        sessionStorage.removeItem('vizml_replay'); // 只执行一次

        // 复杂页面走自定义复现逻辑
        if (typeof mod.customApply === 'function') {
            mod.customApply(rp);
            return;
        }

        const applyFields = (mapping, source) => {
            if (!mapping) return;
            Object.keys(mapping).forEach((k) => {
                if (source[k] === undefined) return;
                const def = mapping[k];
                const id = (typeof def === 'object') ? def.id : def;
                let val = source[k];
                if (typeof def === 'object' && def.transform === 'log10') val = Math.log10(val);
                setFieldValue(document.getElementById(id), val);
            });
        };

        const data = rp.data || {};
        const train = rp.train || {};
        const params = train.params || {};
        applyFields(mod.dataFields, data);
        applyFields(mod.paramFields, params);
        applyFields(mod.extraFields, train);

        const genBtn = mod.generateBtn ? document.getElementById(mod.generateBtn) : null;
        if (genBtn) {
            genBtn.click();
        } else if (typeof window.generateData === 'function') {
            window.generateData();
        }
        if (mod.trainBtn) waitUntilEnabledAndClick(mod.trainBtn, 15000);
    }

    function init() {
        const key = pageKey();
        if (!pages[key]) {
            document.body.classList.add('viz-home-page');
            annotatePage('home');
            return;
        }
        document.body.classList.add('viz-algorithm-page');
        installTopbar(pages[key]);
        enhanceHelp();
        removeDecorativeEmoji();
        annotatePage(key);
        setTimeout(applyReplay, 400);
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, { once: true });
    else init();
})();
