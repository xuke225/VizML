(function () {
    'use strict';

    let lastPruningInfo = null;
    let resizeBound = false;
    let treeZoom = 1;
    let treeSize = null;
    let baselineSize = null;

    function byId(id) {
        return document.getElementById(id);
    }

    function init() {
        const radios = document.querySelectorAll('input[name="pruningMethod"]');
        if (!radios.length) return;
        initTreeComparison();

        radios.forEach((radio) => radio.addEventListener('change', updateControls));
        const alphaInput = byId('ccpAlpha');
        if (alphaInput) {
            alphaInput.addEventListener('input', () => {
                byId('ccpAlphaValue').textContent = formatAlpha(Number(alphaInput.value));
            });
        }

        if (!resizeBound) {
            window.addEventListener('resize', () => {
                if (lastPruningInfo) drawPath(lastPruningInfo.path || []);
            });
            resizeBound = true;
        }
        updateControls();
    }

    function selectedMethod() {
        const selected = document.querySelector('input[name="pruningMethod"]:checked');
        return selected ? selected.value : 'pre';
    }

    function updateControls() {
        const method = selectedMethod();
        document.querySelectorAll('.pruning-strategy').forEach((card) => {
            const radio = card.querySelector('input[name="pruningMethod"]');
            card.classList.toggle('is-active', Boolean(radio && radio.checked));
        });

        document.querySelectorAll('[data-visible-for]').forEach((element) => {
            const modes = element.dataset.visibleFor.split(',');
            element.style.display = modes.includes(method) ? 'flex' : 'none';
        });

        ['maxDepth', 'minSamplesSplit', 'minSamplesLeaf'].forEach((id) => {
            const input = byId(id);
            if (input) input.disabled = method !== 'pre';
        });

        const hint = byId('pruningModeHint');
        const hints = {
            none: '树将不受深度和样本阈值限制，适合观察过拟合。',
            pre: '训练过程中限制树继续分裂，下方参数与上方深度参数共同生效。',
            post_auto: '先生成完整树，再通过交叉验证自动选择代价复杂度系数。',
            post_manual: '先生成完整树，再按你指定的 ccp_alpha 删除弱分支。'
        };
        if (hint) hint.textContent = hints[method];
    }

    function getParams() {
        const method = selectedMethod();
        function readNumber(id, label, minimum, integer = false, optional = false) {
            const raw = byId(id).value.trim();
            if (optional && raw === '') return null;
            const value = raw === '' ? NaN : Number(raw);
            if (!Number.isFinite(value) || value < minimum || (integer && !Number.isInteger(value))) {
                throw new Error(`${label}必须是大于等于 ${minimum} 的${integer ? '整数' : '有限数值'}`);
            }
            return value;
        }
        // 后剪枝对照仍展示一组有效预剪枝配置；只对当前策略的输入要求用户纠正。
        const preParams = method === 'pre' ? {
            max_depth: readNumber('maxDepth', '最大深度', 1, true, true),
            min_samples_split: readNumber('minSamplesSplit', '最小分裂样本数', 2, true),
            min_samples_leaf: readNumber('minSamplesLeaf', '叶节点最小样本数', 1, true),
            max_leaf_nodes: readNumber('maxLeafNodes', '最大叶节点数', 2, true, true),
            min_impurity_decrease: readNumber('minImpurityDecrease', '最小不纯度下降', 0)
        } : {
            max_depth: Number(byId('maxDepth').value) >= 1 && Number.isInteger(Number(byId('maxDepth').value)) ? Number(byId('maxDepth').value) : 5,
            min_samples_split: Number(byId('minSamplesSplit').value) >= 2 && Number.isInteger(Number(byId('minSamplesSplit').value)) ? Number(byId('minSamplesSplit').value) : 2,
            min_samples_leaf: Number(byId('minSamplesLeaf').value) >= 1 && Number.isInteger(Number(byId('minSamplesLeaf').value)) ? Number(byId('minSamplesLeaf').value) : 1,
            max_leaf_nodes: null,
            min_impurity_decrease: 0
        };
        return {
            ...preParams,
            pruning_method: method,
            ccp_alpha: method === 'post_manual' ? readNumber('ccpAlpha', 'ccp_alpha', 0) : 0,
            cv_folds: readNumber('pruningCvFolds', '交叉验证折数', 2, true)
        };
    }

    function render(info) {
        if (!info) return;
        lastPruningInfo = info;
        const panel = byId('pruningResults');
        panel.classList.add('is-visible');
        byId('pruningResultBadge').textContent = info.method_label || '剪枝分析';

        const selected = info.selected_model || {};
        const reduction = info.reduction || {};
        const accuracyChange = Number(reduction.test_accuracy_change || 0);
        const gapChange = Number(reduction.generalization_gap_change || 0);
        byId('pruningSummary').innerHTML = [
            kpi('最终节点数', selected.node_count ?? '--', ''),
            kpi('节点精简比例', percent(reduction.node_reduction_ratio), reduction.node_reduction_ratio >= 0 ? 'is-positive' : 'is-warning'),
            kpi('测试准确率变化', signedPercent(accuracyChange), accuracyChange >= 0 ? 'is-positive' : 'is-warning'),
            kpi('泛化差距变化', signedPercent(gapChange), gapChange <= 0 ? 'is-positive' : 'is-warning')
        ].join('');

        byId('pruningComparisonBody').innerHTML = (info.comparison || []).map((item) => `
            <tr class="${item.selected ? 'is-selected' : ''}">
                <td>${item.selected ? '✓ ' : ''}${item.label}</td>
                <td>${percent(item.train_accuracy)}</td>
                <td>${percent(item.test_accuracy)}</td>
                <td>${percent(item.generalization_gap)}</td>
                <td>${item.depth}</td>
                <td>${item.n_leaves}</td>
                <td>${item.node_count}</td>
            </tr>
        `).join('');

        const notes = Array.isArray(info.parameter_notes) ? info.parameter_notes.join('；') : '';
        const note = byId('pruningExplanation');
        note.replaceChildren();
        const explanation = document.createElement('div');
        explanation.textContent = `选择说明：${info.selection_rule || ''}`;
        const parameters = document.createElement('div');
        parameters.textContent = `当前参数：${notes}`;
        const folds = document.createElement('div');
        folds.textContent = info.cv_folds ? `对照曲线使用 ${info.cv_folds} 折交叉验证；测试集不参与自动选参。` : '样本不足，未执行交叉验证。';
        note.append(explanation, parameters, folds);
        drawPath(info.path || []);
        renderTreeComparison(info.tree_comparison, info.method);
    }

    // Keep the original prediction/path canvas untouched. This additional graph
    // Always displays every selected-model node. Only a strict subtree can use
    // one shared layout; changed splits require a separate baseline graph.
    function initTreeComparison() {
        const results = byId('pruningResults');
        if (!results || byId('pruningTreeComparison')) return;
        const panel = document.createElement('section');
        panel.id = 'pruningTreeComparison';
        panel.className = 'pruning-tree-comparison';
        panel.setAttribute('aria-labelledby', 'pruningTreeTitle');
        panel.innerHTML = `
            <div class="pruning-tree-heading">
                <div><h4 id="pruningTreeTitle">剪枝位置对照图</h4>
                <p id="pruningTreeSubtitle">完整展示当前模型，准确标注可以对应的剪枝位置。</p></div>
                <span id="pruningTreeModeBadge" class="pruning-badge">结构对照</span>
            </div>
            <div class="pruning-tree-toolbar">
                <label id="pruningShowRemovedLabel"><input id="pruningShowRemoved" type="checkbox" checked> 显示灰色分支</label>
                <div class="pruning-tree-zoom">
                    <button id="pruningTreeZoomOut" type="button" aria-label="缩小对照图">−</button>
                    <output id="pruningTreeZoomLabel">100%</output>
                    <button id="pruningTreeZoomIn" type="button" aria-label="放大对照图">＋</button>
                    <button id="pruningTreeFit" type="button">完整视图</button>
                    <button id="pruningTreeActual" type="button">清晰视图</button>
                </div>
            </div>
            <div class="pruning-tree-legend">
                <span><i class="is-retained"></i>当前模型分裂</span>
                <span><i class="is-leaf"></i>当前模型叶节点</span>
                <span><i class="is-cut"></i><b id="pruningCutLegend">剪枝位置</b></span>
                <span id="pruningRemovedLegendItem"><i class="is-removed"></i><b id="pruningRemovedLegend">已剪除</b></span>
            </div>
            <p id="pruningTreeSummary" class="pruning-tree-summary"></p>
            <p id="pruningTreeCaveat" class="pruning-tree-caveat"></p>
            <h5 id="pruningCurrentTreeTitle" class="pruning-tree-view-title"></h5>
            <div id="pruningTreeViewport" class="pruning-tree-viewport" tabindex="0" aria-label="树图区域，可滚动查看并使用按钮缩放">
                <svg id="pruningTreeSvg" xmlns="http://www.w3.org/2000/svg" aria-label="剪枝前后节点位置对照图"></svg>
            </div>
            <div id="pruningTreeDetails" class="pruning-tree-details" aria-live="polite">点击或键盘选中圆圈，查看节点信息。</div>
            <details id="pruningBaselinePanel" class="pruning-baseline-panel" hidden>
                <summary id="pruningBaselineTitle">剪枝前基线（独立对照，不是当前模型）</summary>
                <p class="pruning-tree-caveat">橙色表示分裂规则变化或无法对应的旧结构，不是被剪除的节点。灰色仅表示能确认的停止生长分支；所有当前模型节点都已在上图画出。</p>
                <div class="pruning-tree-zoom">
                    <button id="pruningBaselineFit" type="button">基线全貌</button>
                    <button id="pruningBaselineActual" type="button">基线清晰视图</button>
                </div>
                <div id="pruningBaselineViewport" class="pruning-tree-viewport" tabindex="0" aria-label="剪枝前基线，可滚动查看">
                    <svg id="pruningBaselineSvg" xmlns="http://www.w3.org/2000/svg" aria-label="剪枝前完整基线，不是当前模型"></svg>
                </div>
            </details>`;
        results.insertBefore(panel, results.querySelector('.pruning-result-grid'));
        byId('pruningShowRemoved').addEventListener('change', () => {
            panel.classList.toggle('hide-removed', !byId('pruningShowRemoved').checked);
        });
        byId('pruningTreeZoomOut').addEventListener('click', () => scaleTree(treeZoom / 1.25));
        byId('pruningTreeZoomIn').addEventListener('click', () => scaleTree(treeZoom * 1.25));
        byId('pruningTreeFit').addEventListener('click', fitTree);
        byId('pruningTreeActual').addEventListener('click', () => scaleTree(1));
        byId('pruningBaselineFit').addEventListener('click', () => scaleBaseline(true));
        byId('pruningBaselineActual').addEventListener('click', () => scaleBaseline(false));
    }

    function scaleTree(zoom) {
        if (!treeSize) return;
        treeZoom = Math.max(0.02, Math.min(3, zoom));
        const svg = byId('pruningTreeSvg');
        svg.style.width = `${treeSize.width * treeZoom}px`;
        svg.style.height = `${treeSize.height * treeZoom}px`;
        byId('pruningTreeZoomLabel').textContent = `${Math.round(treeZoom * 100)}%`;
    }

    function fitTree() {
        if (!treeSize) return;
        const viewport = byId('pruningTreeViewport');
        // Fit both dimensions: the initial view must include the bottommost
        // removed nodes, not silently clip them below the scroll viewport.
        scaleTree(Math.min(1, (viewport.clientWidth - 20) / treeSize.width, 520 / treeSize.height));
        viewport.scrollTop = 0;
        viewport.scrollLeft = 0;
    }

    function scaleBaseline(fit) {
        if (!baselineSize) return;
        const viewport = byId('pruningBaselineViewport');
        const scale = fit ? Math.min(1, (viewport.clientWidth - 20) / baselineSize.width, 520 / baselineSize.height) : 1;
        const svg = byId('pruningBaselineSvg');
        svg.style.width = `${baselineSize.width * scale}px`;
        svg.style.height = `${baselineSize.height * scale}px`;
        viewport.scrollTop = 0;
        viewport.scrollLeft = 0;
    }

    function svgElement(tag, attrs, text) {
        const element = document.createElementNS('http://www.w3.org/2000/svg', tag);
        Object.entries(attrs || {}).forEach(([name, value]) => element.setAttribute(name, value));
        if (text !== undefined) element.textContent = text;
        return element;
    }

    function renderTreeComparison(comparison, method) {
        initTreeComparison();
        const panel = byId('pruningTreeComparison');
        panel.hidden = !comparison || !comparison.nodes || !comparison.nodes.length;
        if (panel.hidden) { treeSize = null; return; }
        const svg = byId('pruningTreeSvg');
        svg.replaceChildren();
        byId('pruningBaselineSvg').replaceChildren();
        byId('pruningBaselinePanel').hidden = true;
        baselineSize = null;
        treeSize = null;
        // Do not silently fall back to the old, incomplete matching-only graph.
        const selected = comparison.selected_nodes || [];
        const ids = new Set(selected.map(node => node.id));
        if (selected.length !== comparison.selected_node_count || ids.size !== selected.length) {
            byId('pruningTreeSummary').textContent = '树结构数据不完整，请重启平台服务后重新训练；此处不展示不完整的树图。';
            byId('pruningCurrentTreeTitle').textContent = '等待完整模型数据';
            byId('pruningTreeCaveat').textContent = '';
            byId('pruningTreeDetails').textContent = '';
            return;
        }
        byId('pruningShowRemoved').checked = true;
        panel.classList.remove('hide-removed');
        const counts = comparison.counts;
        const exact = comparison.exact_subtree;
        const selectedById = new Map(selected.map(node => [node.id, node]));
        byId('pruningShowRemovedLabel').hidden = !exact;
        byId('pruningRemovedLegendItem').hidden = !exact;
        byId('pruningCutLegend').textContent = comparison.cut_label;
        byId('pruningRemovedLegend').textContent = comparison.removed_label;
        byId('pruningTreeModeBadge').textContent = exact ? '同图灰圈对照' : '前后双图对照';
        byId('pruningTreeSubtitle').textContent = exact
            ? '彩色圆圈完整展示当前模型；灰色圆圈另行标注被移除的后代。'
            : '结构发生变化：上图是完整的当前模型，下图是独立的剪枝前基线。';
        const caveats = [];
        if (method === 'pre') caveats.push('预剪枝在生长时阻止分裂；灰圈是未剪枝基线中对应的后续节点，不表示先长出来再删除。');
        else caveats.push('灰圈及灰色虚线仅用于对照，不参与当前模型预测；剪枝位置本身保留为叶节点。');
        if (!exact) caveats.push('两棵树并非简单的删除关系，不能把净减少的节点数直接画成同样数量的灰圈。上图包含全部当前节点，不用旧树的橙色节点代替。');
        if (exact && !counts.removed) caveats.push('当前模型与未剪枝基线结构一致，没有被剪除的分支。');
        caveats.push('F 编号属于当前模型，B 编号属于剪枝前基线。大树可滚动查看，或点击完整视图查看全貌。');
        byId('pruningTreeCaveat').textContent = caveats.join(' ');
        byId('pruningTreeDetails').textContent = '点击或键盘选中圆圈，查看节点信息。灰色节点不计入当前模型节点数。';

        treeSize = drawTreeGraph(exact ? comparison.nodes : selected, svg,
            exact ? 'overlay' : 'current', comparison, selectedById, method);
        const actualIds = [...svg.querySelectorAll('[data-model-node-id]')]
            .map(element => Number(element.dataset.modelNodeId));
        if (actualIds.length !== selected.length || new Set(actualIds).size !== selected.length
            || actualIds.some(id => !selectedById.has(id))) {
            svg.replaceChildren();
            treeSize = null;
            byId('pruningCurrentTreeTitle').textContent = '绘图数量校验失败，已停止显示';
            byId('pruningTreeSummary').textContent = '模型统计与实际绘图不一致，请勿使用此图。';
            return;
        }
        byId('pruningCurrentTreeTitle').textContent = `当前模型 · ${actualIds.length} / ${comparison.selected_node_count} 个节点已绘制`;
        byId('pruningTreeSummary').textContent = `未剪枝 ${comparison.baseline_node_count} 个 → 当前模型 ${comparison.selected_node_count} 个 · 净减少 ${comparison.baseline_node_count - comparison.selected_node_count} 个`
            + (exact ? ` · 另示灰色节点 ${counts.removed} 个（不计入当前模型）` : ' · 结构变化，分开对照');
        if (!exact) {
            const baselinePanel = byId('pruningBaselinePanel');
            baselinePanel.hidden = false;
            baselinePanel.open = true;
            byId('pruningBaselineTitle').textContent = `剪枝前基线 · ${comparison.baseline_node_count} 个节点（独立对照，不是当前模型）`;
            baselineSize = drawTreeGraph(comparison.nodes, byId('pruningBaselineSvg'),
                'baseline', comparison, selectedById, method);
            scaleBaseline(false);
        }
        // Readable by default, even when fitting a large baseline would make
        // every label tiny. Full overview remains an explicit user choice.
        const viewport = byId('pruningTreeViewport');
        scaleTree(Math.max(0.8, Math.min(1, (viewport.clientWidth - 20) / treeSize.width, 620 / treeSize.height)));
        viewport.scrollTop = 0;
        viewport.scrollLeft = Math.max(0, treeSize.rootX * treeZoom - viewport.clientWidth / 2);
    }

    function drawTreeGraph(sourceNodes, svg, mode, comparison, selectedById, method) {
        const nodes = sourceNodes.map(node => ({ ...node, children: [] }));
        const byNodeId = new Map(nodes.map(node => [node.id, node]));
        nodes.forEach(node => {
            if (node.parent !== null) byNodeId.get(node.parent).children.push(node);
        });
        let leaves = 0;
        // Input is preorder. Reverse order provides a non-recursive postorder.
        [...nodes].reverse().forEach(node => {
            node.x = node.children.length ? node.children.reduce((sum, child) => sum + child.x, 0) / node.children.length : leaves++ * 88 + 48;
            node.y = node.depth * 100 + 48;
        });
        const width = Math.max(360, leaves * 88 + 8);
        // Reverse the temporary right-to-left leaf coordinates for natural L/R order.
        nodes.forEach(node => { node.x = width - node.x; });
        const height = Math.max(...nodes.map(node => node.y)) + 65;
        svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
        const edges = svgElement('g');
        const circles = svgElement('g');
        svg.append(edges, circles);
        nodes.forEach(node => {
            if (node.parent !== null) {
                const parent = byNodeId.get(node.parent);
                const removed = node.status === 'removed';
                const changed = ['changed', 'unmatched'].includes(node.status);
                const edge = svgElement('path', {
                    d: `M ${parent.x} ${parent.y + 24} L ${node.x} ${node.y - 24}`,
                    class: `pruning-tree-edge${removed ? ' is-removed' : changed ? ' is-changed' : ''}`
                });
                if (mode !== 'baseline' && !removed) {
                    edge.dataset.parentModelNodeId = parent.selected_id;
                    edge.dataset.childModelNodeId = node.selected_id;
                }
                edges.append(edge);
            }
            const currentNode = mode !== 'baseline' && node.status !== 'removed'
                ? selectedById.get(node.selected_id) : null;
            const data = currentNode || node;
            const state = node.status === 'retained' && data.is_leaf ? 'leaf' : node.status;
            const stateLabels = {retained: '分裂节点', leaf: '叶节点', cut: comparison.cut_label,
                removed: comparison.removed_label, changed: '分裂规则变化', unmatched: '无法对应的基线节点'};
            const rule = data.is_leaf ? `预测类别 ${data.prediction}` : `特征 ${data.feature} ≤ ${Number(data.threshold).toPrecision(4)}`;
            const labelId = currentNode ? `F${currentNode.id}` : `B${node.id}`;
            const origin = currentNode ? '当前模型' : '剪枝前基线';
            const description = `${origin}节点 ${labelId} · ${stateLabels[state]} · ${rule} · 训练样本 ${data.samples} · 不纯度 ${Number(data.impurity).toFixed(4)}`;
            const group = svgElement('g', {class: `pruning-tree-node is-${state}`, transform: `translate(${node.x},${node.y})`,
                tabindex: '0', role: 'button', 'aria-label': description, 'data-node-id': node.id});
            if (currentNode) group.dataset.modelNodeId = currentNode.id;
            group.append(svgElement('title', {}, description));
            group.append(svgElement('circle', {r: 24}));
            if (state === 'cut') group.append(svgElement('circle', {r: 29, class: 'pruning-cut-ring'}));
            group.append(svgElement('text', {y: -3, class: 'pruning-tree-id'}, labelId));
            const label = state === 'cut' ? (currentNode ? '转为叶' : '停止点') : state === 'removed' ? (method === 'pre' ? '未生长' : '已剪除') : state === 'changed' || state === 'unmatched' ? '旧结构' : data.is_leaf ? '叶节点' : `特征 ${data.feature}`;
            group.append(svgElement('text', {y: 12, class: 'pruning-tree-label'}, label));
            group.append(svgElement('text', {y: 44, class: 'pruning-tree-samples'}, `${data.samples} 样本`));
            const inspect = () => { byId('pruningTreeDetails').textContent = description; };
            group.addEventListener('click', inspect);
            group.addEventListener('focus', inspect);
            group.addEventListener('keydown', event => {
                if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); inspect(); }
            });
            circles.append(group);
        });
        return {width, height, rootX: nodes[0].x};
    }

    function kpi(label, value, valueClass) {
        return `
            <div class="pruning-kpi">
                <div class="pruning-kpi__label">${label}</div>
                <div class="pruning-kpi__value ${valueClass}">${value}</div>
            </div>
        `;
    }

    function percent(value) {
        const numeric = Number(value);
        return Number.isFinite(numeric) ? `${(numeric * 100).toFixed(1)}%` : '--';
    }

    function signedPercent(value) {
        const numeric = Number(value);
        if (!Number.isFinite(numeric)) return '--';
        const sign = numeric > 0 ? '+' : '';
        return `${sign}${(numeric * 100).toFixed(1)} 个百分点`;
    }

    function formatAlpha(value) {
        if (!Number.isFinite(value)) return '0';
        return value < 0.001 && value > 0 ? value.toExponential(2) : value.toFixed(4);
    }

    function drawPath(points) {
        const canvas = byId('pruningPathCanvas');
        if (!canvas || !points.length) return;
        const rect = canvas.getBoundingClientRect();
        const width = Math.max(220, rect.width || 520);
        const height = 265;
        const ratio = window.devicePixelRatio || 1;
        canvas.width = width * ratio;
        canvas.height = height * ratio;
        const context = canvas.getContext('2d');
        context.setTransform(ratio, 0, 0, ratio, 0, 0);
        context.clearRect(0, 0, width, height);

        const margin = { top: 20, right: 18, bottom: 44, left: 48 };
        const plotWidth = width - margin.left - margin.right;
        const plotHeight = height - margin.top - margin.bottom;
        const scores = points.flatMap((point) => [point.cv_mean, point.test_accuracy])
            .filter((value) => typeof value === 'number' && Number.isFinite(value));
        const minY = Math.max(0, Math.min(...scores) - 0.05);
        const maxY = Math.min(1, Math.max(...scores) + 0.04);
        const epsilon = 1e-7;
        const xValues = points.map((point) => Math.log10(point.alpha + epsilon));
        const minX = Math.min(...xValues);
        const maxX = Math.max(...xValues);

        const x = (index) => margin.left + ((xValues[index] - minX) / (maxX - minX || 1)) * plotWidth;
        const y = (value) => margin.top + (1 - (value - minY) / (maxY - minY || 1)) * plotHeight;

        context.strokeStyle = '#D7DFEA';
        context.fillStyle = '#6B7890';
        context.font = '11px Microsoft YaHei, sans-serif';
        context.lineWidth = 1;
        for (let tick = 0; tick <= 4; tick += 1) {
            const value = minY + ((maxY - minY) * tick) / 4;
            const yPos = y(value);
            context.beginPath();
            context.moveTo(margin.left, yPos);
            context.lineTo(width - margin.right, yPos);
            context.stroke();
            context.fillText(`${(value * 100).toFixed(0)}%`, 7, yPos + 4);
        }

        drawSeries(context, points, x, y, 'cv_mean', '#4F46E5', false);
        drawSeries(context, points, x, y, 'test_accuracy', '#D99A20', true);

        const selectedIndex = points.findIndex((point) => point.selected);
        if (selectedIndex >= 0) {
            context.strokeStyle = '#0F9D8A';
            context.setLineDash([4, 4]);
            context.beginPath();
            context.moveTo(x(selectedIndex), margin.top);
            context.lineTo(x(selectedIndex), height - margin.bottom);
            context.stroke();
            context.setLineDash([]);
            context.fillStyle = '#0F9D8A';
            const selectionLabel = points[selectedIndex].selection_label || '选择位置';
            context.fillText(selectionLabel, Math.min(x(selectedIndex) + 5, width - 68), margin.top + 10);
        }

        context.fillStyle = '#6B7890';
        const labelIndices = [...new Set([0, Math.floor((points.length - 1) / 2), points.length - 1])];
        context.textAlign = 'center';
        labelIndices.forEach((index) => {
            const px = Math.max(margin.left + 14, Math.min(x(index), width - margin.right - 14));
            context.fillText(points[index].alpha === 0 ? '0' : points[index].alpha.toPrecision(2), px, height - margin.bottom + 16);
        });
        context.textAlign = 'left';
        context.fillText('ccp_alpha（对数刻度）', Math.max(margin.left, width / 2 - 58), height - 10);
        context.fillStyle = '#4F46E5';
        context.fillRect(width - 165, 8, 10, 3);
        context.fillText('交叉验证', width - 150, 13);
        context.fillStyle = '#D99A20';
        context.fillRect(width - 82, 8, 10, 3);
        context.fillText('测试集', width - 67, 13);
    }

    function drawSeries(context, points, x, y, key, color, dashed) {
        context.strokeStyle = color;
        context.fillStyle = color;
        context.lineWidth = 2;
        context.setLineDash(dashed ? [5, 4] : []);
        context.beginPath();
        let started = false;
        points.forEach((point, index) => {
            if (typeof point[key] !== 'number' || !Number.isFinite(point[key])) { started = false; return; }
            const px = x(index);
            const py = y(point[key]);
            if (!started) context.moveTo(px, py);
            else context.lineTo(px, py);
            started = true;
        });
        context.stroke();
        context.setLineDash([]);
        points.forEach((point, index) => {
            if (typeof point[key] !== 'number' || !Number.isFinite(point[key])) return;
            context.beginPath();
            context.arc(x(index), y(point[key]), point.selected ? 4 : 2.5, 0, Math.PI * 2);
            context.fill();
        });
    }

    function reset() {
        lastPruningInfo = null;
        treeSize = null;
        baselineSize = null;
        const treePanel = byId('pruningTreeComparison');
        if (treePanel) treePanel.hidden = true;
        const treeSvg = byId('pruningTreeSvg');
        if (treeSvg) treeSvg.replaceChildren();
        const baselineSvg = byId('pruningBaselineSvg');
        if (baselineSvg) baselineSvg.replaceChildren();
        const baselinePanel = byId('pruningBaselinePanel');
        if (baselinePanel) baselinePanel.hidden = true;
        const panel = byId('pruningResults');
        if (panel) panel.classList.remove('is-visible');
        const canvas = byId('pruningPathCanvas');
        if (canvas) canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
    }

    window.DecisionTreePruningUI = { init, getParams, render, reset };
})();
