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
                <a class="viz-home-link" href="/">${homeIcon}<span>算法首页</span></a>
            </div>`;
        document.body.prepend(bar);
    }

    function enhanceHelp() {
        const candidates = document.querySelectorAll('.instructions, .algorithm-explanation');
        candidates.forEach((panel) => {
            if (panel.classList.contains('viz-help')) return;
            const text = panel.textContent || '';
            if (!text.includes('使用说明')) return;
            panel.classList.add('viz-help', 'viz-help--collapsed');
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'viz-help-toggle';
            button.setAttribute('aria-expanded', 'false');
            button.setAttribute('aria-label', '展开使用说明');
            button.textContent = '＋';
            button.addEventListener('click', () => {
                const collapsed = panel.classList.toggle('viz-help--collapsed');
                button.textContent = collapsed ? '＋' : '−';
                button.setAttribute('aria-expanded', String(!collapsed));
                button.setAttribute('aria-label', collapsed ? '展开使用说明' : '收起使用说明');
            });
            panel.appendChild(button);
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
        const emojiPrefix = /^\s*[\p{Extended_Pictographic}\uFE0F\u200D]+\s*/u;
        document.querySelectorAll('button, h3, h4, .explanation-title').forEach((element) => {
            element.childNodes.forEach((node) => {
                if (node.nodeType === Node.TEXT_NODE && emojiPrefix.test(node.textContent || '')) {
                    node.textContent = (node.textContent || '').replace(emojiPrefix, '');
                }
            });
        });
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
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, { once: true });
    else init();
})();
