/* Shared data + six model canvases. Boundaries always come from server predict. */
(() => {
    'use strict';
    const root = document.getElementById('ml-arena');
    if (!root) return;
    const $ = id => document.getElementById('arena-' + id);
    const keys = ['svm', 'knn', 'tree', 'forest', 'mlp', 'bayes'];
    const names = {svm: 'SVM', knn: 'KNN', tree: '决策树', forest: '随机森林', mlp: 'MLP', bayes: '高斯贝叶斯'};
    const modelHints = {
        svm: '核函数控制全局边界；决策分数不等于概率。',
        knn: '点击样本查看真实邻居及所选距离度量的实际距离。',
        tree: '点击样本查看模型实际经过的分裂路径。',
        forest: '最终分类由各树的类别概率平均决定，并非简单多数票。',
        mlp: '使用所选网络结构、激活函数和求解器；显示实际训练状态。',
        bayes: '根据两类的高斯分布与条件独立假设，计算类别后验概率。'
    };
    const colors = ['#6256ce', '#098b87'];
    const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
    const state = {
        base: [], points: [], version: 0, nextId: 0, tool: 'move', hover: null, selected: null,
        split: null, result: null, baseline: null, undo: [], busy: false, sequence: 0,
        generation: 0, controller: null, gesture: null, progress: 1, appearance: 1,
        compare: false, divider: 0.5, valid: false, requestCount: 0,
        schema: null, parametersReady: false, parametersValid: false, parameterSignature: null,
        finalResult: null, checkpoints: new Map(), playback: null, recording: false,
        selectedModels: [...keys], recordingProfiles: {}, recordingEstimate: null, recordingEstimateKey: null
    };
    const activeKeys=()=>keys.filter(key=>state.selectedModels.includes(key));
    const canvases = {data: $('data'), ...Object.fromEntries(keys.map(k => [k, $('plot-' + k)]))};
    const clone = x => JSON.parse(JSON.stringify(x));
    const fmt = x => (x * 100).toFixed(1) + '%';
    const delta = x => (x >= 0 ? '+' : '') + (x * 100).toFixed(1) + ' pp';
    const numeric = id => Number($(id).value);
    const challenge = () => ({noise: numeric('challenge-noise'), corruption: numeric('corruption')});
    const challenged = () => challenge().noise > 0 || challenge().corruption > 0;
    const payload = (withParams = true) => ({
        points: state.base, version: state.version, seed: numeric('seed'), test_size: numeric('test-size'),
        challenge: challenge(), selected_models: activeKeys(), ...(withParams ? {params: parameterValues()} : {})
    });
    function fieldValue(model, field) {
        const input = $(field.id);
        if (field.type === 'boolean') return input.checked;
        if (['gamma','layers','priors'].includes(field.type)) {
            if (input.value === 'custom') {
                const value = $(field.id + '-custom').value.trim();
                return field.type === 'gamma' ? Number(value) : value;
            }
            if (field.type === 'gamma') return input.value;
            if (field.type === 'priors') return null;
            return input.value;
        }
        if (field.type === 'batch') return input.value.trim() === 'auto' ? 'auto' : Number(input.value);
        if (field.type === 'select') return input.value;
        return field.nullable && input.value.trim() === '' ? null : Number(input.value);
    }
    function parameterValues() {
        return Object.fromEntries(activeKeys().map(model => [model, Object.fromEntries(
            state.schema[model].fields.filter(f => !$(f.id).disabled).map(f => [f.name,fieldValue(model,f)])
        )]));
    }
    function parameterConditions() {
        keys.forEach(model => {
            const competing=state.selectedModels.includes(model);
            const fields = state.schema[model].fields;
            const values = Object.fromEntries(fields.map(f => [f.name,fieldValue(model,f)]));
            const matches = conditions => Object.entries(conditions || {}).every(([name,choices]) => choices.includes(values[name]));
            fields.forEach(f => {
                const input = $(f.id), wrapper = input.closest('.arena-param-field');
                const enabled = matches(f.when) && !(f.inactive_when && matches(f.inactive_when));
                wrapper.hidden = !enabled;
                input.disabled = !enabled || !competing;
                const extra = $(f.id+'-custom');
                if (extra) {
                    extra.hidden = input.value !== 'custom';
                    extra.disabled = !enabled || !competing || input.value !== 'custom';
                }
            });
            const panel=$('model-parameters').querySelector('[data-parameter-model="'+model+'"]');
            panel.classList.toggle('is-not-competing',!competing);
            panel.querySelector('summary span').textContent=competing?'完整参数':'未参赛';
        });
        state.parametersValid = [...$('model-parameters').querySelectorAll('input,select')].every(input => input.disabled || input.checkValidity());
    }
    function parameterChanged() {
        parameterConditions();
        const signature = JSON.stringify(parameterValues());
        if (signature === state.parameterSignature) return;
        state.parameterSignature = signature;
        dirty(state.parametersValid ? '模型参数已改变，旧预测失效；重新对决将使用全部实际设置。' : '参数格式或范围有误，请修正后重新对决。');
    }
    async function loadParameters() {
        const response = await fetch('/api/arena/parameters');
        const result = await response.json();
        if (!response.ok || result.status !== 'success' || !keys.every(k => result.models?.[k])) throw new Error('完整参数目录加载失败，请确认已重启最新后端。');
        state.schema = result.models;
        const host = $('model-parameters');
        host.replaceChildren();
        keys.forEach(model => {
            const spec = state.schema[model], panel = document.createElement('details');
            panel.dataset.parameterModel = model;
            panel.open = model === 'svm';
            const summary = document.createElement('summary'), title = document.createElement('b'), link = document.createElement('a');
            title.textContent = spec.title;
            link.textContent = '原实验 ↗'; link.href = spec.lab; link.target = '_blank'; link.rel = 'noopener';
            link.addEventListener('click',event => event.stopPropagation());
            const badge = document.createElement('span');badge.textContent = '完整参数';
            summary.append(title,badge);
            link.className = 'arena-param-lab-link';
            panel.append(summary,link);
            spec.fields.forEach(f => {
                const label = document.createElement('label');label.className = 'arena-param-field';
                const caption = document.createElement('span');caption.textContent = f.label;
                label.append(caption);
                const special = ['gamma','layers','priors'].includes(f.type);
                const isSelect = f.type === 'select' || special;
                const input = document.createElement(isSelect ? 'select' : 'input');
                input.id = 'arena-'+f.id;input.dataset.parameterName = f.name;
                if (isSelect) {
                    let options = f.options || [];
                    if (f.type === 'gamma') options = [['custom','指定数值'],['scale','scale（训练集方差）'],['auto','auto（1 / 特征数）']];
                    if (f.type === 'priors') options = [['auto','自动估计'],['custom','指定 A、B 先验']];
                    if (f.type === 'layers') options = [...options,['custom','自定义层结构']];
                    options.forEach(([value,text]) => input.append(new Option(text,value)));
                    input.value = f.type === 'gamma' ? (typeof f.default === 'number' ? 'custom' : f.default) :
                        f.type === 'priors' ? 'auto' : String(f.default);
                } else {
                    input.type = f.type === 'boolean' ? 'checkbox' : f.type === 'batch' ? 'text' : 'number';
                    if (f.type === 'boolean') input.checked = f.default;
                    else input.value = f.default === null ? '' : f.default;
                    if (f.min !== undefined) input.min = f.min;
                    if (f.max !== undefined) input.max = f.max;
                    input.step = f.step || (f.integer ? '1' : 'any');
                    input.required = !f.nullable && f.type !== 'boolean';
                    if (f.nullable) input.placeholder = '不限';
                    if (f.type === 'batch') input.pattern = 'auto|[1-9][0-9]*';
                }
                label.append(input);
                if (special) {
                    const extra = document.createElement('input');
                    extra.id = input.id+'-custom';
                    extra.type = f.type === 'gamma' ? 'number' : 'text';
                    extra.required = true;
                    extra.setAttribute('aria-label',f.label+'自定义值');
                    extra.value = f.type === 'gamma' ? f.default : f.type === 'layers' ? '64,32,16' : '0.5,0.5';
                    if (f.type === 'gamma') { extra.min=f.min;extra.max=f.max;extra.step=f.step; }
                    label.append(extra);
                }
                panel.append(label);
            });
            if (spec.note) {const note=document.createElement('p');note.className='arena-fairness';note.textContent=spec.note;panel.append(note);}
            panel.addEventListener('toggle',() => {
                if (panel.open) host.querySelectorAll('details').forEach(other => {if(other!==panel)other.open=false;});
            });
            host.append(panel);
        });
        host.addEventListener('input',parameterChanged);
        host.addEventListener('change',parameterChanged);
        state.parametersReady = true;
        parameterConditions();
        state.parameterSignature = JSON.stringify(parameterValues());
    }
    const fresh = () => state.result && state.result.version === state.version;
    const observing = () => Boolean(state.result?.observation);
    keys.forEach(key => {
        const panel = document.createElement('div');
        panel.id = 'arena-process-' + key;
        panel.className = 'arena-process'; panel.hidden = true;
        $('explain-' + key).before(panel);
        const time = document.createElement('div');time.id='arena-track-time-'+key;time.className='arena-track-time';time.hidden=true;
        time.innerHTML='<span></span><div><i></i></div>';
        root.querySelector('[data-model="'+key+'"] header').after(time);
        const article=root.querySelector('[data-model="'+key+'"]');
        const details=document.createElement('details');details.className='arena-model-data';details.id='arena-model-data-'+key;
        const summary=document.createElement('summary');summary.textContent='实验数据与过程详情';
        summary.setAttribute('aria-controls','arena-model-data-body-'+key);
        const body=document.createElement('div');body.className='arena-model-data-body';body.id='arena-model-data-body-'+key;
        body.append(time,$('score-'+key),panel,$('explain-'+key),article.querySelector('.arena-effective-params'));
        details.append(summary,body);article.append(details);
        details.addEventListener('toggle',updateDataDisclosure);
    });
    let layoutFrame=0;
    function participantControls() {
        const selected=activeKeys(), count=selected.length;
        root.querySelectorAll('[data-arena-participant]').forEach(input=>{
            input.checked=selected.includes(input.dataset.arenaParticipant);
            input.disabled=count===2 && input.checked;
        });
        keys.forEach(key=>root.querySelector('[data-model="'+key+'"]').hidden=!selected.includes(key));
        $('participant-count').textContent=count+' / 6 · 至少选 2 个';
        $('participants-all').disabled=count===6;
        root.querySelector('.arena-heading .arena-eyebrow').textContent='ML ARENA / 同一份数据，'+count+' 种答案';
        root.querySelector('.arena-heading p').textContent='拖动几个训练样本，重新对决。观察同一个变化，如何让 '+count+' 种决策边界作出不同响应。';
        root.querySelector('.arena-observe-heading h3').textContent='同一坐标，'+count+' 种决策方式';
        root.querySelector('.arena-inspect-bar>strong').setAttribute('data-participant-count',count);
        updateDataDisclosure();
    }
    function selectParticipants(selected) {
        const next=keys.filter(key=>selected.includes(key));
        if(next.length<2){status('至少需要两个算法才能 PK。','error');participantControls();return;}
        if(next.join(',')===activeKeys().join(','))return;
        // Stop old replay before replacing its participant set.
        leavePlayback(true);
        state.selectedModels=next;
        state.baseline=null;
        if(state.parametersReady){parameterConditions();state.parameterSignature=JSON.stringify(parameterValues());}
        participantControls();setAllDataOpen(false);
        $('metrics-body').innerHTML='<tr><td colspan="8">参赛阵容已改变，请重新对决后查看当前阵容的指标。</td></tr>';
        $('compare-note').textContent='';
        dirty('参赛阵容已更新为 '+next.map(key=>names[key]).join('、')+'；旧结果及演示已失效，请重新对决或记录。');
    }
    function updateDataDisclosure() {
        const selected=activeKeys(),count=selected.filter(key=>$('model-data-'+key).open).length;
        root.classList.toggle('arena-data-collapsed',count===0);
        $('data-fold-state').textContent=count ? '已展开 '+count+' / '+selected.length+' 项详细数据' : '详细数据已全部收起 · '+selected.length+' 模型画布总览';
        $('data-expand').disabled=count===selected.length;
        $('data-collapse').disabled=count===0;
        keys.forEach(key=>$('model-data-'+key).querySelector('summary').setAttribute('aria-expanded',$('model-data-'+key).open));
        scheduleCanvasFit();requestDraw();
    }
    function scheduleCanvasFit() {
        if(layoutFrame)return;
        layoutFrame=requestAnimationFrame(()=>{layoutFrame=0;fitCanvasHeights();});
    }
    function fitCanvasHeights() {
        const topline=root.querySelector('.arena-topline'),bar=$('viewbar');
        const count=activeKeys().length;
        const columns=innerWidth<761?1:innerWidth<=1050?Math.min(2,count):count===4?2:Math.min(3,count);
        root.style.setProperty('--arena-model-columns',columns);
        $('fit-canvases').textContent=innerWidth<761?'定位模型画布 ↓':count+' 画布同屏 ↗';
        const stickyTop=(parseFloat(getComputedStyle(topline).top)||0)+topline.getBoundingClientRect().height;
        bar.style.scrollMarginTop=(stickyTop+8)+'px';
        root.style.setProperty('--arena-controls-scroll-top',(stickyTop+12)+'px');
        if(!root.classList.contains('arena-data-collapsed') || innerWidth<761){root.style.removeProperty('--arena-collapsed-plot-height');return;}
        const grid=root.querySelector('.arena-models'),style=getComputedStyle(grid);
        const actualColumns=style.gridTemplateColumns.split(' ').length, rows=Math.ceil(count/actualColumns);
        const cards=[...grid.children].filter(card=>!card.hidden), gap=parseFloat(style.rowGap)||0;
        let chrome=0;
        for(let row=0;row<rows;row++)chrome+=Math.max(...cards.slice(row*actualColumns,(row+1)*actualColumns).map(card=>
            card.getBoundingClientRect().height-card.querySelector('.arena-model-plot').getBoundingClientRect().height));
        const available=innerHeight-stickyTop-bar.getBoundingClientRect().height-28;
        const height=Math.max(65,Math.min(280,Math.floor((available-chrome-gap*(rows-1))/rows)));
        const value=height+'px';
        if(root.style.getPropertyValue('--arena-collapsed-plot-height')!==value)root.style.setProperty('--arena-collapsed-plot-height',value);
    }
    function setAllDataOpen(open) {
        activeKeys().forEach(key=>$('model-data-'+key).open=open);
        updateDataDisclosure();
    }
    function showSixCanvases() {
        setAllDataOpen(false);
        requestAnimationFrame(()=>{
            fitCanvasHeights();
            $('viewbar').scrollIntoView({block:'start',behavior:'instant'});
            requestDraw();
        });
    }
    function status(message, kind = '') {
        $('status').textContent = message;
        $('status').dataset.state = kind;
    }
    function buttonState() {
        playbackControls();
        updateRecordingEstimate();
        $('run').disabled = state.busy || !state.valid || !state.parametersReady || !state.parametersValid;
        const count=activeKeys().length;
        $('run').textContent = state.busy ? count+' 个模型计算中…' : state.result ? '重新对决 · 更新 '+count+' 个边界 →' : '开始 '+count+' 模型对决 →';
        $('undo').disabled = !state.undo.length;
        $('baseline').disabled = !fresh() || observing();
        root.classList.toggle('is-observing', observing());
        const unavailable = state.busy || !state.valid || !state.parametersReady || !state.parametersValid;
        root.querySelectorAll('[data-observe-step],#arena-step-next,#arena-step-ten,#arena-step-go').forEach(b => b.disabled = unavailable);
        ['step-next','step-ten'].forEach(id=>{
            $(id).disabled=unavailable || Boolean(state.playback?.active);
            $(id).title=state.playback?.active?'时间演示请使用时间轴；选择固定步数可返回单步观察。':'';
        });
        $('step-history').disabled = state.busy || !state.checkpoints.size;
        $('step-final').disabled = state.busy || !observing() || state.finalResult?.version !== state.version;
        $('step-mode').textContent = state.result?.observation?.timeline ? '连续时间演示 · 各模型独立推进 · 不参与排名' : observing() ? '过程快照 · 第 ' + state.result.observation.step + ' 步 · 不参与排名' : '完整对决与过程快照独立，点击步数开始观察';
        root.querySelectorAll('[data-observe-step]').forEach(b=>b.setAttribute('aria-pressed',Boolean(fresh() && observing() && Number(b.dataset.observeStep)===state.result.observation.step)));
        $('version').textContent = '数据 v' + state.version + (fresh() ? ' · 结果已同步' : ' · ' + (state.result ? '需要重新训练' : '未训练'));
        root.classList.toggle('is-stale', Boolean(state.result && !fresh()));
        root.dataset.version = state.version;
        root.dataset.resultVersion = state.result ? state.result.version : '';
        root.dataset.fresh = Boolean(fresh());
        keys.forEach(k => {
            const tag = root.querySelector('[data-model="' + k + '"] .arena-stale-tag');
            tag.hidden = Boolean(fresh() && state.result.models[k]);
            tag.textContent = fresh() && state.result?.observation?.timeline ? '第一步进行中' : state.result ? '旧边界 · 需要重新训练' : '等待真实训练';
            const summary = root.querySelector('[data-model="' + k + '"] .arena-effective-params summary');
            summary.textContent = fresh() ? '本轮实际生效参数' : state.result ? '上一轮参数 · 已过期' : '训练后查看实际参数';
            if (!fresh()) {
                $('rank-' + k).textContent = state.result ? '结果过期' : '待对决';
                delete $('rank-' + k).dataset.rank;
            }
        });
        const comparable = !observing() && state.baseline && fresh() &&
            state.baseline.holdout_fingerprint === state.result.holdout_fingerprint;
        $('compare').hidden = !comparable;
        if (!comparable) {
            state.compare = false;
            $('compare-enabled').checked = false;
        }
    }
    function remember() {
        state.undo.push({base: clone(state.base), challenge: challenge()});
        if (state.undo.length > 30) state.undo.shift();
        buttonState();
    }
    function dirty(message) {
        leavePlayback(true);
        state.version++;
        state.checkpoints.clear();
        state.finalResult = null;
        updateHistory();
        state.sequence++;
        if (state.controller) state.controller.abort();
        state.busy = false;
        state.appearance = 1;
        state.compare = false;
        $('compare-enabled').checked = false;
        state.valid = state.base.length >= 24 && [0, 1].every(c => state.base.filter(p => p.label === c).length >= 6);
        status(message || '数据已改变。旧边界已失效，请重新对决。');
        updateInspector();
        buttonState();
        requestDraw();
    }
    async function api(path, data, signal) {
        state.requestCount++;
        const response = await fetch('/api/arena/' + path, {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data), signal
        });
        let result;
        try { result = await response.json(); } catch (_) { throw new Error('服务返回格式异常，请确认 Flask 正常运行。'); }
        if (!response.ok || result.status !== 'success') throw new Error(result.message || '请求失败，请重试。');
        return result;
    }
    function acceptPrepared(result) {
        state.points = result.points;
        state.split = result.split;
        state.valid = true;
        $('split').textContent = result.split.train_ids.length + ' 训练 / ' + result.split.test_ids.length + ' 测试';
        const ch = result.challenge;
        $('hint').textContent = ch.noise || ch.corruption ?
            '挑战预览：测试集不变 · 污染 ' + ch.corrupted_count + ' 个训练标签' +
            (ch.clipped_count ? ' · ' + ch.clipped_count + ' 点已限于画布内' : '') + '。编辑前请重置挑战。' :
            '实心为训练，空心为测试。拖动不改变划分；增删、换类后重新分层划分。';
        buttonState();
        requestDraw();
    }
    async function prepare() {
        const version = state.version;
        if (!state.valid) {
            state.points = clone(state.base);
            state.split = null;
            $('split').textContent = state.base.length + ' 样本 · 尚不能划分';
            status('需要至少 24 个样本，且 A、B 两类各至少 6 个，才可独立评估。');
            buttonState();
            requestDraw();
            return;
        }
        try {
            const result = await api('prepare', payload(false));
            if (version !== state.version) return;
            acceptPrepared(result);
        } catch (error) {
            if (version !== state.version) return;
            state.valid = false;
            status(error.message, 'error');
            buttonState();
        }
    }
    function resetChallengeValues() {
        $('challenge-noise').value = '0';
        $('corruption').value = '0';
        challengeLabels();
    }
    function challengeLabels() {
        $('challenge-noise-value').value = numeric('challenge-noise').toFixed(2);
        $('corruption-value').value = Math.round(numeric('corruption') * 100) + '%';
    }
    async function generate() {
        const ticket = ++state.generation;
        const kind = $('preset').value;
        if (kind === 'free') {
            remember();
            resetChallengeValues();
            state.base = state.points = [];
            state.selected = state.hover = null;
            state.split = null;
            state.baseline = null;
            state.nextId = 0;
            setTool('a');
            dirty('自由绘制：选择画 A 或画 B，点击或拖动添加真实样本。');
            await prepare();
            return;
        }
        // Invalidate immediately, including an in-flight training request.
        dirty('正在生成真实二维样本…');
        state.valid = false;
        buttonState();
        const version = state.version;
        $('generate').disabled = true;
        try {
            const result = await api('generate', {kind, count: numeric('count'), noise: numeric('noise'), seed: numeric('seed')});
            if (ticket !== state.generation || version !== state.version) return;
            remember();
            resetChallengeValues();
            state.base = result.points;
            state.points = clone(result.points);
            state.nextId = 0;
            state.split = null;
            state.selected = state.hover = null;
            state.baseline = null;
            state.valid = true;
            setTool('move');
            status('数据已生成。选择 2～6 个参赛算法，开始对决以计算真实边界。');
            await prepare();
            const start = performance.now();
            function reveal(now) {
                if (version !== state.version) return;
                state.appearance = reduced ? 1 : Math.min(1, (now - start) / 330);
                requestDraw();
                if (state.appearance < 1) requestAnimationFrame(reveal);
            }
            requestAnimationFrame(reveal);
        } catch (error) {
            if (ticket === state.generation && version === state.version) status(error.message, 'error');
        } finally {
            if (ticket === state.generation) $('generate').disabled = false;
            buttonState();
        }
    }
    async function run() {
        if (!state.valid || state.busy || !state.parametersReady || !state.parametersValid) return;
        leavePlayback();
        const seq = ++state.sequence, version = state.version;
        state.busy = true;
        state.controller = new AbortController();
        status('共同划分已确认，正在计算 '+activeKeys().length+' 个参赛模型及真实预测网格…', 'busy');
        buttonState();
        const input = clone(payload());
        try {
            const result = await api('run', input, state.controller.signal);
            // A data edit or newer request makes the entire old round ineligible.
            if (seq !== state.sequence || version !== state.version) return;
            if (!activeKeys().every(key => result.models?.[key]) || result.model_keys.join(',')!==activeKeys().join(',')) {
                throw new Error('后端返回的参赛阵容不匹配，请重启本地服务后重试。');
            }
            result.layers = buildLayers(result);
            result.input = input;
            state.result = result;
            state.finalResult = result;
            updateHistory();
            acceptPrepared(result);
            if (!state.baseline || state.baseline.holdout_fingerprint !== result.holdout_fingerprint) {
                state.baseline = result;
            }
            names.mlp = result.models.mlp?.fit_info.network_type === 'perceptron' ? '单层感知机' : 'MLP';
            const warnings = activeKeys().filter(key => result.models[key].fit_info.convergence_warning);
            status(activeKeys().length+' 个参赛模型已计算完成 · 同步显现真实边界，非训练迭代。' +
                (warnings.length ? ' ' + warnings.map(key=>names[key]).join('、') + ' 未满足优化器停止条件，显示当前拟合结果。' : ''));
            updateMetrics();
            updateInspector();
            const started = performance.now();
            state.progress = reduced ? 1 : 0;
            function reveal(now) {
                if (seq !== state.sequence || version !== state.version) { state.progress = 1; return; }
                state.progress = reduced ? 1 : Math.min(1, (now - started) / 580);
                // All six are painted by the same animation frame and progress.
                requestDraw();
                if (state.progress < 1) requestAnimationFrame(reveal);
            }
            requestAnimationFrame(reveal);
        } catch (error) {
            if (seq !== state.sequence || error.name === 'AbortError') return;
            status(error.message, 'error');
        } finally {
            if (seq === state.sequence) { state.busy = false; state.controller = null; buttonState(); }
        }
    }
    function updateHistory() {
        const select = $('step-history');
        select.replaceChildren(new Option(state.checkpoints.size ? '选择已记录快照' : '尚无快照', ''));
        [...state.checkpoints.keys()].sort((a,b) => a-b).forEach(step => select.append(new Option('第 ' + step + ' 步',step)));
        select.value = observing() && state.checkpoints.has(state.result.observation.step) ? state.result.observation.step : '';
    }
    function showObservation(result) {
        state.result = result;
        state.compare = false;
        $('compare-enabled').checked = false;
        state.progress = state.appearance = 1;
        $('step-value').value = result.observation.step;
        acceptPrepared(result);
        if (!state.selected || !state.points.some(p => p.id === state.selected)) {
            state.selected = state.points.find(p => p.role === 'test')?.id || state.points[0]?.id;
        }
        updateHistory();updateMetrics();updateInspector();buttonState();requestDraw();
        status('已显示第 ' + result.observation.step + ' 步的真实结果；每个模型按自己的机制推进。点击样本检查，或回看已记录快照。');
    }
    async function observe(step) {
        if (state.busy || !state.valid || !state.parametersReady || !state.parametersValid) return;
        if (!Number.isInteger(step) || step < 1 || step > 10000) { status('观察步数需为 1～10000 的整数。','error');return; }
        leavePlayback();
        if (state.checkpoints.has(step)) { ++state.sequence;showObservation(state.checkpoints.get(step));return; }
        const seq = ++state.sequence, version = state.version;
        state.busy = true;state.controller = new AbortController();
        state.progress = 1;
        status('正在从相同起点计算 '+activeKeys().length+' 个参赛模型第 ' + step + ' 步的真实阶段…','busy');buttonState();
        try {
            const result = await api('observe', {...payload(),step},state.controller.signal);
            if (seq !== state.sequence || version !== state.version) return;
            result.layers = buildLayers(result);
            state.checkpoints.set(step,result);
            if (state.checkpoints.size > 16) state.checkpoints.delete(state.checkpoints.keys().next().value);
            showObservation(result);
        } catch (error) {
            if (seq === state.sequence && error.name !== 'AbortError') status(error.message,'error');
        } finally {
            if (seq === state.sequence) {state.busy=false;state.controller=null;buttonState();}
        }
    }
    function playbackControls() {
        const playback=state.playback, active=playback?.active;
        $('play-record').disabled=state.busy || !state.valid || !state.parametersReady || !state.parametersValid;
        $('play-record').textContent=state.recording?'正在记录真实步骤…':'记录并播放 ▶';
        $('play-cancel').hidden=!state.recording;
        $('play-pause').disabled=state.busy || !active || playback.clock.time>=playback.clock.duration;
        $('play-pause').textContent=active && playback.clock.playing?'暂停 Ⅱ':'继续播放 ▶';
        $('play-restart').disabled=state.busy || !playback || playback.record.version!==state.version;
        $('play-seek').disabled=state.busy || !active;
        keys.forEach(key=>$('track-time-'+key).hidden=!active);
    }
    function leavePlayback(discard=false) {
        if (state.playback) {
            state.playback.clock.pause(performance.now());
            cancelAnimationFrame(state.playback.raf);
            state.playback.active=false;
        }
        state.recording=false;
        if (discard) {state.playback=null;$('play-state').textContent='数据或参数变化后需重新记录';$('play-time').value='0.000 / 0.000 ms';$('play-seek').value=0;}
        else if(state.playback)$('play-state').textContent='时间演示已暂停；可从头播放已记录片段';
        playbackControls();
    }
    function beginPlayback(record) {
        if(!$('play-speed').checkValidity()){status('请先输入 0.00001～10 之间的有效速度倍率。','error');return;}
        const clock=new window.ArenaTimelineClock(record.tracks,numeric('play-speed'));
        state.playback={record,clock,active:true,positions:{},raf:0};
        state.result={...record,models:Object.fromEntries(record.model_keys.map(k=>[k,null])),layers:{},ranks:{},observation:{timeline:true,step:null}};
        state.compare=false;state.progress=state.appearance=1;$('compare-enabled').checked=false;
        acceptPrepared(record);
        if(!state.selected || !state.points.some(p=>p.id===state.selected))state.selected=state.points.find(p=>p.role==='test')?.id;
        $('play-seek').max=clock.duration;
        $('play-steps').value=record.timeline.steps;
        updateHistory();
        clock.play(performance.now());
        if(document.hidden)clock.pause(performance.now());
        renderPlayback(true);schedulePlayback();
        status('按各算法实测耗时独立播放，可暂停、拖动时间轴或实时修改倍率。');
    }
    function recordingSignature(key, input) {
        return JSON.stringify({points:input.points,seed:input.seed,test_size:input.test_size,
            challenge:input.challenge,params:input.params[key]});
    }
    function updateRecordingEstimate() {
        const estimateKey=JSON.stringify([$('play-steps').value,state.version,state.parameterSignature,
            state.parametersReady,state.parametersValid,state.valid,
            activeKeys().map(key=>[key,state.recordingProfiles[key]?.track.recording_ms])]);
        if(state.recordingEstimateKey===estimateKey)return;
        state.recordingEstimateKey=estimateKey;
        const summary=$('record-estimate-summary'), note=$('record-estimate-note'), models=$('record-estimate-models');
        state.recordingEstimate=null;
        models.replaceChildren();
        note.textContent='包含训练、边界预测与快照整理，不含网络传输及动画播放时间；倍率不影响记录等待。输入时不会发起训练。';
        if(!state.parametersReady){summary.textContent='预计记录等待：正在加载参数…';return;}
        if(!state.valid || !state.parametersValid){summary.textContent='预计记录等待：请先补全有效数据和模型参数';return;}
        const steps=numeric('play-steps');
        if(!$('play-steps').value.trim() || !Number.isInteger(steps) || steps<1 || steps>1000){
            summary.textContent='预计记录等待：请输入 1～1000 的整数步数';return;
        }
        const input=payload(), profiles={};
        activeKeys().forEach(key=>{
            const cached=state.recordingProfiles[key];
            if(cached?.signature===recordingSignature(key,input))profiles[key]=cached.track;
        });
        const estimate=window.ArenaTimelineClock.estimateRecording({steps,models:activeKeys(),params:input.params,
            pointCount:state.base.length,trainCount:Math.max(2,Math.floor(state.base.length*(1-input.test_size))),profiles});
        state.recordingEstimate=estimate;
        const duration=ms=>ms<1000?'不足 1 秒':ms<60000?(ms/1000).toFixed(1)+' 秒':(ms/60000).toFixed(1)+' 分钟';
        const range=(low,high)=>high<1000?'不足 1 秒':duration(low)+'～'+duration(high);
        summary.textContent='预计记录等待：约 '+range(estimate.low,estimate.high)+' · 至多 '+estimate.count+' 张快照 · '+
            (estimate.calibrated ? estimate.calibrated+'/'+activeKeys().length+' 个模型已实测校准' : '首次粗估，误差可能较大');
        estimate.tracks.forEach(track=>{
            const span=document.createElement('span');
            span.textContent=names[track.key]+'：'+track.count+' 步 · '+range(track.low,track.high)+(track.calibrated?'（实测校准）':'（粗估）');
            models.append(span);
        });
        note.textContent+=' 各模型按自身上限停止；SVM / L-BFGS 每步独立求解，耗时可能非线性增长。'+
            (estimate.calibrated?'外推与机器负载仍可能造成较大误差。':'完成一次记录后，同一数据与参数会自动使用本机实测校准。');
    }
    async function recordPlayback() {
        if(state.busy || !state.valid || !state.parametersReady || !state.parametersValid)return;
        const steps=numeric('play-steps');
        if(!Number.isInteger(steps)||steps<1||steps>1000){status('连续演示步数需为 1～1000 的整数；更大步数仍可使用单步观察。','error');return;}
        if(!$('play-speed').checkValidity()){status('请先修正速度倍率。','error');return;}
        leavePlayback();
        const seq=++state.sequence,version=state.version;
        state.busy=state.recording=true;state.controller=new AbortController();
        $('play-state').textContent='正在实测并记录，准备阶段不计入播放等待';
        status('正在记录 '+activeKeys().length+' 个参赛模型前 '+steps+' 步的实际操作耗时与边界，完成后开始回放；可取消。','busy');buttonState();
        const input=payload();
        try {
            const record=await api('timeline',{...input,steps},state.controller.signal);
            if(seq!==state.sequence||version!==state.version)return;
            record.model_keys.forEach(key=>{
                const track=record.tracks[key];
                if(Number.isFinite(track.recording_ms))state.recordingProfiles[key]={signature:recordingSignature(key,input),track:{
                    recording_ms:track.recording_ms,setup_ms:track.setup_ms,complete:track.complete,
                    frames:track.frames.map(frame=>({duration_ms:frame.duration_ms}))
                }};
            });
            state.busy=state.recording=false;state.controller=null;
            beginPlayback(record);
        } catch(error) {
            if(seq===state.sequence && error.name!=='AbortError'){status(error.message,'error');$('play-state').textContent='记录失败，可调整设置后重试';}
        } finally {
            if(seq===state.sequence){state.busy=state.recording=false;state.controller=null;buttonState();}
        }
    }
    function renderPlayback(force=false) {
        const playback=state.playback;
        if(!playback?.active || playback.record.version!==state.version)return;
        const {clock,record}=playback, positions=clock.positions();
        let changed=force;
        record.model_keys.forEach(key=>{
            const index=positions[key], track=record.tracks[key];
            if(playback.positions[key]!==index){
                changed=true;
                const frame=index<0?null:track.frames[index];
                const model=frame?{...frame.model,train_ids:record.split.train_ids,test_ids:record.split.test_ids}:null;
                if(model){
                    const bits=atob(model.grid_bits), count=record.grid.size**2;
                    model.grid=Array.from({length:count},(_,i)=>(bits.charCodeAt(i>>3)>>(7-(i&7)))&1);
                    state.result.layers[key]=buildLayers({grid:record.grid,models:{[key]:model}})[key];
                }else state.result.layers[key]=null;
                state.result.models[key]=model;
            }
            const frame=index<0?null:track.frames[index], time=Math.min(clock.time,track.duration_ms);
            const display=$('track-time-'+key);
            display.querySelector('span').textContent=time.toFixed(3)+' / '+track.duration_ms.toFixed(3)+' ms · '+
                (frame?'本步 '+frame.duration_ms.toFixed(3)+' ms':'准备 '+track.setup_ms.toFixed(3)+' ms')+
                (index===track.frames.length-1?' · 本段完成':'');
            display.querySelector('i').style.width=(time/Math.max(Number.EPSILON,track.duration_ms)*100)+'%';
        });
        playback.positions=positions;
        $('play-seek').value=clock.time;
        $('play-time').value=clock.time.toFixed(3)+' / '+clock.duration.toFixed(3)+' ms';
        $('play-state').textContent=(clock.time>=clock.duration?'本段演示结束 · 不代表所有模型已收敛':clock.playing?'播放中 · 各模型按实测时间推进':'已暂停 · 时间与模型进度保持')+' · 前 '+record.timeline.steps+' 步片段';
        if(changed){updateMetrics();updateInspector();buttonState();requestDraw();}
        else playbackControls();
    }
    function schedulePlayback() {
        const playback=state.playback;
        if(!playback?.active || !playback.clock.playing)return;
        cancelAnimationFrame(playback.raf);
        playback.raf=requestAnimationFrame(now=>{
            if(state.playback!==playback || !playback.active)return;
            playback.clock.tick(now);renderPlayback();schedulePlayback();
        });
    }
    function playbackSpeed(value, fromSlider=false) {
        if(!Number.isFinite(value)||value<0.00001||value>10){$('play-speed').setCustomValidity('倍率必须在 0.00001～10 之间');return;}
        $('play-speed').setCustomValidity('');
        if(fromSlider){value=Number(value.toPrecision(5));$('play-speed').value=value;}
        else $('play-speed-slider').value=Math.log10(value);
        const slowdown=value<1?'耗时放慢 '+Number((1/value).toPrecision(5))+' 倍':value===1?'实测原速':'加速 '+Number(value.toPrecision(5))+' 倍';
        $('play-speed-note').textContent=Number(value.toPrecision(5))+'× · '+slowdown;
        if(state.playback?.active){state.playback.clock.setSpeed(value,performance.now());renderPlayback();schedulePlayback();}
    }
    function processPanel(key, model) {
        const panel = $('process-' + key), info = model.observation;
        panel.hidden = !info;
        if (!info) return;
        panel.replaceChildren();
        const heading = document.createElement('strong');
        heading.textContent = info.title + ' · ' + info.current + ' / ' + info.total + ' ' + info.unit + (info.complete ? ' · 已到上限' : '');
        const progress = document.createElement('progress');progress.max = Math.max(1,info.total);progress.value = info.current;
        progress.setAttribute('aria-label',info.title);
        const note = document.createElement('p');note.textContent = info.note;
        const stat = document.createElement('span');stat.className = 'arena-process-stat';
        const details = [];
        if (info.timing) details.push('本步实测 '+info.timing.step_ms.toFixed(3)+' ms · 准备 '+info.timing.setup_ms.toFixed(3)+' ms · 累计 '+info.timing.cumulative_ms.toFixed(3)+' ms');
        if (info.actual_iterations !== undefined) details.push('求解器实际 ' + info.actual_iterations + ' 次迭代');
        if (info.support_count !== undefined) details.push(info.support_count + ' 个支持向量（画布圆环）');
        if (info.frontier_impurity !== undefined) details.push('当前终端节点加权不纯度 ' + info.frontier_impurity.toFixed(4));
        if (info.seen_count !== undefined) details.push('已使用 ' + info.seen_count + ' 个训练样本');
        if (info.weight_norm !== undefined) details.push('权重范数 ' + info.weight_norm.toFixed(4));
        if (info.training_pool_log_loss !== undefined) details.push('训练池 Log loss ' + info.training_pool_log_loss.toFixed(4) + '（评估值）');
        stat.textContent = details.join(' · ');
        panel.append(heading,progress,note,stat);
        if (key === 'knn' || key === 'forest') {
            const vote = document.createElement('div');
            vote.id='arena-vote-'+key;vote.className='arena-process-vote';panel.append(vote);
        }
        let values, label;
        if (info.loss_curve?.length) {
            values = info.loss_curve.map((v,i) => [i+1,v]);label = '真实逐轮训练损失（含正则项；不保证单调）';
        } else if (key !== 'knn' && state.playback?.active) {
            values=state.playback.record.tracks[key].frames.slice(0,state.playback.positions[key]+1).map(f=>[f.step,f.model.train.accuracy]);
            label='实测步骤：训练池准确率（非优化目标）';
        } else if (key !== 'knn') {
            values = [...state.checkpoints.entries()].sort((a,b)=>a[0]-b[0]).filter(([step])=>step<=state.result.observation.step)
                .map(([step,r]) => [step,r.models[key].train.accuracy]);
            label = '已记录快照：训练池准确率（非优化目标）';
        }
        if (values?.length) {
            const ns = 'http://www.w3.org/2000/svg', svg = document.createElementNS(ns,'svg');
            svg.setAttribute('viewBox','0 0 300 48');svg.setAttribute('role','img');svg.setAttribute('aria-label',label);
            const min = Math.min(...values.map(v=>v[1])), max = Math.max(...values.map(v=>v[1]));
            const coordinates = values.map(([x,y])=>[5+(x-values[0][0])/Math.max(1,values.at(-1)[0]-values[0][0])*290,42-(y-min)/Math.max(1e-9,max-min)*34]);
            const line = document.createElementNS(ns,'polyline');
            line.setAttribute('points',coordinates.map(v=>v.join(',')).join(' '));svg.append(line);
            // A single checkpoint is a point, never a fabricated horizontal curve.
            if (coordinates.length === 1) {const dot=document.createElementNS(ns,'circle');dot.setAttribute('cx',coordinates[0][0]);dot.setAttribute('cy',coordinates[0][1]);dot.setAttribute('r','3');svg.append(dot);}
            const caption = document.createElement('small');caption.textContent = label + ' · ' + values[0][0] + '→' + values.at(-1)[0] + ' · ' + values.at(-1)[1].toFixed(4);
            panel.append(svg,caption);
        }
    }
    function buildLayers(result) {
        return Object.fromEntries(keys.map(key => {
            if(!result.models[key])return [key,null];
            const n = result.grid.size, values = result.models[key].grid;
            const layer = document.createElement('canvas');
            layer.width = layer.height = n;
            const ctx = layer.getContext('2d'), raster = ctx.createImageData(n, n);
            const palette = [[233, 229, 249], [216, 240, 235]];
            values.forEach((v, i) => {
                raster.data.set([...palette[v], 255], i * 4);
            });
            ctx.putImageData(raster, 0, 0);
            // Marching squares on REAL sampled predictions; no preset boundary paths.
            const segments = [];
            for (let row = 0; row < n - 1; row++) for (let col = 0; col < n - 1; col++) {
                const a = values[row*n+col], b = values[row*n+col+1];
                const c = values[(row+1)*n+col+1], d = values[(row+1)*n+col];
                const edges = [];
                if (a !== b) edges.push([col+0.5, row]);
                if (b !== c) edges.push([col+1, row+0.5]);
                if (c !== d) edges.push([col+0.5, row+1]);
                if (d !== a) edges.push([col, row+0.5]);
                for (let i = 0; i + 1 < edges.length; i += 2) segments.push([edges[i], edges[i+1]]);
            }
            return [key, {raster: layer, segments, size: n}];
        }));
    }
    function updateMetrics() {
        const result = state.result;
        if (!result) return;
        const comparable = !observing() && state.baseline && state.baseline.holdout_fingerprint === result.holdout_fingerprint;
        keys.forEach(key => {
            const model = result.models[key];
            if(!model){
                $('effective-'+key).textContent='等待该算法完成第 1 步。';
                $('rank-'+key).textContent='第一步进行中';delete $('rank-'+key).dataset.rank;
                $('score-'+key).innerHTML='<strong>—<small>等待真实阶段</small></strong><span>实测时间尚未到达第 1 步</span>';
                $('process-'+key).hidden=true;return;
            }
            $('effective-' + key).textContent = JSON.stringify({
                '有效配置': model.configured_params, '实际模型参数': model.effective_params, '预处理': model.preprocessing
            },null,2);
            $('rank-' + key).textContent = model.observation ? model.observation.current + ' ' + model.observation.unit : '#' + result.ranks[key] + ' · Macro F1';
            if (model.observation) delete $('rank-' + key).dataset.rank;
            else $('rank-' + key).dataset.rank = result.ranks[key];
            processPanel(key,model);
            $('score-' + key).innerHTML = '<strong>' + fmt(model.test.f1) +
                '<small>TEST MACRO F1</small></strong><span>Accuracy ' + fmt(model.test.accuracy) +
                '<br>' + (observing() ? '训练池' : '训练') + ' Accuracy ' + fmt(model.train.accuracy) + '</span>';
        });
        const neuralTitle = root.querySelector('[data-model="mlp"] header strong');
        neuralTitle.textContent = result.models.mlp?.fit_info.network_type === 'perceptron' ? 'PERCEPTRON' : 'MLP';
        names.mlp = result.models.mlp?.fit_info.network_type === 'perceptron' ? '单层感知机' : 'MLP';
        $('metrics-body').innerHTML = activeKeys().map(key => {
            const m = result.models[key];
            if(!m)return '<tr><td>'+names[key]+'</td><td colspan="7">第一步进行中 · 尚无完成的阶段结果</td></tr>';
            return '<tr><td>' + names[key] + '</td><td>' + fmt(m.train.accuracy) +
                '</td><td>' + fmt(m.test.accuracy) + '</td><td>' + fmt(m.test.precision) +
                '</td><td>' + fmt(m.test.recall) + '</td><td>' + fmt(m.test.f1) +
                '</td><td>' + m.train_ms.toFixed(2) + ' ms</td><td>' +
                (comparable ? delta(m.test.f1 - state.baseline.models[key].test.f1) : observing() ? '过程不排名 / 不比较基线' : '测试集已改变') + '</td></tr>';
        }).join('');
        root.querySelector('.arena-metrics th:nth-child(2)').textContent = observing() ? '训练池 Acc（含未见样本）' : '训练 Acc';
        root.querySelector('.arena-metrics th:nth-child(7)').textContent = result.observation?.timeline ? '累计操作耗时（含准备）' : '训练耗时';
        $('compare-note').textContent = comparable ? '同一测试集 · 基线 v' + state.baseline.version + ' → 本轮 v' + result.version + ' · ' + activeKeys().map(key =>
            names[key] + ' F1 ' + delta(result.models[key].test.f1 - state.baseline.models[key].test.f1)).join(' / ') : '';
    }
    function updateInspector() {
        const result = state.result, p = state.points.find(p => p.id === state.selected);
        const current = fresh();
        $('inspect-title').textContent = p ?
            '样本 #' + p.id + ' · ' + (p.role === 'test' ? '测试' : '训练') + ' · 真实类别 ' + (p.clean_label === 1 ? 'B' : p.clean_label === 0 ? 'A' : p.label ? 'B' : 'A') :
            '◎ 点击空心测试点，比较预测；悬停任意样本可同步高亮';
        keys.forEach(key => {
            const article = root.querySelector('[data-model="' + key + '"]');
            article.classList.remove('is-error');
            const vote = $('vote-'+key);
            if (vote) {
                vote.replaceChildren();
                const explanation = p && current && result.models[key]?.explanations[p.id];
                const probabilities = explanation && (key === 'knn' ? explanation.partial_probabilities : explanation.probabilities);
                if (probabilities) {
                    const label=document.createElement('span');
                    label.textContent='#'+p.id+' · '+(key==='knn'?'当前部分投票':'当前树概率均值')+' · A '+fmt(probabilities[0])+' / B '+fmt(probabilities[1]);
                    const bar=document.createElement('div');bar.className='arena-vote-bar';
                    bar.setAttribute('role','img');bar.setAttribute('aria-label',label.textContent);
                    probabilities.forEach((v,i)=>{const segment=document.createElement('i');segment.style.width=(v*100)+'%';segment.style.background=colors[i];bar.append(segment);});
                    vote.append(label,bar);
                } else vote.textContent = current ? '点击一个样本观察投票。' : '旧过程已过期，请重新观察。';
            }
            if(current && !result.models[key]){$('explain-'+key).textContent='该算法正在准备或计算第 1 步；尚未显示任何未完成的预测。';article.classList.remove('has-fit-warning');return;}
            if (!p || !result || !current) {
                $('explain-' + key).textContent = !current && result ? '数据或参数已变，重新对决后才能检查预测。' :
                    (key === 'mlp' && current && !observing() ? mlpSummary(result.models.mlp.fit_info) : modelHints[key]);
                article.classList.toggle('has-fit-warning', Boolean(current && result.models[key]?.fit_info.convergence_warning));
                return;
            }
            const m = result.models[key], pred = m.predictions[p.id], explanation = m.explanations[p.id];
            if (!explanation) return;
            const wrong = p.role === 'test' && pred !== p.clean_label;
            article.classList.toggle('is-error', wrong);
            let text = (state.compare ? '本轮 · ' : '') + (wrong ? '× 误判' : '✓ ') + '预测 ' + (pred ? 'B' : 'A') + ' · ';
            if (key === 'svm') text += '决策分数 ' + explanation.decision_score.toFixed(3) + '（正值 → B，非概率）';
            if (key === 'knn') {
                const neighbors = explanation.revealed_neighbors || explanation.neighbors;
                text += neighbors.length + ' 个真实训练邻居：' + neighbors.map(n => '#' + n.id + ' d=' + n.distance.toFixed(2)).join('，');
                if (explanation.partial_probabilities) text += ' · 部分投票 P(B) ' + fmt(explanation.partial_probabilities[1]) + ' → ' + (explanation.partial_prediction ? 'B' : 'A') + '（背景仍为完整 K）';
            }
            if (key === 'tree') text += explanation.path.map(s => (s.feature ? 'X₂' : 'X₁') + s.direction + s.threshold.toFixed(2)).join(' → ') + (observing() ? ' → 当前终端节点 ' : ' → 叶 ') + explanation.leaf;
            if (explanation.probabilities) {
                text += 'P(A) ' + fmt(explanation.probabilities[0]) + ' / P(B) ' + fmt(explanation.probabilities[1]);
            }
            if (key === 'forest') text += ' · ' + m.fit_info.tree_count + ' 棵树概率的平均。单树分类倾向 A ' +
                explanation.tree_votes[0] + ' / B ' + explanation.tree_votes[1] + '（不作为最终多数票）。';
            if (key === 'mlp') text += (explanation.decision_score !== undefined ? '决策分数 ' + explanation.decision_score.toFixed(3) + '（无概率输出）' : '') + ' · ' +
                (observing() ? '网络 ' + m.fit_info.network.join('→') + ' · ' + m.fit_info.solver + ' · ' + m.fit_info.iterations + (m.observation.mode === 'incremental_epochs' ? ' 轮完整遍历' : ' 次实际迭代') +
                    (m.fit_info.final_loss === null ? ' · 无交叉熵损失输出' : ' · 当前训练损失 ' + m.fit_info.final_loss.toFixed(4)) : mlpSummary(m.fit_info));
            if (key === 'bayes') text += ' · 高斯条件独立假设；概率是模型输出，不代表已校准。';
            article.classList.toggle('has-fit-warning', Boolean(m.fit_info.convergence_warning));
            $('explain-' + key).textContent = text;
        });
    }
    function mlpSummary(info) {
        return '网络 ' + info.network.join('→') + ' · ' + info.solver + ' 实际 ' + info.iterations +
            ' 次迭代' + (info.final_loss === null ? ' · 不提供交叉熵损失' : ' · 末次损失 ' + info.final_loss.toFixed(4)) +
            (info.convergence_warning ? ' · ⚠ 未满足停止条件（迭代或评估上限）' : ' · 已满足优化器停止条件');
    }
    let frame = 0;
    function requestDraw() {
        if (!frame) frame = requestAnimationFrame(() => { frame = 0; Object.keys(canvases).forEach(draw); });
    }
    function geometry(canvas) {
        const width = canvas.clientWidth, height = canvas.clientHeight;
        const dpr = Math.min(devicePixelRatio || 1, 2);
        if (canvas.width !== Math.round(width*dpr) || canvas.height !== Math.round(height*dpr)) {
            canvas.width = Math.round(width*dpr);
            canvas.height = Math.round(height*dpr);
        }
        return {width, height, dpr, left: 25, top: 12, w: Math.max(1, width-37), h: Math.max(1, height-34)};
    }
    const project = (p, g) => [g.left + (p.x+3)/6*g.w, g.top + (3-p.y)/6*g.h];
    function world(event, canvas) {
        const rect = canvas.getBoundingClientRect(), g = geometry(canvas);
        return {x: Math.max(-2.95, Math.min(2.95, (event.clientX-rect.left-g.left)/g.w*6-3)),
            y: Math.max(-2.95, Math.min(2.95, 3-(event.clientY-rect.top-g.top)/g.h*6)),
            px: event.clientX-rect.left, py: event.clientY-rect.top};
    }
    function nearest(event, canvas) {
        const pos = world(event, canvas), g = geometry(canvas);
        let found = null, distance = event.pointerType === 'touch' ? 20 : 11;
        for (const p of state.points) {
            const xy = project(p, g), d = Math.hypot(xy[0]-pos.px, xy[1]-pos.py);
            if (d < distance) { found = p; distance = d; }
        }
        return {point: found, pos};
    }
    function paintBoundary(ctx, layer, g) {
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(layer.raster, g.left, g.top, g.w, g.h);
        ctx.beginPath();
        layer.segments.forEach(([a,b]) => {
            ctx.moveTo(g.left+a[0]/(layer.size-1)*g.w, g.top+a[1]/(layer.size-1)*g.h);
            ctx.lineTo(g.left+b[0]/(layer.size-1)*g.w, g.top+b[1]/(layer.size-1)*g.h);
        });
        ctx.strokeStyle = '#778e9e';
        ctx.lineWidth = 0.85;
        ctx.stroke();
    }
    function draw(key) {
        const canvas = canvases[key], g = geometry(canvas), ctx = canvas.getContext('2d');
        if (!g.width || !g.height) return;
        ctx.setTransform(g.dpr, 0, 0, g.dpr, 0, 0);
        ctx.clearRect(0,0,g.width,g.height);
        ctx.fillStyle = '#f9fbfe';
        ctx.fillRect(0,0,g.width,g.height);
        const current = fresh(), result = state.result, isModel = key !== 'data';
        if (isModel && result?.layers[key]) {
            ctx.save();
            ctx.globalAlpha = current ? state.progress : 0.22;
            paintBoundary(ctx, result.layers[key], g);
            if (state.compare && current && state.baseline) {
                ctx.save();
                ctx.beginPath();
                ctx.rect(g.left,g.top,g.w*state.divider,g.h);
                ctx.clip();
                paintBoundary(ctx, state.baseline.layers[key], g);
                ctx.restore();
            }
            ctx.restore();
        }
        // The same world domain and grid ticks are used for every plot.
        ctx.strokeStyle = '#8096b023';
        ctx.lineWidth = .7;
        for (let tick = -3; tick <= 3; tick++) {
            const xy = project({x:tick,y:tick},g);
            ctx.beginPath(); ctx.moveTo(xy[0],g.top); ctx.lineTo(xy[0],g.top+g.h);
            ctx.moveTo(g.left,xy[1]); ctx.lineTo(g.left+g.w,xy[1]); ctx.stroke();
        }
        ctx.fillStyle = '#9aa8ba';
        ctx.font = '9px system-ui';
        ctx.textAlign = 'center';
        [-2,0,2].forEach(tick => {
            const xy = project({x:tick,y:tick},g);
            ctx.fillText(tick,xy[0],g.top+g.h+15);
            ctx.fillText(tick,g.left-13,xy[1]+3);
        });
        // KNN lines are exact neighbor IDs returned by kneighbors.
        if (key === 'knn' && current && state.selected && $('show-links').checked) {
            const selected = state.points.find(p => p.id === state.selected);
            const explanation = result.models.knn?.explanations[state.selected];
            if (selected && explanation) {
                const from = project(selected,g);
                ctx.save();
                if (state.compare) {
                    ctx.beginPath();ctx.rect(g.left+g.w*state.divider,g.top,g.w*(1-state.divider),g.h);ctx.clip();
                }
                ctx.setLineDash([3,3]); ctx.strokeStyle = '#56627ea0'; ctx.lineWidth = 1;
                (explanation.revealed_neighbors || explanation.neighbors).forEach(n => {
                    const p = state.points.find(p => p.id === n.id);
                    if (p) { const to = project(p,g); ctx.beginPath(); ctx.moveTo(...from); ctx.lineTo(...to); ctx.stroke(); }
                });
                ctx.restore();
            }
        }
        if (isModel && current && result.observation && result.models[key]) paintProcess(ctx,g,key,result.models[key].observation);
        let points = state.points;
        if (isModel && state.compare && current) {
            // Left half also uses baseline samples, not current samples on an old region.
            ctx.save(); ctx.beginPath(); ctx.rect(g.left,g.top,g.w*state.divider,g.h); ctx.clip();
            paintPoints(ctx,state.baseline.points,g,key,state.baseline);
            ctx.restore();
            ctx.save(); ctx.beginPath(); ctx.rect(g.left+g.w*state.divider,g.top,g.w*(1-state.divider),g.h); ctx.clip();
            paintPoints(ctx,points,g,key,result);
            ctx.restore();
            const x = g.left+g.w*state.divider;
            ctx.strokeStyle = '#4c426a'; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(x,g.top); ctx.lineTo(x,g.top+g.h); ctx.stroke();
            ctx.fillStyle = '#fffef1ed';ctx.fillRect(g.left+4,g.top+g.h-19,42,15);ctx.fillRect(g.left+g.w-46,g.top+g.h-19,42,15);
            ctx.fillStyle = '#61536f';ctx.font='9px system-ui';ctx.fillText('基线',g.left+25,g.top+g.h-8);ctx.fillText('本轮',g.left+g.w-25,g.top+g.h-8);
        } else paintPoints(ctx,points,g,key,current ? result : null);
        if (!points.length) {
            ctx.fillStyle = '#8b9bb0';ctx.font='12px system-ui';ctx.textAlign='center';
            ctx.fillText('选择画 A / 画 B，点击或拖动添加样本',g.width/2,g.height/2);
        }
        canvas.dataset.bounds = '-3,3,-3,3';
        canvas.dataset.highlight = state.hover || state.selected || '';
        canvas.dataset.boundaryVersion = result ? result.version : '';
    }
    function paintProcess(ctx,g,key,info) {
        ctx.save();
        ctx.beginPath();ctx.rect(g.left,g.top,g.w,g.h);ctx.clip();
        if (key === 'svm') {
            const supports = new Set(info.support_ids);
            ctx.strokeStyle = '#a56b31';ctx.lineWidth = 1.2;
            state.points.filter(p=>supports.has(p.id)).forEach(p=>{const xy=project(p,g);ctx.beginPath();ctx.arc(...xy,6.5,0,Math.PI*2);ctx.stroke();});
        }
        if (key === 'tree') info.segments.forEach((segment,i) => {
            const a=project({x:segment.a[0],y:segment.a[1]},g), b=project({x:segment.b[0],y:segment.b[1]},g);
            ctx.strokeStyle=i===info.segments.length-1?'#d07831':'#445778a0';ctx.lineWidth=i===info.segments.length-1?2.5:1;
            ctx.setLineDash(i===info.segments.length-1?[]:[4,3]);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();
        });
        if (key === 'bayes') info.gaussians.forEach(distribution => {
            const xy=project({x:distribution.mean[0],y:distribution.mean[1]},g);
            ctx.strokeStyle=colors[distribution.label];ctx.lineWidth=2;ctx.setLineDash([5,3]);
            ctx.beginPath();ctx.ellipse(...xy,distribution.std[0]/6*g.w,distribution.std[1]/6*g.h,0,0,Math.PI*2);ctx.stroke();
            ctx.setLineDash([]);ctx.beginPath();ctx.moveTo(xy[0]-6,xy[1]);ctx.lineTo(xy[0]+6,xy[1]);ctx.moveTo(xy[0],xy[1]-6);ctx.lineTo(xy[0],xy[1]+6);ctx.stroke();
        });
        ctx.restore();
    }
    function paintPoints(ctx,points,g,key,result) {
        const model = result && result.models[key], active = state.hover || state.selected;
        const seen = model?.observation?.seen_ids ? new Set(model.observation.seen_ids) : null;
        points.forEach(p => {
            const [x,y] = project(p,g), highlighted = p.id === active;
            const wrong = model && p.role === 'test' && model.predictions[p.id] !== p.clean_label;
            ctx.globalAlpha = ($('errors-only').checked && model && !wrong && !highlighted ? .22 : 1) * state.appearance;
            if (seen && p.role === 'train' && !seen.has(p.id) && !highlighted) ctx.globalAlpha *= .18;
            ctx.beginPath();
            if (p.corrupted) {
                ctx.moveTo(x,y-5);ctx.lineTo(x+5,y);ctx.lineTo(x,y+5);ctx.lineTo(x-5,y);ctx.closePath();
            } else ctx.arc(x,y,key === 'data' ? 3.8 : 3.3,0,Math.PI*2);
            ctx.fillStyle = p.role === 'test' ? '#fff' : colors[p.label];
            ctx.strokeStyle = p.corrupted ? '#bc5e2c' : colors[p.label];
            ctx.lineWidth = p.role === 'test' || p.corrupted ? 1.6 : .6;
            ctx.fill();ctx.stroke();
            if (wrong) {
                ctx.strokeStyle='#c6502e';ctx.lineWidth=1.4;
                ctx.beginPath();ctx.arc(x,y,6.2,0,Math.PI*2);ctx.stroke();
                ctx.beginPath();ctx.moveTo(x+4,y-8);ctx.lineTo(x+9,y-3);ctx.moveTo(x+4,y-3);ctx.lineTo(x+9,y-8);ctx.stroke();
            }
            if (highlighted) {
                ctx.globalAlpha = 1;
                ctx.strokeStyle='#fff';ctx.lineWidth=4;ctx.beginPath();ctx.arc(x,y,8,0,Math.PI*2);ctx.stroke();
                ctx.strokeStyle='#293950';ctx.lineWidth=1.5;ctx.stroke();
            }
        });
        ctx.globalAlpha = 1;
    }
    function setTool(tool) {
        state.tool = tool;
        root.querySelectorAll('[data-tool]').forEach(b => b.setAttribute('aria-pressed', b.dataset.tool === tool));
        canvases.data.style.cursor = tool === 'move' ? 'grab' : tool === 'inspect' ? 'crosshair' : tool === 'erase' ? 'not-allowed' : 'crosshair';
    }
    function updateHover(event, canvas) {
        const {point, pos} = nearest(event, canvas);
        state.hover = point ? point.id : null;
        if (canvas === canvases.data) {
            $('tooltip').hidden = !point;
            if (point) {
                $('tooltip').textContent = '#' + point.id + ' · ' + (point.role === 'test' ? '测试' : '训练') +
                    ' · ' + (point.label ? 'B' : 'A') + ' (' + point.x.toFixed(2) + ', ' + point.y.toFixed(2) + ')';
                $('tooltip').style.left = Math.max(5,Math.min(canvas.clientWidth-170,pos.px+12)) + 'px';
                $('tooltip').style.top = Math.max(5,pos.py-34) + 'px';
            }
        }
        requestDraw();
    }
    function selectPoint(p) {
        state.selected = p ? p.id : null;
        updateInspector();
        requestDraw();
    }
    function editBegin() {
        if (challenged()) {
            status('挑战正在预览。请先“重置挑战”再编辑原始数据；仍可悬停和点击检查。');
            return false;
        }
        remember();
        return true;
    }
    function addPoint(pos) {
        if (state.base.length >= 600) { status('最多支持 600 个样本，以保持画布和计算流畅。'); return; }
        let id;
        do { id = 'draw-' + state.nextId++; } while(state.base.some(p => p.id === id));
        state.base.push({id, x:pos.x, y:pos.y, label:state.tool === 'b' ? 1 : 0});
        state.points = state.base.map(p => ({...p, ...roleFor(p.id)}));
    }
    function roleFor(id) {
        const old = state.points.find(p => p.id === id);
        return old ? {role:old.role,clean_label:old.clean_label} : {};
    }
    const dataCanvas = canvases.data;
    dataCanvas.addEventListener('pointerdown', event => {
        if (event.button !== 0) return;
        const {point, pos} = nearest(event,dataCanvas);
        if (state.tool === 'inspect') { selectPoint(point); return; }
        if (state.tool === 'move' && point) {
            selectPoint(point);
            if (!editBegin()) return;
            state.gesture = {kind:'move',id:point.id,start:pos,offset:{x:point.x-pos.x,y:point.y-pos.y},changed:false};
        } else if (['a','b'].includes(state.tool)) {
            if (!editBegin()) return;
            state.gesture = {kind:'draw',last:pos,changed:true};
            dirty();
            addPoint(pos);
        } else if (point && ['erase','flip'].includes(state.tool)) {
            if (!editBegin()) return;
            if (state.tool === 'erase') {
                state.base = state.base.filter(p => p.id !== point.id);
                if (state.selected === point.id) state.selected = null;
            } else state.base.find(p => p.id === point.id).label ^= 1;
            dirty('样本成员或类别已改变，将按同一种子重新分层划分；旧基线不再可比。');
            state.points = clone(state.base);
            prepare();
        }
        if (state.gesture) {
            dataCanvas.setPointerCapture(event.pointerId);
            event.preventDefault();
        }
        requestDraw();
    });
    dataCanvas.addEventListener('pointermove', event => {
        const gesture = state.gesture;
        if (!gesture) { updateHover(event,dataCanvas); return; }
        const pos = world(event,dataCanvas);
        if (gesture.kind === 'move') {
            if (!gesture.changed && Math.hypot(pos.px-gesture.start.px,pos.py-gesture.start.py) < 3) return;
            if (!gesture.changed) {
                dirty('位置已改变，旧边界待重新训练。样本成员及类别不变，因此保留相同划分。');
                gesture.changed = true;
            }
            const p = state.base.find(p => p.id === gesture.id);
            p.x = Math.max(-2.95,Math.min(2.95,pos.x+gesture.offset.x));
            p.y = Math.max(-2.95,Math.min(2.95,pos.y+gesture.offset.y));
            const current = state.points.find(p => p.id === gesture.id);
            Object.assign(current,{x:p.x,y:p.y});
        } else if (Math.hypot(pos.px-gesture.last.px,pos.py-gesture.last.py) >= 10) {
            addPoint(pos);gesture.last=pos;
        }
        requestDraw();
    });
    function endGesture(event) {
        if (!state.gesture) return;
        const changed = state.gesture.changed;
        if (!changed) state.undo.pop();
        state.gesture = null;
        if (dataCanvas.hasPointerCapture(event.pointerId)) dataCanvas.releasePointerCapture(event.pointerId);
        if (changed) {
            state.valid = state.base.length >= 24 && [0,1].every(c => state.base.filter(p => p.label===c).length>=6);
            prepare();
        }
        buttonState();
        requestDraw();
    }
    dataCanvas.addEventListener('pointerup',endGesture);
    dataCanvas.addEventListener('pointercancel',endGesture);
    Object.entries(canvases).forEach(([key,canvas]) => {
        if (key !== 'data') {
            canvas.addEventListener('pointermove', e => updateHover(e,canvas));
            canvas.addEventListener('pointerdown', e => selectPoint(nearest(e,canvas).point));
        }
        canvas.addEventListener('pointerleave',() => {
            if (!state.gesture) { state.hover=null;$('tooltip').hidden=true;requestDraw(); }
        });
        canvas.addEventListener('keydown',event => {
            if (!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key)) return;
            event.preventDefault();
            if (!state.points.length) return;
            const ordered = [...state.points].sort((a,b) => (a.role==='test'?-1:1)-(b.role==='test'?-1:1));
            const index = ordered.findIndex(p => p.id===state.selected), direction = ['ArrowLeft','ArrowUp'].includes(event.key)?-1:1;
            selectPoint(ordered[(index+direction+ordered.length)%ordered.length]);
        });
    });
    root.querySelectorAll('[data-tool]').forEach(button => button.addEventListener('click',() => setTool(button.dataset.tool)));
    $('run').addEventListener('click',run);
    root.querySelectorAll('[data-arena-participant]').forEach(input=>input.addEventListener('change',()=>{
        selectParticipants([...root.querySelectorAll('[data-arena-participant]:checked')].map(input=>input.dataset.arenaParticipant));
    }));
    $('participants-all').addEventListener('click',()=>selectParticipants(keys));
    $('data-expand').addEventListener('click',()=>setAllDataOpen(true));
    $('data-collapse').addEventListener('click',showSixCanvases);
    $('fit-canvases').addEventListener('click',showSixCanvases);
    $('play-record').addEventListener('click',recordPlayback);
    $('play-steps').addEventListener('input',updateRecordingEstimate);
    $('play-pause').addEventListener('click',()=>{
        const playback=state.playback;if(!playback?.active || state.busy)return;
        if(playback.clock.playing){playback.clock.pause(performance.now());cancelAnimationFrame(playback.raf);}
        else playback.clock.play(performance.now());
        renderPlayback();schedulePlayback();
    });
    $('play-restart').addEventListener('click',()=>{
        if(state.busy || !state.playback || state.playback.record.version!==state.version)return;
        const record=state.playback.record;leavePlayback();++state.sequence;beginPlayback(record);
    });
    $('play-cancel').addEventListener('click',()=>{
        if(!state.recording)return;
        ++state.sequence;if(state.controller)state.controller.abort();state.controller=null;state.busy=state.recording=false;
        $('play-state').textContent='已取消本次记录（服务器已开始的计算可能仍会完成，但结果不会应用）';
        status('已取消记录，保留当前数据。');buttonState();
    });
    $('play-speed').addEventListener('input',()=>playbackSpeed(numeric('play-speed')));
    $('play-speed-slider').addEventListener('input',()=>playbackSpeed(10**numeric('play-speed-slider'),true));
    $('play-seek').addEventListener('input',()=>{
        const playback=state.playback;if(!playback?.active)return;
        playback.clock.pause(performance.now());cancelAnimationFrame(playback.raf);
        const value=numeric('play-seek'), end=playback.clock.duration;
        // Native range serialization can round the endpoint down by a few ulps.
        playback.clock.seek(end-value<=Math.max(1e-9,end*1e-12)?end:value,performance.now());renderPlayback();
    });
    document.addEventListener('visibilitychange',()=>{
        if(document.hidden && state.playback?.active && state.playback.clock.playing){
            state.playback.clock.pause(performance.now());cancelAnimationFrame(state.playback.raf);renderPlayback();
        }
    });
    root.querySelectorAll('[data-observe-step]').forEach(button=>button.addEventListener('click',()=>observe(Number(button.dataset.observeStep))));
    $('step-next').addEventListener('click',()=>observe((observing() ? state.result.observation.step : 0)+1));
    $('step-ten').addEventListener('click',()=>observe((observing() ? state.result.observation.step : 0)+10));
    $('step-go').addEventListener('click',()=>observe(numeric('step-value')));
    $('step-value').addEventListener('keydown',event=>{if(event.key==='Enter')observe(numeric('step-value'));});
    $('step-history').addEventListener('change',()=>{if($('step-history').value)observe(Number($('step-history').value));});
    $('step-final').addEventListener('click',()=>{
        if (state.busy || state.finalResult?.version !== state.version) return;
        leavePlayback();
        ++state.sequence;
        state.result=state.finalResult;state.progress=1;
        updateHistory();updateMetrics();updateInspector();buttonState();requestDraw();
        status('已返回完整对决结果；过程快照仍可回看。');
    });
    $('generate').addEventListener('click',generate);
    $('preset').addEventListener('change',generate);
    $('noise').addEventListener('input',() => { $('noise-value').value=numeric('noise').toFixed(2); });
    $('noise').addEventListener('change',() => { status('生成噪声已设置；点击“重新生成数据”应用。原有数据尚未改变。'); });
    $('clear').addEventListener('click',() => {
        remember();resetChallengeValues();state.base=state.points=[];state.split=null;state.selected=null;
        dirty('画布已清空。选择画 A / 画 B 开始创建分类问题。');
        prepare();
    });
    $('undo').addEventListener('click',() => {
        const previous=state.undo.pop();if(!previous)return;
        state.base=previous.base;
        $('challenge-noise').value=previous.challenge.noise;$('corruption').value=previous.challenge.corruption;
        challengeLabels();state.points=clone(state.base);state.selected=null;
        dirty('已撤销修改，请重新对决确认边界。');prepare();
    });
    ['seed','test-size'].forEach(id => $(id).addEventListener('change',() => {
        dirty('划分条件已改变，将重新分层划分。');prepare();
    }));
    ['challenge-noise','corruption'].forEach(id => {
        $(id).addEventListener('input',() => {
            challengeLabels();
            // Mark old predictions stale as soon as the thumb moves, not on release.
            dirty('训练条件已改变，旧边界已失效。释放滑块后预览，重新对决后比较。');
        });
        $(id).addEventListener('change',prepare);
    });
    $('reset-challenge').addEventListener('click',() => {
        if (!challenged()) return;
        remember();resetChallengeValues();dirty('挑战已重置，点击重新对决回到干净训练条件。');prepare();
    });
    $('baseline').addEventListener('click',() => {
        if (!fresh() || observing()) return;
        state.baseline=state.result;updateMetrics();buttonState();requestDraw();
        status('当前真实结果已设为基线。测试集不变时可比较后续对决。');
    });
    $('compare-enabled').addEventListener('change',() => { state.compare=$('compare-enabled').checked;updateInspector();requestDraw(); });
    $('divider').addEventListener('input',() => { state.divider=numeric('divider')/100;requestDraw(); });
    $('errors-only').addEventListener('change',requestDraw);
    $('show-links').addEventListener('change',requestDraw);
    function focusMode(on) {
        root.classList.toggle('arena-focused',on);
        document.body.classList.toggle('arena-focus-active',on);
        $('focus').textContent=on?'退出专注 · Esc':'专注模式 ↗';
        $('focus').setAttribute('aria-pressed',on);
        scheduleCanvasFit();
        requestDraw();
    }
    $('focus').addEventListener('click',() => focusMode(!root.classList.contains('arena-focused')));
    document.addEventListener('keydown',e => { if(e.key==='Escape')focusMode(false); });
    new ResizeObserver(()=>{requestDraw();scheduleCanvasFit();}).observe(root);
    window.addEventListener('resize',scheduleCanvasFit);
    // Read-only diagnostic snapshots make browser regression checks possible.
    window.VizMLArena = {snapshot: () => ({
        version:state.version, resultVersion:state.result?.version, fresh:Boolean(fresh()),
        busy:state.busy, revealProgress:state.progress, parametersReady:state.parametersReady,
        parameters:state.parametersReady ? parameterValues() : null,
        points:clone(state.points), base:clone(state.base), split:clone(state.split), requestCount:state.requestCount,
        selected:state.selected, hover:state.hover, checkpoints:[...state.checkpoints.keys()], result:state.result ? {
            models:state.result.models, ranks:state.result.ranks, fingerprint:state.result.fingerprint,
            observation:state.result.observation || null,
            holdout_fingerprint:state.result.holdout_fingerprint, bounds:state.result.bounds,
            model_keys:state.result.model_keys
        } : null,
        playback:state.playback ? {
            active:state.playback.active,playing:state.playback.clock.playing,time:state.playback.clock.time,
            duration:state.playback.clock.duration,speed:state.playback.clock.speed,positions:{...state.playback.positions},
            tracks:Object.fromEntries(state.playback.record.model_keys.map(key=>[key,{
                setup_ms:state.playback.record.tracks[key].setup_ms,
                duration_ms:state.playback.record.tracks[key].duration_ms,
                frames:state.playback.record.tracks[key].frames.map(f=>({step:f.step,duration_ms:f.duration_ms,elapsed_ms:f.elapsed_ms}))
            }]))
        } : null, recording:state.recording, recordingEstimate:clone(state.recordingEstimate),
        expandedData:activeKeys().filter(key=>$('model-data-'+key).open), selectedModels:activeKeys()
    })};
    participantControls();buttonState();
    loadParameters().then(() => { buttonState();return generate(); }).catch(error => status(error.message,'error'));
})();
