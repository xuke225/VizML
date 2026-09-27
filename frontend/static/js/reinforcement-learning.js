(function () {
  'use strict';

  const $ = id => document.getElementById(id);
  const canvas = $('gridCanvas');
  const ctx = canvas.getContext('2d');
  const state = { environments: {}, result: null, index: 0, view: 'value', route: 'demo', chart: null, learnTimer: null, pathTimer: null, agentStep: null, valueMin: 0, valueMax: 1 };
  const directions = [[-1, 0], [0, 1], [1, 0], [0, -1]];
  const palette = [[233, 239, 251], [188, 213, 237], [156, 217, 213], [132, 199, 184], [229, 210, 123]];

  function status(message, kind) {
    $('statusBar').textContent = message;
    $('statusBar').dataset.state = kind || '';
  }

  function fillRange(input) {
    const span = Number(input.max) - Number(input.min);
    const percentage = span > 0 ? ((Number(input.value) - Number(input.min)) / span) * 100 : 0;
    input.style.setProperty('--fill', `${percentage}%`);
  }

  [['episodes', 'episodesVal', value => String(value)], ['alpha', 'alphaVal', value => value.toFixed(2)],
    ['gamma', 'gammaVal', value => value.toFixed(2)], ['epsilon', 'epsilonVal', value => value.toFixed(2)],
    ['decay', 'decayVal', value => value.toFixed(3)]].forEach(([id, output, format]) => {
    const input = $(id);
    const update = () => { $(output).textContent = format(Number(input.value)); fillRange(input); };
    input.addEventListener('input', update);
    update();
  });

  async function loadEnvironments() {
    try {
      const response = await fetch('/api/rl/info');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      if (data.status !== 'success' || !data.environments) throw new Error(data.message || VizMLI18n.t('场景数据不可用'));
      state.environments = data.environments;
      const select = $('envSel');
      select.replaceChildren();
      Object.entries(data.environments).forEach(([key, item]) => select.add(new Option(item.name, key)));
      select.disabled = false;
      $('trainBtn').disabled = false;
      showEnvironment();
      status(VizMLI18n.t('场景已就绪。调整参数后开始训练。'));
    } catch (error) {
      $('envSel').replaceChildren(new Option(VizMLI18n.t('场景载入失败'), ''));
      status(VizMLI18n.t('无法载入场景：') + (error.message), 'error');
    }
  }

  function stopLearning() {
    if (state.learnTimer !== null) clearInterval(state.learnTimer);
    state.learnTimer = null;
    $('playLearnBtn').querySelector('span').textContent = VizMLI18n.t('播放学习过程');
    $('playLearnBtn').firstChild.textContent = '▶ ';
  }

  function stopPath() {
    if (state.pathTimer !== null) clearInterval(state.pathTimer);
    state.pathTimer = null;
    state.agentStep = null;
    $('playPathBtn').querySelector('span').textContent = VizMLI18n.t('播放当前轨迹');
  }

  function resetResults() {
    stopLearning(); stopPath();
    state.result = null;
    state.index = 0;
    state.valueMin = 0; state.valueMax = 1;
    $('scaleLow').textContent = VizMLI18n.t('低 Q'); $('scaleHigh').textContent = VizMLI18n.t('高 Q');
    if (state.chart) { state.chart.destroy(); state.chart = null; }
    $('chartEmpty').classList.remove('is-hidden');
    $('episodeLabel').textContent = VizMLI18n.t('等待训练');
    $('chartEpisode').textContent = '—';
    $('ckptSlider').value = 0; $('ckptSlider').max = 0; $('ckptSlider').disabled = true;
    fillRange($('ckptSlider'));
    $('playLearnBtn').disabled = true; $('playPathBtn').disabled = true;
    ['stSuccess', 'stReward', 'stEps', 'stPath'].forEach(id => { $(id).textContent = '—'; });
    $('pathDetail').textContent = VizMLI18n.t('从起点尝试抵达目标');
    $('stageCaption').dataset.tone = '';
    $('stageCaption').querySelector('.rl-insight__badge').textContent = VizMLI18n.t('观察重点');
    $('stageCaption').querySelector('p').textContent = VizMLI18n.t('训练后拖动时间线，查看每个阶段的价值热力、贪心动作和智能体轨迹。');
  }

  function showEnvironment() {
    resetResults();
    const item = state.environments[$('envSel').value];
    if (!item) return;
    $('episodes').value = item.recommended?.episodes || 500;
    $('alpha').value = item.recommended?.alpha || 0.1;
    ['episodes', 'alpha'].forEach(id => $(id).dispatchEvent(new Event('input')));
    $('envDesc').textContent = item.description;
    $('mapTitle').textContent = item.name;
    const hasWind = item.grid.wind?.some(value => value > 0);
    $('mapSubtitle').textContent = (item.grid.rows) + ' × ' + (item.grid.cols) + VizMLI18n.t(' 网格 · 起点 S → 目标 G') + (hasWind ? [VizMLI18n.t(' · 蓝色数字标示向上风力')]: '');
    $('windLegend').hidden = !hasWind;
    renderMap();
  }
  $('envSel').addEventListener('change', showEnvironment);

  async function train() {
    stopLearning(); stopPath();
    const button = $('trainBtn');
    const select = $('envSel');
    button.disabled = true; select.disabled = true;
    button.querySelector('span').textContent = VizMLI18n.t('正在训练…');
    status(VizMLI18n.t('正在训练智能体并记录检查点，请稍候。'));
    try {
      const payload = { env: select.value, params: {
        episodes: Number($('episodes').value), alpha: Number($('alpha').value),
        gamma: Number($('gamma').value), epsilon: Number($('epsilon').value),
        epsilon_decay: Number($('decay').value)
      } };
      const response = await fetch('/api/rl/train', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      const data = await response.json();
      if (!response.ok || data.status !== 'success' || !data.results?.checkpoints?.length) throw new Error(data.message || VizMLI18n.t('训练数据不完整'));
      resetResults();
      state.result = data.results;
      const values = data.results.checkpoints.flatMap(checkpoint => checkpoint.heatmap.flat().filter(Number.isFinite));
      state.valueMin = Math.min(...values);
      state.valueMax = Math.max(...values);
      if (state.valueMin === state.valueMax) state.valueMax = state.valueMin + 1;
      $('scaleLow').textContent = VizMLI18n.t('低 Q ') + (state.valueMin.toFixed(0));
      $('scaleHigh').textContent = VizMLI18n.t('高 Q ') + (state.valueMax.toFixed(0));
      $('ckptSlider').max = data.results.checkpoints.length - 1;
      $('ckptSlider').disabled = data.results.checkpoints.length < 2;
      $('playLearnBtn').disabled = data.results.checkpoints.length < 2;
      $('playPathBtn').disabled = false;
      drawChart(data.results.rewards);
      selectCheckpoint(data.results.checkpoints.length - 1);
      status(VizMLI18n.t('训练完成 · ') + (data.results.stats.episodes) + VizMLI18n.t(' 回合 · 最近 100 回合成功率 ') + ((data.results.stats.recent_success_rate * 100).toFixed(0)) + '%', 'success');
    } catch (error) {
      status(VizMLI18n.t('训练失败：') + (error.message), 'error');
    } finally {
      button.disabled = false; select.disabled = false;
      button.querySelector('span').textContent = VizMLI18n.t('重新训练');
    }
  }
  $('trainBtn').addEventListener('click', train);

  function checkpoint() { return state.result?.checkpoints[state.index] || null; }
  function routePath() {
    const current = checkpoint();
    return current ? (state.route === 'greedy' ? current.greedy_path : current.demo_path) : null;
  }

  function selectCheckpoint(index) {
    if (!state.result) return;
    stopPath();
    state.index = Math.max(0, Math.min(index, state.result.checkpoints.length - 1));
    $('ckptSlider').value = state.index;
    fillRange($('ckptSlider'));
    const current = checkpoint();
    $('episodeLabel').textContent = VizMLI18n.t('第 ') + (current.episode) + ' / ' + (state.result.stats.episodes) + VizMLI18n.t(' 回合');
    $('chartEpisode').textContent = `EP ${current.episode}`;
    $('stSuccess').textContent = `${Math.round(current.success_rate * 100)}%`;
    $('stReward').textContent = current.avg_reward.toFixed(1);
    $('stEps').textContent = current.epsilon.toFixed(3);
    const goal = state.result.grid.goal;
    const path = current.greedy_path || [];
    const reached = path.length > 0 && path[path.length - 1][0] === goal[0] && path[path.length - 1][1] === goal[1];
    $('stPath').textContent = reached ? (path.length - 1) + [VizMLI18n.t(' 步')]: VizMLI18n.t('未抵达');
    $('pathDetail').textContent = reached ? [VizMLI18n.t('当前贪心策略可到达目标')]: VizMLI18n.t('当前策略可能停滞或重复');
    const early = current.success_rate < .3;
    const strong = current.success_rate >= .8;
    $('stageCaption').dataset.tone = early ? 'warm' : strong ? 'good' : '';
    $('stageCaption').querySelector('.rl-insight__badge').textContent = early ? [VizMLI18n.t('探索起步')]: strong ? [VizMLI18n.t('策略成形')]: VizMLI18n.t('持续学习');
    const detail = state.route === 'demo'
      ? VizMLI18n.t('演示轨迹走了 ') + (current.demo_steps) + VizMLI18n.t(' 步，奖励 ') + (current.demo_reward.toFixed(0)) + (current.demo_fell ? VizMLI18n.t('，掉崖 ') + (current.demo_fell) + [VizMLI18n.t(' 次')]: '') + [VizMLI18n.t('。轨迹含随机探索，箭头表示已尝试的贪心动作。')]: reached ? VizMLI18n.t('当前贪心策略经过 ') + (path.length - 1) + [VizMLI18n.t(' 步到达目标。箭头对应本阶段的 Q 值。')]: VizMLI18n.t('当前贪心策略尚不能到达目标；继续观察后续检查点的变化。');
    $('stageCaption').querySelector('p').textContent = detail;
    if (state.chart) state.chart.update('none');
    renderMap();
  }

  $('ckptSlider').addEventListener('input', event => { stopLearning(); selectCheckpoint(Number(event.target.value)); });
  $('playLearnBtn').addEventListener('click', () => {
    if (!state.result) return;
    if (state.learnTimer !== null) { stopLearning(); return; }
    stopPath();
    let next = state.index >= state.result.checkpoints.length - 1 ? 0 : state.index;
    selectCheckpoint(next);
    $('playLearnBtn').querySelector('span').textContent = VizMLI18n.t('暂停播放');
    $('playLearnBtn').firstChild.textContent = 'Ⅱ ';
    state.learnTimer = setInterval(() => {
      next += 1;
      if (next >= state.result.checkpoints.length) { stopLearning(); return; }
      selectCheckpoint(next);
    }, 1050);
  });
  $('playPathBtn').addEventListener('click', () => {
    if (state.pathTimer !== null) { stopPath(); renderMap(); return; }
    stopLearning();
    const path = routePath();
    if (!path || path.length < 2) return;
    state.agentStep = 0;
    $('playPathBtn').querySelector('span').textContent = VizMLI18n.t('停止轨迹');
    renderMap();
    state.pathTimer = setInterval(() => {
      state.agentStep += 1;
      renderMap();
      if (state.agentStep >= path.length - 1) { stopPath(); renderMap(); }
    }, Math.max(100, Math.min(230, 5000 / path.length)));
  });
  document.querySelectorAll('[data-view]').forEach(button => button.addEventListener('click', () => {
    state.view = button.dataset.view;
    document.querySelectorAll('[data-view]').forEach(item => { item.classList.toggle('is-active', item === button); item.setAttribute('aria-pressed', String(item === button)); });
    renderMap();
  }));
  document.querySelectorAll('[data-route]').forEach(button => button.addEventListener('click', () => {
    stopPath();
    state.route = button.dataset.route;
    document.querySelectorAll('[data-route]').forEach(item => { item.classList.toggle('is-active', item === button); item.setAttribute('aria-pressed', String(item === button)); });
    if (state.result) selectCheckpoint(state.index); else renderMap();
  }));

  function roundedRect(x, y, width, height, radius) {
    ctx.beginPath(); ctx.roundRect(x, y, width, height, radius);
  }
  function heatColor(value) {
    const position = Math.max(0, Math.min(1, (value - state.valueMin) / (state.valueMax - state.valueMin))) * (palette.length - 1);
    const index = Math.min(palette.length - 2, Math.floor(position));
    const fraction = position - index;
    const channel = n => Math.round(palette[index][n] + (palette[index + 1][n] - palette[index][n]) * fraction);
    return `rgb(${channel(0)}, ${channel(1)}, ${channel(2)})`;
  }

  function renderMap() {
    const item = state.environments[$('envSel').value];
    const grid = state.result?.grid || item?.grid;
    if (!grid) return;
    const surface = document.querySelector('.rl-map-surface');
    const maxWidth = Math.min(650, Math.max(180, surface.clientWidth - 36));
    const cell = Math.min(62, maxWidth / grid.cols);
    const width = Math.round(cell * grid.cols), height = Math.round(cell * grid.rows);
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.style.width = `${width}px`; canvas.style.height = `${height}px`;
    canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const current = checkpoint();
    const types = new Map(grid.cells.map(cellInfo => [`${cellInfo.row},${cellInfo.col}`, cellInfo.type]));
    const gap = Math.max(2, cell * .055);
    $('valueScale').classList.toggle('is-hidden', !current || state.view !== 'value');

    for (let row = 0; row < grid.rows; row += 1) for (let col = 0; col < grid.cols; col += 1) {
      const type = types.get(`${row},${col}`);
      const x = col * cell + gap, y = row * cell + gap;
      const size = cell - gap * 2;
      let fill = '#f1f4f9';
      if (current && state.view === 'value' && Number.isFinite(current.heatmap[row][col])) fill = heatColor(current.heatmap[row][col]);
      if (type === 'wall') fill = '#475570';
      if (type === 'cliff') fill = '#f1c5c4';
      if (type === 'start') fill = '#56c2ad';
      if (type === 'goal') fill = '#efb465';
      roundedRect(x, y, size, size, Math.min(8, cell * .15)); ctx.fillStyle = fill; ctx.fill();
      if (type === 'cliff') {
        ctx.save(); roundedRect(x, y, size, size, Math.min(8, cell * .15)); ctx.clip();
        ctx.strokeStyle = 'rgba(188,77,81,.28)'; ctx.lineWidth = 1.2;
        for (let offset = -size; offset < size * 2; offset += 9) { ctx.beginPath(); ctx.moveTo(x + offset, y); ctx.lineTo(x + offset + size, y + size); ctx.stroke(); }
        ctx.restore();
      }
      if (type === 'start' || type === 'goal') {
        ctx.fillStyle = '#fff'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.font = `800 ${Math.max(12, Math.min(22, cell * .37))}px sans-serif`;
        ctx.fillText(type === 'start' ? 'S' : 'G', x + size / 2, y + size / 2 + 1);
      }
      if (row === 0 && grid.wind?.[col] > 0) {
        ctx.fillStyle = '#287bb8'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
        ctx.font = `800 ${Math.max(9, Math.min(13, cell * .22))}px sans-serif`;
        ctx.fillText(`↑${grid.wind[col]}`, x + size / 2, y + 4);
      }
    }

    if (current?.policy) for (let row = 0; row < grid.rows; row += 1) for (let col = 0; col < grid.cols; col += 1) {
      const action = current.policy[row][col];
      if (action < 0 || types.get(`${row},${col}`) !== 'normal') continue;
      const [dr, dc] = directions[action];
      const centerX = (col + .5) * cell, centerY = (row + .5) * cell;
      const length = cell * .18;
      const endX = centerX + dc * length, endY = centerY + dr * length;
      ctx.beginPath(); ctx.moveTo(centerX - dc * length * .6, centerY - dr * length * .6); ctx.lineTo(endX, endY);
      ctx.strokeStyle = '#263650'; ctx.lineWidth = Math.max(1.6, cell * .04); ctx.lineCap = 'round'; ctx.stroke();
      const angle = Math.atan2(dr, dc), wing = cell * .1;
      ctx.beginPath(); ctx.moveTo(endX, endY); ctx.lineTo(endX - wing * Math.cos(angle - .7), endY - wing * Math.sin(angle - .7));
      ctx.moveTo(endX, endY); ctx.lineTo(endX - wing * Math.cos(angle + .7), endY - wing * Math.sin(angle + .7)); ctx.stroke();
    }

    const path = routePath();
    if (path?.length) {
      const lastStep = state.agentStep === null ? path.length - 1 : state.agentStep;
      const visible = path.slice(0, lastStep + 1);
      ctx.beginPath();
      visible.forEach((position, index) => {
        const x = (position[1] + .5) * cell, y = (position[0] + .5) * cell;
        const previous = visible[index - 1];
        const teleport = previous && types.get(`${previous[0]},${previous[1]}`) === 'cliff' && position[0] === grid.start[0] && position[1] === grid.start[1];
        if (index === 0 || teleport) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.strokeStyle = 'rgba(255,255,255,.86)'; ctx.lineWidth = Math.max(5, cell * .16); ctx.lineJoin = 'round'; ctx.lineCap = 'round'; ctx.stroke();
      ctx.strokeStyle = state.route === 'demo' ? 'rgba(108,83,217,.64)' : 'rgba(37,104,186,.74)';
      ctx.lineWidth = Math.max(2.5, cell * .085); ctx.stroke();
      const position = visible[visible.length - 1];
      const x = (position[1] + .5) * cell, y = (position[0] + .5) * cell;
      ctx.beginPath(); ctx.arc(x, y, Math.max(5, cell * .14), 0, Math.PI * 2);
      ctx.fillStyle = state.route === 'demo' ? '#674bd6' : '#286db4'; ctx.fill();
      ctx.strokeStyle = '#fff'; ctx.lineWidth = Math.max(2, cell * .045); ctx.stroke();
    }
  }

  const checkpointLine = { id: 'rlCheckpointLine', afterDraw(chart) {
    const current = checkpoint();
    if (!current || !chart.chartArea) return;
    const x = chart.scales.x.getPixelForValue(current.episode - 1);
    const { top, bottom } = chart.chartArea;
    const c = chart.ctx;
    c.save(); c.beginPath(); c.moveTo(x, top); c.lineTo(x, bottom);
    c.strokeStyle = '#5b52d8'; c.lineWidth = 1.5; c.setLineDash([4, 4]); c.stroke(); c.restore();
  } };

  function drawChart(rewards) {
    if (typeof Chart === 'undefined') return;
    let sum = 0;
    const average = rewards.map((reward, index) => {
      sum += reward;
      if (index >= 25) sum -= rewards[index - 25];
      return sum / Math.min(index + 1, 25);
    });
    const chartContext = $('rewardChart').getContext('2d');
    const gradient = chartContext.createLinearGradient(0, 0, 0, 220);
    gradient.addColorStop(0, 'rgba(90,83,211,.18)'); gradient.addColorStop(1, 'rgba(90,83,211,0)');
    $('chartEmpty').classList.add('is-hidden');
    state.chart = new Chart(chartContext, { type: 'line', data: { labels: rewards.map((_, index) => index + 1), datasets: [
      { label: VizMLI18n.t('单轮奖励'), data: rewards, borderColor: 'rgba(159,175,198,.43)', borderWidth: 1, pointRadius: 0, tension: 0 },
      { label: VizMLI18n.t('25 回合移动平均'), data: average, borderColor: '#5a53d3', backgroundColor: gradient, borderWidth: 2.5, pointRadius: 0, tension: .2, fill: true }
    ] }, plugins: [checkpointLine], options: { responsive: true, maintainAspectRatio: false, animation: false,
      interaction: { mode: 'index', intersect: false },
      plugins: { legend: { display: true, position: 'bottom', align: 'start', labels: { color: '#7f8ba0', boxWidth: 11, boxHeight: 3, padding: 12, font: { size: 10 } } },
        tooltip: { callbacks: { title: items => VizMLI18n.t('第 ') + (items[0].label) + VizMLI18n.t(' 回合'), label: item => `${item.dataset.label}: ${item.parsed.y.toFixed(1)}` } } },
      scales: { x: { grid: { display: false }, border: { display: false }, ticks: { color: '#9ca8b7', maxTicksLimit: 6, font: { size: 10 } } },
        y: { grid: { color: '#eef1f5' }, border: { display: false }, ticks: { color: '#9ca8b7', maxTicksLimit: 5, font: { size: 10 } } } }
    } });
  }

  if (typeof ResizeObserver !== 'undefined') new ResizeObserver(() => renderMap()).observe(document.querySelector('.rl-map-surface'));
  else window.addEventListener('resize', renderMap);
  loadEnvironments();
})();
