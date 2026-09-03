/** VizML homepage interactions: lightweight canvas demo, filters, and mobile navigation. */
(function () {
    'use strict';

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    function initNavigation() {
        const toggle = document.querySelector('.nav-toggle');
        const nav = document.querySelector('.primary-nav');
        if (!toggle || !nav) return;

        const close = () => {
            toggle.setAttribute('aria-expanded', 'false');
            nav.classList.remove('is-open');
        };

        toggle.addEventListener('click', () => {
            const open = toggle.getAttribute('aria-expanded') !== 'true';
            toggle.setAttribute('aria-expanded', String(open));
            nav.classList.toggle('is-open', open);
        });
        nav.querySelectorAll('a').forEach((link) => link.addEventListener('click', close));
        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape') close();
        });
        document.addEventListener('click', (event) => {
            if (!nav.contains(event.target) && !toggle.contains(event.target)) close();
        });
    }

    function initFilters() {
        const buttons = Array.from(document.querySelectorAll('.filter-button'));
        const cards = Array.from(document.querySelectorAll('.lab-card'));
        const status = document.getElementById('filterStatus');
        if (!buttons.length || !cards.length) return;

        const labels = { all: '全部', supervised: '监督学习', unsupervised: '无监督学习', advanced: '进阶模型' };
        buttons.forEach((button) => {
            button.addEventListener('click', () => {
                const category = button.dataset.category;
                let count = 0;
                buttons.forEach((item) => {
                    const active = item === button;
                    item.classList.toggle('is-active', active);
                    item.setAttribute('aria-pressed', String(active));
                });
                cards.forEach((card) => {
                    const visible = category === 'all' || card.dataset.category === category;
                    card.hidden = !visible;
                    if (visible) count += 1;
                });
                if (status) status.textContent = `当前显示${labels[category]}的 ${count} 个算法实验`;
            });
        });
    }

    class HeroExperiment {
        constructor(canvas) {
            this.canvas = canvas;
            this.context = canvas.getContext('2d');
            this.wrap = canvas.closest('.canvas-wrap');
            this.mode = 'moons';
            this.noise = 0.18;
            this.seed = 20260903;
            this.points = [];
            this.progress = prefersReducedMotion ? 1 : 0;
            this.phase = 0;
            this.paused = false;
            this.frame = null;
            this.lastTime = 0;
            this.width = 0;
            this.height = 0;
            this.pauseButton = document.getElementById('demoPause');
            this.modelName = document.getElementById('modelName');
            this.accuracy = document.getElementById('accuracyMetric');
            this.noiseControl = document.getElementById('noiseControl');
            this.noiseValue = document.getElementById('noiseValue');
            this.runButton = document.getElementById('demoRun');
        }

        init() {
            if (!this.context) {
                this.wrap?.classList.add('canvas-unavailable');
                return;
            }
            this.bindControls();
            this.resize();
            this.generate();
            if ('ResizeObserver' in window) {
                this.resizeObserver = new ResizeObserver(() => this.resize());
                this.resizeObserver.observe(this.wrap);
            } else {
                window.addEventListener('resize', () => this.resize(), { passive: true });
            }
            if (prefersReducedMotion) this.draw();
            else this.frame = requestAnimationFrame((time) => this.animate(time));
        }

        bindControls() {
            document.querySelectorAll('.demo-chip').forEach((button) => {
                button.addEventListener('click', () => {
                    document.querySelectorAll('.demo-chip').forEach((item) => {
                        const active = item === button;
                        item.classList.toggle('is-active', active);
                        item.setAttribute('aria-pressed', String(active));
                    });
                    this.mode = button.dataset.demo;
                    this.seed = 20260903;
                    this.restart();
                });
            });

            this.noiseControl?.addEventListener('input', () => {
                this.noise = Number(this.noiseControl.value) / 100;
                this.noiseValue.value = this.noise.toFixed(2);
                this.seed = 20260903;
                this.restart(false);
            });

            this.runButton?.addEventListener('click', () => {
                this.seed += 97;
                this.restart();
                this.runButton.animate(
                    [{ transform: 'scale(1)' }, { transform: 'scale(.97)' }, { transform: 'scale(1)' }],
                    { duration: 260, easing: 'ease-out' }
                );
            });

            this.pauseButton?.addEventListener('click', () => {
                this.paused = !this.paused;
                this.pauseButton.setAttribute('aria-pressed', String(this.paused));
                this.pauseButton.setAttribute('aria-label', this.paused ? '继续动画' : '暂停动画');
                const use = this.pauseButton.querySelector('use');
                const label = this.pauseButton.querySelector('span');
                use?.setAttribute('href', this.paused ? '#icon-play' : '#icon-pause');
                if (label) label.textContent = this.paused ? '继续' : '暂停';
                if (!this.paused && !prefersReducedMotion) {
                    this.lastTime = performance.now();
                    this.frame = requestAnimationFrame((time) => this.animate(time));
                }
            });

            document.addEventListener('visibilitychange', () => {
                if (!document.hidden && !this.paused && !prefersReducedMotion && !this.frame) {
                    this.lastTime = performance.now();
                    this.frame = requestAnimationFrame((time) => this.animate(time));
                }
            });
        }

        random() {
            this.seed |= 0;
            this.seed = (this.seed + 0x6D2B79F5) | 0;
            let value = this.seed;
            value = Math.imul(value ^ (value >>> 15), 1 | value);
            value ^= value + Math.imul(value ^ (value >>> 7), 61 | value);
            return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
        }

        gaussian() {
            const first = Math.max(this.random(), Number.EPSILON);
            const second = this.random();
            return Math.sqrt(-2 * Math.log(first)) * Math.cos(2 * Math.PI * second);
        }

        generate() {
            const points = [];
            const jitter = this.noise * 0.32;
            if (this.mode === 'moons') {
                for (let i = 0; i < 100; i += 1) {
                    const angle = Math.PI * (i / 99) + this.gaussian() * jitter;
                    points.push({ x: .31 + .25 * Math.cos(angle) + this.gaussian() * jitter, y: .43 + .27 * Math.sin(angle) + this.gaussian() * jitter, group: 0 });
                    points.push({ x: .67 - .25 * Math.cos(angle) + this.gaussian() * jitter, y: .57 - .27 * Math.sin(angle) + this.gaussian() * jitter, group: 1 });
                }
            } else if (this.mode === 'circles') {
                for (let i = 0; i < 100; i += 1) {
                    const angle = Math.PI * 2 * i / 100 + this.gaussian() * jitter;
                    points.push({ x: .5 + (.17 + this.gaussian() * jitter) * Math.cos(angle), y: .5 + (.23 + this.gaussian() * jitter) * Math.sin(angle), group: 0 });
                    points.push({ x: .5 + (.36 + this.gaussian() * jitter) * Math.cos(angle), y: .5 + (.41 + this.gaussian() * jitter) * Math.sin(angle), group: 1 });
                }
            } else {
                for (let i = 0; i < 100; i += 1) {
                    points.push({ x: .31 + this.gaussian() * (.08 + jitter), y: .62 + this.gaussian() * (.11 + jitter), group: 0 });
                    points.push({ x: .69 + this.gaussian() * (.09 + jitter), y: .36 + this.gaussian() * (.1 + jitter), group: 1 });
                }
            }
            this.points = points.map((point) => ({ x: Math.max(.04, Math.min(.96, point.x)), y: Math.max(.06, Math.min(.94, point.y)), group: point.group, delay: this.random() * .55 }));
            const base = this.mode === 'clusters' ? 97.2 : this.mode === 'circles' ? 92.6 : 94.8;
            const score = Math.max(78, base - Math.max(0, this.noise - .12) * 29);
            if (this.accuracy) this.accuracy.textContent = `${score.toFixed(1)}%`;
            if (this.modelName) this.modelName.textContent = this.mode === 'clusters' ? 'K-Means' : 'RBF SVM';
        }

        restart(animate = true) {
            this.generate();
            this.progress = prefersReducedMotion || !animate ? 1 : 0;
            this.phase = 0;
            this.draw();
            if (!this.paused && !prefersReducedMotion && !this.frame) {
                this.lastTime = performance.now();
                this.frame = requestAnimationFrame((time) => this.animate(time));
            }
        }

        resize() {
            const bounds = this.canvas.getBoundingClientRect();
            if (!bounds.width || !bounds.height) return;
            const ratio = Math.min(window.devicePixelRatio || 1, 2);
            this.width = bounds.width;
            this.height = bounds.height;
            this.canvas.width = Math.round(bounds.width * ratio);
            this.canvas.height = Math.round(bounds.height * ratio);
            this.context.setTransform(ratio, 0, 0, ratio, 0, 0);
            this.draw();
        }

        animate(time) {
            this.frame = null;
            if (this.paused || document.hidden) return;
            const delta = this.lastTime ? Math.min(32, time - this.lastTime) : 16;
            this.lastTime = time;
            this.progress = Math.min(1, this.progress + delta / 1250);
            this.phase += delta / 1000;
            this.draw();
            this.frame = requestAnimationFrame((nextTime) => this.animate(nextTime));
        }

        drawGrid() {
            const ctx = this.context;
            ctx.strokeStyle = '#e8edf4';
            ctx.lineWidth = 1;
            const step = 28;
            ctx.beginPath();
            for (let x = .5; x < this.width; x += step) { ctx.moveTo(x, 0); ctx.lineTo(x, this.height); }
            for (let y = .5; y < this.height; y += step) { ctx.moveTo(0, y); ctx.lineTo(this.width, y); }
            ctx.stroke();
        }

        drawRegions() {
            const ctx = this.context;
            const w = this.width;
            const h = this.height;
            ctx.save();
            ctx.globalAlpha = .68 * Math.min(1, this.progress * 1.8);
            if (this.mode === 'circles') {
                ctx.fillStyle = 'rgba(231,111,81,.11)';
                ctx.fillRect(0, 0, w, h);
                ctx.beginPath();
                ctx.ellipse(w * .5, h * .5, w * .265, h * .34, 0, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(81,71,217,.14)';
                ctx.fill();
                ctx.strokeStyle = `rgba(15,157,138,${.58 + Math.sin(this.phase * 2) * .08})`;
                ctx.lineWidth = 2;
                ctx.setLineDash([6, 5]);
                ctx.stroke();
            } else if (this.mode === 'clusters') {
                const gradient = ctx.createLinearGradient(0, h, w, 0);
                gradient.addColorStop(0, 'rgba(81,71,217,.13)');
                gradient.addColorStop(.49, 'rgba(81,71,217,.08)');
                gradient.addColorStop(.51, 'rgba(231,111,81,.08)');
                gradient.addColorStop(1, 'rgba(231,111,81,.13)');
                ctx.fillStyle = gradient;
                ctx.fillRect(0, 0, w, h);
                ctx.beginPath();
                ctx.moveTo(w * .12, 0);
                ctx.lineTo(w * .88, h);
                ctx.strokeStyle = `rgba(15,157,138,${.58 + Math.sin(this.phase * 2) * .08})`;
                ctx.lineWidth = 2;
                ctx.setLineDash([7, 5]);
                ctx.stroke();
            } else {
                const yAt = (x) => h * (.52 + Math.sin((x / w) * Math.PI * 2 - .7) * .14);
                ctx.beginPath();
                ctx.moveTo(0, 0);
                ctx.lineTo(w, 0);
                for (let x = w; x >= 0; x -= 5) ctx.lineTo(x, yAt(x));
                ctx.closePath();
                ctx.fillStyle = 'rgba(81,71,217,.13)';
                ctx.fill();
                ctx.beginPath();
                ctx.moveTo(0, yAt(0));
                for (let x = 0; x <= w; x += 5) ctx.lineTo(x, yAt(x));
                ctx.lineTo(w, h);
                ctx.lineTo(0, h);
                ctx.closePath();
                ctx.fillStyle = 'rgba(231,111,81,.1)';
                ctx.fill();
                ctx.beginPath();
                for (let x = 0; x <= w; x += 5) {
                    const y = yAt(x);
                    if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                }
                ctx.strokeStyle = `rgba(15,157,138,${.62 + Math.sin(this.phase * 2) * .08})`;
                ctx.lineWidth = 2;
                ctx.setLineDash([7, 5]);
                ctx.stroke();
            }
            ctx.restore();
        }

        drawPoints() {
            const ctx = this.context;
            const padding = 17;
            const usableWidth = this.width - padding * 2;
            const usableHeight = this.height - padding * 2;
            this.points.forEach((point) => {
                const localProgress = Math.max(0, Math.min(1, (this.progress - point.delay) / .45));
                if (localProgress <= 0) return;
                const eased = 1 - Math.pow(1 - localProgress, 3);
                const x = padding + point.x * usableWidth;
                const y = padding + point.y * usableHeight;
                ctx.beginPath();
                ctx.arc(x, y, 3.6 * eased, 0, Math.PI * 2);
                ctx.fillStyle = point.group === 0 ? '#5147d9' : '#e76f51';
                ctx.fill();
                ctx.strokeStyle = 'rgba(255,255,255,.9)';
                ctx.lineWidth = 1.2;
                ctx.stroke();
            });
        }

        draw() {
            if (!this.context || !this.width || !this.height) return;
            this.context.clearRect(0, 0, this.width, this.height);
            this.context.fillStyle = '#fbfcff';
            this.context.fillRect(0, 0, this.width, this.height);
            this.drawGrid();
            this.drawRegions();
            this.drawPoints();
        }
    }

    function initCanvas() {
        const canvas = document.getElementById('heroCanvas');
        if (canvas) new HeroExperiment(canvas).init();
    }

    function init() {
        initNavigation();
        initFilters();
        initCanvas();
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, { once: true });
    else init();
})();
