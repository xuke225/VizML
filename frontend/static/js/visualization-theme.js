/** VizML shared visualization tokens and canvas helpers. */
(function () {
    'use strict';

    const palette = Object.freeze([
        '#4F46E5', '#0F9D8A', '#E07A3F', '#D64C7F', '#2B8AC6',
        '#7C6CE7', '#7B8F31', '#C15FBC', '#586779', '#D99A20'
    ]);

    const theme = {
        palette,
        colors: Object.freeze({
            canvas: '#FBFCFE',
            grid: '#E7EDF5',
            gridStrong: '#D8E1EC',
            axis: '#95A3B8',
            text: '#172033',
            muted: '#6B7890',
            primary: '#4F46E5',
            accent: '#0F9D8A',
            warning: '#D99A20',
            danger: '#C2415D',
            pointStroke: '#FFFFFF'
        }),

        color(index, alpha) {
            const hex = palette[Math.abs(Number(index) || 0) % palette.length];
            return alpha === undefined ? hex : this.withAlpha(hex, alpha);
        },

        withAlpha(hex, alpha) {
            const clean = String(hex).replace('#', '');
            const expanded = clean.length === 3 ? clean.split('').map((x) => x + x).join('') : clean;
            const value = parseInt(expanded.slice(0, 6), 16);
            const red = (value >> 16) & 255;
            const green = (value >> 8) & 255;
            const blue = value & 255;
            return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
        },

        clear(ctx, width, height) {
            ctx.save();
            ctx.clearRect(0, 0, width, height);
            ctx.fillStyle = this.colors.canvas;
            ctx.fillRect(0, 0, width, height);
            ctx.restore();
        },

        drawGrid(ctx, width, height, spacing = 24) {
            ctx.save();
            ctx.strokeStyle = this.colors.grid;
            ctx.lineWidth = 1;
            ctx.beginPath();
            for (let x = .5; x <= width; x += spacing) {
                ctx.moveTo(x, 0);
                ctx.lineTo(x, height);
            }
            for (let y = .5; y <= height; y += spacing) {
                ctx.moveTo(0, y);
                ctx.lineTo(width, y);
            }
            ctx.stroke();
            ctx.restore();
        },

        drawPlotFrame(ctx, width, height, padding = 28) {
            ctx.save();
            ctx.strokeStyle = this.colors.gridStrong;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(padding, padding);
            ctx.lineTo(padding, height - padding);
            ctx.lineTo(width - padding, height - padding);
            ctx.stroke();
            ctx.restore();
        },

        drawPoint(ctx, x, y, color, radius = 4, selected = false) {
            ctx.save();
            if (selected) {
                ctx.beginPath();
                ctx.arc(x, y, radius + 5, 0, Math.PI * 2);
                ctx.fillStyle = this.withAlpha(color, .14);
                ctx.fill();
            }
            ctx.beginPath();
            ctx.arc(x, y, radius, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();
            ctx.strokeStyle = this.colors.pointStroke;
            ctx.lineWidth = 1.5;
            ctx.stroke();
            ctx.restore();
        },

        plotlyLayout(overrides = {}) {
            return Object.assign({
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: this.colors.canvas,
                font: { family: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", sans-serif', color: this.colors.text, size: 12 },
                colorway: palette,
                margin: { l: 52, r: 24, t: 42, b: 48 },
                hoverlabel: { bgcolor: '#172033', bordercolor: '#172033', font: { color: '#FFFFFF' } },
                xaxis: { gridcolor: this.colors.grid, zerolinecolor: this.colors.gridStrong, linecolor: this.colors.axis, tickfont: { color: this.colors.muted } },
                yaxis: { gridcolor: this.colors.grid, zerolinecolor: this.colors.gridStrong, linecolor: this.colors.axis, tickfont: { color: this.colors.muted } }
            }, overrides);
        },

        applyChartDefaults() {
            if (!window.Chart || !window.Chart.defaults) return;
            Chart.defaults.color = this.colors.muted;
            Chart.defaults.borderColor = this.colors.grid;
            Chart.defaults.backgroundColor = this.colors.canvas;
            Chart.defaults.font.family = 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", sans-serif';
            Chart.defaults.plugins.legend.labels.usePointStyle = true;
            Chart.defaults.plugins.legend.labels.boxWidth = 8;
            Chart.defaults.plugins.tooltip.backgroundColor = '#172033';
            Chart.defaults.plugins.tooltip.titleColor = '#FFFFFF';
            Chart.defaults.plugins.tooltip.bodyColor = '#E7EDF5';
            Chart.defaults.plugins.tooltip.cornerRadius = 8;
            Chart.defaults.plugins.tooltip.padding = 10;
        }
    };

    window.VizTheme = Object.freeze(theme);
    theme.applyChartDefaults();
    window.addEventListener('load', () => theme.applyChartDefaults(), { once: true });
})();
