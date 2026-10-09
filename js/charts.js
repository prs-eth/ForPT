/* Lightweight interactive line-chart renderer (no external deps).
   Charts render lazily the first time their containing tab panel becomes
   active, since a hidden (display:none) panel has no layout box and
   path.getTotalLength() would return 0. */

(function () {
    'use strict';

    var SVG_NS = 'http://www.w3.org/2000/svg';

    /* groupId -> array of {isolate, clear} from every chart that opted into
       a shared external legend instead of building its own (see cfg.legendGroup
       and renderSharedLegend). */
    var LEGEND_GROUPS = {};

    var CHART_DATA = {
        'sem-data-efficiency': {
            xLabel: 'Training forest area (ha)',
            yLabel: 'Test mIoU (%)',
            xScale: 'log',
            xDomain: [0.13, 7.2],
            xValues: [0.15, 0.34, 0.81, 2.59, 5.71],
            xTicks: [
                { value: 0.15, label: '0.15' },
                { value: 0.34, label: '0.34' },
                { value: 0.81, label: '0.81' },
                { value: 2.59, label: '2.59' },
                { value: 5.71, label: '5.71' }
            ],
            yDomain: [55, 91],
            yTicks: [55, 60, 65, 70, 75, 80, 85, 90],
            markerMode: 'all',
            series: [
                { key: 'scratch', label: 'Scratch', color: '#3987e5', shape: 'circle',
                  values: [57.86, 78.83, 83.31, 86.15, 87.2] },
                { key: 'decoder', label: 'Decoder probing', color: '#d95926', shape: 'diamond',
                  values: [61.2, 81.56, 82.65, 84.22, 87.26] },
                { key: 'knn', label: 'kNN', color: '#199e70', shape: 'triangle',
                  values: [68.5, 78.2, 80.5, 81.5, 83.2] },
                { key: 'linear', label: 'Linear probing', color: '#c98500', shape: 'square',
                  values: [59.25, 69.25, 74.31, 80.66, 84.18] },
                { key: 'finetune', label: 'Fine-tuning', color: '#d55181', shape: 'star',
                  values: [61.28, 82.2, 85.26, 86.54, 88.33] }
            ]
        },
        'sem-val-curve': {
            xLabel: 'Epoch',
            yLabel: 'Validation mIoU (%)',
            xScale: 'linear',
            xDomain: [1, 100],
            xValues: null, /* filled below: 1..100 */
            xTicks: [
                { value: 1, label: '1' },
                { value: 20, label: '20' },
                { value: 40, label: '40' },
                { value: 60, label: '60' },
                { value: 80, label: '80' },
                { value: 100, label: '100' }
            ],
            yDomain: [55, 91],
            yTicks: [55, 60, 65, 70, 75, 80, 85, 90],
            markerMode: 'end',
            crossoverAnnotation: { referenceKey: 'scratch', targetKey: 'finetune' },
            series: [
                { key: 'scratch', label: 'Scratch', color: '#3987e5', shape: 'circle',
                  values: [57.67, 71.6, 69.87, 76.44, 78.84, 82.74, 83.23, 74.27, 60.98, 82.27, 83.26, 83.88, 84.3, 84.39, 83.76, 81.93, 85.22, 85.12, 85.32, 85.25, 85.73, 85.53, 86.04, 85.56, 84.08, 85.84, 84.96, 84.64, 85.97, 85.1, 85.24, 86.46, 86.61, 86.56, 86.78, 86.71, 86.8, 86.64, 78.32, 82.88, 85.33, 82.47, 84.33, 86.08, 85.67, 86.55, 86.3, 86.74, 86.75, 86.49, 86.71, 86.92, 86.61, 86.9, 86.73, 87.17, 87.0, 87.35, 87.34, 87.02, 87.36, 87.29, 87.46, 87.43, 87.47, 87.47, 86.97, 87.23, 86.84, 87.47, 87.36, 87.47, 87.61, 87.37, 87.44, 87.73, 87.54, 87.54, 87.85, 87.78, 87.47, 87.76, 87.78, 87.79, 87.7, 87.73, 87.67, 87.74, 87.73, 87.72, 87.81, 87.74, 87.79, 87.82, 87.89, 87.87, 87.81, 87.75, 87.77, 87.72] },
                { key: 'linear', label: 'Linear probing', color: '#c98500', shape: 'square',
                  values: [73.95, 80.48, 82.45, 82.69, 81.7, 83.15, 81.18, 83.96, 83.53, 83.97, 83.97, 83.99, 84.13, 84.27, 81.55, 82.84, 83.05, 83.61, 83.89, 83.33, 83.14, 84.44, 83.96, 83.25, 83.97, 84.35, 84.19, 84.14, 83.91, 83.8, 83.98, 83.8, 83.65, 84.34, 84.3, 84.25, 80.69, 84.1, 83.45, 84.31, 84.51, 84.24, 84.43, 84.17, 84.11, 83.75, 84.51, 84.13, 84.41, 83.66, 83.96, 83.76, 84.32, 84.39, 82.65, 83.72, 84.15, 83.63, 84.55, 84.05, 83.88, 83.59, 84.49, 84.26, 82.82, 84.24, 84.32, 84.33, 84.28, 84.22, 83.69, 84.03, 84.1, 83.58, 83.74, 83.91, 84.33, 83.17, 83.96, 84.53, 84.26, 83.67, 83.96, 84.18, 84.04, 84.15, 84.15, 84.1, 84.29, 84.12, 84.01, 84.34, 84.33, 84.38, 84.32, 84.22, 84.29, 84.09, 84.22, 84.17] },
                { key: 'finetune', label: 'Fine-tuning', color: '#d55181', shape: 'star',
                  values: [83.05, 85.82, 85.43, 86.24, 86.74, 87.48, 86.59, 87.16, 85.22, 88.13, 88.42, 87.31, 87.67, 88.6, 88.19, 88.23, 88.07, 88.56, 88.65, 88.39, 88.77, 88.28, 88.26, 88.73, 88.07, 88.27, 88.54, 88.84, 88.53, 88.92, 88.92, 88.92, 89.04, 88.77, 89.07, 88.52, 88.77, 88.87, 88.39, 88.78, 88.8, 88.48, 88.68, 88.95, 89.05, 88.73, 88.94, 88.99, 88.87, 89.15, 89.01, 89.04, 89.08, 89.09, 88.91, 89.06, 88.96, 89.0, 88.97, 88.89, 89.05, 88.91, 88.92, 88.91, 88.85, 88.79, 89.01, 89.09, 88.8, 89.11, 89.14, 89.1, 88.74, 88.86, 88.88, 88.95, 88.82, 88.91, 88.89, 89.05, 88.97, 89.01, 88.99, 89.04, 88.98, 88.9, 89.01, 89.06, 89.06, 89.0, 88.95, 89.0, 89.0, 88.94, 89.05, 88.98, 89.0, 88.9, 88.94, 88.97] }
            ]
        },
        'species-val-curve': {
            xLabel: 'Epoch',
            yLabel: 'Validation accuracy (%)',
            xScale: 'linear',
            xDomain: [1, 100],
            xValues: null, /* filled below: 1..100 */
            xTicks: [
                { value: 1, label: '1' },
                { value: 20, label: '20' },
                { value: 40, label: '40' },
                { value: 60, label: '60' },
                { value: 80, label: '80' },
                { value: 100, label: '100' }
            ],
            yDomain: [38, 90],
            yTicks: [40, 50, 60, 70, 80, 90],
            markerMode: 'end',
            crossoverAnnotation: { referenceKey: 'scratch', targetKey: 'finetune' },
            series: [
                { key: 'scratch', label: 'Scratch', color: '#3987e5', shape: 'circle',
                  values: [40.91, 51.69, 51.75, 60.89, 59.62, 68.17, 71.11, 67.97, 72.77, 75.93, 75.73, 75.51, 76.33, 76.98, 77.82, 77.03, 79.4, 78.5, 80.25, 79.6, 80.22, 80.87, 79.49, 80.45, 81.6, 81.12, 81.18, 82.67, 83.01, 82.73, 82.93, 83.04, 83.07, 82.79, 83.24, 82.7, 82.7, 83.07, 83.18, 83.47, 83.89, 83.32, 84.0, 84.0, 84.9, 83.66, 84.37, 84.42, 84.06, 84.2, 84.68, 83.66, 85.13, 84.51, 84.31, 84.0, 84.31, 85.3, 85.5, 84.9, 84.62, 84.68, 84.79, 84.79, 85.02, 85.64, 85.19, 84.65, 84.62, 85.3, 85.24, 85.36, 85.07, 85.38, 85.24, 85.38, 85.5, 85.55, 85.44, 85.38, 86.15, 85.61, 85.72, 85.55, 85.38, 86.0, 85.75, 85.69, 86.03, 85.81, 85.81, 85.67, 85.72, 85.61, 85.72, 85.61, 85.75, 85.55, 85.61, 85.5] },
                { key: 'linear', label: 'Linear probing', color: '#c98500', shape: 'square',
                  values: [55.76, 63.18, 66.08, 64.9, 66.53, 67.92, 66.11, 69.67, 70.63, 70.29, 70.94, 68.71, 69.84, 69.98, 70.54, 70.46, 70.74, 70.77, 69.75, 70.01, 71.47, 71.08, 70.4, 70.26, 70.4, 70.09, 70.85, 70.82, 71.33, 69.78, 69.67, 71.05, 70.26, 70.29, 71.11, 70.77, 71.44, 71.61, 69.78, 70.26, 72.38, 71.59, 71.25, 72.07, 70.06, 69.86, 70.4, 71.08, 71.59, 70.54, 70.99, 71.42, 71.67, 71.98, 71.02, 69.33, 70.57, 72.4, 70.94, 72.07, 71.36, 70.77, 71.73, 71.39, 71.08, 71.02, 71.3, 71.08, 71.39, 71.44, 72.32, 72.12, 72.38, 72.8, 71.87, 72.04, 72.01, 71.39, 71.9, 72.38, 71.92, 72.12, 71.67, 72.6, 71.92, 72.29, 71.98, 72.46, 72.21, 72.57, 72.46, 72.21, 72.12, 72.4, 72.4, 72.46, 72.38, 72.26, 72.09, 72.23] },
                { key: 'finetune', label: 'Fine-tuning', color: '#d55181', shape: 'star',
                  values: [53.02, 58.83, 64.31, 64.7, 71.36, 71.84, 75.23, 77.14, 75.06, 78.78, 79.57, 79.09, 79.03, 80.59, 80.64, 79.6, 80.78, 81.97, 80.73, 80.87, 82.48, 81.77, 83.3, 82.14, 82.34, 82.84, 82.22, 83.35, 84.59, 83.18, 83.61, 83.1, 83.86, 83.35, 84.17, 83.8, 83.94, 84.93, 83.75, 83.94, 83.47, 84.62, 84.14, 83.63, 83.86, 84.17, 84.2, 85.07, 85.13, 84.99, 84.71, 84.34, 85.02, 85.55, 83.97, 85.24, 85.02, 85.52, 84.79, 84.82, 84.96, 85.44, 84.73, 85.05, 84.93, 85.41, 84.59, 85.5, 85.44, 85.61, 85.89, 85.64, 85.78, 85.78, 85.44, 85.84, 85.67, 85.72, 85.78, 85.92, 85.72, 86.06, 86.03, 85.55, 86.0, 86.12, 86.26, 86.0, 86.15, 86.0, 86.15, 86.12, 86.17, 86.2, 86.17, 86.23, 86.26, 86.26, 86.43, 86.17] }
            ]
        },
        'xfer-sem-seg': {
            legendGroup: 'xfer',
            xLabel: 'Training forest area (ha)', yLabel: 'Test mIoU (%)', xScale: 'log',
            xDomain: [0.13, 7.2], xValues: [0.15, 0.34, 0.81, 2.59, 5.71],
            xTicks: [{ value: 0.15, label: '0.15' }, { value: 0.34, label: '0.34' }, { value: 0.81, label: '0.81' },
                     { value: 2.59, label: '2.59' }, { value: 5.71, label: '5.71' }],
            yDomain: [45, 87], yTicks: [50, 60, 70, 80], markerMode: 'all',
            series: [
                { key: 'ssl', label: 'SSL pretrain', color: '#4878CF', shape: 'circle',
                  values: [59.25, 69.25, 74.31, 80.66, 84.18] },
                { key: 'species', label: 'Species Cls. pretrain', color: '#B47CC7', shape: 'triangle',
                  values: [63.21, 78.44, 79.67, 81.57, 82.21] },
                { key: 'age', label: 'Age Reg. pretrain', color: '#C4AD66', shape: 'triangle-down',
                  values: [47.67, 76.16, 77.68, 79.07, 79.55] }
            ]
        },
        'xfer-ins-seg': {
            legendGroup: 'xfer',
            xLabel: 'Training forest area (ha)', yLabel: 'Test mAP (%)', xScale: 'log',
            xDomain: [0.24, 7.2], xValues: [0.29, 0.55, 1.35, 2.90, 5.71],
            xTicks: [{ value: 0.29, label: '0.29' }, { value: 0.55, label: '0.55' }, { value: 1.35, label: '1.35' },
                     { value: 2.90, label: '2.90' }, { value: 5.71, label: '5.71' }],
            yDomain: [0, 40], yTicks: [0, 10, 20, 30, 40], markerMode: 'all',
            series: [
                { key: 'ssl', label: 'SSL pretrain', color: '#4878CF', shape: 'circle',
                  values: [4.3, 12.0, 20.3, 33.3, 38.7] },
                { key: 'species', label: 'Species Cls. pretrain', color: '#B47CC7', shape: 'triangle',
                  values: [2.3, 6.2, 10.3, 17.0, 22.4] },
                { key: 'age', label: 'Age Reg. pretrain', color: '#C4AD66', shape: 'triangle-down',
                  values: [1.0, 5.0, 9.2, 19.3, 24.6] }
            ]
        },
        'xfer-species-cls': {
            legendGroup: 'xfer',
            xLabel: 'Number of training trees', yLabel: 'Val overall accuracy (%)', xScale: 'log',
            xDomain: [120, 17000], xValues: [141, 708, 1416, 2833, 7082, 14165],
            xTicks: [{ value: 141, label: '141' }, { value: 708, label: '708' }, { value: 1416, label: '1.4k' },
                     { value: 2833, label: '2.8k' }, { value: 7082, label: '7.1k' }, { value: 14165, label: '14.2k' }],
            yDomain: [38, 76], yTicks: [40, 50, 60, 70], markerMode: 'all',
            series: [
                { key: 'ssl', label: 'SSL pretrain', color: '#4878CF', shape: 'circle',
                  values: [50.14, 61.96, 64.98, 67.35, 71.08, 72.80] },
                { key: 'sem', label: 'Sem. Seg. pretrain', color: '#6ACC65', shape: 'square',
                  values: [40.86, 52.17, 56.94, 61.60, 66.48, 68.76] },
                { key: 'ins', label: 'Inst. Seg. pretrain', color: '#D65F5F', shape: 'diamond',
                  values: [42.27, 53.19, 57.17, 62.10, 66.90, 70.06] },
                { key: 'age', label: 'Age Reg. pretrain', color: '#C4AD66', shape: 'triangle-down',
                  values: [42.58, 56.55, 61.17, 64.95, 68.74, 70.85] }
            ]
        },
        'xfer-age-reg': {
            legendGroup: 'xfer',
            xLabel: 'Number of training trees', yLabel: 'Val RMSE (years)', xScale: 'log',
            xDomain: [50, 1500], xValues: [62, 125, 250, 625, 1250],
            xTicks: [{ value: 62, label: '62' }, { value: 125, label: '125' }, { value: 250, label: '250' },
                     { value: 625, label: '625' }, { value: 1250, label: '1.25k' }],
            yDomain: [28, 37], yTicks: [30, 32, 34, 36], markerMode: 'all',
            series: [
                { key: 'ssl', label: 'SSL pretrain', color: '#4878CF', shape: 'circle',
                  values: [32.4, 31.25, 30.77, 29.73, 29.14] },
                { key: 'sem', label: 'Sem. Seg. pretrain', color: '#6ACC65', shape: 'square',
                  values: [35.81, 32.84, 31.86, 31.39, 30.92] },
                { key: 'ins', label: 'Inst. Seg. pretrain', color: '#D65F5F', shape: 'diamond',
                  values: [34.95, 31.93, 31.2, 30.81, 30.56] },
                { key: 'species', label: 'Species Cls. pretrain', color: '#B47CC7', shape: 'triangle',
                  values: [34.72, 32.79, 31.92, 31.2, 30.81] }
            ]
        }
    };
    CHART_DATA['sem-val-curve'].xValues = Array.from({ length: 100 }, function (_, i) { return i + 1; });
    CHART_DATA['species-val-curve'].xValues = Array.from({ length: 100 }, function (_, i) { return i + 1; });

    function svgEl(tag, attrs) {
        var el = document.createElementNS(SVG_NS, tag);
        if (attrs) {
            for (var k in attrs) { el.setAttribute(k, attrs[k]); }
        }
        return el;
    }

    function fmt(v) { return v.toFixed(1) + '%'; }

    /* Shape glyphs: one fixed shape per series, doubling color as a
       colorblind-safe / grayscale-safe identity channel (mirrors the
       marker used in the legend and the tooltip key). */
    function starPoints(cx, cy, outerR, innerR) {
        var pts = [];
        for (var i = 0; i < 10; i++) {
            var r = i % 2 === 0 ? outerR : innerR;
            var angle = -Math.PI / 2 + i * Math.PI / 5;
            pts.push((cx + r * Math.cos(angle)).toFixed(2) + ',' + (cy + r * Math.sin(angle)).toFixed(2));
        }
        return pts.join(' ');
    }

    function createShape(shape, cx, cy, scale) {
        scale = scale || 1;
        switch (shape) {
            case 'square': {
                var s = 8 * scale;
                return svgEl('rect', { x: cx - s / 2, y: cy - s / 2, width: s, height: s, rx: 1 });
            }
            case 'diamond': {
                var d = 6.2 * scale;
                var pts = [[cx, cy - d], [cx + d, cy], [cx, cy + d], [cx - d, cy]]
                    .map(function (p) { return p[0] + ',' + p[1]; }).join(' ');
                return svgEl('polygon', { points: pts });
            }
            case 'triangle': {
                var t = 6 * scale;
                var tp = [[cx, cy - t], [cx + t * 0.95, cy + t * 0.75], [cx - t * 0.95, cy + t * 0.75]]
                    .map(function (p) { return p[0] + ',' + p[1]; }).join(' ');
                return svgEl('polygon', { points: tp });
            }
            case 'star':
                return svgEl('polygon', { points: starPoints(cx, cy, 6.6 * scale, 2.9 * scale) });
            case 'triangle-down': {
                var td = 6 * scale;
                var tdp = [[cx, cy + td], [cx + td * 0.95, cy - td * 0.75], [cx - td * 0.95, cy - td * 0.75]]
                    .map(function (p) { return p[0] + ',' + p[1]; }).join(' ');
                return svgEl('polygon', { points: tdp });
            }
            case 'cross': {
                var cs = 5.6 * scale, cw = 2.1 * scale;
                var cp = [
                    [cx - cw, cy - cs], [cx + cw, cy - cs], [cx + cw, cy - cw],
                    [cx + cs, cy - cw], [cx + cs, cy + cw], [cx + cw, cy + cw],
                    [cx + cw, cy + cs], [cx - cw, cy + cs], [cx - cw, cy + cw],
                    [cx - cs, cy + cw], [cx - cs, cy - cw], [cx - cw, cy - cw]
                ].map(function (p) { return p[0] + ',' + p[1]; }).join(' ');
                return svgEl('polygon', { points: cp, transform: 'rotate(45 ' + cx + ' ' + cy + ')' });
            }
            case 'circle':
            default:
                return svgEl('circle', { cx: cx, cy: cy, r: 4.6 * scale });
        }
    }

    function buildScales(cfg, plot) {
        var logMin = Math.log(cfg.xDomain[0]);
        var logMax = Math.log(cfg.xDomain[1]);
        function xScale(x) {
            var t = cfg.xScale === 'log'
                ? (Math.log(x) - logMin) / (logMax - logMin)
                : (x - cfg.xDomain[0]) / (cfg.xDomain[1] - cfg.xDomain[0]);
            return plot.left + t * plot.w;
        }
        function xInvert(px) {
            var t = (px - plot.left) / plot.w;
            return cfg.xScale === 'log' ? Math.exp(logMin + t * (logMax - logMin))
                                         : cfg.xDomain[0] + t * (cfg.xDomain[1] - cfg.xDomain[0]);
        }
        function yScale(y) {
            var t = (y - cfg.yDomain[0]) / (cfg.yDomain[1] - cfg.yDomain[0]);
            return plot.top + (1 - t) * plot.h;
        }
        return { xScale: xScale, xInvert: xInvert, yScale: yScale };
    }

    function nearestIndex(cfg, dataX) {
        var xs = cfg.xValues, best = 0, bestDist = Infinity;
        for (var i = 0; i < xs.length; i++) {
            var d = Math.abs(xs[i] - dataX);
            if (d < bestDist) { bestDist = d; best = i; }
        }
        return best;
    }

    function initChart(container) {
        var id = container.dataset.chartId;
        var cfg = CHART_DATA[id];
        if (!cfg) { return; }

        var W = 640, H = 320;
        var plot = { left: 74, top: 12, w: W - 74 - 16, h: H - 12 - 50 };
        var scales = buildScales(cfg, plot);

        var seriesGroups = {};
        var legendButtons = {};
        var isolated = null;

        function applyIsolation() {
            cfg.series.forEach(function (s) {
                var muted = isolated !== null && isolated !== s.key;
                seriesGroups[s.key].classList.toggle('is-muted', muted);
                if (legendButtons[s.key]) { legendButtons[s.key].classList.toggle('is-muted', muted); }
            });
        }
        function isolate(key) { isolated = key; applyIsolation(); }
        function clearIsolation() { isolated = null; applyIsolation(); }

        if (cfg.legendGroup) {
            /* A shared legend outside this chart drives isolation instead -
               see renderSharedLegend(). Register so it can reach this chart. */
            LEGEND_GROUPS[cfg.legendGroup] = LEGEND_GROUPS[cfg.legendGroup] || [];
            LEGEND_GROUPS[cfg.legendGroup].push({ isolate: isolate, clear: clearIsolation });
        }

        /* Legend: shape glyph (matches the series' plot marker) + label.
           Hover (and keyboard focus) dims every other series - no click
           needed, mirroring the reference chart's interaction. */
        var legend = document.createElement('div');
        legend.className = 'viz-legend';
        if (!cfg.legendGroup) {
            cfg.series.forEach(function (s) {
                var item = document.createElement('span');
                item.className = 'viz-legend-item';
                item.tabIndex = 0;

                var iconSvg = svgEl('svg', { class: 'viz-legend-icon', viewBox: '0 0 16 16' });
                var icon = createShape(s.shape, 8, 8, 0.85);
                icon.setAttribute('fill', s.color);
                iconSvg.appendChild(icon);

                var label = document.createElement('span');
                label.textContent = s.label;

                item.appendChild(iconSvg);
                item.appendChild(label);
                item.addEventListener('mouseenter', function () { isolate(s.key); });
                item.addEventListener('mouseleave', clearIsolation);
                item.addEventListener('focus', function () { isolate(s.key); });
                item.addEventListener('blur', clearIsolation);

                legend.appendChild(item);
                legendButtons[s.key] = item;
            });
        }

        /* Chart wrap: svg + tooltip */
        var chartWrap = document.createElement('div');
        chartWrap.className = 'viz-chart-wrap';

        var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H, preserveAspectRatio: 'xMidYMid meet' });

        /* Gridlines + y labels */
        cfg.yTicks.forEach(function (yt) {
            var y = scales.yScale(yt);
            svg.appendChild(svgEl('line', {
                class: 'viz-gridline', x1: plot.left, x2: plot.left + plot.w, y1: y, y2: y
            }));
            var label = svgEl('text', {
                class: 'viz-axis-label', x: plot.left - 10, y: y + 5, 'text-anchor': 'end'
            });
            label.textContent = String(yt);
            svg.appendChild(label);
        });

        /* Axis baselines (bottom + left), a step brighter than the gridlines */
        svg.appendChild(svgEl('line', {
            class: 'viz-axis-baseline', x1: plot.left, x2: plot.left + plot.w,
            y1: plot.top + plot.h, y2: plot.top + plot.h
        }));
        svg.appendChild(svgEl('line', {
            class: 'viz-axis-baseline', x1: plot.left, x2: plot.left,
            y1: plot.top, y2: plot.top + plot.h
        }));

        /* X ticks + short tick marks */
        cfg.xTicks.forEach(function (xt) {
            var x = scales.xScale(xt.value);
            svg.appendChild(svgEl('line', {
                class: 'viz-axis-baseline', x1: x, x2: x,
                y1: plot.top + plot.h, y2: plot.top + plot.h + 4
            }));
            var label = svgEl('text', {
                class: 'viz-axis-label', x: x, y: plot.top + plot.h + 21, 'text-anchor': 'middle'
            });
            label.textContent = xt.label;
            svg.appendChild(label);
        });

        /* Axis titles */
        var xTitle = svgEl('text', {
            class: 'viz-axis-title', x: plot.left + plot.w / 2, y: H - 8, 'text-anchor': 'middle'
        });
        xTitle.textContent = cfg.xLabel;
        svg.appendChild(xTitle);

        var yTitle = svgEl('text', {
            class: 'viz-axis-title', x: 0, y: 0, 'text-anchor': 'middle',
            transform: 'translate(31,' + (plot.top + plot.h / 2) + ') rotate(-90)'
        });
        yTitle.textContent = cfg.yLabel;
        svg.appendChild(yTitle);

        /* Crossover annotation: a dashed "cross" showing the target series
           reaching the reference series' final value early - drawn under
           the data lines so the lines themselves stay the focal layer. */
        var crossoverDot = null;
        if (cfg.crossoverAnnotation) {
            var ann = cfg.crossoverAnnotation;
            var refSeries = null, targetSeries = null;
            cfg.series.forEach(function (s) {
                if (s.key === ann.referenceKey) { refSeries = s; }
                if (s.key === ann.targetKey) { targetSeries = s; }
            });
            if (refSeries && targetSeries) {
                var refValue = refSeries.values[refSeries.values.length - 1];
                var crossIdx = -1;
                for (var ci = 0; ci < targetSeries.values.length; ci++) {
                    if (targetSeries.values[ci] >= refValue) { crossIdx = ci; break; }
                }
                if (crossIdx !== -1) {
                    var refY = scales.yScale(refValue);
                    var crossPx = scales.xScale(cfg.xValues[crossIdx]);
                    var crossPy = scales.yScale(targetSeries.values[crossIdx]);

                    var hline = svgEl('line', {
                        class: 'viz-annotation-line', x1: plot.left, x2: plot.left + plot.w, y1: refY, y2: refY
                    });
                    hline.style.stroke = refSeries.color;
                    svg.appendChild(hline);

                    var vline = svgEl('line', {
                        class: 'viz-annotation-line', x1: crossPx, x2: crossPx, y1: plot.top, y2: crossPy
                    });
                    vline.style.stroke = targetSeries.color;
                    svg.appendChild(vline);

                    crossoverDot = svgEl('circle', {
                        class: 'viz-annotation-dot', cx: crossPx, cy: crossPy, r: 4.5, fill: targetSeries.color
                    });
                }
            }
        }

        /* Series lines + markers */
        var pathsToAnimate = [];
        cfg.series.forEach(function (s) {
            var group = svgEl('g', { class: 'viz-series-group' });
            var d = '';
            cfg.xValues.forEach(function (x, i) {
                var px = scales.xScale(x), py = scales.yScale(s.values[i]);
                d += (i === 0 ? 'M' : 'L') + px + ',' + py + ' ';
            });
            var path = svgEl('path', { class: 'viz-line', d: d, stroke: s.color });
            group.appendChild(path);
            pathsToAnimate.push(path);

            cfg.xValues.forEach(function (x, i) {
                var isLast = i === cfg.xValues.length - 1;
                if (cfg.markerMode === 'all' || (cfg.markerMode === 'end' && isLast)) {
                    var px = scales.xScale(x), py = scales.yScale(s.values[i]);
                    var marker = createShape(s.shape, px, py, 1.35);
                    marker.setAttribute('class', 'viz-marker');
                    marker.setAttribute('fill', s.color);
                    group.appendChild(marker);
                }
            });

            svg.appendChild(group);
            seriesGroups[s.key] = group;
        });

        if (crossoverDot) { svg.appendChild(crossoverDot); }

        /* Crosshair + hit layer */
        var crosshair = svgEl('line', {
            class: 'viz-crosshair', x1: 0, x2: 0, y1: plot.top, y2: plot.top + plot.h
        });
        svg.appendChild(crosshair);

        var tooltip = document.createElement('div');
        tooltip.className = 'viz-tooltip';
        chartWrap.appendChild(svg);
        chartWrap.appendChild(tooltip);

        var tooltipRows = {};
        var tooltipX = document.createElement('div');
        tooltipX.className = 'viz-tooltip-x';
        tooltip.appendChild(tooltipX);
        cfg.series.forEach(function (s) {
            var row = document.createElement('div');
            row.className = 'viz-tooltip-row';
            var keySvg = svgEl('svg', { class: 'viz-tooltip-key', viewBox: '0 0 16 16' });
            var keyIcon = createShape(s.shape, 8, 8, 0.85);
            keyIcon.setAttribute('fill', s.color);
            keySvg.appendChild(keyIcon);
            var name = document.createElement('span');
            name.className = 'viz-tooltip-name';
            name.textContent = s.label;
            var value = document.createElement('span');
            value.className = 'viz-tooltip-value';
            row.appendChild(keySvg);
            row.appendChild(name);
            row.appendChild(value);
            tooltip.appendChild(row);
            tooltipRows[s.key] = value;
        });

        var hitLayer = svgEl('rect', {
            class: 'viz-hit-layer', x: plot.left, y: plot.top, width: plot.w, height: plot.h
        });
        svg.appendChild(hitLayer);

        function pointerToChartX(evt) {
            var pt = svg.createSVGPoint();
            pt.x = evt.clientX; pt.y = evt.clientY;
            var loc = pt.matrixTransform(svg.getScreenCTM().inverse());
            return loc.x;
        }

        function showTooltip(evt) {
            var px = pointerToChartX(evt);
            var dataX = scales.xInvert(px);
            var idx = nearestIndex(cfg, dataX);
            var snappedX = cfg.xValues[idx];
            var snappedPx = scales.xScale(snappedX);

            crosshair.setAttribute('x1', snappedPx);
            crosshair.setAttribute('x2', snappedPx);
            crosshair.style.opacity = '1';

            tooltipX.textContent = cfg.xScale === 'log'
                ? snappedX + ' ha'
                : 'Epoch ' + snappedX;
            cfg.series.forEach(function (s) {
                tooltipRows[s.key].textContent = fmt(s.values[idx]);
            });

            var svgRect = svg.getBoundingClientRect();
            var renderScale = svgRect.width / W;
            var left = snappedPx * renderScale + 14;
            if (left + 160 > svgRect.width) { left = snappedPx * renderScale - 174; }
            tooltip.style.left = Math.max(0, left) + 'px';
            tooltip.style.top = (plot.top * renderScale) + 'px';
            tooltip.classList.add('is-visible');
        }

        function hideTooltip() {
            crosshair.style.opacity = '0';
            tooltip.classList.remove('is-visible');
        }

        hitLayer.addEventListener('pointermove', showTooltip);
        hitLayer.addEventListener('pointerleave', hideTooltip);

        if (!cfg.legendGroup) { container.appendChild(legend); }
        container.appendChild(chartWrap);

        /* Draw-in animation, staggered per series */
        pathsToAnimate.forEach(function (path, i) {
            var len = path.getTotalLength();
            path.style.strokeDasharray = len;
            path.style.strokeDashoffset = len;
            setTimeout(function () {
                path.style.strokeDashoffset = '0';
            }, 60 + i * 110);
        });
    }

    /* ---------------------------------------------------------------- */
    /* Scatter chart (single series - no legend needed, per the "one     */
    /* series names itself in the title" rule). Used for the species     */
    /* long-tail gain-vs-frequency plot: one dot per class, plus a       */
    /* dashed zero-line and a dashed OLS trend line as annotations.      */
    /* ---------------------------------------------------------------- */

    var SCATTER_DATA = {
        'species-gain-vs-freq': {
            xLabel: 'Training samples per class (log scale)',
            yLabel: 'Accuracy gain, fine-tune − scratch (%)',
            xDomain: [25, 3000],
            xTicks: [50, 100, 500, 1000, 2500],
            yDomain: [-20, 35],
            yTicks: [-20, -10, 0, 10, 20, 30],
            color: '#2fae3a',
            trend: { slope: -0.004324151022940693, intercept: 5.847321189089543, r: -0.20, p: 0.261 },
            points: [
                { class: 'Pinus_sylvestris', x: 2648, scratch: 92.75, ft: 92.44, gain: -0.31 },
                { class: 'Fagus_sylvatica', x: 1975, scratch: 77.51, ft: 86.19, gain: 8.68 },
                { class: 'Picea_abies', x: 1550, scratch: 90.76, ft: 88.68, gain: -2.08 },
                { class: 'Acer_campestre', x: 1018, scratch: 85.14, ft: 81.98, gain: -3.15 },
                { class: 'Carpinus_betulus', x: 991, scratch: 57.14, ft: 40.48, gain: -16.67 },
                { class: 'Quercus_faginea', x: 805, scratch: 86.67, ft: 80.48, gain: -6.19 },
                { class: 'Betula_pendula', x: 577, scratch: 81.01, ft: 81.65, gain: 0.63 },
                { class: 'Pseudotsuga_menziesii', x: 525, scratch: 89.47, ft: 86.47, gain: -3.01 },
                { class: 'Acer_pseudoplatanus', x: 471, scratch: 81.90, ft: 92.38, gain: 10.48 },
                { class: 'Pinus_nigra', x: 377, scratch: 91.95, ft: 90.80, gain: -1.15 },
                { class: 'Tilia_cordata', x: 367, scratch: 63.64, ft: 57.14, gain: -6.49 },
                { class: 'Quercus_petraea', x: 288, scratch: 83.56, ft: 80.82, gain: -2.74 },
                { class: 'Pinus_contorta', x: 257, scratch: 92.42, ft: 92.42, gain: 0.00 },
                { class: 'Quercus_ilex', x: 248, scratch: 80.39, ft: 94.12, gain: 13.73 },
                { class: 'Eucalyptus_miniata', x: 233, scratch: 95.74, ft: 95.74, gain: 0.00 },
                { class: 'Ulmus_laevis', x: 215, scratch: 3.17, ft: 30.16, gain: 26.98 },
                { class: 'Crataegus_monogyna', x: 190, scratch: 58.97, ft: 61.54, gain: 2.56 },
                { class: 'Quercus_robur', x: 157, scratch: 78.95, ft: 78.95, gain: 0.00 },
                { class: 'Picea_glauca', x: 139, scratch: 100.00, ft: 100.00, gain: 0.00 },
                { class: 'Pinus_resinosa', x: 132, scratch: 97.22, ft: 100.00, gain: 2.78 },
                { class: 'Quercus_rubra', x: 119, scratch: 80.49, ft: 70.73, gain: -9.76 },
                { class: 'Abies_alba', x: 99, scratch: 55.00, ft: 75.00, gain: 20.00 },
                { class: 'Fraxinus_excelsior', x: 93, scratch: 0.00, ft: 18.18, gain: 18.18 },
                { class: 'Acer_saccharum', x: 91, scratch: 100.00, ft: 89.47, gain: -10.53 },
                { class: 'Populus_tremuloides', x: 89, scratch: 78.95, ft: 89.47, gain: 10.53 },
                { class: 'Euonymus_europaeus', x: 86, scratch: 27.78, ft: 61.11, gain: 33.33 },
                { class: 'Fraxinus_angustifolia', x: 80, scratch: 51.85, ft: 51.85, gain: 0.00 },
                { class: 'Pinus_pinaster', x: 79, scratch: 33.33, ft: 66.67, gain: 33.33 },
                { class: 'Larix_decidua', x: 78, scratch: 43.75, ft: 43.75, gain: 0.00 },
                { class: 'Pinus_radiata', x: 73, scratch: 91.67, ft: 83.33, gain: -8.33 },
                { class: 'Populus_deltoides', x: 46, scratch: 100.00, ft: 100.00, gain: 0.00 },
                { class: 'Corylus_avellana', x: 38, scratch: 50.00, ft: 80.00, gain: 30.00 },
                { class: 'Prunus_avium', x: 31, scratch: 54.55, ft: 45.45, gain: -9.09 }
            ]
        }
    };

    function initScatterChart(container) {
        var id = container.dataset.chartId;
        var cfg = SCATTER_DATA[id];
        if (!cfg) { return; }

        var W = 640, H = 320;
        var plot = { left: 74, top: 12, w: W - 74 - 16, h: H - 12 - 50 };
        var logMin = Math.log(cfg.xDomain[0]), logMax = Math.log(cfg.xDomain[1]);

        function xScale(x) {
            var t = (Math.log(x) - logMin) / (logMax - logMin);
            return plot.left + t * plot.w;
        }
        function yScale(y) {
            var t = (y - cfg.yDomain[0]) / (cfg.yDomain[1] - cfg.yDomain[0]);
            return plot.top + (1 - t) * plot.h;
        }

        var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H, preserveAspectRatio: 'xMidYMid meet' });

        /* Gridlines + y labels */
        cfg.yTicks.forEach(function (yt) {
            var y = yScale(yt);
            svg.appendChild(svgEl('line', {
                class: 'viz-gridline', x1: plot.left, x2: plot.left + plot.w, y1: y, y2: y
            }));
            var label = svgEl('text', {
                class: 'viz-axis-label', x: plot.left - 10, y: y + 5, 'text-anchor': 'end'
            });
            label.textContent = yt;
            svg.appendChild(label);
        });

        /* Axis baselines */
        svg.appendChild(svgEl('line', {
            class: 'viz-axis-baseline', x1: plot.left, x2: plot.left + plot.w,
            y1: plot.top + plot.h, y2: plot.top + plot.h
        }));
        svg.appendChild(svgEl('line', {
            class: 'viz-axis-baseline', x1: plot.left, x2: plot.left,
            y1: plot.top, y2: plot.top + plot.h
        }));

        /* X ticks (log-scale) */
        cfg.xTicks.forEach(function (xt) {
            var x = xScale(xt);
            svg.appendChild(svgEl('line', {
                class: 'viz-axis-baseline', x1: x, x2: x,
                y1: plot.top + plot.h, y2: plot.top + plot.h + 4
            }));
            var label = svgEl('text', {
                class: 'viz-axis-label', x: x, y: plot.top + plot.h + 21, 'text-anchor': 'middle'
            });
            label.textContent = xt >= 1000 ? (xt / 1000) + 'k' : String(xt);
            svg.appendChild(label);
        });

        /* Axis titles */
        var xTitle = svgEl('text', {
            class: 'viz-axis-title', x: plot.left + plot.w / 2, y: H - 8, 'text-anchor': 'middle'
        });
        xTitle.textContent = cfg.xLabel;
        svg.appendChild(xTitle);

        var yTitle = svgEl('text', {
            class: 'viz-axis-title', x: 0, y: 0, 'text-anchor': 'middle',
            transform: 'translate(31,' + (plot.top + plot.h / 2) + ') rotate(-90)'
        });
        yTitle.textContent = cfg.yLabel;
        svg.appendChild(yTitle);

        /* Zero reference line */
        var zeroY = yScale(0);
        var zeroLine = svgEl('line', {
            class: 'viz-annotation-line', x1: plot.left, x2: plot.left + plot.w, y1: zeroY, y2: zeroY
        });
        zeroLine.style.stroke = '#898781';
        svg.appendChild(zeroLine);

        /* OLS trend line, sampled across the x domain */
        var trendPath = '';
        var steps = 40;
        for (var i = 0; i <= steps; i++) {
            var xt2 = Math.exp(logMin + (i / steps) * (logMax - logMin));
            var yt2 = cfg.trend.slope * xt2 + cfg.trend.intercept;
            trendPath += (i === 0 ? 'M' : 'L') + xScale(xt2) + ',' + yScale(yt2) + ' ';
        }
        var trendLine = svgEl('path', { class: 'viz-annotation-line', d: trendPath, fill: 'none' });
        trendLine.style.stroke = cfg.color;
        trendLine.style.opacity = '0.85';
        svg.appendChild(trendLine);

        /* r/p annotation */
        var annotation = svgEl('text', {
            class: 'viz-axis-label', x: plot.left + plot.w, y: plot.top + 14, 'text-anchor': 'end'
        });
        annotation.textContent = 'r = ' + cfg.trend.r.toFixed(2) + ', p = ' + cfg.trend.p.toFixed(3);
        svg.appendChild(annotation);

        /* Points */
        var pointEls = [];
        cfg.points.forEach(function (p) {
            var px = xScale(p.x), py = yScale(p.gain);
            var dot = svgEl('circle', { class: 'viz-marker', cx: px, cy: py, r: 5, fill: cfg.color });
            svg.appendChild(dot);
            pointEls.push({ p: p, px: px, py: py, el: dot });
        });

        /* Hover: nearest-point layer (dense scatter - pointer only needs
           to be closest, not dead-center on an individual dot). */
        var chartWrap = document.createElement('div');
        chartWrap.className = 'viz-chart-wrap';
        var tooltip = document.createElement('div');
        tooltip.className = 'viz-tooltip';
        var ttClass = document.createElement('div');
        ttClass.className = 'viz-tooltip-x';
        var ttSamples = document.createElement('div');
        ttSamples.className = 'viz-tooltip-name';
        ttSamples.style.marginBottom = '4px';
        var rowScratch = document.createElement('div');
        rowScratch.className = 'viz-tooltip-row';
        var rowFt = document.createElement('div');
        rowFt.className = 'viz-tooltip-row';
        var rowGain = document.createElement('div');
        rowGain.className = 'viz-tooltip-row';
        function makeRow(row, label) {
            var name = document.createElement('span');
            name.className = 'viz-tooltip-name';
            name.textContent = label;
            var value = document.createElement('span');
            value.className = 'viz-tooltip-value';
            row.appendChild(name);
            row.appendChild(value);
            return value;
        }
        var valScratch = makeRow(rowScratch, 'Scratch');
        var valFt = makeRow(rowFt, 'Fine-tune');
        var valGain = makeRow(rowGain, 'Gain');
        tooltip.appendChild(ttClass);
        tooltip.appendChild(ttSamples);
        tooltip.appendChild(rowScratch);
        tooltip.appendChild(rowFt);
        tooltip.appendChild(rowGain);

        chartWrap.appendChild(svg);
        chartWrap.appendChild(tooltip);

        var hitLayer = svgEl('rect', {
            class: 'viz-hit-layer', x: plot.left, y: plot.top, width: plot.w, height: plot.h
        });
        svg.appendChild(hitLayer);

        var activeDot = null;
        function showTooltip(evt) {
            var pt = svg.createSVGPoint();
            pt.x = evt.clientX; pt.y = evt.clientY;
            var loc = pt.matrixTransform(svg.getScreenCTM().inverse());

            var nearest = null, bestDist = Infinity;
            pointEls.forEach(function (item) {
                var dx = item.px - loc.x, dy = item.py - loc.y;
                var d = dx * dx + dy * dy;
                if (d < bestDist) { bestDist = d; nearest = item; }
            });
            if (!nearest) { return; }

            if (activeDot) { activeDot.setAttribute('r', 5); }
            nearest.el.setAttribute('r', 7);
            activeDot = nearest.el;

            ttClass.textContent = nearest.p.class.replace(/_/g, ' ');
            ttSamples.textContent = nearest.p.x + ' training samples';
            valScratch.textContent = nearest.p.scratch.toFixed(1) + '%';
            valFt.textContent = nearest.p.ft.toFixed(1) + '%';
            valGain.textContent = (nearest.p.gain >= 0 ? '+' : '') + nearest.p.gain.toFixed(1) + '%';
            valGain.style.color = nearest.p.gain >= 0 ? '#4ade80' : '#f87171';

            var svgRect = svg.getBoundingClientRect();
            var renderScale = svgRect.width / W;
            var left = nearest.px * renderScale + 14;
            if (left + 160 > svgRect.width) { left = nearest.px * renderScale - 174; }
            var top = nearest.py * renderScale - 90;
            tooltip.style.left = Math.max(0, left) + 'px';
            tooltip.style.top = Math.max(0, top) + 'px';
            tooltip.classList.add('is-visible');
        }
        function hideTooltip() {
            if (activeDot) { activeDot.setAttribute('r', 5); activeDot = null; }
            tooltip.classList.remove('is-visible');
        }
        hitLayer.addEventListener('pointermove', showTooltip);
        hitLayer.addEventListener('pointerleave', hideTooltip);

        container.appendChild(chartWrap);
    }

    /* ---------------------------------------------------------------- */
    /* Parity (predicted-vs-reference) charts: several stacked panels    */
    /* sharing one species legend, a 1:1 reference line, and a per-panel */
    /* RMSE / MD / R^2 readout - the web equivalent of the paper figure. */
    /* ---------------------------------------------------------------- */

    var PARITY_DATA = {
        'age-pred-vs-ref': {
            unit: 'years',
            domain: [0, 300],
            ticks: [0, 100, 200, 300],
            species: {
                p: { label: 'Pinus sylvestris', color: '#e08b4c' },
                s: { label: 'Picea abies', color: '#5b9bd5' }
            },
            panels: [
                { key: 'linear', title: 'Linear probing', metrics: { rmse: 29.73, bias: -2.95, r2: 0.52 },
                  points: [[56.0,69.7,'p'],[55.0,56.53,'p'],[58.0,51.65,'p'],[34.0,65.01,'p'],[53.0,61.6,'p'],[87.0,73.5,'s'],[84.0,72.93,'s'],[61.0,54.53,'s'],[72.0,64.7,'s'],[72.0,86.96,'p'],[56.0,76.53,'p'],[56.0,72.29,'p'],[63.0,88.43,'p'],[60.0,56.86,'p'],[59.0,73.51,'p'],[68.0,71.15,'p'],[60.0,70.27,'p'],[66.0,76.67,'p'],[55.0,70.51,'p'],[55.0,52.28,'p'],[69.0,74.13,'p'],[70.0,85.76,'p'],[160.0,91.31,'s'],[131.0,71.37,'s'],[138.0,82.37,'s'],[136.0,66.88,'s'],[100.0,85.51,'s'],[114.0,53.16,'s'],[59.0,78.95,'p'],[57.0,83.71,'p'],[52.0,69.46,'p'],[52.0,64.05,'p'],[53.0,59.44,'p'],[54.0,65.04,'p'],[56.0,76.29,'p'],[57.0,60.3,'p'],[63.0,72.16,'p'],[70.0,68.98,'p'],[70.0,54.37,'p'],[65.0,64.29,'p'],[56.0,62.43,'p'],[55.0,48.92,'p'],[58.0,47.21,'p'],[34.0,63.06,'p'],[53.0,56.49,'p'],[87.0,77.9,'s'],[84.0,87.78,'s'],[61.0,71.13,'s'],[72.0,63.94,'s'],[72.0,89.05,'p'],[56.0,75.32,'p'],[56.0,84.17,'p'],[63.0,83.24,'p'],[60.0,52.62,'p'],[59.0,71.7,'p'],[68.0,69.01,'p'],[60.0,69.29,'p'],[66.0,63.65,'p'],[55.0,61.26,'p'],[55.0,53.07,'p'],[69.0,76.28,'p'],[70.0,92.6,'p'],[160.0,105.33,'s'],[131.0,74.37,'s'],[138.0,84.62,'s'],[136.0,76.94,'s'],[100.0,79.68,'s'],[114.0,63.0,'s'],[59.0,71.76,'p'],[57.0,60.3,'p'],[52.0,55.45,'p'],[52.0,67.93,'p'],[53.0,70.36,'p'],[54.0,66.86,'p'],[56.0,76.68,'p'],[57.0,57.46,'p'],[63.0,64.4,'p'],[70.0,68.81,'p'],[70.0,54.51,'p'],[65.0,58.67,'p'],[145.0,82.91,'s'],[239.0,68.04,'s'],[165.0,66.99,'s'],[280.0,67.26,'s'],[278.0,56.36,'s'],[64.0,64.72,'p'],[81.0,73.9,'p'],[62.0,48.74,'p'],[67.0,65.7,'p'],[77.0,45.31,'p'],[62.0,57.34,'p'],[58.0,66.55,'p'],[51.0,46.51,'p'],[58.0,31.33,'p'],[112.0,77.7,'p'],[42.0,47.31,'p'],[112.0,96.95,'p'],[96.0,83.27,'p'],[65.0,58.89,'p'],[50.0,57.96,'p'],[58.0,79.64,'p'],[88.0,91.2,'p'],[69.0,45.06,'p'],[74.0,45.1,'p'],[84.0,68.93,'p'],[69.0,39.2,'p'],[119.0,75.17,'p'],[119.0,108.47,'p'],[72.0,58.61,'p'],[57.0,59.7,'p'],[65.0,69.75,'p'],[89.0,127.38,'p'],[73.0,138.87,'p'],[81.0,128.03,'p'],[72.0,124.69,'p'],[81.0,129.28,'p'],[99.0,126.01,'p'],[52.0,71.51,'s'],[56.0,66.89,'s'],[61.0,40.64,'s'],[67.0,53.15,'s'],[52.0,43.11,'s'],[9.0,7.95,'s'],[37.0,56.66,'s'],[17.0,34.93,'s'],[21.0,11.49,'s'],[9.0,-1.71,'s'],[13.0,27.5,'s'],[8.0,-9.54,'s'],[1.0,3.7,'p'],[10.0,-3.14,'s'],[35.0,67.49,'s'],[10.0,46.78,'s'],[38.0,40.8,'s'],[7.0,3.69,'s'],[1.0,-4.5,'p'],[11.0,11.3,'s'],[12.0,27.21,'s'],[11.0,3.58,'s'],[11.0,2.12,'s'],[16.0,34.4,'s'],[7.0,6.36,'p'],[8.0,4.4,'p'],[11.0,18.12,'s'],[11.0,29.12,'s'],[8.0,8.14,'s'],[5.0,3.5,'p'],[8.0,18.85,'p'],[3.0,-2.51,'p'],[7.0,-2.11,'p'],[7.0,13.98,'p'],[9.0,13.03,'s'],[4.0,7.59,'p'],[4.0,13.87,'p'],[4.0,-2.96,'p'],[16.0,-0.45,'s'],[2.0,-23.56,'p'],[6.0,13.48,'p'],[11.0,21.14,'p'],[13.0,-2.09,'s'],[4.0,3.63,'p'],[1.0,4.34,'s'],[6.0,11.36,'p'],[5.0,9.45,'p'],[9.0,-1.78,'p'],[12.0,-9.39,'s'],[1.0,-0.59,'p'],[10.0,16.04,'s'],[13.0,0.72,'s'],[15.0,25.63,'s'],[17.0,25.05,'p'],[18.0,12.63,'s'],[20.0,17.95,'p'],[17.0,30.49,'s'],[12.0,6.39,'s'],[10.0,-2.61,'s'],[16.0,21.47,'p'],[15.0,19.75,'p'],[7.0,7.01,'s'],[7.0,-1.4,'s'],[13.0,10.14,'s'],[28.0,18.19,'s'],[39.0,42.75,'s'],[35.0,47.63,'s'],[25.0,45.48,'s'],[25.0,61.06,'s'],[38.0,69.08,'s'],[35.0,47.87,'s'],[15.0,23.38,'s'],[14.0,6.67,'s'],[12.0,-10.65,'s'],[14.0,-3.11,'s'],[20.0,31.03,'s'],[11.0,9.22,'s'],[8.0,7.8,'s'],[36.0,34.77,'s'],[16.0,29.6,'s'],[20.0,16.98,'s'],[8.0,9.35,'s'],[12.0,31.66,'s'],[7.0,1.46,'s'],[9.0,8.56,'s'],[34.0,57.55,'s'],[9.0,8.9,'s'],[37.0,46.76,'s'],[6.0,14.93,'s'],[10.0,16.22,'s'],[11.0,34.36,'s'],[10.0,0.8,'s'],[10.0,24.53,'s'],[15.0,34.49,'s'],[6.0,29.05,'p'],[7.0,-4.63,'p'],[10.0,2.64,'s'],[10.0,18.28,'s'],[7.0,2.14,'s'],[4.0,2.99,'p'],[7.0,-12.85,'p'],[2.0,-17.67,'p'],[6.0,15.61,'p'],[6.0,-10.27,'p'],[8.0,15.11,'s'],[3.0,-10.73,'p'],[3.0,-15.51,'p'],[3.0,-16.54,'p'],[15.0,21.79,'s'],[1.0,-26.54,'p'],[5.0,0.7,'p'],[10.0,7.8,'p'],[12.0,1.3,'s'],[3.0,-18.21,'p'],[5.0,19.81,'p'],[4.0,15.35,'p'],[8.0,7.66,'p'],[11.0,12.29,'s'],[9.0,22.31,'s'],[12.0,10.16,'s'],[14.0,6.49,'s'],[16.0,-3.39,'p'],[17.0,12.01,'s'],[19.0,4.93,'p'],[16.0,23.17,'s'],[11.0,14.61,'s'],[9.0,19.56,'s'],[15.0,16.04,'p'],[14.0,5.81,'p'],[6.0,11.06,'s'],[6.0,8.46,'s'],[12.0,7.75,'s'],[27.0,7.42,'s'],[38.0,44.18,'s'],[34.0,27.65,'s'],[24.0,28.45,'s'],[24.0,41.7,'s'],[37.0,49.92,'s'],[34.0,36.98,'s'],[14.0,19.24,'s'],[13.0,18.89,'s'],[11.0,11.16,'s'],[13.0,8.99,'s'],[19.0,27.59,'s'],[10.0,-8.45,'s']] },
                { key: 'scratch', title: 'Scratch', metrics: { rmse: 24.09, bias: 2.56, r2: 0.69 },
                  points: [[56.0,64.76,'p'],[55.0,53.93,'p'],[58.0,71.0,'p'],[34.0,67.25,'p'],[53.0,59.49,'p'],[87.0,80.98,'s'],[84.0,79.34,'s'],[61.0,81.51,'s'],[72.0,79.35,'s'],[72.0,61.71,'p'],[56.0,65.02,'p'],[56.0,64.69,'p'],[63.0,62.73,'p'],[60.0,63.37,'p'],[59.0,55.47,'p'],[68.0,64.84,'p'],[60.0,64.09,'p'],[66.0,61.41,'p'],[55.0,59.88,'p'],[55.0,60.0,'p'],[69.0,63.46,'p'],[70.0,58.77,'p'],[160.0,89.7,'s'],[131.0,90.77,'s'],[138.0,89.97,'s'],[136.0,82.65,'s'],[100.0,84.55,'s'],[114.0,95.41,'s'],[59.0,79.3,'p'],[57.0,53.97,'p'],[52.0,72.07,'p'],[52.0,54.73,'p'],[53.0,69.54,'p'],[54.0,64.14,'p'],[56.0,69.88,'p'],[57.0,49.96,'p'],[63.0,73.48,'p'],[70.0,62.81,'p'],[70.0,60.73,'p'],[65.0,51.75,'p'],[56.0,59.54,'p'],[55.0,54.34,'p'],[58.0,50.27,'p'],[34.0,60.13,'p'],[53.0,58.06,'p'],[87.0,85.67,'s'],[84.0,87.59,'s'],[61.0,82.26,'s'],[72.0,77.12,'s'],[72.0,62.37,'p'],[56.0,64.18,'p'],[56.0,61.88,'p'],[63.0,58.05,'p'],[60.0,58.22,'p'],[59.0,62.91,'p'],[68.0,69.89,'p'],[60.0,63.58,'p'],[66.0,64.16,'p'],[55.0,59.0,'p'],[55.0,58.27,'p'],[69.0,63.44,'p'],[70.0,68.8,'p'],[160.0,92.37,'s'],[131.0,88.92,'s'],[138.0,90.71,'s'],[136.0,89.18,'s'],[100.0,86.39,'s'],[114.0,95.98,'s'],[59.0,65.39,'p'],[57.0,54.35,'p'],[52.0,60.28,'p'],[52.0,51.04,'p'],[53.0,65.94,'p'],[54.0,57.63,'p'],[56.0,62.47,'p'],[57.0,62.23,'p'],[63.0,60.4,'p'],[70.0,55.98,'p'],[70.0,59.42,'p'],[65.0,47.45,'p'],[145.0,110.49,'s'],[239.0,136.51,'s'],[165.0,126.37,'s'],[280.0,131.04,'s'],[278.0,149.48,'s'],[64.0,100.24,'p'],[81.0,99.33,'p'],[62.0,60.66,'p'],[67.0,72.53,'p'],[77.0,67.05,'p'],[62.0,63.24,'p'],[58.0,72.44,'p'],[51.0,66.76,'p'],[58.0,22.25,'p'],[112.0,66.06,'p'],[42.0,40.5,'p'],[112.0,82.92,'p'],[96.0,96.58,'p'],[65.0,82.64,'p'],[50.0,48.2,'p'],[58.0,64.47,'p'],[88.0,69.11,'p'],[69.0,47.46,'p'],[74.0,43.23,'p'],[84.0,48.91,'p'],[69.0,58.03,'p'],[119.0,50.71,'p'],[119.0,73.89,'p'],[72.0,154.16,'p'],[57.0,58.32,'p'],[65.0,60.84,'p'],[89.0,114.98,'p'],[73.0,115.62,'p'],[81.0,122.47,'p'],[72.0,138.5,'p'],[81.0,123.52,'p'],[99.0,149.93,'p'],[52.0,70.02,'s'],[56.0,69.09,'s'],[61.0,85.27,'s'],[67.0,73.76,'s'],[52.0,77.11,'s'],[9.0,19.07,'s'],[37.0,74.19,'s'],[17.0,49.41,'s'],[21.0,19.17,'s'],[9.0,18.2,'s'],[13.0,23.23,'s'],[8.0,19.56,'s'],[1.0,4.3,'p'],[10.0,6.48,'s'],[35.0,56.79,'s'],[10.0,23.22,'s'],[38.0,57.28,'s'],[7.0,4.68,'s'],[1.0,4.29,'p'],[11.0,17.39,'s'],[12.0,24.44,'s'],[11.0,18.43,'s'],[11.0,19.22,'s'],[16.0,59.0,'s'],[7.0,19.17,'p'],[8.0,22.46,'p'],[11.0,4.83,'s'],[11.0,21.24,'s'],[8.0,14.19,'s'],[5.0,4.39,'p'],[8.0,17.2,'p'],[3.0,4.41,'p'],[7.0,11.54,'p'],[7.0,7.66,'p'],[9.0,15.27,'s'],[4.0,4.34,'p'],[4.0,4.42,'p'],[4.0,4.44,'p'],[16.0,20.56,'s'],[2.0,4.23,'p'],[6.0,5.81,'p'],[11.0,21.9,'p'],[13.0,22.88,'s'],[4.0,11.81,'p'],[1.0,4.32,'s'],[6.0,4.77,'p'],[5.0,5.8,'p'],[9.0,19.23,'p'],[12.0,21.66,'s'],[1.0,4.31,'p'],[10.0,21.99,'s'],[13.0,8.47,'s'],[15.0,25.04,'s'],[17.0,11.61,'p'],[18.0,21.68,'s'],[20.0,23.73,'p'],[17.0,21.35,'s'],[12.0,22.84,'s'],[10.0,15.22,'s'],[16.0,20.17,'p'],[15.0,18.91,'p'],[7.0,20.71,'s'],[7.0,17.92,'s'],[13.0,13.64,'s'],[28.0,29.1,'s'],[39.0,60.01,'s'],[35.0,40.08,'s'],[25.0,52.13,'s'],[25.0,74.49,'s'],[38.0,65.92,'s'],[35.0,48.43,'s'],[15.0,16.45,'s'],[14.0,20.7,'s'],[12.0,15.31,'s'],[14.0,19.19,'s'],[20.0,111.85,'s'],[11.0,6.18,'s'],[8.0,18.25,'s'],[36.0,72.31,'s'],[16.0,22.47,'s'],[20.0,20.96,'s'],[8.0,16.72,'s'],[12.0,17.36,'s'],[7.0,16.68,'s'],[9.0,6.43,'s'],[34.0,67.78,'s'],[9.0,22.52,'s'],[37.0,65.42,'s'],[6.0,4.94,'s'],[10.0,21.63,'s'],[11.0,22.75,'s'],[10.0,22.62,'s'],[10.0,19.62,'s'],[15.0,33.65,'s'],[6.0,5.73,'p'],[7.0,18.26,'p'],[10.0,7.52,'s'],[10.0,24.67,'s'],[7.0,11.74,'s'],[4.0,5.4,'p'],[7.0,9.51,'p'],[2.0,4.35,'p'],[6.0,6.69,'p'],[6.0,7.88,'p'],[8.0,11.49,'s'],[3.0,5.03,'p'],[3.0,5.5,'p'],[3.0,4.88,'p'],[15.0,18.75,'s'],[1.0,4.53,'p'],[5.0,7.81,'p'],[10.0,22.76,'p'],[12.0,21.24,'s'],[3.0,7.3,'p'],[5.0,7.34,'p'],[4.0,5.85,'p'],[8.0,8.93,'p'],[11.0,21.79,'s'],[9.0,16.3,'s'],[12.0,12.8,'s'],[14.0,23.55,'s'],[16.0,14.4,'p'],[17.0,17.48,'s'],[19.0,20.5,'p'],[16.0,22.17,'s'],[11.0,22.59,'s'],[9.0,8.37,'s'],[15.0,20.98,'p'],[14.0,17.32,'p'],[6.0,21.8,'s'],[6.0,15.32,'s'],[12.0,15.99,'s'],[27.0,34.53,'s'],[38.0,74.08,'s'],[34.0,38.5,'s'],[24.0,47.8,'s'],[24.0,52.31,'s'],[37.0,49.68,'s'],[34.0,61.73,'s'],[14.0,16.72,'s'],[13.0,23.69,'s'],[11.0,7.47,'s'],[13.0,14.89,'s'],[19.0,24.58,'s'],[10.0,5.81,'s']] },
                { key: 'finetune', title: 'Fine-tuning', metrics: { rmse: 21.61, bias: 1.78, r2: 0.75 },
                  points: [[56.0,55.8,'p'],[55.0,39.12,'p'],[58.0,73.61,'p'],[34.0,52.81,'p'],[53.0,65.11,'p'],[87.0,93.38,'s'],[84.0,83.4,'s'],[61.0,93.08,'s'],[72.0,86.34,'s'],[72.0,63.07,'p'],[56.0,58.79,'p'],[56.0,70.9,'p'],[63.0,58.89,'p'],[60.0,81.09,'p'],[59.0,51.88,'p'],[68.0,71.12,'p'],[60.0,72.59,'p'],[66.0,73.79,'p'],[55.0,77.4,'p'],[55.0,82.54,'p'],[69.0,57.74,'p'],[70.0,57.44,'p'],[160.0,105.72,'s'],[131.0,90.76,'s'],[138.0,112.02,'s'],[136.0,92.17,'s'],[100.0,89.72,'s'],[114.0,107.59,'s'],[59.0,65.67,'p'],[57.0,49.34,'p'],[52.0,59.2,'p'],[52.0,40.93,'p'],[53.0,50.41,'p'],[54.0,55.59,'p'],[56.0,49.53,'p'],[57.0,42.37,'p'],[63.0,61.3,'p'],[70.0,57.79,'p'],[70.0,51.51,'p'],[65.0,44.94,'p'],[56.0,50.33,'p'],[55.0,41.86,'p'],[58.0,40.73,'p'],[34.0,52.48,'p'],[53.0,48.98,'p'],[87.0,103.14,'s'],[84.0,89.48,'s'],[61.0,96.97,'s'],[72.0,74.47,'s'],[72.0,78.23,'p'],[56.0,50.21,'p'],[56.0,68.99,'p'],[63.0,54.8,'p'],[60.0,91.06,'p'],[59.0,60.85,'p'],[68.0,84.77,'p'],[60.0,83.4,'p'],[66.0,77.69,'p'],[55.0,64.04,'p'],[55.0,76.43,'p'],[69.0,68.7,'p'],[70.0,81.43,'p'],[160.0,94.41,'s'],[131.0,108.45,'s'],[138.0,88.56,'s'],[136.0,111.88,'s'],[100.0,116.2,'s'],[114.0,121.02,'s'],[59.0,80.34,'p'],[57.0,46.11,'p'],[52.0,50.25,'p'],[52.0,45.24,'p'],[53.0,52.97,'p'],[54.0,54.42,'p'],[56.0,46.99,'p'],[57.0,52.16,'p'],[63.0,76.05,'p'],[70.0,58.04,'p'],[70.0,64.98,'p'],[65.0,45.81,'p'],[145.0,160.48,'s'],[239.0,127.96,'s'],[165.0,154.61,'s'],[280.0,158.18,'s'],[278.0,160.04,'s'],[64.0,60.09,'p'],[81.0,94.23,'p'],[62.0,38.75,'p'],[67.0,46.58,'p'],[77.0,139.6,'p'],[62.0,52.28,'p'],[58.0,47.69,'p'],[51.0,34.2,'p'],[58.0,21.84,'p'],[112.0,107.56,'p'],[42.0,26.84,'p'],[112.0,115.34,'p'],[96.0,104.91,'p'],[65.0,138.59,'p'],[50.0,38.33,'p'],[58.0,49.3,'p'],[88.0,70.06,'p'],[69.0,78.98,'p'],[74.0,78.51,'p'],[84.0,56.15,'p'],[69.0,103.02,'p'],[119.0,109.81,'p'],[119.0,125.15,'p'],[72.0,186.13,'p'],[57.0,83.4,'p'],[65.0,113.36,'p'],[89.0,119.3,'p'],[73.0,129.1,'p'],[81.0,129.64,'p'],[72.0,118.78,'p'],[81.0,119.31,'p'],[99.0,144.84,'p'],[52.0,92.43,'s'],[56.0,57.76,'s'],[61.0,48.71,'s'],[67.0,85.73,'s'],[52.0,72.71,'s'],[9.0,3.07,'s'],[37.0,60.76,'s'],[17.0,11.54,'s'],[21.0,17.31,'s'],[9.0,1.08,'s'],[13.0,6.5,'s'],[8.0,12.68,'s'],[1.0,6.09,'p'],[10.0,7.37,'s'],[35.0,31.77,'s'],[10.0,18.84,'s'],[38.0,40.1,'s'],[7.0,7.29,'s'],[1.0,12.03,'p'],[11.0,3.88,'s'],[12.0,14.82,'s'],[11.0,17.97,'s'],[11.0,13.82,'s'],[16.0,10.42,'s'],[7.0,15.15,'p'],[8.0,11.06,'p'],[11.0,11.99,'s'],[11.0,8.65,'s'],[8.0,10.88,'s'],[5.0,8.27,'p'],[8.0,10.9,'p'],[3.0,10.61,'p'],[7.0,10.62,'p'],[7.0,12.23,'p'],[9.0,9.02,'s'],[4.0,11.02,'p'],[4.0,10.6,'p'],[4.0,11.85,'p'],[16.0,12.7,'s'],[2.0,9.37,'p'],[6.0,7.41,'p'],[11.0,13.84,'p'],[13.0,14.63,'s'],[4.0,11.55,'p'],[1.0,5.79,'s'],[6.0,14.94,'p'],[5.0,8.28,'p'],[9.0,14.31,'p'],[12.0,10.01,'s'],[1.0,5.86,'p'],[10.0,14.87,'s'],[13.0,14.53,'s'],[15.0,14.06,'s'],[17.0,23.58,'p'],[18.0,14.94,'s'],[20.0,14.46,'p'],[17.0,9.9,'s'],[12.0,15.8,'s'],[10.0,12.56,'s'],[16.0,6.35,'p'],[15.0,3.07,'p'],[7.0,11.12,'s'],[7.0,10.54,'s'],[13.0,4.0,'s'],[28.0,25.44,'s'],[39.0,28.8,'s'],[35.0,23.79,'s'],[25.0,21.47,'s'],[25.0,40.03,'s'],[38.0,52.14,'s'],[35.0,22.2,'s'],[15.0,20.59,'s'],[14.0,12.52,'s'],[12.0,10.54,'s'],[14.0,13.91,'s'],[20.0,19.31,'s'],[11.0,11.03,'s'],[8.0,12.88,'s'],[36.0,34.24,'s'],[16.0,19.79,'s'],[20.0,16.56,'s'],[8.0,14.75,'s'],[12.0,20.39,'s'],[7.0,16.59,'s'],[9.0,12.32,'s'],[34.0,58.15,'s'],[9.0,17.21,'s'],[37.0,47.5,'s'],[6.0,10.38,'s'],[10.0,18.3,'s'],[11.0,22.98,'s'],[10.0,13.39,'s'],[10.0,15.15,'s'],[15.0,25.25,'s'],[6.0,10.68,'p'],[7.0,13.45,'p'],[10.0,13.17,'s'],[10.0,15.05,'s'],[7.0,13.16,'s'],[4.0,10.43,'p'],[7.0,10.67,'p'],[2.0,8.6,'p'],[6.0,13.01,'p'],[6.0,11.77,'p'],[8.0,12.61,'s'],[3.0,10.5,'p'],[3.0,10.07,'p'],[3.0,9.91,'p'],[15.0,14.61,'s'],[1.0,7.07,'p'],[5.0,11.05,'p'],[10.0,14.34,'p'],[12.0,11.56,'s'],[3.0,10.0,'p'],[5.0,6.7,'p'],[4.0,10.77,'p'],[8.0,14.08,'p'],[11.0,14.85,'s'],[9.0,16.4,'s'],[12.0,18.1,'s'],[14.0,16.83,'s'],[16.0,9.33,'p'],[17.0,18.21,'s'],[19.0,14.23,'p'],[16.0,20.58,'s'],[11.0,17.38,'s'],[9.0,13.05,'s'],[15.0,15.35,'p'],[14.0,15.03,'p'],[6.0,14.15,'s'],[6.0,15.19,'s'],[12.0,14.27,'s'],[27.0,28.04,'s'],[38.0,36.31,'s'],[34.0,28.06,'s'],[24.0,30.06,'s'],[24.0,39.67,'s'],[37.0,39.42,'s'],[34.0,38.31,'s'],[14.0,9.52,'s'],[13.0,17.57,'s'],[11.0,9.37,'s'],[13.0,16.42,'s'],[19.0,20.97,'s'],[10.0,11.96,'s']] }
            ]
        }
    };

    function initParityChart(container) {
        var id = container.dataset.chartId;
        var cfg = PARITY_DATA[id];
        if (!cfg) { return; }

        var lo = cfg.domain[0], hi = cfg.domain[1];
        var W = 460, H = 460;
        var plot = { left: 58, top: 34, w: W - 58 - 14, h: H - 34 - 44 };

        function xScale(v) { return plot.left + ((v - lo) / (hi - lo)) * plot.w; }
        function yScale(v) { return plot.top + (1 - (v - lo) / (hi - lo)) * plot.h; }

        /* Shared species legend, once for the whole group of panels */
        var legend = document.createElement('div');
        legend.className = 'viz-legend';
        Object.keys(cfg.species).forEach(function (key) {
            var sp = cfg.species[key];
            var item = document.createElement('span');
            item.className = 'viz-legend-item';
            var iconSvg = svgEl('svg', { class: 'viz-legend-icon', viewBox: '0 0 16 16' });
            var icon = svgEl('circle', { cx: 8, cy: 8, r: 5, fill: sp.color });
            iconSvg.appendChild(icon);
            var label = document.createElement('span');
            label.style.fontStyle = 'italic';
            label.textContent = sp.label;
            item.appendChild(iconSvg);
            item.appendChild(label);
            legend.appendChild(item);
        });
        container.appendChild(legend);

        var panelsWrap = document.createElement('div');
        panelsWrap.style.display = 'flex';
        panelsWrap.style.flexDirection = 'row';
        panelsWrap.style.flexWrap = 'wrap';
        panelsWrap.style.gap = '10px';

        cfg.panels.forEach(function (panel, panelIdx) {
            var chartWrap = document.createElement('div');
            chartWrap.className = 'viz-chart-wrap';
            chartWrap.style.flex = '1 1 0';
            chartWrap.style.minWidth = '220px';

            var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H, preserveAspectRatio: 'xMidYMid meet' });
            var clipId = 'parity-clip-' + id + '-' + panelIdx;
            var defs = svgEl('defs');
            var clipPath = svgEl('clipPath', { id: clipId });
            clipPath.appendChild(svgEl('rect', { x: plot.left, y: plot.top, width: plot.w, height: plot.h }));
            defs.appendChild(clipPath);
            svg.appendChild(defs);

            /* Gridlines + labels (shared x/y ticks, square domain) */
            cfg.ticks.forEach(function (t) {
                var y = yScale(t);
                svg.appendChild(svgEl('line', {
                    class: 'viz-gridline', x1: plot.left, x2: plot.left + plot.w, y1: y, y2: y
                }));
                var ylabel = svgEl('text', {
                    class: 'viz-axis-label', x: plot.left - 8, y: y + 4, 'text-anchor': 'end'
                });
                ylabel.textContent = t;
                svg.appendChild(ylabel);

                var x = xScale(t);
                var xlabel = svgEl('text', {
                    class: 'viz-axis-label', x: x, y: plot.top + plot.h + 16, 'text-anchor': 'middle'
                });
                xlabel.textContent = t;
                svg.appendChild(xlabel);
            });

            svg.appendChild(svgEl('line', {
                class: 'viz-axis-baseline', x1: plot.left, x2: plot.left + plot.w,
                y1: plot.top + plot.h, y2: plot.top + plot.h
            }));
            svg.appendChild(svgEl('line', {
                class: 'viz-axis-baseline', x1: plot.left, x2: plot.left,
                y1: plot.top, y2: plot.top + plot.h
            }));

            var xTitle = svgEl('text', {
                class: 'viz-axis-title', x: plot.left + plot.w / 2, y: H - 4, 'text-anchor': 'middle'
            });
            xTitle.textContent = 'Reference age (' + cfg.unit + ')';
            svg.appendChild(xTitle);

            var yTitle = svgEl('text', {
                class: 'viz-axis-title', x: 0, y: 0, 'text-anchor': 'middle',
                transform: 'translate(14,' + (plot.top + plot.h / 2) + ') rotate(-90)'
            });
            yTitle.textContent = 'Predicted age (' + cfg.unit + ')';
            svg.appendChild(yTitle);

            /* Panel title */
            var title = svgEl('text', {
                x: plot.left, y: 16, 'text-anchor': 'start'
            });
            title.setAttribute('style', 'fill:#ffffff; font-size:15px; font-weight:700; font-family:inherit;');
            title.textContent = panel.title;
            svg.appendChild(title);

            /* 1:1 reference line */
            svg.appendChild(svgEl('line', {
                class: 'viz-annotation-line', x1: xScale(lo), y1: yScale(lo), x2: xScale(hi), y2: yScale(hi),
                style: 'stroke:#7f7f7f; opacity:0.8;'
            }));

            /* Points, clipped to the plot rect (matches the paper's ylim(0, hi)) */
            var pointsGroup = svgEl('g', { 'clip-path': 'url(#' + clipId + ')' });
            var pointEls = [];
            panel.points.forEach(function (row) {
                var ref = row[0], pred = row[1], spKey = row[2];
                var sp = cfg.species[spKey];
                var px = xScale(ref), py = yScale(pred);
                var dot = svgEl('circle', { class: 'viz-marker', cx: px, cy: py, r: 3.4, fill: sp.color });
                dot.style.opacity = '0.8';
                pointsGroup.appendChild(dot);
                pointEls.push({ ref: ref, pred: pred, sp: sp, px: px, py: py, el: dot });
            });
            svg.appendChild(pointsGroup);

            /* Metrics readout, top-left */
            var metricsLines = [
                'RMSE = ' + panel.metrics.rmse + ' ' + cfg.unit,
                'MD = ' + panel.metrics.bias + ' ' + cfg.unit,
                'R² = ' + panel.metrics.r2
            ];
            metricsLines.forEach(function (line, i) {
                var t = svgEl('text', {
                    class: 'viz-axis-label', x: plot.left + 8, y: plot.top + 16 + i * 15, 'text-anchor': 'start'
                });
                t.textContent = line;
                svg.appendChild(t);
            });

            var tooltip = document.createElement('div');
            tooltip.className = 'viz-tooltip';
            var ttClass = document.createElement('div');
            ttClass.className = 'viz-tooltip-x';
            ttClass.style.fontStyle = 'italic';
            var rowRef = document.createElement('div');
            rowRef.className = 'viz-tooltip-row';
            var rowPred = document.createElement('div');
            rowPred.className = 'viz-tooltip-row';
            function makeRow(row, label) {
                var name = document.createElement('span');
                name.className = 'viz-tooltip-name';
                name.textContent = label;
                var value = document.createElement('span');
                value.className = 'viz-tooltip-value';
                row.appendChild(name);
                row.appendChild(value);
                return value;
            }
            var valRef = makeRow(rowRef, 'Reference');
            var valPred = makeRow(rowPred, 'Predicted');
            tooltip.appendChild(ttClass);
            tooltip.appendChild(rowRef);
            tooltip.appendChild(rowPred);

            chartWrap.appendChild(svg);
            chartWrap.appendChild(tooltip);

            var hitLayer = svgEl('rect', {
                class: 'viz-hit-layer', x: plot.left, y: plot.top, width: plot.w, height: plot.h
            });
            svg.appendChild(hitLayer);

            var activeDot = null;
            function showTooltip(evt) {
                var pt = svg.createSVGPoint();
                pt.x = evt.clientX; pt.y = evt.clientY;
                var loc = pt.matrixTransform(svg.getScreenCTM().inverse());

                var nearest = null, bestDist = Infinity;
                pointEls.forEach(function (item) {
                    var dx = item.px - loc.x, dy = item.py - loc.y;
                    var d = dx * dx + dy * dy;
                    if (d < bestDist) { bestDist = d; nearest = item; }
                });
                if (!nearest) { return; }

                if (activeDot) { activeDot.setAttribute('r', 3.4); }
                nearest.el.setAttribute('r', 5.5);
                activeDot = nearest.el;

                ttClass.textContent = nearest.sp.label;
                valRef.textContent = nearest.ref.toFixed(0) + ' ' + cfg.unit;
                valPred.textContent = nearest.pred.toFixed(1) + ' ' + cfg.unit;

                var svgRect = svg.getBoundingClientRect();
                var renderScale = svgRect.width / W;
                var left = nearest.px * renderScale + 12;
                if (left + 140 > svgRect.width) { left = nearest.px * renderScale - 152; }
                var top = nearest.py * renderScale - 60;
                tooltip.style.left = Math.max(0, left) + 'px';
                tooltip.style.top = Math.max(0, top) + 'px';
                tooltip.classList.add('is-visible');
            }
            function hideTooltip() {
                if (activeDot) { activeDot.setAttribute('r', 3.4); activeDot = null; }
                tooltip.classList.remove('is-visible');
            }
            hitLayer.addEventListener('pointermove', showTooltip);
            hitLayer.addEventListener('pointerleave', hideTooltip);

            panelsWrap.appendChild(chartWrap);
        });

        container.appendChild(panelsWrap);
    }

    /* One legend shared across every chart registered under groupId (see
       cfg.legendGroup) - hovering an item isolates that series on ALL of
       them at once, instead of each panel repeating its own legend. Call
       after the charts in the group have already been initialized. */
    window.renderSharedLegend = function (groupId, containerEl, seriesList) {
        if (!containerEl) { return; }
        containerEl.innerHTML = '';
        containerEl.className = 'viz-legend';

        function isolateAll(key) {
            (LEGEND_GROUPS[groupId] || []).forEach(function (inst) { inst.isolate(key); });
        }
        function clearAll() {
            (LEGEND_GROUPS[groupId] || []).forEach(function (inst) { inst.clear(); });
        }

        seriesList.forEach(function (s) {
            var item = document.createElement('span');
            item.className = 'viz-legend-item';
            item.tabIndex = 0;

            var iconSvg = svgEl('svg', { class: 'viz-legend-icon', viewBox: '0 0 16 16' });
            var icon = createShape(s.shape, 8, 8, 0.85);
            icon.setAttribute('fill', s.color);
            iconSvg.appendChild(icon);

            var label = document.createElement('span');
            label.textContent = s.label;

            item.appendChild(iconSvg);
            item.appendChild(label);
            item.addEventListener('mouseenter', function () { isolateAll(s.key); });
            item.addEventListener('mouseleave', clearAll);
            item.addEventListener('focus', function () { isolateAll(s.key); });
            item.addEventListener('blur', clearAll);

            containerEl.appendChild(item);
        });
    };

    window.initChartsIn = function (rootEl) {
        if (!rootEl) { return; }
        var charts = rootEl.querySelectorAll('.viz-chart[data-chart-id]:not([data-rendered])');
        charts.forEach(function (el) {
            el.dataset.rendered = '1';
            var chartId = el.dataset.chartId;
            if (CHART_DATA[chartId]) {
                initChart(el);
            } else if (SCATTER_DATA[chartId]) {
                initScatterChart(el);
            } else if (PARITY_DATA[chartId]) {
                initParityChart(el);
            }
        });
    };
})();
