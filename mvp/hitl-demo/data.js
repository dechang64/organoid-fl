/* =========================================================================
   类器官标注平台 · HITL MVP —— 数据与指标层
   纯前端、无依赖。样本图像由程序化合成（确定性种子），指标为真实几何计算。
   ========================================================================= */
(function (global) {
  'use strict';

  /* ---------------- 随机数 ---------------- */
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /* ---------------- 标签体系（与 utils/taxonomy.py 的 4 分类一致） ---------------- */
  const TAXONOMY = [
    { id: 'CYSTIC',  name: '囊性类器官', short: '囊性', color: '#4cc9f0', rgb: [76, 201, 240] },
    { id: 'SOLID',   name: '实性类器官', short: '实性', color: '#f472b6', rgb: [244, 114, 182] },
    { id: 'BUDDING', name: '出芽类器官', short: '出芽', color: '#ffb703', rgb: [255, 183, 3] },
    { id: 'DEBRIS',  name: '碎片/伪影', short: '碎片', color: '#94a3b8', rgb: [148, 163, 184] }
  ];
  const CLS = {}; TAXONOMY.forEach(c => CLS[c.id] = c);

  /* ---------------- 几何 ---------------- */
  function blob(cx, cy, R, rnd, squash) {
    const n = 26, pts = [];
    const a1 = 0.09 + rnd() * 0.10, a2 = 0.04 + rnd() * 0.07;
    const p1 = rnd() * 6.283, p2 = rnd() * 6.283;
    const sq = squash == null ? 0.90 + rnd() * 0.12 : squash;
    for (let i = 0; i < n; i++) {
      const t = (i / n) * Math.PI * 2;
      const r = R * (1 + a1 * Math.sin(3 * t + p1) + a2 * Math.sin(5 * t + p2));
      pts.push([cx + Math.cos(t) * r, cy + Math.sin(t) * r * sq]);
    }
    return pts;
  }
  function centroid(pts) {
    let x = 0, y = 0;
    for (const p of pts) { x += p[0]; y += p[1]; }
    return [x / pts.length, y / pts.length];
  }
  function area(pts) {
    let s = 0;
    for (let i = 0; i < pts.length; i++) {
      const a = pts[i], b = pts[(i + 1) % pts.length];
      s += a[0] * b[1] - b[0] * a[1];
    }
    return Math.abs(s) / 2;
  }
  function scaleAbout(pts, k, cx, cy) {
    return pts.map(p => [cx + (p[0] - cx) * k, cy + (p[1] - cy) * k]);
  }
  function shift(pts, dx, dy) {
    return pts.map(p => [p[0] + dx, p[1] + dy]);
  }
  function jitter(pts, rnd, amp) {
    const [cx, cy] = centroid(pts);
    return pts.map(p => {
      const dx = (rnd() - 0.5) * amp, dy = (rnd() - 0.5) * amp;
      const d = Math.hypot(p[0] - cx, p[1] - cy) || 1;
      const wr = 1 + ((rnd() - 0.5) * amp) / d; // 沿径向的轻微形变
      return [cx + (p[0] - cx) * wr + dx, cy + (p[1] - cy) * wr + dy];
    });
  }
  function convexHull(a, b) {
    const pts = a.concat(b).slice().sort((p, q) => p[0] - q[0] || p[1] - q[1]);
    const cross = (o, p, q) => (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0]);
    const lower = [], upper = [];
    for (const p of pts) {
      while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
      lower.push(p);
    }
    for (let i = pts.length - 1; i >= 0; i--) {
      const p = pts[i];
      while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
      upper.push(p);
    }
    lower.pop(); upper.pop();
    const hull = lower.concat(upper);
    return hull.length >= 3 ? hull : a;
  }

  /* ---------------- 样本合成 ---------------- */
  const STRATA = {
    easy:         { label: '容易',   weight: 1.0, color: '#37d39b' },
    borderline:   { label: '模糊边界', weight: 2.0, color: '#ffb703' },
    low_contrast: { label: '低对比',  weight: 2.0, color: '#a78bfa' },
    clustered:    { label: '粘连团簇', weight: 2.0, color: '#ff5d73' }
  };

  function buildSamples() {
    const spec = [
      { id: 'S-001', stratum: 'easy',         n: 5, R: 0.085, cls: ['CYSTIC', 'SOLID', 'CYSTIC', 'BUDDING', 'CYSTIC'] },
      { id: 'S-002', stratum: 'borderline',   n: 6, R: 0.075, cls: ['CYSTIC', 'CYSTIC', 'SOLID', 'BUDDING', 'CYSTIC', 'DEBRIS'] },
      { id: 'S-003', stratum: 'low_contrast', n: 4, R: 0.100, cls: ['SOLID', 'CYSTIC', 'CYSTIC', 'BUDDING'] },
      { id: 'S-004', stratum: 'clustered',    n: 7, R: 0.070, cls: ['CYSTIC', 'CYSTIC', 'CYSTIC', 'SOLID', 'BUDDING', 'CYSTIC', 'DEBRIS'] },
      { id: 'S-005', stratum: 'easy',         n: 4, R: 0.095, cls: ['BUDDING', 'CYSTIC', 'SOLID', 'CYSTIC'] }
    ];
    return spec.map((s, si) => {
      const rnd = mulberry32(1000 + si * 77);
      const polys = [];
      const positions = [];
      // 手工布点，保证不重叠过多（clustered 例外）
      const anchors = [
        [[0.28, 0.30], [0.68, 0.26], [0.30, 0.68], [0.72, 0.66], [0.50, 0.48], [0.15, 0.52], [0.85, 0.45]],
        [[0.26, 0.28], [0.70, 0.24], [0.28, 0.70], [0.72, 0.70], [0.50, 0.46], [0.86, 0.50], [0.14, 0.50]],
        [[0.30, 0.32], [0.70, 0.34], [0.34, 0.70], [0.66, 0.68], [0.50, 0.50], [0.16, 0.60], [0.84, 0.34]],
        [[0.30, 0.36], [0.42, 0.40], [0.62, 0.30], [0.66, 0.58], [0.30, 0.66], [0.52, 0.72], [0.80, 0.70]],
        [[0.28, 0.34], [0.70, 0.30], [0.32, 0.68], [0.70, 0.66], [0.50, 0.50], [0.16, 0.52], [0.84, 0.44]]
      ][si];
      for (let i = 0; i < s.n; i++) {
        const [ax, ay] = anchors[i % anchors.length];
        const jx = (rnd() - 0.5) * 0.03, jy = (rnd() - 0.5) * 0.03;
        const R = s.R * (0.78 + rnd() * 0.5);
        polys.push({ cls: s.cls[i], pts: blob(ax + jx, ay + jy, R, rnd) });
      }
      // 让 S-002 出现一对轻度粘连、S-004 出现重度粘连团簇
      if (s.stratum === 'borderline') {
        const c = centroid(polys[0].pts);
        polys[1].pts = blob(c[0] + s.R * 1.35, c[1] + s.R * 0.20, s.R * 0.95, mulberry32(7));
      }
      if (s.stratum === 'clustered') {
        const c = centroid(polys[0].pts);
        polys[1].pts = blob(c[0] + s.R * 1.25, c[1] + s.R * 0.30, s.R * 1.05, mulberry32(11));
        polys[2].pts = blob(c[0] + s.R * 0.70, c[1] - s.R * 0.95, s.R * 0.90, mulberry32(13));
      }
      return {
        id: s.id,
        stratum: s.stratum,
        imageSeed: 5000 + si * 31,
        polys: polys.map((p, i) => ({ id: s.id + '-T' + (i + 1), cls: p.cls, pts: p.pts }))
      };
    });
  }

  /* ---------------- 荧光图像渲染 ---------------- */
  function renderFluoro(cv, sample, opt) {
    opt = opt || {};
    const W = cv.width, H = cv.height;
    const g = cv.getContext('2d');
    const dim = sample.stratum === 'low_contrast' ? 0.52 : 1.0;

    const bg = g.createRadialGradient(W * 0.5, H * 0.46, W * 0.05, W * 0.5, H * 0.5, W * 0.78);
    bg.addColorStop(0, '#08161c');
    bg.addColorStop(0.55, '#050d12');
    bg.addColorStop(1, '#02060a');
    g.fillStyle = bg; g.fillRect(0, 0, W, H);

    // 本底纹理
    const rnd = mulberry32(sample.imageSeed);
    for (let i = 0; i < 1400; i++) {
      const x = rnd() * W, y = rnd() * H, r = rnd() * 1.6 + 0.3;
      g.fillStyle = 'rgba(120,180,210,' + (0.012 + rnd() * 0.05).toFixed(3) + ')';
      g.beginPath(); g.arc(x, y, r, 0, 6.283); g.fill();
    }

    // 实例
    for (const inst of sample.polys) {
      const pts = inst.pts.map(p => [p[0] * W, p[1] * H]);
      const c = centroid(pts);
      let R = 0;
      for (const p of pts) R = Math.max(R, Math.hypot(p[0] - c[0], p[1] - c[1]));

      g.save();
      g.beginPath();
      pts.forEach((p, i) => i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]));
      g.closePath();

      const grd = g.createRadialGradient(c[0], c[1], R * 0.10, c[0], c[1], R * 1.02);
      grd.addColorStop(0, 'rgba(70,150,190,' + (0.16 * dim).toFixed(3) + ')');
      grd.addColorStop(0.62, 'rgba(60,150,200,' + (0.30 * dim).toFixed(3) + ')');
      grd.addColorStop(0.93, 'rgba(150,235,255,' + (0.62 * dim).toFixed(3) + ')');
      grd.addColorStop(1, 'rgba(190,250,255,' + (0.80 * dim).toFixed(3) + ')');
      g.fillStyle = grd;
      g.shadowColor = 'rgba(90,210,255,' + (0.55 * dim).toFixed(3) + ')';
      g.shadowBlur = W * 0.020;
      g.fill();
      g.shadowBlur = 0;

      g.lineWidth = Math.max(1.2, W * 0.0032);
      g.strokeStyle = 'rgba(200,250,255,' + (0.85 * dim).toFixed(3) + ')';
      g.stroke();

      // 内部结构（类器官腔隙感）
      g.globalAlpha = 0.30 * dim;
      const iv = 3 + Math.floor(mulberry32(inst.id.length * 97 + Math.round(c[0]))() * 3);
      for (let k = 0; k < iv; k++) {
        const a = (k / iv) * 6.283 + 0.6;
        const rr = R * (0.18 + 0.34 * ((k * 37) % 10) / 10);
        g.fillStyle = 'rgba(10,40,60,.65)';
        g.beginPath();
        g.arc(c[0] + Math.cos(a) * rr, c[1] + Math.sin(a) * rr * 0.9, R * 0.14, 0, 6.283);
        g.fill();
      }
      g.globalAlpha = 1;
      g.restore();
    }

    // 暗角
    const vg = g.createRadialGradient(W * 0.5, H * 0.5, W * 0.30, W * 0.5, H * 0.5, W * 0.76);
    vg.addColorStop(0, 'rgba(0,0,0,0)');
    vg.addColorStop(1, 'rgba(0,0,0,.55)');
    g.fillStyle = vg; g.fillRect(0, 0, W, H);

    if (opt.hideWatermark !== true) {
      g.fillStyle = 'rgba(180,220,240,.42)';
      g.font = '11px "Cascadia Mono",monospace';
      g.fillText(sample.id + '  ·  ' + (sample.stratum), 10, H - 9);
    }
  }

  /* ---------------- 叠加层：轮廓 ---------------- */
  function drawInstances(cv, instances, opt) {
    opt = opt || {};
    const W = cv.width, H = cv.height;
    const g = cv.getContext('2d');
    const lw = opt.lw || Math.max(1.4, W * 0.0045);
    for (const inst of instances) {
      const c = CLS[inst.cls] || CLS.CYSTIC;
      const pts = inst.pts.map(p => [p[0] * W, p[1] * H]);
      g.beginPath();
      pts.forEach((p, i) => i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]));
      g.closePath();
      if (opt.fill !== false) {
        g.fillStyle = 'rgba(' + c.rgb.join(',') + ',' + (opt.fillAlpha == null ? 0.13 : opt.fillAlpha) + ')';
        g.fill();
      }
      g.lineWidth = lw;
      g.strokeStyle = c.color;
      if (opt.glow) { g.shadowColor = c.color; g.shadowBlur = 8; }
      g.stroke();
      g.shadowBlur = 0;
      if (opt.labels !== false) {
        const [cx, cy] = centroid(pts);
        g.fillStyle = 'rgba(4,10,16,.78)';
        const t = c.short;
        g.font = '600 ' + Math.round(W * 0.026) + 'px "Segoe UI","Microsoft YaHei",sans-serif';
        const tw = g.measureText(t).width + 8;
        g.fillRect(cx - tw / 2, cy - 9, tw, 17);
        g.fillStyle = c.color;
        g.fillText(t, cx - tw / 2 + 4, cy + 3);
      }
    }
  }

  /* ---------------- 掩膜与匹配 ---------------- */
  const MASK_N = 200;
  function polyMask(pts, N) {
    const cv = document.createElement('canvas');
    cv.width = cv.height = N;
    const g = cv.getContext('2d');
    g.fillStyle = '#fff';
    g.beginPath();
    pts.forEach((p, i) => i ? g.lineTo(p[0] * N, p[1] * N) : g.moveTo(p[0] * N, p[1] * N));
    g.closePath(); g.fill();
    const d = g.getImageData(0, 0, N, N).data;
    const m = new Uint8Array(N * N);
    for (let i = 0; i < N * N; i++) m[i] = d[i * 4 + 3] > 127 ? 1 : 0;
    return m;
  }
  function maskIoU(a, b) {
    let inter = 0, uni = 0;
    for (let i = 0; i < a.length; i++) {
      const x = a[i], y = b[i];
      if (x && y) inter++;
      if (x || y) uni++;
    }
    return uni ? inter / uni : 0;
  }
  function diceOf(a, b) {
    let inter = 0, sa = 0, sb = 0;
    for (let i = 0; i < a.length; i++) {
      if (a[i]) { sa++; if (b[i]) inter++; }
      if (b[i]) sb++;
    }
    return (sa + sb) ? (2 * inter) / (sa + sb) : 0;
  }

  function annotateMasks(instances) {
    return instances.map(i => ({ inst: i, mask: polyMask(i.pts, MASK_N) }));
  }

  /** 贪心匹配：按 IoU 降序连接，阈值默认 0.5 */
  function matchSets(A, B, thr) {
    thr = thr == null ? 0.5 : thr;
    const pairs = [];
    for (let i = 0; i < A.length; i++) {
      for (let j = 0; j < B.length; j++) {
        const iou = maskIoU(A[i].mask, B[j].mask);
        if (iou >= thr) pairs.push({ i, j, iou, dice: diceOf(A[i].mask, B[j].mask) });
      }
    }
    pairs.sort((p, q) => q.iou - p.iou);
    const usedA = new Set(), usedB = new Set(), out = [];
    for (const p of pairs) {
      if (usedA.has(p.i) || usedB.has(p.j)) continue;
      usedA.add(p.i); usedB.add(p.j);
      out.push(p);
    }
    return { pairs: out, unmatchedA: A.length - usedA.size, unmatchedB: B.length - usedB.size };
  }

  function boundaryPts(pts, n) {
    n = n || 56;
    const segs = [];
    let total = 0;
    for (let i = 0; i < pts.length; i++) {
      const a = pts[i], b = pts[(i + 1) % pts.length];
      const L = Math.hypot(b[0] - a[0], b[1] - a[1]);
      segs.push({ a, b, L }); total += L;
    }
    const step = total / n, out = [];
    let si = 0, acc = 0;
    for (let k = 0; k < n; k++) {
      const target = k * step;
      while (si < segs.length - 1 && acc + segs[si].L < target) { acc += segs[si].L; si++; }
      const t = segs[si].L ? (target - acc) / segs[si].L : 0;
      out.push([segs[si].a[0] + (segs[si].b[0] - segs[si].a[0]) * t,
                segs[si].a[1] + (segs[si].b[1] - segs[si].a[1]) * t]);
    }
    return out;
  }
  function hd95(a, b) {
    const A = boundaryPts(a), B = boundaryPts(b);
    const d1 = A.map(p => Math.min.apply(null, B.map(q => Math.hypot(p[0] - q[0], p[1] - q[1]))));
    const d2 = B.map(p => Math.min.apply(null, A.map(q => Math.hypot(p[0] - q[0], p[1] - q[1]))));
    const all = d1.concat(d2).sort((x, y) => x - y);
    return all[Math.min(all.length - 1, Math.floor(0.95 * all.length))] * 1000; // 归一化→‰
  }

  /* ---------------- 指标汇总 ---------------- */
  function compareRevisions(rA, rB) {
    const A = annotateMasks(rA.instances), B = annotateMasks(rB.instances);
    const m = matchSets(A, B, 0.5);
    const dice = m.pairs.map(p => p.dice);
    const hd = m.pairs.map(p => hd95(A[p.i].inst.pts, B[p.j].inst.pts));
    const tp = m.pairs.length;
    const fp = m.unmatchedB, fn = m.unmatchedA;
    const f1 = (2 * tp) / (2 * tp + fp + fn) || 0;
    return {
      meanDice: dice.length ? dice.reduce((a, b) => a + b, 0) / dice.length : 0,
      minDice: dice.length ? Math.min.apply(null, dice) : 0,
      meanHD95: hd.length ? hd.reduce((a, b) => a + b, 0) / hd.length : 0,
      detF1: f1,
      matched: tp, fp, fn
    };
  }

  /* ---- Krippendorff's α（名义，可缺测） ---- */
  function alphalNominal(units) {
    // units: [{labels: [l1,l2,l3]}]  null 表示该编码者在此单元缺测
    const values = [];
    const seen = {};
    for (const u of units) {
      for (const l of u.labels) {
        if (l == null) continue;
        if (!seen[l]) { seen[l] = values.length; values.push(l); }
      }
    }
    if (values.length === 0) return 1;
    const V = values.length;
    const o = [];
    for (let i = 0; i < V; i++) o.push(new Array(V).fill(0));
    let n = 0;
    for (const u of units) {
      const ls = u.labels.filter(l => l != null);
      const mu = ls.length;
      if (mu < 2) continue;
      n += mu;
      for (let a = 0; a < mu; a++) {
        for (let b = 0; b < mu; b++) {
          if (a === b) continue;
          o[seen[ls[a]]][seen[ls[b]]] += 1 / (mu - 1);
        }
      }
    }
    if (n === 0) return 1;
    let sumDiag = 0, sumAll = 0;
    const nv = new Array(V).fill(0);
    for (let i = 0; i < V; i++) {
      for (let j = 0; j < V; j++) {
        sumAll += o[i][j];
        if (i === j) sumDiag += o[i][j];
        nv[i] += o[i][j];
      }
    }
    const Do = sumAll - sumDiag;
    let sumSq = 0;
    for (let i = 0; i < V; i++) sumSq += nv[i] * nv[i];
    const De = (n * n - sumSq) / (n - 1);
    if (De <= 1e-12) return 1;
    return 1 - Do / De;
  }

  /** 用匹配关系建立跨编码者的"同一实例"单元（并查集） */
  function buildUnits(revisions) {
    const nodes = [];
    const index = {};
    revisions.forEach((r, ci) => {
      r.instances.forEach((inst, ii) => {
        const key = ci + '#' + ii;
        index[key] = nodes.length;
        nodes.push({ coder: ci, inst: inst, parent: nodes.length });
      });
    });
    function find(x) { while (nodes[x].parent !== x) { nodes[x].parent = nodes[nodes[x].parent].parent; x = nodes[x].parent; } return x; }
    function union(a, b) { const ra = find(a), rb = find(b); if (ra !== rb) nodes[rb].parent = ra; }

    for (let i = 0; i < revisions.length; i++) {
      for (let j = i + 1; j < revisions.length; j++) {
        const A = annotateMasks(revisions[i].instances);
        const B = annotateMasks(revisions[j].instances);
        const m = matchSets(A, B, 0.5);
        for (const p of m.pairs) union(index[i + '#' + p.i], index[j + '#' + p.j]);
      }
    }
    const comps = {};
    nodes.forEach((n, i) => {
      const r = find(i);
      (comps[r] = comps[r] || []).push(n);
    });
    return Object.keys(comps).map(k => {
      const group = comps[k];
      const labels = revisions.map(() => null);
      revisions.forEach((_, ci) => {
        const mine = group.filter(n => n.coder === ci);
        if (!mine.length) return;
        const cnt = {};
        mine.forEach(n => cnt[n.inst.cls] = (cnt[n.inst.cls] || 0) + 1);
        labels[ci] = Object.keys(cnt).sort((a, b) => cnt[b] - cnt[a])[0];
      });
      return { labels: labels };
    });
  }

  function computeAgreement(revisions) {
    const pairs = [[0, 1], [0, 2], [1, 2]];
    const per = pairs.map(p => compareRevisions(revisions[p[0]], revisions[p[1]]));
    const meanDice = per.reduce((a, b) => a + b.meanDice, 0) / per.length;
    const meanF1 = per.reduce((a, b) => a + b.detF1, 0) / per.length;
    const meanHD = per.reduce((a, b) => a + b.meanHD95, 0) / per.length;
    const units = buildUnits(revisions);
    const alpha = alphalNominal(units);
    // 类别层面的完全一致比例
    let full = 0, counted = 0;
    units.forEach(u => {
      const ls = u.labels.filter(l => l != null);
      if (ls.length < 2) return;
      counted++;
      if (ls.every(l => l === ls[0])) full++;
    });
    const labelAgree = counted ? full / counted : 1;
    const controversy = Math.max(0, Math.min(1,
      (1 - meanDice) * 1.25 * 0.62 + (1 - labelAgree) * 0.38));
    return {
      pairs: per, meanDice, meanF1, meanHD95: meanHD, alpha, labelAgree,
      units: units.length, controversy
    };
  }

  /* ---------------- 预标注与模拟专家 ---------------- */
  function modelPreannotate(sample) {
    const rnd = mulberry32(sample.imageSeed + 991);
    let insts = sample.polys.map((p, i) => ({
      id: 'M-' + (i + 1), cls: p.cls,
      pts: jitter(scaleAbout(p.pts, 1.032, centroid(p.pts)[0], centroid(p.pts)[1]), rnd, 0.006)
    }));
    insts = insts.map(x => ({ id: x.id, cls: x.cls, pts: shift(x.pts, 0.0035, -0.0028) }));

    if (sample.stratum === 'borderline' || sample.stratum === 'clustered') {
      // 模型把粘连的一对合成一个（真实模型最常见的失败模式之一）
      const a = insts[0].pts, b = insts[1].pts;
      const hull = convexHull(a, b);
      insts = [{ id: 'M-1', cls: insts[0].cls, pts: hull }].concat(insts.slice(2));
    }
    if (sample.stratum === 'low_contrast') {
      // 低对比下漏检最小实例
      let mi = 0, ma = 1e9;
      insts.forEach((x, i) => { const a = area(x.pts); if (a < ma) { ma = a; mi = i; } });
      insts = insts.filter((_, i) => i !== mi);
    }
    return { coder: 'MODEL', engine: 'organoid-det-v0.9', instances: insts };
  }

  function scriptedExpert(sample, coder) {
    const seedAdd = coder === 'B' ? 41 : 83;
    const rnd = mulberry32(sample.imageSeed + seedAdd);
    const amp = coder === 'B' ? 0.0075 : 0.0145;
    let insts = sample.polys.map((p, i) => ({
      id: coder + '-' + (i + 1), cls: p.cls, pts: jitter(p.pts, rnd, amp)
    }));
    if (coder === 'C' && (sample.stratum === 'clustered' || sample.stratum === 'borderline')) {
      insts = [{ id: 'C-1', cls: insts[0].cls, pts: convexHull(insts[0].pts, insts[1].pts) }]
        .concat(insts.slice(2));
    }
    if (coder === 'C' && sample.stratum === 'low_contrast') {
      // 低对比时把碎片标成实性
      insts = insts.map(x => x.cls === 'DEBRIS' ? { id: x.id, cls: 'SOLID', pts: x.pts } : x);
    }
    return { coder: coder, instances: insts };
  }

  /* ---------------- 哈希（演示用，非密码学） ---------------- */
  function q4(x) { return Math.round(x * 1e4) / 1e4; }
  function canon(v) {
    if (v === null || v === undefined) return 'null';
    if (typeof v === 'number') return Number.isFinite(v) ? String(q4(v)) : 'null';
    if (typeof v === 'string') return JSON.stringify(v);
    if (typeof v === 'boolean') return v ? 'true' : 'false';
    if (Array.isArray(v)) return '[' + v.map(canon).join(',') + ']';
    const keys = Object.keys(v).sort();
    return '{' + keys.map(k => JSON.stringify(k) + ':' + canon(v[k])).join(',') + '}';
  }
  function fnv1a(str, basis) {
    let h = basis >>> 0;
    for (let i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 16777619) >>> 0;
    }
    return ('00000000' + h.toString(16)).slice(-8);
  }
  function hash64(str) {
    return fnv1a(str, 0x811c9dc5) + fnv1a(str + '#s', 0x01000193);
  }

  function newChain() {
    return { entries: [], prev: '0000000000000000' };
  }
  function pushAudit(chain, actor, event, payload) {
    const body = { seq: chain.entries.length + 1, at: payload._at || nowISO(), actor: actor, event: event, payload: payload };
    delete body.payload._at;
    const preimage = canon({ prev: chain.prev, seq: body.seq, at: body.at, actor: body.actor, event: body.event, payload: body.payload });
    body.prev = chain.prev;
    body.hash = hash64(preimage);
    chain.entries.push(body);
    chain.prev = body.hash;
    return body;
  }
  function verifyChain(chain) {
    let prev = '0000000000000000';
    for (let i = 0; i < chain.entries.length; i++) {
      const e = chain.entries[i];
      const preimage = canon({ prev: prev, seq: e.seq, at: e.at, actor: e.actor, event: e.event, payload: e.payload });
      const expect = hash64(preimage);
      if (e.prev !== prev || e.hash !== expect) return { ok: false, badAt: i + 1 };
      prev = e.hash;
    }
    return { ok: true, badAt: 0 };
  }

  function nowISO() {
    const d = new Date();
    const p = n => ('0' + n).slice(-2);
    return d.getFullYear() + '-' + p(d.getMonth() + 1) + '-' + p(d.getDate()) + 'T' +
           p(d.getHours()) + ':' + p(d.getMinutes()) + ':' + p(d.getSeconds()) + '+08:00';
  }
  function fmtDur(ms) {
    const s = Math.max(0, Math.round(ms / 1000));
    return Math.floor(s / 60) + ':' + ('0' + (s % 60)).slice(-2);
  }

  global.ORG = {
    TAXONOMY, CLS, STRATA,
    mulberry32, blob, centroid, area, convexHull,
    buildSamples, renderFluoro, drawInstances,
    modelPreannotate, scriptedExpert,
    polyMask, maskIoU, diceOf, matchSets, compareRevisions, computeAgreement,
    alphalNominal, buildUnits, hd95,
    canon, hash64, newChain, pushAudit, verifyChain, nowISO, fmtDur, q4
  };
})(window);
