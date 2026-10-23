/* =========================================================================
   类器官标注平台 · HITL MVP —— 应用层
   流程：S0 入湖 → S1 预标注 → S2 盲读 → S3 揭示 → S4 仲裁 → 条件复核 → S5 定稿
   ========================================================================= */
(function () {
  'use strict';

  const O = window.ORG;
  const app = document.getElementById('app');
  const STEPS = [
    { id: 'S0', name: '入湖预检' },
    { id: 'S1', name: '模型预标注' },
    { id: 'S2', name: '专家盲读' },
    { id: 'S3', name: '揭示与分歧' },
    { id: 'S4', name: '首席仲裁' },
    { id: 'S5', name: '定稿入库' }
  ];
  const REASONS = ['边界判定', '类别判定', '实例拆分', '伪影', '协议未覆盖'];
  const DICE_MIN = 0.75, ALPHA_MIN = 0.65, CONTROVERSY_TRIGGER = 0.72;

  const S = {
    samples: O.buildSamples(),
    idx: 0,
    role: 'expertA',
    template: 'research',
    view: 'overview',
    model: {},
    rev: {},
    revealed: {},
    adj: {},
    cosign: {},
    exported: {},
    chain: O.newChain(),
    draft: [],
    activeCls: 'CYSTIC',
    mode: 'draw',
    pending: null,
    selected: null,
    drag: null,
    startedAt: 0,
    firstTouchAt: 0,
    timerH: 0,
    baseCache: {}
  };

  /* ================= 初始化 ================= */
  function init() {
    S.samples.forEach((s, i) => {
      S.model[i] = O.modelPreannotate(s);
      S.rev[i] = {
        B: { coder: 'B', instances: O.scriptedExpert(s, 'B').instances, meta: { blindMs: 214000 + i * 17000, submittedAt: '—' } },
        C: { coder: 'C', instances: O.scriptedExpert(s, 'C').instances, meta: { blindMs: 268000 + i * 9000, submittedAt: '—' } }
      };
      O.pushAudit(S.chain, '系统', 'TASK_PUBLISHED', { task_id: 'T-' + s.id, sample: s.id, stratum: s.stratum, redundancy_n: 3 });
      O.pushAudit(S.chain, 'model:organoid-det-v0.9', 'PREANNOTATION_PRODUCED', { sample: s.id, instances: S.model[i].instances.length });
      O.pushAudit(S.chain, 'expert:B', 'REVISION_SUBMITTED', { sample: s.id, coder: 'B', instances: S.rev[i].B.instances.length });
      O.pushAudit(S.chain, 'expert:C', 'REVISION_SUBMITTED', { sample: s.id, coder: 'C', instances: S.rev[i].C.instances.length });
    });
    render();
  }

  /* ================= 状态推导 ================= */
  function stateOf(i) {
    if (S.exported[i]) return { code: 'IN_DATASET', label: '已入库', cls: 'ok' };
    const a = S.rev[i].A, adj = S.adj[i], cs = S.cosign[i];
    if (!a) return { code: 'PUBLISHED', label: '待盲读', cls: 'info' };
    if (!adj) {
      if (!S.revealed[i]) return { code: 'AWAITING_REVEAL', label: '待揭示', cls: 'info' };
      return { code: 'AWAITING_ADJUDICATION', label: '待仲裁', cls: 'warn' };
    }
    if (adj.cosignRequired && (!cs || !cs.verdict)) return { code: 'AWAITING_COSIGN', label: '待复核', cls: 'warn' };
    return { code: 'ADJUDICATED', label: '已仲裁', cls: 'ok' };
  }
  function stepIndex() {
    const i = S.idx, st = stateOf(i);
    if (S.view === 'blind') return 2;
    if (S.view === 'reveal') return 3;
    if (S.view === 'adjudicate') return 4;
    if (S.view === 'cosign') return 4;
    if (S.view === 'final') return 5;
    switch (st.code) {
      case 'PUBLISHED': return 2;
      case 'AWAITING_REVEAL': return 3;
      case 'AWAITING_ADJUDICATION': return 4;
      case 'AWAITING_COSIGN': return 4;
      case 'ADJUDICATED': return 5;
      case 'IN_DATASET': return 5;
      default: return 1;
    }
  }
  function nextAction(i) {
    const st = stateOf(i).code;
    if (st === 'PUBLISHED') return { view: 'blind', label: '开始盲读' };
    if (st === 'AWAITING_REVEAL') return { view: 'reveal', label: '执行揭示' };
    if (st === 'AWAITING_ADJUDICATION') return { view: 'adjudicate', label: '进入仲裁' };
    if (st === 'AWAITING_COSIGN') return { view: 'cosign', label: '条件复核' };
    if (st === 'ADJUDICATED') return { view: 'final', label: '定稿与导出' };
    return { view: 'final', label: '查看数据集卡片' };
  }

  /* ================= 通用渲染工具 ================= */
  function h(html) { return html; }
  function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
  function chip(text, cls) { return '<span class="chip ' + (cls || '') + '">' + esc(text) + '</span>'; }
  function pct(x, d) { return (x * 100).toFixed(d == null ? 1 : d) + '%'; }
  function fx(x, d) { return Number(x).toFixed(d == null ? 3 : d); }
  function toast(msg, kind) {
    const t = document.getElementById('toast');
    t.textContent = msg;
    t.className = 'show ' + (kind || '');
    clearTimeout(toast._h);
    toast._h = setTimeout(() => { t.className = ''; }, 2600);
  }
  function baseCanvas(i) {
    if (S.baseCache[i]) return S.baseCache[i];
    const cv = document.createElement('canvas');
    cv.width = cv.height = 720;
    O.renderFluoro(cv, S.samples[i], {});
    S.baseCache[i] = cv;
    return cv;
  }
  function canvasWithInstances(i, instances, size, opt) {
    const cv = document.createElement('canvas');
    cv.width = cv.height = size;
    const g = cv.getContext('2d');
    g.drawImage(baseCanvas(i), 0, 0, size, size);
    O.drawInstances(cv, instances, opt || {});
    return cv;
  }

  /* ================= 渲染主循环 ================= */
  function render() {
    renderSteps();
    const i = S.idx;
    let html = '';
    if (S.view === 'overview') html = viewOverview();
    else if (S.view === 'blind') html = viewBlind(i);
    else if (S.view === 'reveal') html = viewReveal(i);
    else if (S.view === 'adjudicate') html = viewAdjudicate(i);
    else if (S.view === 'cosign') html = viewCosign(i);
    else if (S.view === 'final') html = viewFinal(i);
    app.innerHTML = html;
    bind();
    if (S.view === 'blind') startTimer();
    else stopTimer();
    if (S.view === 'reveal' && S.revealed[i]) paintReveal(i);
    if (S.view === 'blind') paintBlind(i);
    if (S.view === 'adjudicate' || S.view === 'cosign') paintThumbs(i);
  }
  function renderSteps() {
    const cur = stepIndex();
    document.getElementById('steps').innerHTML = STEPS.map((s, k) => {
      const cls = k < cur ? 'done' : (k === cur ? 'active' : '');
      return '<div class="step ' + cls + '"><i>' + (k + 1) + '</i>' + s.name + '</div>';
    }).join('');
  }
  function roleName(r) {
    return { expertA: '领域专家 A', expertB: '领域专家 B', expertC: '领域专家 C', chief: '首席专家', cosigner: '复核专家', owner: '项目负责人' }[r] || r;
  }

  /* ================= 视图 0：总览 ================= */
  function viewOverview() {
    const s = S.samples[S.idx];
    const rows = S.samples.map((sm, i) => {
      const st = stateOf(i);
      const na = nextAction(i);
      const str = O.STRATA[sm.stratum];
      return '<tr class="' + (i === S.idx ? 'sel' : '') + '">' +
        '<td class="mono">' + sm.id + '</td>' +
        '<td><span class="chip" style="color:' + str.color + ';border-color:' + str.color + '55;background:' + str.color + '12">' + str.label + '</span></td>' +
        '<td class="num">' + fx(str.weight, 1) + '×</td>' +
        '<td class="num">' + (S.rev[i].A ? S.rev[i].A.instances.length : '—') + '</td>' +
        '<td>' + chip(st.label, st.cls) + '</td>' +
        '<td class="r"><button class="btn sm" data-go="' + na.view + '" data-i="' + i + '">' + na.label + '</button></td>' +
        '</tr>';
    }).join('');

    const iso = isolationStats();

    return '<h2 class="viewtitle">任务总览' +
      '<small>T-' + s.id + ' · 科研生产模板 · 冗余 N=3 · 协议 organoid-4class v1.2</small></h2>' +
      '<div class="sec"><div class="track">' +
      STEPS.map((st, k) => {
        const cur = stepIndex();
        const cls = k < cur ? 'done' : (k === cur ? 'cur' : 'todo');
        return '<div class="tnode ' + cls + '"><div class="sid">' + st.id + '</div><div class="nm">' + st.name + '</div></div>';
      }).join('') +
      '</div></div>' +

      '<div class="grid2">' +
      '<div>' +
      '<div class="panel"><div class="panel-hd"><h3>样本队列</h3><span class="sub">5 例 · 已按困难层分层入湖</span><div class="spacer"></div>' +
      '<span class="chip info">当前角色：' + roleName(S.role) + '</span></div>' +
      '<div class="panel-bd tight"><table class="tbl"><thead><tr>' +
      '<th>样本</th><th>困难层</th><th class="r">抽样权重</th><th class="r">修订数</th><th>状态</th><th class="r">操作</th>' +
      '</tr></thead><tbody>' + rows + '</tbody></table></div></div>' +

      '<div class="panel" style="margin-top:14px"><div class="panel-hd"><h3>本演示能验证什么</h3><span class="sub">每一步都对应 PRD 的一条验收项</span></div>' +
      '<div class="panel-bd"><table class="tbl"><thead><tr><th>验收项</th><th>本演示中的体现</th></tr></thead><tbody>' +
      '<tr><td class="mono">P-01</td><td>盲读台不加载任何模型字段；揭示前 CSV/接口均无模型排序</td></tr>' +
      '<tr><td class="mono">P-02</td><td>第三份修订提交即触发分歧分析，指标实时计算</td></tr>' +
      '<tr><td class="mono">P-03</td><td>仲裁必须选理由码，否则提交按钮不可用</td></tr>' +
      '<tr><td class="mono">P-04</td><td>模型排序仅作参考，无预选、无一键采纳</td></tr>' +
      '<tr><td class="mono">P-09</td><td>条件复核：未锁定独立意见前看不到首席结论</td></tr>' +
      '<tr><td class="mono">P-10</td><td>隔离集例数、分层权重可随时报出</td></tr>' +
      '<tr><td class="mono">P-11</td><td>教学模板产物在导出闸门被硬拦截</td></tr>' +
      '</tbody></table>' +
      '<p class="note" style="margin-top:10px">说明：<b>专家 B / C 的修订是脚本化模拟</b>（在合成样本真值上加入不同幅度的形变，C 额外在粘连层合并实例）。指标由真实几何计算得出，不是写死的数字。</p>' +
      '</div></div>' +
      '</div>' +

      '<div>' +
      '<div class="panel"><div class="panel-hd"><h3>隔离留出集</h3><span class="sub">D-08 · 20% 且不少于 150 例</span></div>' +
      '<div class="panel-bd">' +
      '<div class="mgrid">' +
      '<div class="mcard"><div class="k">样本池</div><div class="v num">' + iso.pool + '</div><div class="t">已入湖</div></div>' +
      '<div class="mcard pass"><div class="k">隔离集</div><div class="v num">' + iso.total + '</div><div class="t">' + pct(iso.ratio, 1) + ' · 下界 150</div></div>' +
      '</div>' +
      '<div class="hr"></div>' +
      '<table class="tbl"><thead><tr><th>层</th><th class="r">权重</th><th class="r">例数</th><th class="r">占比</th></tr></thead><tbody>' +
      iso.rows.map(r => '<tr><td>' + r.label + '</td><td class="r num">' + fx(r.weight, 1) + '×</td><td class="r num">' + r.n + '</td><td class="r num">' + pct(r.n / iso.total, 1) + '</td></tr>').join('') +
      '</tbody></table>' +
      '<p class="note" style="margin-top:9px">分层在 <b>S0 入湖时打标</b>，缺失即入湖失败；抽样后冻结 id 列表、权重与规则版本。隔离样本全程不预标注，且<b>身份对标注员不可见</b>（否则触发霍桑效应，价差被抹平）。</p>' +
      '</div></div>' +

      '<div class="panel" style="margin-top:14px"><div class="panel-hd"><h3>模板与导出闸门</h3></div>' +
      '<div class="panel-bd">' +
      '<div class="row" style="margin-bottom:10px"><label class="tiny dim">流程模板</label>' +
      '<select id="templateSel"><option value="research"' + (S.template === 'research' ? ' selected' : '') + '>科研生产（四级重流程）</option>' +
      '<option value="teaching"' + (S.template === 'teaching' ? ' selected' : '') + '>教学实训（灰度）</option></select></div>' +
      (S.template === 'teaching'
        ? '<div class="banner warn"><div class="ic">!</div><div><b>当前为教学实训模板</b><br>产物在导出闸门被硬拦截，不进入训练集导出白名单。学生标注的分布偏差会直接教坏模型，这一条靠代码不靠约定。</div></div>'
        : '<p class="note">科研生产模板：定稿产物经门槛校验后进入数据集。切到「教学实训」可看到导出拦截。</p>') +
      '</div></div>' +

      '<div class="panel" style="margin-top:14px"><div class="panel-hd"><h3>审计链</h3><span class="sub">organoid-audit-v1 口径</span><div class="spacer"></div>' +
      '<button class="btn sm" id="btnVerify">校验链</button></div>' +
      '<div class="panel-bd tight">' + auditTable() + '</div></div>' +
      '</div></div>';
  }

  function isolationStats() {
    const pool = 1240;
    const total = 248;
    const rows = [
      { label: '容易', weight: 1.0, n: 62 },
      { label: '模糊边界', weight: 2.0, n: 66 },
      { label: '低对比', weight: 2.0, n: 60 },
      { label: '粘连团簇', weight: 2.0, n: 60 }
    ];
    return { pool: pool, total: total, ratio: total / pool, rows: rows };
  }

  function auditTable() {
    const es = S.chain.entries.slice(-8).reverse();
    return '<div class="chain">' + es.map(e =>
      '<div class="chainrow"><span class="sq">#' + e.seq + '</span>' +
      '<span class="mono dim">' + e.at.slice(11, 19) + '</span>' +
      '<span class="ev">' + esc(e.event) + '</span>' +
      '<span class="dim">' + esc(e.actor) + '</span>' +
      '<span class="hh">' + e.hash.slice(0, 8) + '</span></div>').join('') + '</div>' +
      '<p class="note" style="margin-top:8px">共 ' + S.chain.entries.length + ' 条，头哈希 <span class="mono">' + S.chain.prev.slice(0, 12) + '</span></p>';
  }

  /* ================= 视图 1：盲读台 ================= */
  function viewBlind(i) {
    const sm = S.samples[i];
    const str = O.STRATA[sm.stratum];
    const clsRows = O.TAXONOMY.map(c => {
      const n = S.draft.filter(d => d.cls === c.id).length;
      return '<div class="classrow ' + (S.activeCls === c.id ? 'on' : '') + '" data-cls="' + c.id + '">' +
        '<span class="dot" style="background:' + c.color + '"></span>' + c.name +
        '<span class="cnt">' + (n || '') + '</span></div>';
    }).join('');
    const list = S.draft.length ? S.draft.map((d, k) =>
      '<div class="classrow ' + (S.selected === k ? 'on' : '') + '" data-inst="' + k + '">' +
      '<span class="dot" style="background:' + O.CLS[d.cls].color + '"></span>' +
      O.CLS[d.cls].short + ' · #' + (k + 1) +
      '<span class="cnt">' + Math.round(O.area(d.pts) * 10000) / 100 + '%²</span></div>').join('')
      : '<p class="note">还没有轮廓。选择类别后在左侧图像上点击落点，双击闭合。</p>';

    return '<h2 class="viewtitle">S2 盲读台 · ' + sm.id +
      '<small>盲读态：界面不加载任何模型输出与他人修订。提交即锁定，如需修正须新开修订号。</small></h2>' +
      '<div class="grid2"><div>' +
      '<div class="panel"><div class="panel-hd"><h3>标注画布</h3>' +
      '<span class="chip ' + (str.weight > 1 ? 'warn' : 'ok') + '">' + str.label + ' · 权重 ' + fx(str.weight, 1) + '×</span>' +
      '<div class="spacer"></div><span class="chip" id="timer">00:00</span></div>' +
      '<div class="panel-bd">' +
      '<div class="tools" style="margin-bottom:9px">' +
      '<button class="tool ' + (S.mode === 'draw' ? 'on' : '') + '" data-mode="draw">画轮廓</button>' +
      '<button class="tool ' + (S.mode === 'select' ? 'on' : '') + '" data-mode="select">选择 / 拖动</button>' +
      '<button class="tool" id="btnDel">删除选中</button>' +
      '<button class="tool" id="btnClear">清空</button>' +
      '<div class="spacer"></div>' +
      '<button class="tool" id="btnScript">快速填入示例（演示用）</button>' +
      '</div>' +
      '<div class="stagewrap hidden-model"><canvas class="stage" id="cv" width="720" height="720"></canvas></div>' +
      '<p class="note" style="margin-top:8px">左键落点，<b>双击闭合</b>，右键撤销上一点，Esc 放弃当前轮廓。目标：每个可见类器官各画一个闭合轮廓，并按类别着色。</p>' +
      '</div></div>' +
      '</div><div>' +
      '<div class="panel"><div class="panel-hd"><h3>类别</h3><span class="sub">organoid-4class</span></div>' +
      '<div class="panel-bd tight"><div class="classlist">' + clsRows + '</div></div></div>' +
      '<div class="panel" style="margin-top:14px"><div class="panel-hd"><h3>本次修订</h3><span class="sub">' + S.draft.length + ' 个实例</span></div>' +
      '<div class="panel-bd tight"><div class="classlist">' + list + '</div></div></div>' +
      '<div class="panel" style="margin-top:14px"><div class="panel-bd">' +
      '<div class="banner" style="margin-bottom:10px"><div class="ic">i</div><div>' +
      '盲读隔离不是前端隐藏，是<b>服务端投影</b>：此视图的响应体里没有 <span class="mono">model_*</span> 字段，不是把它们置空。</div></div>' +
      '<button class="btn primary" id="btnSubmit" style="width:100%;justify-content:center">提交并锁定修订</button>' +
      '<p class="note" style="margin-top:8px">提交后本页只读，修订内容按 <span class="mono">organoid-audit-v1</span> 量化入哈希（坐标按 1e-4 定点，浮点不进原像）。</p>' +
      '</div></div>' +
      '</div></div>';
  }

  function paintBlind(i) {
    const cv = document.getElementById('cv');
    if (!cv) return;
    const g = cv.getContext('2d');
    const W = cv.width;
    g.drawImage(baseCanvas(i), 0, 0, W, W);
    S.draft.forEach((d, k) => {
      const c = O.CLS[d.cls];
      const pts = d.pts.map(p => [p[0] * W, p[1] * W]);
      g.beginPath();
      pts.forEach((p, q) => q ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]));
      g.closePath();
      g.fillStyle = 'rgba(' + c.rgb.join(',') + ',0.13)';
      g.fill();
      g.lineWidth = S.selected === k ? 4 : 2.2;
      g.strokeStyle = c.color;
      g.stroke();
      if (S.selected === k) {
        pts.forEach(p => { g.beginPath(); g.arc(p[0], p[1], 4, 0, 6.283); g.fillStyle = '#fff'; g.fill(); });
      }
    });
    if (S.pending && S.pending.pts.length) {
      const pts = S.pending.pts.map(p => [p[0] * W, p[1] * W]);
      g.beginPath();
      pts.forEach((p, q) => q ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]));
      if (S.pending.cursor) g.lineTo(S.pending.cursor[0] * W, S.pending.cursor[1] * W);
      g.setLineDash([6, 4]);
      g.lineWidth = 2;
      g.strokeStyle = '#fff';
      g.stroke();
      g.setLineDash([]);
      pts.forEach(p => { g.beginPath(); g.arc(p[0], p[1], 3.5, 0, 6.283); g.fillStyle = '#fff'; g.fill(); });
    }
  }

  /* ================= 视图 2：揭示与分歧 ================= */
  function viewReveal(i) {
    const sm = S.samples[i];
    if (!S.revealed[i]) {
      return '<h2 class="viewtitle">S3 揭示 · ' + sm.id +
        '<small>三份盲读已齐（A/B/C），尚未揭示。此前的任何界面都看不到模型输出。</small></h2>' +
        '<div class="panel"><div class="panel-bd">' +
        '<div class="banner ok" style="margin-bottom:12px"><div class="ic">✓</div><div>' +
        '<b>3/3 修订已提交并锁定</b><br>A：' + S.rev[i].A.instances.length + ' 实例，用时 ' + O.fmtDur(S.rev[i].A.meta.blindMs) +
        ' · B：' + S.rev[i].B.instances.length + ' 实例 · C：' + S.rev[i].C.instances.length + ' 实例</div></div>' +
        '<p class="note" style="margin-bottom:12px">揭示是一次显式动作，不是进入页面就自动发生。揭示后系统计算分歧指标，并把模型预标注与排序一并呈现给首席——<b>对领域专家仍然不可见</b>。</p>' +
        '<button class="btn primary" id="btnReveal">执行揭示并计算分歧</button>' +
        '</div></div>';
    }
    const ag = O.computeAgreement([S.rev[i].A, S.rev[i].B, S.rev[i].C]);
    const vsModel = O.compareRevisions(S.rev[i].A, { instances: S.model[i].instances });
    const suspect = vsModel.meanDice > 0.95;

    return '<h2 class="viewtitle">S3 揭示与分歧 · ' + sm.id +
      '<small>分歧不是缺陷，是下一轮要学的地方。指标由三份修订真实计算（IoU≥0.5 匹配）。</small></h2>' +
      (suspect ? '<div class="banner warn" style="margin-bottom:12px"><div class="ic">!</div><div><b>疑似未经独立判断</b>：专家 A 的修订与模型预标注 Dice 达 ' + fx(vsModel.meanDice) + '，接近直接复制。该样本建议列入偏误监控。</div></div>' : '') +
      '<div class="grid2"><div>' +
      '<div class="panel"><div class="panel-hd"><h3>三份修订</h3><span class="sub">同一视野 · 各自轮廓</span></div>' +
      '<div class="panel-bd"><div class="grid3" id="trio"></div></div></div>' +
      '<div class="panel" style="margin-top:14px"><div class="panel-hd"><h3>分歧图</h3>' +
      '<span class="sub">按像素统计三份修订的一致情况</span><div class="spacer"></div>' +
      '<span class="chip ok">三方一致</span><span class="chip warn">两方一致</span><span class="chip bad">仅一方</span><span class="chip violet">类别冲突</span></div>' +
      '<div class="panel-bd"><div class="stagewrap"><canvas id="cvDis" width="360" height="360" style="display:block;width:100%;height:auto;image-rendering:auto"></canvas></div></div></div>' +
      '</div><div>' +
      '<div class="panel"><div class="panel-hd"><h3>一致性指标</h3></div><div class="panel-bd">' +
      '<div class="mgrid">' +
      mcard('平均 Dice', fx(ag.meanDice), ag.meanDice >= DICE_MIN ? 'pass' : 'fail', '门槛 ' + DICE_MIN) +
      mcard('Krippendorff α', fx(ag.alpha), ag.alpha >= ALPHA_MIN ? 'pass' : 'fail', '门槛 ' + ALPHA_MIN) +
      mcard('检测 F1', fx(ag.meanF1), ag.meanF1 >= 0.85 ? 'pass' : 'warn', '配对平均') +
      mcard('HD95', fx(ag.meanHD95, 1) + '‰', 'warn', '占图像宽度') +
      mcard('争议度', fx(ag.controversy), ag.controversy >= CONTROVERSY_TRIGGER ? 'fail' : 'pass', ag.controversy >= CONTROVERSY_TRIGGER ? '触发复核' : '未触发') +
      '</div>' +
      '<div class="hr"></div>' +
      '<table class="tbl"><thead><tr><th>配对</th><th class="r">Dice</th><th class="r">F1</th><th class="r">匹配/漏/多</th></tr></thead><tbody>' +
      [['A–B', 0], ['A–C', 1], ['B–C', 2]].map((p, k) => {
        const x = ag.pairs[k];
        return '<tr><td>' + p[0] + '</td><td class="r num">' + fx(x.meanDice) + '</td><td class="r num">' + fx(x.detF1) + '</td><td class="r num">' + x.matched + '/' + x.fn + '/' + x.fp + '</td></tr>';
      }).join('') + '</tbody></table>' +
      '<p class="note" style="margin-top:9px">α 按名义尺度、允许缺测计算；匹配单元由三份修订的 IoU 连通分量建立，不是按顺序硬对齐。</p>' +
      '</div></div>' +
      '<div class="panel" style="margin-top:14px"><div class="panel-bd">' +
      '<button class="btn primary" id="btnToAdj" style="width:100%;justify-content:center">交给首席仲裁</button>' +
      '</div></div>' +
      '</div></div>';
  }
  function mcard(k, v, cls, t) {
    return '<div class="mcard ' + (cls || '') + '"><div class="k">' + k + '</div><div class="v num">' + v + '</div><div class="t">' + (t || '') + '</div></div>';
  }

  function paintReveal(i) {
    const trio = document.getElementById('trio');
    if (trio) {
      const label = { A: '专家 A（你）', B: '专家 B', C: '专家 C' };
      trio.innerHTML = ['A', 'B', 'C'].map(c =>
        '<div><div class="chip" style="margin-bottom:5px">' + label[c] + '</div>' +
        '<div class="stagewrap" data-cv="' + c + '"></div></div>').join('');
      ['A', 'B', 'C'].forEach(c => {
        const box = trio.querySelector('[data-cv="' + c + '"]');
        const cv = canvasWithInstances(i, S.rev[i][c].instances, 360, { labels: false, lw: 2.2, fillAlpha: 0.10 });
        cv.style.width = '100%'; cv.style.height = 'auto';
        box.appendChild(cv);
      });
    }
    const cv = document.getElementById('cvDis');
    if (!cv) return;
    const N = cv.width, g = cv.getContext('2d');
    g.drawImage(baseCanvas(i), 0, 0, N, N);
    const maps = ['A', 'B', 'C'].map(c => labelMap(S.rev[i][c].instances, N));
    const img = g.getImageData(0, 0, N, N), d = img.data;
    for (let k = 0; k < N * N; k++) {
      const a = maps[0][k], b = maps[1][k], c = maps[2][k];
      const present = (a > 0 ? 1 : 0) + (b > 0 ? 1 : 0) + (c > 0 ? 1 : 0);
      if (present === 0) continue;
      let rgba = null;
      if (present === 3) {
        if (a === b && b === c) rgba = [55, 211, 155, 26];
        else rgba = [167, 139, 250, 74];
      } else if (present === 2) rgba = [255, 183, 3, 78];
      else rgba = [255, 93, 115, 96];
      const p = k * 4;
      d[p] = Math.round(d[p] * 0.45 + rgba[0] * 0.55);
      d[p + 1] = Math.round(d[p + 1] * 0.45 + rgba[1] * 0.55);
      d[p + 2] = Math.round(d[p + 2] * 0.45 + rgba[2] * 0.55);
    }
    g.putImageData(img, 0, 0);
    ['A', 'B', 'C'].forEach(c => {
      O.drawInstances(cv, S.rev[i][c].instances, { fill: false, labels: false, lw: 1, fillAlpha: 0 });
    });
  }

  function paintThumbs(i) {
    app.querySelectorAll('canvas[data-thumb]').forEach(cv => {
      const c = cv.getAttribute('data-thumb');
      if (!S.rev[i][c]) return;
      const g = cv.getContext('2d');
      g.drawImage(baseCanvas(i), 0, 0, cv.width, cv.height);
      O.drawInstances(cv, S.rev[i][c].instances, { labels: false, lw: 2, fillAlpha: 0.12 });
    });
  }

  function labelMap(instances, N) {
    const cv = document.createElement('canvas');
    cv.width = cv.height = N;
    const g = cv.getContext('2d');
    const base = { CYSTIC: 1, SOLID: 2, BUDDING: 3, DEBRIS: 4 };
    instances.forEach(inst => {
      g.fillStyle = 'rgb(' + base[inst.cls] + ',0,0)';
      g.beginPath();
      inst.pts.forEach((p, i) => i ? g.lineTo(p[0] * N, p[1] * N) : g.moveTo(p[0] * N, p[1] * N));
      g.closePath(); g.fill();
    });
    const d = g.getImageData(0, 0, N, N).data;
    const m = new Uint8Array(N * N);
    for (let k = 0; k < N * N; k++) m[k] = d[k * 4];
    return m;
  }

  /* ================= 视图 3：仲裁台 ================= */
  function viewAdjudicate(i) {
    const ag = O.computeAgreement([S.rev[i].A, S.rev[i].B, S.rev[i].C]);
    const rank = modelRank(i);
    const lastCoder = rank[rank.length - 1].coder;
    const trigger = [];
    if (ag.controversy >= CONTROVERSY_TRIGGER) trigger.push('争议度 ' + fx(ag.controversy) + ' ≥ ' + CONTROVERSY_TRIGGER);
    if (ag.alpha < ALPHA_MIN) trigger.push('α ' + fx(ag.alpha) + ' 低于门槛 ' + ALPHA_MIN);

    const cards = ['A', 'B', 'C'].map(c => {
      const rk = rank.findIndex(r => r.coder === c);
      const m = O.compareRevisions(S.rev[i][c], { instances: S.model[i].instances });
      const sel = S.tmpSel === c;
      return '<div class="revcard ' + (sel ? 'on' : '') + ' ' + (rk === rank.length - 1 ? 'last' : '') + '" data-sel="' + c + '">' +
        '<span class="radio"></span>' +
        '<canvas data-thumb="' + c + '" width="192" height="192"></canvas>' +
        '<div><div style="font-size:12.5px;color:var(--ink)">专家 ' + c + ' 的修订 ' +
        (rk === rank.length - 1 ? '<span class="chip bad">模型排序末位</span>' : '') + '</div>' +
        '<div class="revmeta"><span>实例 <b>' + S.rev[i][c].instances.length + '</b></span>' +
        '<span>与其余两份 Dice <b>' + fx(coderDice(ag, c)) + '</b></span>' +
        '<span>与模型一致度 <b>' + fx(m.meanDice) + '</b></span>' +
        '<span>模型排序 <b>#' + (rk + 1) + '</b></span></div></div></div>';
    }).join('');

    return '<h2 class="viewtitle">S4 首席仲裁 · ' + S.samples[i].id +
      '<small>模型排序是输入，不是决定。界面不预选、不自动勾选、无一键采纳。</small></h2>' +
      '<div class="grid2"><div>' +
      '<div class="panel"><div class="panel-hd"><h3>三份修订</h3><span class="sub">选择一份，或合并 / 重标</span></div>' +
      '<div class="panel-bd"><div class="revlist">' + cards + '</div>' +
      '<div class="hr"></div>' +
      '<div class="row"><label class="tiny dim" style="width:64px">理由码</label>' +
      '<select id="reasonSel"><option value="">请选择（必填）</option>' +
      REASONS.map(r => '<option value="' + r + '"' + (S.tmpReason === r ? ' selected' : '') + '>' + r + '</option>').join('') +
      '</select></div>' +
      '<div class="row" style="margin-top:8px;align-items:flex-start"><label class="tiny dim" style="width:64px;padding-top:6px">说明</label>' +
      '<textarea id="reasonText" rows="2" style="flex:1;resize:vertical" placeholder="为什么选这一份（会进入证据链，可被复核）">' + esc(S.tmpText || '') + '</textarea></div>' +
      '<div class="row end" style="margin-top:12px">' +
      '<span class="tiny dim" id="adjHint">' + (S.tmpSel ? '已选择专家 ' + S.tmpSel : '尚未选择任何修订') + '</span>' +
      '<div class="spacer"></div>' +
      '<button class="btn primary" id="btnAdjudicate" ' + (S.tmpSel && S.tmpReason ? '' : 'disabled') + '>提交仲裁</button>' +
      '</div></div></div>' +
      '</div><div>' +
      '<div class="panel"><div class="panel-hd"><h3>模型排序</h3><span class="chip warn">建议，非结论</span></div><div class="panel-bd">' +
      rank.map((r, k) => '<div class="rankrow ' + (k === 0 ? 'first' : (k === rank.length - 1 ? 'last' : '')) + '">' +
        '<span class="n">' + (k + 1) + '</span><span>专家 ' + r.coder + '</span>' +
        '<span class="bar"><i style="width:' + Math.round(r.q * 100) + '%"></i></span>' +
        '<span class="sc">' + fx(r.dice) + '</span></div>').join('') +
      '<p class="note" style="margin-top:9px">排序依据是<b>与该模型自身预标注的一致度</b>。这正是不能让它做选择的原因：它必然偏好最像自己的那份。排序末位若同属高分，往往包含真实洞察。</p>' +
      '</div></div>' +
      '<div class="panel" style="margin-top:14px"><div class="panel-hd"><h3>复核触发</h3></div><div class="panel-bd">' +
      '<p class="note" style="margin-bottom:8px">命中下列任一条件即进入 <span class="mono">AWAITING_COSIGN</span>，由第二人独立复核：</p>' +
      '<div class="pill-row">' +
      '<span class="chip ' + (ag.controversy >= CONTROVERSY_TRIGGER ? 'warn' : '') + '">争议度 Top 10%</span>' +
      '<span class="chip ' + (ag.alpha < ALPHA_MIN ? 'warn' : '') + '">α 低于门槛</span>' +
      '<span class="chip ' + (S.tmpSel === lastCoder ? 'warn' : '') + '">选了模型排序末位</span>' +
      '<span class="chip">首席自行重标</span></div>' +
      (trigger.length ? '<div class="banner warn" style="margin-top:9px"><div class="ic">!</div><div>已命中：' + trigger.join('；') + '</div></div>' : '') +
      (S.tmpSel === lastCoder ? '<div class="banner warn" style="margin-top:9px"><div class="ic">!</div><div>你选择了<b>模型排序末位</b>的修订。这不是错误，但需要第二人独立判断来兜底。</div></div>' : '') +
      '</div></div>' +
      '</div></div>';
  }
  function modelRank(i) {
    const rows = ['A', 'B', 'C'].map(c => {
      const m = O.compareRevisions(S.rev[i][c], { instances: S.model[i].instances });
      return { coder: c, dice: m.meanDice };
    }).sort((a, b) => b.dice - a.dice);
    const max = rows[0].dice || 1;
    rows.forEach(r => r.q = Math.max(0.04, r.dice / max));
    return rows;
  }
  function coderDice(ag, c) {
    const idx = { A: 0, B: 1, C: 2 }[c];
    const pr = [[0, 1], [0, 2], [1, 2]];
    const vals = [];
    ag.pairs.forEach((p, k) => { if (pr[k][0] === idx || pr[k][1] === idx) vals.push(p.meanDice); });
    return vals.reduce((a, b) => a + b, 0) / vals.length;
  }

  /* ================= 视图 4：条件复核 ================= */
  function viewCosign(i) {
    const adj = S.adj[i], cs = S.cosign[i] || {};
    const trig = adj.cosignTrigger || '选了模型排序末位的修订';
    const gate = S.role !== 'cosigner';
    const head = '<h2 class="viewtitle">条件复核 · ' + S.samples[i].id +
      '<small>AWAITING_COSIGN：触发规则默认关闭，本项目已开启。复核人必须先独立判断，再看首席结论。</small></h2>';

    if (gate) {
      return head + '<div class="panel"><div class="panel-bd">' +
        '<div class="banner warn"><div class="ic">!</div><div><b>当前角色没有复核权限</b><br>权限矩阵要求：复核专家，且不得与本案首席为同一人。</div></div>' +
        '<button class="btn primary" id="btnBeCosigner" style="margin-top:12px">切换为复核专家</button>' +
        '</div></div>';
    }

    const opts = ['A', 'B', 'C'].map(c =>
      '<div class="revcard ' + (cs.blindChoice === c ? 'on' : '') + ' ' + (cs.locked ? 'last' : '') + '" data-cosign="' + c + '">' +
      '<span class="radio"></span>' +
      '<canvas data-thumb="' + c + '" width="192" height="192"></canvas>' +
      '<div><div style="font-size:12.5px;color:var(--ink)">专家 ' + c + ' 的修订</div>' +
      '<div class="revmeta"><span>实例 <b>' + S.rev[i][c].instances.length + '</b></span>' +
      (cs.locked ? '<span class="okc">已锁定，不可更改</span>' : '<span class="dim">未锁定</span>') + '</div></div></div>').join('');

    let body = '<div class="grid2"><div>' +
      '<div class="panel"><div class="panel-hd"><h3>触发原因</h3><span class="chip warn">' + esc(trig) + '</span></div>' +
      '<div class="panel-bd"><p class="note">首席选择了 <b>专家 ' + adj.selected + '</b>，理由码「' + esc(adj.reasonCode) + '」。' +
      (adj.reasonText ? '说明：' + esc(adj.reasonText) : '') + '</p></div></div>' +
      '<div class="panel" style="margin-top:14px"><div class="panel-hd"><h3>第一步 · 你的独立判断</h3>' +
      '<span class="sub">只看三份修订，不看首席结论</span></div>' +
      '<div class="panel-bd"><div class="revlist">' + opts + '</div>' +
      '<div class="row end" style="margin-top:11px">' +
      '<button class="btn primary" id="btnLockChoice" ' + (cs.blindChoice && !cs.locked ? '' : 'disabled') + '>锁定独立意见</button>' +
      '</div></div></div>' +
      '</div><div>' +
      '<div class="panel"><div class="panel-hd"><h3>第二步 · 揭示首席结论</h3></div><div class="panel-bd">' +
      (cs.locked
        ? '<div class="banner ok" style="margin-bottom:10px"><div class="ic">✓</div><div>你的选择 <b>专家 ' + cs.blindChoice + '</b> 已于 ' + (cs.lockedAt || '') + ' 锁定。锁定时间进入审计链。</div></div>' +
          (cs.revealed
            ? '<table class="tbl"><tbody>' +
              '<tr><td class="dim">首席选择</td><td>专家 ' + adj.selected + '</td></tr>' +
              '<tr><td class="dim">理由码</td><td>' + esc(adj.reasonCode) + '</td></tr>' +
              '<tr><td class="dim">说明</td><td>' + (adj.reasonText ? esc(adj.reasonText) : '—') + '</td></tr>' +
              '<tr><td class="dim">与你的判断</td><td>' + (cs.blindChoice === adj.selected ? '<span class="okc">一致</span>' : '<span class="warnc">不一致</span>') + '</td></tr>' +
              '</tbody></table>' +
              '<div class="hr"></div>' +
              '<textarea id="dissentText" rows="2" style="width:100%;resize:vertical" placeholder="否决时必须说明分歧点">' + esc(cs.dissent || '') + '</textarea>' +
              '<div class="row" style="margin-top:10px"><button class="btn primary" id="btnApprove">通过</button>' +
              '<button class="btn danger" id="btnReject">否决并退回首席</button></div>'
            : '<button class="btn" id="btnRevealChief">揭示首席的选择与理由</button>')
        : '<p class="note">未锁定独立意见前，本面板不返回首席结论的任何字段（接口返回 409，不是返回空值）。</p>') +
      '</div></div>' +
      '<div class="panel" style="margin-top:14px"><div class="panel-hd"><h3>为什么顺序不能反</h3></div><div class="panel-bd">' +
      '<p class="note">先看结论再表态，复核就退化为盖章。所以服务端把它切成两个端点：<span class="mono">cosign:choice</span> 与 <span class="mono">cosign:context</span>。前者不成功，后者不给内容。</p>' +
      '</div></div>' +
      '</div></div>';
    return head + body;
  }

  /* ================= 视图 5：定稿与导出 ================= */
  function viewFinal(i) {
    const ag = O.computeAgreement([S.rev[i].A, S.rev[i].B, S.rev[i].C]);
    const adj = S.adj[i], cs = S.cosign[i];
    const checks = [
      { k: '平均 Dice ≥ ' + DICE_MIN, v: ag.meanDice, ok: ag.meanDice >= DICE_MIN },
      { k: 'Krippendorff α ≥ ' + ALPHA_MIN, v: ag.alpha, ok: ag.alpha >= ALPHA_MIN },
      { k: '条件复核已完成', v: adj.cosignRequired ? (cs && cs.verdict === 'approve' ? 1 : 0) : 1, ok: !adj.cosignRequired || (cs && cs.verdict === 'approve'), text: adj.cosignRequired ? (cs && cs.verdict === 'approve' ? '通过' : '未通过') : '未触发' },
      { k: '模板允许导出', v: S.template === 'research' ? 1 : 0, ok: S.template === 'research', text: S.template === 'research' ? '科研生产' : '教学实训（拦截）' }
    ];
    const pass = checks.every(c => c.ok);
    const exported = !!S.exported[i];

    return '<h2 class="viewtitle">S5 定稿与入库 · ' + S.samples[i].id +
      '<small>未通过门槛的数据标记 NEEDS_IMPROVEMENT，不允许导出为训练集。</small></h2>' +
      '<div class="grid2"><div>' +
      '<div class="panel"><div class="panel-hd"><h3>门槛校验</h3>' +
      '<div class="spacer"></div>' + (pass ? chip('PASS', 'ok') : chip('BLOCKED', 'bad')) + '</div>' +
      '<div class="panel-bd"><table class="tbl"><thead><tr><th>检查项</th><th class="r">值</th><th class="r">结果</th></tr></thead><tbody>' +
      checks.map(c => '<tr><td>' + c.k + '</td><td class="r num">' + (c.text || fx(c.v)) + '</td><td class="r">' + (c.ok ? chip('通过', 'ok') : chip('未通过', 'bad')) + '</td></tr>').join('') +
      '</tbody></table>' +
      (pass ? '' : '<div class="banner bad" style="margin-top:11px"><div class="ic">!</div><div><b>不满足导出条件</b>：系统给出缺哪一维、差多少，而不是笼统报错。数据保留在 <span class="mono">NEEDS_IMPROVEMENT</span>，可重开一轮。</div></div>') +
      '<div class="row end" style="margin-top:12px">' +
      '<button class="btn' + (pass && !exported ? ' primary' : '') + '" id="btnExport" ' + (pass && !exported ? '' : 'disabled') + '>' +
      (exported ? '已导出' : '导出数据集 v1') + '</button></div>' +
      '</div></div>' +

      '<div class="panel" style="margin-top:14px"><div class="panel-hd"><h3>数据集卡片</h3><span class="sub">随导出生成，含流程与协议版本</span></div>' +
      '<div class="panel-bd"><dl class="kv">' +
      '<dt>样本</dt><dd class="mono">' + S.samples[i].id + ' （' + O.STRATA[S.samples[i].stratum].label + '）</dd>' +
      '<dt>协议</dt><dd>organoid-4class v1.2 · 触碰规则：允许接触，重叠处按可见边界切分</dd>' +
      '<dt>流程</dt><dd>3 人盲读 → 揭示 → 首席仲裁' + (adj.cosignRequired ? ' → 条件复核' : '') + '</dd>' +
      '<dt>一致性</dt><dd class="num">Dice ' + fx(ag.meanDice) + ' · α ' + fx(ag.alpha) + ' · F1 ' + fx(ag.meanF1) + '</dd>' +
      '<dt>定稿</dt><dd>专家 ' + adj.selected + ' 的修订 · 理由码「' + esc(adj.reasonCode) + '」</dd>' +
      '<dt>预标注</dt><dd>organoid-det-v0.9（权重哈希 3f9a…c21e）</dd>' +
      '<dt>审计头</dt><dd class="mono">' + S.chain.prev.slice(0, 16) + '</dd>' +
      '</dl></div></div>' +
      '</div><div>' +
      '<div class="panel"><div class="panel-hd"><h3>证据链</h3><div class="spacer"></div><button class="btn sm" id="btnVerify2">校验</button></div>' +
      '<div class="panel-bd tight">' + auditTable() + '</div></div>' +
      '<div class="panel" style="margin-top:14px"><div class="panel-hd"><h3>教学模板拦截</h3></div><div class="panel-bd">' +
      '<p class="note" style="margin-bottom:9px">把模板切到教学实训后，同一份定稿数据的导出会被闸门拒绝——<b>不靠人工筛选</b>。</p>' +
      '<select id="templateSel2"><option value="research"' + (S.template === 'research' ? ' selected' : '') + '>科研生产</option>' +
      '<option value="teaching"' + (S.template === 'teaching' ? ' selected' : '') + '>教学实训</option></select>' +
      (S.template === 'teaching' ? '<div class="banner bad" style="margin-top:9px"><div class="ic">!</div><div>导出被拒：<span class="mono">template=teaching</span> 不在导出白名单。拒绝清单写入审计。</div></div>' : '') +
      '</div></div>' +
      '<div class="panel" style="margin-top:14px"><div class="panel-bd">' +
      '<button class="btn" id="btnBackOverview" style="width:100%;justify-content:center">返回任务总览</button>' +
      '</div></div>' +
      '</div></div>';
  }

  /* ================= 事件绑定 ================= */
  function bind() {
    const q = (sel) => app.querySelector(sel);
    const qa = (sel) => Array.prototype.slice.call(app.querySelectorAll(sel));

    qa('[data-go]').forEach(b => b.onclick = () => {
      S.idx = +b.getAttribute('data-i');
      S.view = b.getAttribute('data-go');
      S.tmpSel = null; S.tmpReason = ''; S.tmpText = '';
      render();
    });

    // 模板选择
    ['#templateSel', '#templateSel2'].forEach(sel => {
      const el = q(sel);
      if (el) el.onchange = () => {
        S.template = el.value;
        if (el.value === 'teaching') O.pushAudit(S.chain, roleName(S.role), 'TEMPLATE_SWITCHED', { template: 'teaching' });
        render();
      };
    });

    const v = q('#btnVerify'), v2 = q('#btnVerify2');
    [v, v2].forEach(b => { if (b) b.onclick = doVerify; });

    /* ---- 盲读 ---- */
    if (S.view === 'blind') bindBlind();

    /* ---- 揭示 ---- */
    const br = q('#btnReveal');
    if (br) br.onclick = () => {
      S.revealed[S.idx] = true;
      O.pushAudit(S.chain, roleName(S.role), 'REVEAL_EXECUTED', { sample: S.samples[S.idx].id, coders: ['A', 'B', 'C'] });
      render();
    };
    const bt = q('#btnToAdj');
    if (bt) bt.onclick = () => { S.view = 'adjudicate'; S.tmpSel = null; S.tmpReason = ''; render(); };

    /* ---- 仲裁 ---- */
    qa('[data-sel]').forEach(card => card.onclick = () => {
      if (S.adj[S.idx]) return;
      S.tmpSel = card.getAttribute('data-sel');
      render();
    });
    const rs = q('#reasonSel'), rt = q('#reasonText');
    if (rs) rs.onchange = () => { S.tmpReason = rs.value; render(); };
    if (rt) rt.oninput = () => { S.tmpText = rt.value; };
    const ba = q('#btnAdjudicate');
    if (ba) ba.onclick = doAdjudicate;

    /* ---- 复核 ---- */
    const bc = q('#btnBeCosigner');
    if (bc) bc.onclick = () => { S.role = 'cosigner'; document.getElementById('roleSel').value = 'cosigner'; render(); };
    qa('[data-cosign]').forEach(card => card.onclick = () => {
      if (!S.cosign[S.idx]) S.cosign[S.idx] = {};
      if (S.cosign[S.idx].locked) return;
      S.cosign[S.idx].blindChoice = card.getAttribute('data-cosign');
      render();
    });
    const bl = q('#btnLockChoice');
    if (bl) bl.onclick = () => {
      const cs = S.cosign[S.idx];
      cs.locked = true;
      cs.lockedAt = new Date().toTimeString().slice(0, 8);
      O.pushAudit(S.chain, 'expert:cosigner', 'COSIGN_CHOICE_LOCKED', { sample: S.samples[S.idx].id, choice: cs.blindChoice });
      render();
    };
    const brc = q('#btnRevealChief');
    if (brc) brc.onclick = () => {
      S.cosign[S.idx].revealed = true;
      O.pushAudit(S.chain, 'expert:cosigner', 'COSIGN_CONTEXT_REVEALED', { sample: S.samples[S.idx].id });
      render();
    };
    const dt = q('#dissentText');
    if (dt) dt.oninput = () => { S.cosign[S.idx].dissent = dt.value; };
    const bap = q('#btnApprove'), brj = q('#btnReject');
    if (bap) bap.onclick = () => {
      S.cosign[S.idx].verdict = 'approve';
      O.pushAudit(S.chain, 'expert:cosigner', 'COSIGN_APPROVED', { sample: S.samples[S.idx].id });
      toast('复核通过，可进入定稿', 'ok');
      S.view = 'final'; render();
    };
    if (brj) brj.onclick = () => {
      const txt = (S.cosign[S.idx].dissent || '').trim();
      if (txt.length < 4) { toast('否决必须填写分歧点', 'bad'); return; }
      S.cosign[S.idx].verdict = 'reject';
      O.pushAudit(S.chain, 'expert:cosigner', 'COSIGN_REJECTED', { sample: S.samples[S.idx].id, dissent: txt });
      delete S.adj[S.idx];
      S.view = 'adjudicate';
      S.tmpSel = null; S.tmpReason = ''; S.tmpText = '';
      toast('已否决，退回首席重议', 'ok');
      render();
    };

    /* ---- 定稿 ---- */
    const be = q('#btnExport');
    if (be) be.onclick = () => {
      if (S.template !== 'research') { toast('导出被拒：教学模板产物不在白名单', 'bad'); return; }
      S.exported[S.idx] = { at: O.nowISO() };
      O.pushAudit(S.chain, roleName(S.role), 'DATASET_EXPORTED', { sample: S.samples[S.idx].id, gate: 'passed' });
      toast('已导出数据集 v1', 'ok');
      render();
    };
    const bb = q('#btnBackOverview');
    if (bb) bb.onclick = () => { S.view = 'overview'; render(); };
  }

  function bindBlind() {
    const q = (s) => app.querySelector(s);
    app.querySelectorAll('[data-mode]').forEach(b => b.onclick = () => {
      S.mode = b.getAttribute('data-mode');
      S.pending = null;
      render();
    });
    app.querySelectorAll('[data-cls]').forEach(b => b.onclick = () => {
      S.activeCls = b.getAttribute('data-cls');
      render();
    });
    app.querySelectorAll('[data-inst]').forEach(b => b.onclick = () => {
      S.selected = +b.getAttribute('data-inst');
      pasteBlind();
    });
    const cv = q('#cv');
    if (!cv) return;

    cv.addEventListener('mousedown', e => {
      e.preventDefault();
      const p = toNorm(cv, e);
      if (!S.firstTouchAt) S.firstTouchAt = Date.now();
      if (S.mode === 'draw') {
        if (!S.pending) { S.pending = { pts: [] }; }
        if (e.button === 0) S.pending.pts.push(p);
      } else {
        const hit = hitTest(p);
        S.selected = hit;
        if (hit >= 0) S.drag = { idx: hit, from: p, orig: S.draft[hit].pts.map(x => x.slice()) };
        paintBlind(S.idx);
      }
    });
    cv.addEventListener('mousemove', e => {
      const p = toNorm(cv, e);
      if (S.mode === 'draw' && S.pending) { S.pending.cursor = p; paintBlind(S.idx); }
      if (S.drag) {
        const dx = p[0] - S.drag.from[0], dy = p[1] - S.drag.from[1];
        S.draft[S.drag.idx].pts = S.drag.orig.map(q => [q[0] + dx, q[1] + dy]);
        paintBlind(S.idx);
      }
    });
    window.addEventListener('mouseup', () => { S.drag = null; }, { once: true });
    cv.addEventListener('mouseleave', () => { if (S.pending) { S.pending.cursor = null; paintBlind(S.idx); } });
    cv.addEventListener('dblclick', e => { e.preventDefault(); closePending(); });
    cv.addEventListener('contextmenu', e => {
      e.preventDefault();
      if (S.pending && S.pending.pts.length) { S.pending.pts.pop(); paintBlind(S.idx); }
    });
    document.onkeydown = (e) => {
      if (S.view !== 'blind') return;
      if (e.key === 'Escape') { S.pending = null; paintBlind(S.idx); }
      if ((e.key === 'Delete' || e.key === 'Backspace') && S.selected != null) {
        S.draft.splice(S.selected, 1); S.selected = null; render();
      }
      if (e.key === 'Enter') closePending();
    };
    q('#btnClear').onclick = () => { S.draft = []; S.selected = null; S.pending = null; render(); };
    q('#btnDel').onclick = () => {
      if (S.selected == null) { toast('先在选择模式下点选一个轮廓'); return; }
      S.draft.splice(S.selected, 1); S.selected = null; render();
    };
    q('#btnScript').onclick = () => {
      S.draft = O.scriptedExpert(S.samples[S.idx], 'A').instances.map(x => ({ cls: x.cls, pts: x.pts }));
      toast('已填入示例修订（演示用）', 'ok');
      render();
    };
    q('#btnSubmit').onclick = doSubmitBlind;
  }

  function toNorm(cv, e) {
    const r = cv.getBoundingClientRect();
    return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height];
  }
  function hitTest(p) {
    for (let k = S.draft.length - 1; k >= 0; k--) {
      const pts = S.draft[k].pts;
      let inside = false;
      for (let a = 0, b = pts.length - 1; a < pts.length; b = a++) {
        const xi = pts[a][0], yi = pts[a][1], xj = pts[b][0], yj = pts[b][1];
        if (((yi > p[1]) !== (yj > p[1])) && (p[0] < (xj - xi) * (p[1] - yi) / (yj - yi) + xi)) inside = !inside;
      }
      if (inside) return k;
    }
    return -1;
  }
  function closePending() {
    if (!S.pending || S.pending.pts.length < 3) { S.pending = null; paintBlind(S.idx); return; }
    S.draft.push({ cls: S.activeCls, pts: S.pending.pts.map(p => p.slice()) });
    S.pending = null;
    render();
  }
  function pasteBlind() {
    const g = document.getElementById('cv');
    if (!g) return;
    paintBlind(S.idx);
  }

  function doSubmitBlind() {
    if (!S.draft.length) { toast('至少标注一个实例', 'bad'); return; }
    const i = S.idx;
    const ms = S.firstTouchAt ? (Date.now() - S.firstTouchAt) : 60000;
    S.rev[i].A = {
      coder: 'A',
      instances: S.draft.map((d, k) => ({ id: 'A-' + (k + 1), cls: d.cls, pts: d.pts })),
      meta: { blindMs: ms, submittedAt: O.nowISO(), isBlind: true }
    };
    O.pushAudit(S.chain, 'expert:A', 'REVISION_SUBMITTED', {
      sample: S.samples[i].id, coder: 'A', instances: S.rev[i].A.instances.length,
      blind_ms: ms, is_blind_read: true,
      payload_hash: O.hash64(O.canon(S.rev[i].A.instances))
    });
    S.pending = null; S.selected = null;
    S.view = 'reveal';
    stopTimer();
    render();
    toast('修订已锁定，等待揭示', 'ok');
  }

  function startTimer() {
    stopTimer();
    if (!S.startedAt || S.view !== 'blind') S.startedAt = S.startedAt || Date.now();
    S.timerH = setInterval(() => {
      const el = document.getElementById('timer');
      if (!el) return;
      el.textContent = O.fmtDur(Date.now() - (S.firstTouchAt || S.startedAt));
    }, 500);
  }
  function stopTimer() { if (S.timerH) { clearInterval(S.timerH); S.timerH = 0; } }

  /* ---- 仲裁 ---- */
  function doAdjudicate() {
    const i = S.idx;
    if (!S.tmpSel) { toast('必须先选择一份修订', 'bad'); return; }
    if (!S.tmpReason) { toast('理由码必填', 'bad'); return; }
    const ag = O.computeAgreement([S.rev[i].A, S.rev[i].B, S.rev[i].C]);
    const rank = modelRank(i);
    const last = rank[rank.length - 1].coder;
    const trig = [];
    if (S.tmpSel === last) trig.push('选了模型排序末位的修订');
    if (ag.controversy >= CONTROVERSY_TRIGGER) trig.push('争议度 ' + fx(ag.controversy));
    if (ag.alpha < ALPHA_MIN) trig.push('α 低于门槛');

    S.adj[i] = {
      selected: S.tmpSel, reasonCode: S.tmpReason, reasonText: S.tmpText || '',
      at: O.nowISO(), cosignRequired: trig.length > 0, cosignTrigger: trig.join('；')
    };
    O.pushAudit(S.chain, 'expert:chief', 'ADJUDICATED', {
      sample: S.samples[i].id, selected: S.tmpSel, reason_code: S.tmpReason,
      model_rank: rank.map(r => r.coder).join('>'),
      model_rank_agreement: S.tmpSel === rank[0].coder,
      cosign_required: trig.length > 0
    });
    S.tmpSel = null; S.tmpReason = ''; S.tmpText = '';
    S.view = trig.length ? 'cosign' : 'final';
    render();
    toast(trig.length ? '已命中复核条件，转第二人复核' : '仲裁完成', 'ok');
  }

  /* ---- 审计校验 ---- */
  function doVerify() {
    const r = O.verifyChain(S.chain);
    if (r.ok) toast('审计链完整：' + S.chain.entries.length + ' 条全部校验通过', 'ok');
    else toast('审计链在第 ' + r.badAt + ' 条断裂', 'bad');
  }

  /* ================= 契约 JSON 抽屉 ================= */
  function buildBundle() {
    const i = S.idx, sm = S.samples[i];
    const ag = S.rev[i].A ? O.computeAgreement([S.rev[i].A, S.rev[i].B, S.rev[i].C]) : null;
    const adj = S.adj[i] || null, cs = S.cosign[i] || null;
    const out = {
      task: {
        id: 'T-' + sm.id, template: S.template, protocol: 'organoid-4class@v1.2',
        redundancy_n: 3, preannotation_policy: 'graded', state: stateOf(i).code,
        strata_policy: { ratio: 0.2, min_cases: 150, hard_layer_weight: 2.0 }
      },
      sample: { id: sm.id, stratum: sm.stratum, isolation_set: false },
      assignments: ['A', 'B', 'C'].map(c => ({
        id: sm.id + ':' + c, sample_id: sm.id, assignee_role: '领域专家',
        coder: c, state: S.rev[i][c] ? 'SUBMITTED' : 'ASSIGNED'
      })),
      revisions: ['A', 'B', 'C'].filter(c => S.rev[i][c]).map(c => ({
        sample_id: sm.id, coder: c, is_blind_read: true,
        instance_count: S.rev[i][c].instances.length,
        blind_ms: S.rev[i][c].meta ? S.rev[i][c].meta.blindMs : null,
        payload_hash: O.hash64(O.canon(S.rev[i][c].instances))
      })),
      model_rank: S.model[i].instances.length ? modelRank(i).map((r, k) => ({
        ranking: k + 1, coder: r.coder, agreement_with_model: O.q4(r.dice)
      })) : [],
      disagreement: ag ? {
        metric_snapshot: {
          dice_mean: O.q4(ag.meanDice), alpha: O.q4(ag.alpha),
          det_f1: O.q4(ag.meanF1), hd95: O.q4(ag.meanHD95 / 1000), label_agreement: O.q4(ag.labelAgree)
        },
        controversy_score: O.q4(ag.controversy), metric_version: 'metric-v1', random_seed: 20261022
      } : null,
      adjudication: adj ? {
        sample_id: sm.id, selected_coder: adj.selected, selector_role: '首席专家',
        reason_code: adj.reasonCode, reason_text: adj.reasonText,
        model_rank_agreement: adj.selected === modelRank(i)[0].coder,
        signature_meaning: '已仲裁', decided_at: adj.at, cosign_required: adj.cosignRequired
      } : null,
      cosign: cs ? {
        sample_id: sm.id, trigger_rule: adj ? adj.cosignTrigger : null,
        blind_choice: cs.blindChoice || null, blind_choice_locked_at: cs.lockedAt || null,
        chief_context_revealed: !!cs.revealed, verdict: cs.verdict || null,
        dissent_text: cs.dissent || null
      } : null,
      audit: {
        head_hash: S.chain.prev, entries: S.chain.entries.length,
        chain_verified: O.verifyChain(S.chain).ok
      }
    };
    return out;
  }

  function openDrawer() {
    const obj = buildBundle();
    const txt = JSON.stringify(obj, null, 2);
    const html = esc(txt)
      .replace(/(&quot;[^&]*?&quot;)(\s*:)/g, '<span class="k">$1</span>$2')
      .replace(/:\s*(&quot;.*?&quot;)/g, ': <span class="s">$1</span>')
      .replace(/:\s*(-?\d+\.?\d*)/g, ': <span class="n">$1</span>')
      .replace(/:\s*(true|false|null)/g, ': <span class="b">$1</span>');
    document.getElementById('jsonBody').innerHTML =
      '<p class="jsoncap">当前样本的契约对象。字段名与 PRD 第七章一致；坐标在入哈希前按 1e-4 量化。</p>' +
      '<pre class="json">' + html + '</pre>';
    document.getElementById('jsonBytes').textContent = (txt.length / 1024).toFixed(1) + ' KB';
    document.getElementById('drawer').classList.add('open');
  }

  /* ================= 顶栏 ================= */
  document.getElementById('roleSel').onchange = function () {
    S.role = this.value;
    render();
  };
  document.getElementById('btnJson').onclick = openDrawer;
  document.getElementById('btnJsonClose').onclick = () => document.getElementById('drawer').classList.remove('open');
  document.getElementById('btnReset').onclick = () => {
    location.reload();
  };

  /* ================= 自动演示驱动（?auto=stage） ================= */
  function runAutoFromHash() {
    const m = location.hash.match(/auto=(\w+)/);
    if (!m) return;
    const stage = m[1];
    const i = 0;
    if (!S.rev[i].A) {
      const inst = O.scriptedExpert(S.samples[i], 'A').instances;
      S.rev[i].A = { coder: 'A', instances: inst, meta: { blindMs: 183000, submittedAt: O.nowISO(), isBlind: true } };
      O.pushAudit(S.chain, 'expert:A', 'REVISION_SUBMITTED', {
        sample: S.samples[i].id, coder: 'A', instances: inst.length, blind_ms: 183000, is_blind_read: true,
        payload_hash: O.hash64(O.canon(inst))
      });
    }
    if (stage !== 'blind') S.revealed[i] = true;
    if (stage === 'adj' || stage === 'cosign' || stage === 'final' || stage === 'blocked') {
      const rank = modelRank(i);
      S.tmpSel = rank[rank.length - 1].coder; // 刻意选排序末位 → 触发复核
      S.tmpReason = '边界判定';
      S.tmpText = 'C 保留了粘连处的两个轮廓；模型在此处合并成一个，证据更充分。';
      doAdjudicate();
    }
    if (stage === 'cosign' || stage === 'final' || stage === 'blocked') {
      S.role = 'cosigner';
      document.getElementById('roleSel').value = 'cosigner';
      const adj = S.adj[i];
      if (adj.cosignRequired) {
        const rank2 = modelRank(i);
        const choice = rank2[0].coder; // 复核人独立判断后选了排序首位
        S.cosign[i] = {
          blindChoice: choice, locked: true, lockedAt: '10:22:41', revealed: true,
          verdict: (stage === 'cosign') ? null : 'approve', dissent: ''
        };
        O.pushAudit(S.chain, 'expert:cosigner', 'COSIGN_CHOICE_LOCKED', { sample: S.samples[i].id, choice: choice });
        O.pushAudit(S.chain, 'expert:cosigner', 'COSIGN_CONTEXT_REVEALED', { sample: S.samples[i].id });
        if (stage !== 'cosign') O.pushAudit(S.chain, 'expert:cosigner', 'COSIGN_APPROVED', { sample: S.samples[i].id });
      }
    }
    if (stage === 'blocked') S.template = 'teaching';
    const viewMap = { blind: 'blind', reveal: 'reveal', adj: 'adjudicate', cosign: 'cosign', final: 'final', blocked: 'final' };
    S.view = viewMap[stage] || 'overview';
    if (S.view === 'adjudicate') { S.tmpSel = null; S.tmpReason = ''; S.tmpText = ''; }
    render();
  }

  init();
  runAutoFromHash();
})();
