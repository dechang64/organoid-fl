const puppeteer = require('puppeteer-core');
const sleep = ms => new Promise(r => setTimeout(r, ms));
(async () => {
  const browser = await puppeteer.launch({
    executablePath: 'C://Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
    headless: 'new', args: ['--no-sandbox', '--disable-gpu'],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1500, height: 1400 });
  const url = 'file:///C:/Users/decha/organoid-fl/mvp/hitl-demo/index.html';
  const text = () => page.evaluate(() => document.body.innerText);
  const drawerJSON = () => page.evaluate(() => {
    const pre = document.querySelector('#jsonBody pre.json');
    return pre ? pre.innerText : '';
  });
  async function gotoStage(hash) {
    await page.goto('about:blank');
    await page.goto(url + (hash ? '#' + hash : ''), { waitUntil: 'networkidle2' });
    await sleep(hash ? 1100 : 800);
  }
  let pass = 0, fail = 0;
  const check = (name, cond) => { console.log((cond ? 'PASS ' : 'FAIL ') + name); cond ? pass++ : fail++; };

  /* ===== A. 复核台投影 ===== */
  await gotoStage('auto=cosign');
  let t = await text();
  check('A1 触发原因面板不含首席选择', !t.includes('首席选择了'));
  check('A2 显示「锁定并揭示之前不可见」占位', t.includes('锁定独立意见并执行揭示之前不可见'));
  await page.click('#btnJson'); await sleep(400);
  let d = await drawerJSON();
  check('A3 抽屉含投影占位标记', d.includes('chief_context_withheld_until_cosign_reveal'));
  check('A4 抽屉 JSON 不含 selected_coder', !d.includes('selected_coder'));
  await page.click('#btnJsonClose'); await sleep(300);

  /* ===== E. 盲读期抽屉投影 ===== */
  await gotoStage('');
  await page.evaluate(() => { document.querySelector('[data-go="blind"][data-i="0"]').click(); });
  await sleep(500);
  await page.click('#btnScript'); await sleep(300);
  await page.click('#btnScript'); await sleep(500);
  await page.click('#btnJson'); await sleep(400);
  d = await drawerJSON();
  check('E1 盲读期抽屉 revisions 仅含自己(A)', !d.includes('"coder": "B"') && !d.includes('"coder": "C"'));
  check('E2 盲读期抽屉 disagreement 为 null', d.includes('"disagreement": null'));
  check('E3 盲读期抽屉 model_rank 为空', d.includes('"model_rank": []'));
  check('E4 盲读期抽屉 assignments 仅自己 + 占位', d.includes('peer_assignments_withheld_during_blind_phase'));
  await page.click('#btnJsonClose'); await sleep(200);

  /* ===== G. 总览审计链投影 ===== */
  await gotoStage('');
  t = await text();
  check('G1 标注员盲读期总览审计链无预标注事件', !t.includes('PREANNOTATION_PRODUCED'));
  check('G2 投影可见条数标注', t.includes('当前角色/阶段投影可见'));

  /* ===== B. 主流程 + 跨样本隔离 ===== */
  await page.evaluate(() => { document.querySelector('[data-go="blind"][data-i="0"]').click(); });
  await sleep(400);
  t = await text();
  check('B0 盲读草稿为空（不再预建 A 修订）', t.includes('还没有轮廓'));
  await page.click('#btnScript'); await sleep(300);
  await page.click('#btnScript'); await sleep(500);
  await page.click('#btnSubmit'); await sleep(600);
  await page.click('#btnReveal'); await sleep(500);
  await page.click('#btnToAdj'); await sleep(500);
  const guardEl = await page.$('#btnSwitchChief');
  if (guardEl) { await guardEl.click(); await sleep(400); }
  const cards = await page.$$('.revcard[data-sel]');
  await cards[cards.length - 1].click(); await sleep(300);
  await page.select('#reasonSel', '边界判定'); await sleep(400);
  await page.click('#btnAdjudicate'); await sleep(600);
  t = await text();
  check('B2 进入复核台', t.includes('条件复核'));
  await page.click('#btnBeCosigner'); await sleep(400);
  const ccards = await page.$$('.revcard[data-cosign]');
  await ccards[0].click(); await sleep(300);
  await page.click('#btnLockChoice'); await sleep(400);
  await page.click('#btnRevealChief'); await sleep(400);
  t = await text();
  check('B3 揭示后可见首席结论', t.includes('首席选择了'));
  await page.click('#btnApprove'); await sleep(500);
  t = await text();
  check('B4 演示辅助入审计', t.includes('DEMO_ASSIST_FILLED'));
  // 切回领域专家再开抽屉（F1 检查的是标注员的投影）
  await page.select('#roleSel', 'expertA'); await sleep(500);
  await page.click('#btnJson'); await sleep(400);
  d = await drawerJSON();
  check('F1 揭示后领域专家抽屉 model_rank 仍为空（全程不可见）', d.includes('"model_rank": []'));
  await page.click('#btnJsonClose'); await sleep(200);
  await page.click('#btnBackOverview'); await sleep(600);
  await page.evaluate(() => { document.querySelector('[data-go="blind"][data-i="1"]').click(); });
  await sleep(500);
  t = await text();
  check('B5 S-002 盲读草稿为空（无跨样本注入）', t.includes('还没有轮廓'));

  /* ===== H. 仲裁台标注员守卫 ===== */
  await gotoStage('auto=adj');
  t = await text();
  check('H1 标注员被挡在仲裁台外', t.includes('仲裁台仅首席专家可用'));
  await page.click('#btnSwitchChief'); await sleep(400);
  t = await text();
  check('H2 切换首席后可见模型排序面板', t.includes('模型排序'));
  await page.click('#btnJson'); await sleep(400);
  d = await drawerJSON();
  check('F2 首席抽屉含 model_rank 条目', d.includes('"agreement_with_model"'));
  await page.click('#btnJsonClose'); await sleep(200);

  /* ===== D. NEEDS_IMPROVEMENT 状态机 ===== */
  await gotoStage('auto=blocked');
  t = await text();
  check('D1 教学模板 → 闸门拦截 + 重开按钮', t.includes('重开一轮'));
  await page.click('#btnBackOverview'); await sleep(600);
  t = await text();
  check('D2 总览状态芯片为「门槛未过」', t.includes('门槛未过'));
  await page.evaluate(() => { const b=[...document.querySelectorAll('[data-go]')].find(x=>x.getAttribute('data-go')==='final'); b && b.click(); });
  await sleep(500);
  await page.click('#btnReopen'); await sleep(600);
  t = await text();
  check('D3 重开后状态回到「待盲读」', t.includes('待盲读'));
  check('D4 重开动作入审计', t.includes('TASK_REOPENED'));

  await browser.close();
  console.log('----');
  console.log('pass=' + pass + ' fail=' + fail);
  process.exit(fail ? 1 : 0);
})().catch(e => { console.error('ERROR', e.message); process.exit(2); });
