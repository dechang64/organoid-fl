# HITL MVP 端到端回归

`e2e_regression.js`：23 项断言，覆盖 PRD v7.1 10.8 记录的两轮审计修复。

运行（需本机 Edge + puppeteer-core）：
```bash
npm install puppeteer-core
NODE_PATH=./node_modules node mvp/hitl-demo/tests/e2e_regression.js
```

覆盖面：
- A1–A4 复核台与契约抽屉的投影（P-09）
- E1–E4 盲读期抽屉投影：同伴修订/分歧快照不出现（P-01 / S-12）
- G1–G2 总览审计链按角色投影（S-12）
- B0–B5 主流程 + 跨样本草稿隔离
- F1–F2 模型排序全生命周期可见性（P-01：对领域专家全程不可见）
- H1–H2 仲裁台角色守卫
- D1–D4 NEEDS_IMPROVEMENT 状态机与重开路径（PRD 4.8 / 4.9）

产品化时按 PRD 10.9 迁移为 Playwright 用例纳入 CI。
