# HITL 标注流水线 MVP（演示版）

对应《PRD v7 HITL 流水线》《实施方案 v1》的可交互演示。**纯前端、零依赖、无构建**——直接用浏览器打开 `index.html`（或单文件版 `类器官标注平台_HITL_MVP_单文件.html`）即可。

## 文件

| 文件 | 说明 |
|---|---|
| `index.html` + `styles.css` + `data.js` + `app.js` | 开发用四文件版（改代码改这套） |
| `类器官标注平台_HITL_MVP_单文件.html` | 外发/演示用单文件版（CSS/JS 已内联，与四文件版内容一致） |

重新生成单文件版：

```bash
node -e "
const fs=require('fs');
let h=fs.readFileSync('index.html','utf8');
h=h.replace('<link rel=\"stylesheet\" href=\"styles.css\">','<style>\n'+fs.readFileSync('styles.css','utf8')+'\n</style>');
h=h.replace('<script src=\"data.js\"></script>\n<script src=\"app.js\"></script>','<script>\n'+fs.readFileSync('data.js','utf8')+'\n</script>\n<script>\n'+fs.readFileSync('app.js','utf8')+'\n</script>');
fs.writeFileSync('类器官标注平台_HITL_MVP_单文件.html',h);
"
```

## 演示动线

1. **总览** → 隔离集面板、任务看板
2. **盲读台** → 画轮廓（或「快速填入示例」），界面不含任何模型字段
3. **揭示** → 三份修订并列 + 按像素分歧图 + 配对 Dice/F1/HD95 + Krippendorff's α
4. **仲裁台** → 选择修订、理由码必填；选了模型排序末位的会触发条件复核
5. **条件复核** → 复核专家先锁定独立意见，才能看到首席结论
6. **定稿** → 门槛 PASS；切到教学模板再导出会被硬拦截

快速跳转：URL 加 `#auto=blind|reveal|adj|cosign|final|blocked`。

## 实现边界（诚实声明）

- 样本图像为**程序化合成**（确定性种子，mulberry32），非真实显微图像。
- 指标为**真实几何计算**：200px 掩膜栅格化 → IoU≥0.5 贪心匹配 → 配对 Dice/F1/HD95；α 经「IoU 连通分量 + 并查集」建跨编码者单元后按名义尺度计算（可缺测），非对真值循环论证。
- 专家 B/C 为脚本化模拟（真值 + 不同幅度形变；C 在粘连层合并实例、低对比层改变类别）。
- 审计哈希用 FNV-1a 双基，**非密码学**，页面内有标注；正式实现按 `analysis/audit_spec.py` 的 `organoid-audit-v1`。
- 契约对象字段名与 PRD 第七章一致（Task / Revision / Disagreement / Adjudication / Cosign / StratumLabel），坐标入哈希前按 1e-4 量化。

## 下一步

- 多边形顶点编辑与撤销栈（当前只能重画）
- 导出契约 JSON 下载
- S-002~S-005 全链路演示聚焦
- 与本仓库契约层（`analysis/schema.py`、`analysis/audit_spec.py`、`analysis/contracts/`）的真实对接——届时把指标与哈希换成共享实现，前端只做投影。
