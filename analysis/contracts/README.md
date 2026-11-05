# 契约层（M0）

PRD 10.4 的 M0 出口是「契约冻结」，验收方式是机器而非自评。这个目录就是被冻结的那一层，
本文件记录它冻结了什么、怎么验证、以及冻结过程中发现的四处 PRD 口径问题。

## 1. 交付内容

| 模块 | 冻结的对象 | 对应 PRD |
| --- | --- | --- |
| `enums.py` | 11 个样本状态、5 个 Assignment 状态、10 个角色、4 个困难层、2 个模板 id、3 个投影阶段 | 3.1、4.8、7.2、D-08 |
| `entities.py` | 9 个新实体（`AnnotationProtocol`／`Task`／`Assignment`／`Revision`／`Disagreement`／`ModelRank`／`Adjudication`／`Cosign`／`StratumLabel`） | 7.2 |
| `protocol.py` | 协议构造、标签注册表绑定、规则哈希与漂移检测 | 7.2、C-05 |
| `templates.py` | 科研／教学两套编排，差异全部落在配置字段 | 2.3、D-05、D-10 |
| `state_machine.py` | 12 条工作流边 + 9 条行政终止边、guard、角色铸造、Assignment 级联 | 4.8、D-09 |
| `exchange.py` | 交换信封、完整性哈希、跨实体校验、`template=teaching` 导出硬拦截 | 7.2、P-11、D-10 |
| `projection.py` | 按角色 + 阶段投影（11 处演示缺陷的根因修复，G-3 的参考实现） | 3.2、P-01、P-09 |
| `hashing.py` | 复用 `analysis/audit_spec.py` 的规范 JSON，新增加域分隔符的内容哈希 | 8.2、P-06 |

三个设计约束贯穿全层：**零第三方依赖**（可从裸解释器导入，不需要 pandas-before-torch 守卫）、
**实体不可变**（`frozen=True`，盲读版无法被编辑，P-06 由构造保证）、**读端严格**
（未知字段报错，不静默丢弃）。

## 2. 验证

```bash
.venv/Scripts/python.exe -m pytest tests/test_contracts_*.py -q     # 140 passed
.venv/Scripts/python.exe -m pytest tests -q                          # 477 passed
.venv/Scripts/python.exe -m ruff check analysis tests tools           # All checks passed
cargo test --all-targets                                             # 16 passed
node tools/verify_audit_vectors.mjs                                  # 8/8 vectors
```

M0 出口准则的逐条对应：

- **9 个实体** → `contract_report()["entities"]`，并断言其不随新增模板变化；
- **状态机含默认不可达的 AWAITING_COSIGN** → `test_cosign_state_is_unreachable_with_the_default_configuration`（图遍历，不是断言文字）；
- **交换格式 schema** → `test_round_trip_preserves_the_envelope`、`test_dump_is_canonical_and_reproducible`；
- **标注协议 schema** → `test_protocol_hash_covers_rules_but_not_version`；
- **模板抽象** → `test_a_third_template_needs_configuration_only`：注册一个代码里不存在的模板，跑完整状态机、往返交换信封，并断言实体清单逐字未变；
- **90 有序对穷举（G-4）** → 见下节，实际是 110 对，原因写清楚；
- **哈希三方对齐（G-1）** → Python / Rust / Node 同读 `tests/fixtures/audit_hash_vectors.json`，Node 侧本轮补上了浮点域拒绝。

## 3. 四处 PRD 口径问题（已按下面的口径落地，需回改文档）

### 3.1 4.8 的状态数与 90/12/78 自相矛盾

4.8 写「十状态」，但同一张表把 `IN_DATASET` 当成 `ADJUDICATED` 的出口却没列进状态表。
按列出的数一遍是 10 个，按用到的数一遍是 11 个；而表里的合法边是 12 条——12 条只与 11 个状态相容
（因为 `ADJUDICATED → IN_DATASET` 就是其中一条）。于是 G-4 的「90 有序对（12 合法 + 78 非法）」
无法与它描述的表同时成立。

**落地口径：11 状态。** `IN_DATASET` 承载别的状态都不承载的语义——只有它能出现在导出清单里，
P-10 的分层例数统计也依赖它；把「已签」和「可导出」合并成同一个谓词，正是让未签字样本进训练集的路径。
实际计数：**110 有序对 = 12 工作流 + 9 行政终止 + 89 非法**，由 `state_machine_report()` 计算并在测试中断言。

### 3.2 `REJECTED` 没有入边

4.8 给 `REJECTED` 的触发是「数据或协议问题 → 终止」，但没写可以从哪些状态进入。严格读下来它会永远不可达，
变成一个装饰性状态。**落地口径：新增 `EdgeKind.ABORT` 边类，任何非终态都可行政终止**，与工作流边分开计数、
分开评审；这样 12 这个数字仍然可核对，终止策略的变化也不会被误读成流程变化。

### 3.3 Assignment 链的顺序与同段文字冲突

4.8 列的链是 `ASSIGNED → BLIND_DRAFTED → REVEALED → SUBMITTED → LOCKED`，同段又说
「REVEALED 只在任务进入 IN_REVIEW 时授予」，即全员提交之后。两句话不能同时成立。
**落地口径：`BLIND_DRAFTED → SUBMITTED → REVEALED → LOCKED`**——先提交后揭示是 P-01 本身；
按字面顺序实现，会让专家在锁定盲读之前就看到预标注。

### 3.4 `reason_code` 没有词表

7.2 有该字段，但没有一节定义取值。**落地口径：只约束形状（码非空 + 文本非空，P-03），不约束词表**，
`REASON_CODE_SUGGESTIONS` 是 UI 的种子列表，不是契约。闭集一旦写死，第一个真实临床理由就会被拒。
这条与 Q-07（困难层判定规则）同类：属方法学决策，不该由契约层代决。

### 3.5 附带发现：预标注本体不在 9 个实体里

`ModelRank` 只表达「排序与分数」，模型预标注的**内容**（掩膜/框）没有对应实体。投影层目前只能给出
`preannotation.available` 布尔值，而 D-03 的「揭示」语义要求把预标注内容呈现给已盲读的人。
M1 需要补一个 `PreAnnotation` 载荷引用（或明确它挂在 `Revision` 的 `provenance=MODEL_PREANNOTATION` 下）。
这是本次契约化暴露出的**真实建模缺口**，不是文档笔误。

## 4. M0 未做的事（对应 10.9 其余缺口）

| 缺口 | 为什么不在 M0 |
| --- | --- |
| G-2 认证与授权 | 需要服务端；本层提供 `Machine.apply(actor_role=...)` 的角色校验，鉴权本体在 M1 |
| G-5 真实数据 | 需要图像 loader 与真实预标注；契约层只定义形状，不碰数据 |
| G-7 争议度滚动分位 | 需要任务分布；`Disagreement.metric_snapshot` 已可承载，规则属 M2 |
| G-8 `revision_no` 递增 | 已进实体（`Revision.revision_no` + `derived_from`），重开路径的编排在 M1 |
| G-9 持久化 | M1 出口条件；本层全是纯函数与不可变记录，落库时不需要改形状 |

## 5. 加东西的规矩

- **加字段**：可以，但必须同时更新 `tests/contracts_factory.py` 与 `test_every_entity_round_trips_through_its_own_dict`。
- **改字段语义 / 删字段**：不行。10.6 把「数据模型与协议语义」划为不可委托项，这类变更要人工决策并升 `CONTRACT_VERSION`。
- **加模板**：只改 `templates.py` 的注册表，不碰 `entities.py`。`test_a_third_template_needs_configuration_only` 会证明这一点。
- **加投影规则**：改 `projection.py` 的 `_RULES` 表；`test_no_role_ever_receives_the_ranking_unless_the_policy_says_so` 会对角色 × 状态 × 锁定态全组合复核。
