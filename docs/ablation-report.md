# 消融对照结果

> 由 `scripts/ablation.py --report` 生成，数字全部来自 `eval/ablation.json`。
> 这份报告回答的是「关掉某个模块之后，什么变了」，**不是**「系统有多准」。

## 离线对照（不需要模型）

图：4988 节点 / 5128 边。

| 对照项 | 问题 | 判读 |
|---|---|---|
| `alias_merge` | 别名合并对连通性的贡献 | 合并 407 个节点，分量 1256 → 1087，最大分量 2905 → 2829 |
| `type_constraint` | 八类实体约束的遵守率（上界，非因果） | 552 个节点（11.1%）不在八类之内，牵动 622 条边；约束是提示词，没有代码闸门 |
| `curated_ontology` | 策展本体对六个维度的贡献 | reservoir 286→316 · fault 62→87 · caprock 18→58 · wellbore 107→132 · monitoring 186→213 · regulation 149→171 |

### 别名合并的连通性影响

| 口径 | 节点 | 边 | 连通分量 | 最大分量 | 占比 | 孤立节点 |
|---|---:|---:|---:|---:|---:|---:|
| 合并后（当前） | 4419 | 4763 | 1087 | 2829 | 64.0% | 822 |
| 不合并 | 4826 | 4850 | 1256 | 2905 | 60.2% | 929 |

差：分量 169、最大分量 -76、合并节点 407。

## 在线对照（需要 Ollama）

| 组 | arm | 用例集 | 用例数 | 通过 | 失败 | 闸门状态 | 物理状态 | 重写次数 |
|---|---|---|---:|---:|---:|---|---|---:|
| `baseline` | `baseline` | golden | 1 | 1 | 0 | passed | unverifiable | 1 |
| `baseline@probe` | `baseline` | probe | 1 | 0 | 1 | failed | violation | 2 |
| `no_physics_gate@probe` | `no_physics_gate` | probe | 1 | 1 | 0 | passed | pass | 2 |
| `no_rewrite` | `no_rewrite` | golden | 1 | 1 | 0 | passed | warning | 1 |
| `no_rewrite@probe` | `no_rewrite` | probe | 1 | 0 | 1 | failed | violation | 1 |

环境：本地模型（cloud 客户端已禁用，三组同口径）

注意：小样本诊断，不是基准。看的是闸门是否仍然拦得住，不是 Faithfulness 的绝对值。

## 复现

```bash
python scripts/ablation.py --offline --write          # 离线三组
python scripts/ablation.py --online --probe --arm baseline --write
python scripts/ablation.py --online --probe --arm no_physics_gate --write
python scripts/ablation.py --online --probe --arm no_rewrite --write
python scripts/ablation.py --report docs/ablation-report.md
```
