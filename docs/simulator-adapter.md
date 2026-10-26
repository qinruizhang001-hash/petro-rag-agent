# 储层模拟器适配契约（`ccus-sim-result/2`）

> **不兼容更新**：`/1` 没有记录输入算例；把旧结果文件传给新算例也会被当成模拟器结论。
> `/2` 要求 `case` 包含 `SimulationCase` 的全部字段，运行时逐字段对照。
> 旧文件不会被猜测升级；必须从原始输入与模拟结果重新生成。契约通过只表示格式与输入匹配，**不代表经过真实油田验证**。

本文件是 `utils/reservoir_backend.py` 的外部契约说明。代码里两处引用它
（`ExternalResultBackend` 的类说明与 `BackendUnavailable` 的消息），所以它
不是可选的补充材料 —— 缺了它，报错信息就是在指一个不存在的地方。

## 为什么需要契约，而不是"接一个模拟器"

没有契约时，对接的真实问题不是"跑不起来"，而是**跑起来了但单位不对**：

| 模拟器 | 压力 | 时间 | 饱和度 |
|---|---|---|---|
| ECLIPSE | bar / psi | day / month / year | 小数 |
| CMG GEM | kPa | day | 小数 |
| TOUGH2 | Pa | second | 小数 |
| 各种后处理脚本 | MPa | day | **百分数** |

psi 与 bar 差 14.5 倍，百分数与小数差 100 倍。两类错误都不会抛异常，只会
让"盖层封得住"变成"封不住"（或者反过来）。所以契约的第一条不是"字段齐不齐"，
而是**单位必须显式声明，且不做换算**。

## 输入契约：`SimulationCase`

```python
SimulationCase(
    name="case-A",
    depth_m=2500.0,
    injection_pressure_bar=250.0,
    temperature_celsius=60.0,
    thickness_m=50.0,          # 默认 50
    porosity=0.2,              # 默认 0.2
    permeability_md=100.0,     # 默认 100
    plume_radius_m=500.0,      # 默认 500
    injection_rate_m3_per_day=None,
    rho_brine=1030.0,
    fault_dip_deg=60.0,
    fault_azimuth_deg=90.0,
)
```

`validate()` 返回**全部**问题而不是抛第一个 —— 否则改一个字段跑一次，
一轮下来要跑五遍。校验范围与理由：

| 字段 | 范围 | 超出时说明什么 |
|---|---|---|
| `depth_m` | 50 – 9000 m | 不是"合理深度"的争论，而是单位错的信号（2500 写成 2.5） |
| `temperature_celsius` | −20 – 200 ℃ | 同上（华氏度、开尔文混用） |
| `thickness_m` | 1 – 1000 m | 0 会让后续除法炸掉，1000 以上多半是 mm |
| `porosity` | 0.01 – 0.45 | 0.2 vs 20（百分数）在这里就会被拦下 |
| `permeability_md` / `plume_radius_m` / 注入速率 | > 0 | 负值无物理意义 |
| `fault_dip_deg` | (0, 90) | 90° 是垂直面，剪切分量为零，不在模型适用域内 |

## 输出契约：`ccus-sim-result/2`

外部适配器需要产出一个 JSON 文件：

```json
{
  "schema": "ccus-sim-result/2",
  "case": {
    "name": "case-A", "depth_m": 2500.0, "injection_pressure_bar": 250.0,
    "temperature_celsius": 60.0, "thickness_m": 50.0, "porosity": 0.2,
    "permeability_md": 100.0, "plume_radius_m": 500.0,
    "injection_rate_m3_per_day": null, "rho_brine": 1030.0,
    "fault_dip_deg": 60.0, "fault_azimuth_deg": 90.0
  },
  "backend": "eclipse-2021",
  "converged": true,
  "units": {"pressure": "bar", "length": "m", "time": "day", "saturation": "fraction"},
  "time_days": [0, 30, 90, 180, 365],
  "co2_saturation": [0.0, 0.12, 0.24, 0.31, 0.36],
  "pressure_bar": [253.0, 262.4, 271.9, 278.5, 283.1],
  "co2_column_height_m": 18.5,
  "entry_pressure_bar": 43.3,
  "notes": ["网格 50×50×10，相对渗透率取 Brooks-Corey"]
}
```

逐项校验规则（全部来自 `SimulationResult.validate()`）：

1. **`schema` 必须精确等于** `ccus-sim-result/2`。不做版本兼容猜测 —— 字段
   含义变了却"看起来还能解析"，比直接拒绝更危险。
2. **`case` 必须与本次 `SimulationCase` 完全相同**（含可选字段的实际值）。
   这只能证明文件声称属于相同输入，不能证明输出确实由模拟器算出。
3. **`converged` 必须是 JSON 布尔值 `true`**；字符串 `"false"` 不是 true。
   `false` 时结果里的数字仍然存在、看起来完全
   正常，但那是迭代失败后的残值。把它当结论写进报告，等于用噪声做安全判定。
   这是 fail-closed，与质量闸门同一条原则：拒绝，而不是打个警告放行。
4. **`units` 四项必须全部显式给出且等于约定值**。缺一个就拒绝。不做换算，
   因为"猜单位"是这类集成里最容易出错的一步 —— psi 与 bar 两个数字都
   "看起来合理"。
5. **时间严格递增且有限**，长度一致。回退的时间轴说明后处理脚本拼接错了。
6. **饱和度是有限的 [0, 1] 内数**。出现 20 说明给的是百分数。
7. **压力必须为有限正数**。NaN/Infinity（包括 Python JSON 默认宽容的非标准写法）直接拒绝。

## 两种后端，一个是真的

| 后端 | `kind` | 能做什么 | 不能做什么 |
|---|---|---|---|
| `AnalyticalScreeningBackend` | `analytical` | 组合 `relperm` + `flow1d` + `caprock` 真的算出数，用于证明契约可满足 | 不是储层模拟器：一维、不可混、无重力分异、无溶解 |
| `ExternalResultBackend` | `external` | 按契约校验并消费外部 JSON | **不运行**模拟器；没有在线耦合、没有时间步握手 |

两者都被**同一套** `SimulationResult.validate()` 检查。这一点是刻意的：
契约只有被两侧同时使用，才不是"只有文档在声称"。

`run_case()` 的返回明确区分谁算的、走的哪条通路：

```python
{
  "ok": True,
  "backend": "eclipse-2021",     # 谁算的（外部结果里写的名字）
  "adapter": "external-json",    # 走哪条通路
  "degraded": False,
  "degraded_reason": None,
  "result": {...},
  "summary": "...",
}
```

指定外部结果但文件缺失或不合法时，**默认 `ok=False`，不把解析结果伪装成成功**。
调用方确实允许退回筛查时，显式传 `allow_screening_fallback=True`；这时
`degraded=True` 且 `degraded_reason` 写明缺什么、怎么补，返回的 `backend`
仍是 `analytical-screening`。

## 写一个适配器：三步

1. 让模拟器输出上面那个 JSON：`case` 可用 `dataclasses.asdict(case)` 生成，
   单位在适配器侧明确换算，不在这里猜。
2. 用 `ExternalResultBackend(path).validate(case)` 探测 —— 它返回问题列表，
   不抛异常，适合放进 CI。
3. 通过后接进 `run_case(case, external_path=path)`。

最小自检：

```python
from utils.reservoir_backend import ExternalResultBackend, SimulationCase

case = SimulationCase(name="case-A", depth_m=2500, injection_pressure_bar=250,
                      temperature_celsius=60)
problems = ExternalResultBackend("out/eclipse_case_A.json").validate(case)
print("\n".join(problems) or "契约通过")
```

## 可重复的跨模型对照（尚无真实参考数据）

`compare_screening_to_external` 是**对照工具，不是已经完成的验证**。在有独立
模拟器结果时，先让其通过上述 case/单位/收敛校验，再在文件里声明：

```json
"comparison_basis": {
  "pressure_bar": "injection_well_bottomhole",
  "co2_saturation": "same_control_volume_mean"
}
```

这两个声明必须由导出人员核对：若数值模拟器输出的是**储层平均压力**而筛查模型是
**井底压力**，或两条饱和度曲线覆盖不同体积，即使数值相近也不能比较。
函数无法独立证明声明是真的，结果里因此始终是
`"source_identity_verified": false`，**不颁发场地级通过证书**。

```python
from utils.reservoir_backend import compare_screening_to_external

result = compare_screening_to_external(
    case, "out/eclipse_case_A.json",
    pressure_tolerance_bar=2.0,  # 示意，实际容差须在看结果前由工程师确定
    saturation_tolerance=0.02,   # 示意，不是本项目的内置及格线
)
# status: rejected / not_comparable / compared
```

返回两条曲线在筛查模型全部时间点上的 MAE 和最大绝对误差；只有参考时间轴**完整覆盖**
筛查时间轴时才线性插值，绝不外推或挑一段有利的重叠区间。容差没有默认值，比较是否
落在用户容差内也不代表模型通过了真实油田验证。仓库现有测试仅用**合成曲线**确认
比较器逻辑，**没有**独立 CMG/ECLIPSE 输出或现场监测数据。

## 这个契约故意不包含的东西

- **在线耦合**：没有"调一次求解器、拿到一步、反馈压力再调下一次"的握手。
  那需要异步执行、预算控制和求解器进程管理，属于另一个量级的工作。
- **网格与相态信息**：只取时间序列上的标量（饱和度、压力、柱高）。三维场
  数据不进入报告，也就不进入契约 —— 传进来只会被丢掉。
- **不确定性**：模拟器结果按确定性输入处理。要把参数不确定性传播进去，
  需要在外层做多次 `run_case` 并汇总，而不是在契约里加分布字段。

这三条写在 Roadmap 里，而不是写成"已支持储层模拟器"。
