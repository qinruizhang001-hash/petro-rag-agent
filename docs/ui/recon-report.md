# GitHub 开源项目侦察报告

**对象**：CCUS 全链条工业 AI 界面（贾维斯式）
**核查日期**：2026-10-27
**核查方式**：GitHub REST API（`/repos`、`/search/repositories`、`/git/trees`、`/contents`、`/readme`）+ npm registry 逐项实测。表中星数、许可证、最后推送时间均为当日接口返回值，不是记忆或估算。

---

## 0. 核查口径

四项判据，任何一项不合格就不进推荐：

| 判据 | 阈值 | 为什么是硬门槛 |
|---|---|---|
| **许可证可商用** | MIT / Apache-2.0 / ISC / BSD | 这是要卖给油企的产品。GPL/AGPL 会让你的交付物必须开源，无许可证等于"保留所有权利"，法务过不去 |
| **维护活跃度** | 6 个月内有提交 | 一次性发布的仓库等于"作者交完作业就走了"，你 fork 之后所有 bug 都是你的 |
| **形态匹配** | 是组件/库，不是单体应用 | 抄一个单体应用的代码比自己写还慢，而且它的架构假设会绑住你 |
| **工业适配** | 不是游戏/个人助理风格 | 见第 2 节的判断 |

一条附加规则：**没有 `owner/repo` 形式的"项目名"不进入推荐**。因为无法核查许可证，而许可证决定这产品能不能卖。你给的线索里有 3 条属于这种，见下一节。

---

## 1. 结论先行

**三句话：**

1. **A 类（HUD 外壳）有可商用资产，但没有一个能直接当依赖装。** 唯一合格的那个必须 fork 进私有仓库自持，且要拆掉它的地图依赖。
2. **B 类（语音/对话）有一个明确的赢家**：`elevenlabs/ui`，MIT、2432 星、11 天前还在推、registry 里有 50 个可直接 `add` 的条目。**但它只支持 React** —— 这一条直接替你决定了技术选型。
3. **C 类（CCUS 业务面板）在 GitHub 上是空集。** 不是没搜到，是搜到了但全是课程作业级别的东西。这部分必须自建，建在通用工业可视化底座上。

**一个必须先说的坏消息**：你简报里的 4 条"重点核查对象"，有 3 条无法核查（没有 owner/repo），1 条许可证致命。这说明那份线索清单里的 `[reference:N]` 引用不可信——**别把它当调研结果用**。

---

## 2. 你给的线索逐条裁决

| # | 线索原文 | 能否核查 | 实测结果 | 裁决 |
|---|---|---|---|---|
| 1 | `xxjun9527/jarvis-holographic` | ✅ 存在 | MIT · 44★ · 24 fork · TypeScript · 2025-11-25 建库，**最后提交 2025-12-16** | 视觉参考，不作依赖 |
| 2 | `@//masterdeepak15/jarvis-ui` | ✅ 存在（路径写法有误，应为 `masterdeepak15/jarvis-ui`） | **MIT**（GitHub 显示 None，见下）· 2★ · TypeScript · 69 个组件 · npm v1.1.4 @ 2026-07-02 | **可用，但必须 fork** |
| 3 | `kaustav991a/J.A.R.V.I.S` | ✅ 存在 | **GPL-3.0** · 0★ · Python · 2026-10-24 有提交 | **排除**（许可证 + 形态都不对） |
| 4 | Blink `JARVIS HUD Interface` | ❌ 无法核查 | 无 owner/repo。搜 `jarvis hud blink` 全网 1 条结果、0★ | 不采信 |
| 5 | `ElevenLabs UI` | ✅ 存在 | MIT · 2432★ · 176 fork · **2026-10-16 推送** · 50 个 registry 条目 | **采用**（但"支持 Vue"是错的） |
| 6 | `JHCWColin/AIChat-UI` | ✅ 存在 | MIT · 5★ · HTML 单页 49.9 MB · 2026-10-25 推送 | 实现参考，不作组件库 |
| 7 | `AI-Based-Carbon-Capture` | ❌ 无法核查 | 无 owner/repo。`carbon capture dashboard` 全站 19 个仓库，**最高 1★** | 不采信 |
| 8 | `CO₂-EOR Digital Twin` | ❌ 无法核查 | 无 owner/repo。`co2 eor digital twin` **命中 0** | 不采信 |

### 2.1 关于 `jarvis-ui` 的许可证，需要说清楚

GitHub API 返回 `license: None`，README 却挂着 MIT 徽章。这不是作者写错，是 **GitHub 的许可证探测器只扫仓库根目录**，而它的 LICENSE 在 `packages/jarvis-ui/LICENSE`（monorepo 子包）。我直接读了文件：

```
MIT License
Copyright (c) 2026 Deepak Chougale
```

npm registry 上 `@masterdeepak15/jarvis-ui` 的元数据 `license` 字段也是 `MIT`。**所以它确实可以商用。**

但这条恰好说明一件事：**许可证不能看徽章、也不能看 GitHub 的 license 字段**。你以后每引入一个依赖，都要点开 LICENSE 文件读第一段。

### 2.2 关于 `kaustav991a/J.A.R.V.I.S`，两个独立的排除理由

- **GPL-3.0**：你把它作为组件链进产品，整个产品就是衍生作品，卖的时候必须一并开源。对 B 端私有化交付这是死路。
- **形态不对**：它做的是"唤醒词 → 流式 STT → 推理 → 本地 TTS"的**后端管线**，不是 UI 组件。就算换成 MIT，它能给你的也只是 Python 侧的语音链路代码，跟界面无关。

**即使许可证没问题，这条线索也解决不了你界面上的任何问题。** 这是"看起来很像贾维斯"和"能拿来用"之间的典型差距。

---

## 3. A 类：HUD / 科幻界面框架

按星数排序，全部实测：

| 仓库 | 许可证 | ★ | 语言 | 最后推送 | 形态 | 裁决 |
|---|---|---|---|---|---|---|
| `eadmin2/jarvis_ai` | MIT | 197 | Python | 2026-06-13 | Hermes Agent 的语音外壳（65 KB） | 参考 |
| `Gmasterzhangxinyang/jarvis-amber-protocol-hud` | **None** | 64 | HTML | 2025-12-02 | 单文件 HUD，支持接 Dify | **排除** |
| `xxjun9527/jarvis-holographic` | MIT | 44 | TypeScript | 2025-12-16 | React 18 全息界面 | 参考 |
| `jincocodev/openclaw-jarvis-ui` | ISC | 39 | JavaScript | 2026-02-22 | Three.js orb + chat + 系统监控 | 参考 |
| `MultiX0/jarvis` | Apache-2.0 | 23 | **Dart** | 2026-09-04 | Flutter Windows 桌面 HUD | **排除**（栈不符） |
| `masterdeepak15/jarvis-ui` | **MIT** | 2 | TypeScript | 2026-07-01 | **React 组件库，69 个** | **采用（fork）** |
| `sam3gp8/jarvis-aio` | MIT | 31 | — | 2026-10-26 | 智能家居管家 | 排除 |
| `BlaineONG/HomeCore` | MIT | 3 | — | 2026-03-14 | Home Assistant 主题 | 排除 |

### 3.1 `masterdeepak15/jarvis-ui` —— 唯一合格项，但有三个坑

**实测**：69 个非测试组件（131 个 `.tsx` 含测试），peer 依赖 `react >=18`，依赖 `recharts` / `leaflet` / `react-leaflet` / `@react-google-maps/api` / `react-day-picker`。npm v1.1.4 发布于 2026-07-02，仓库最后推送 2026-07-01（包比仓库略新）。

**可直接复用的模块**（真正对工业界面有用的）：

```
外壳   JHudFrame  JHudFrameCard  JHudCanvas  JHudBar  JHudLabel  JHudClock
指标   JStatCard  JKpiTicker  JArcMeter  JGaugeChart  JProgress  JStatusPill  JBadge
图表   JLineChart  JBarChart  JDonutChart  JRadarChart  JSparkline  JHeatmap
交互   JRadialMenu  JRadialItem  JNodeGraph  JCommandPalette  JControlPanel
音频   JWaveform  JOrb
数据   JTable  JDataRow  JAlert  JActivityFeed
```

**坑 1：69 个组件里超过一半是"浏览器里的假操作系统"。**

`JDesktop` `JWindow` `JWindowManager` `JTaskbar` `JStartMenu` `JBootScreen` `JOSThemeProvider` `JFileExplorer` —— 这是一整套仿 Windows 的桌面壳。**直接搬进工业产品，会让你的界面看起来像游戏启动器**，正好撞上你要求里禁掉的"夜店赛博风"。这一层必须整体丢掉，只保留上表那 25 个左右。

**坑 2：`@react-google-maps/api` 依赖。**

它的 Tactical Map 页面用 Google Maps。两个致命问题：

- 中国境内 `maps.googleapis.com` 不可用；
- 私有化部署在内网，出不了公网。

**这个依赖必须删掉。** 换 Leaflet + 自建瓦片（或天地图/高德企业版）。注意 `leaflet` 本身是 BSD-2，没问题，但**瓦片源是服务，不是代码**，要单独采购或自建。

**坑 3：2 星、单人、4 个月没更新。**

这不是"不能用"，是"不能当依赖用"。`npm i` 之后上游不再修 bug，你的生产环境就被冻住了。**正确做法是 fork 进自己的 monorepo，删掉不需要的组件和地图依赖，之后按自己的节奏改。** MIT 允许这么做，而且这是唯一负责任的做法。

### 3.2 `xxjun9527/jarvis-holographic` —— 好看，但是死的

React 18 + TypeScript + LLM 对话 + 分层 HUD 面板，MIT。**建库 2025-11-25，最后提交 2025-12-16，三周后就没人了**，至今 10 个月。

它的价值是**视觉参考**：可以打开它的 demo 截图，看它怎么处理 HUD 分层面板与光环动效，然后自己在 shadcn 上重画。不要 clone 它的代码——一个三周的 demo，架构基本没有，你会花更多时间删它的东西。

### 3.3 排除项的理由

- **`jarvis-amber-protocol-hud`（64★）**：许可证 `None`。中文项目、单 HTML 文件、能接 Dify，看起来挺合适——**但无许可证就是不能商用**。64 星在这里没有意义。可以看它的视觉，不能拿它的代码。
- **`MultiX0/jarvis`（Apache-2.0）**：Dart/Flutter，Windows 桌面应用。你不可能把 Flutter widget 塞进 React 网页。许可证没问题，栈不对。
- **`eadmin2/jarvis_ai`（197★，本类最高）**：MIT，但只有 65 KB，是给 Hermes Agent 套的 Python 语音壳（Whisper STT + ElevenLabs TTS）。星数高是因为"贾维斯"这个词的流量，不是因为它是 UI 库。

---

## 4. B 类：语音交互组件

### 4.1 `elevenlabs/ui` —— 本报告唯一的"直接采用"级发现

| 项 | 实测值 |
|---|---|
| 许可证 | **MIT**（可商用，可闭源分发） |
| 星数 | 2432 |
| 最后推送 | 2026-10-16（11 天前） |
| 底座 | shadcn/ui + Tailwind |
| 安装方式 | `npx shadcn@latest add https://ui.elevenlabs.io/r/all.json` |
| 前置 | Node 18+、已 init shadcn/ui、已配 Tailwind |
| registry 条目 | 50（含 demo 与 hooks，实际组件约 24 个） |

**可复用的组件**（按工业界面的用途重排，不是按它的分类）：

```
状态指示   orb（待机/聆听/思考）  matrix  shimmering-text
波形       waveform  live-waveform  bar-visualizer  scrub-bar
对话       conversation  message  response  conversation-bar
输入       speech-input  mic-selector  voice-picker  voice-button
回放       audio-player  transcript-viewer
整块模板   voice-chat-01 / 02 / 03  voice-form-01  voice-nav-01
```

**必须纠正的一点：它不支持 Vue。**

你的简报写"支持 React / Vue / Next.js"。实测它建在 **shadcn/ui** 上，而 shadcn/ui 是 React + Radix 的产物，没有 Vue 版本。它的 README 通篇只说 Next.js。**这是一个会直接影响技术选型的错误信息，不能带进决策。**

**必须区分的另一点：组件库 ≠ 云服务。**

- `elevenlabs/ui` 是**代码**，MIT，可以离线打包进私有化部署，**不收钱、不联网**。
- ElevenLabs 的**语音合成服务**是云 API，按字符计费，私有化场景用不了（内网出不去，而且把工况数据发到境外 API 本身就是合规问题）。

**结论：组件白拿，服务必须换成本地 TTS**（Piper / CosyVoice / GPT-SoVITS 之类）。这个区分如果不写清楚，交付到客户现场会变成一个"为什么语音功能打不开"的事故。

### 4.2 `JHCWColin/AIChat-UI` —— 值得读，不值得装

MIT · 5★ · 2026-10-25 活跃 · HTML 单页 **49.9 MB**。

它的价值在**架构思路**，不在代码：不依赖 realtime 模型，用 HTTP + STT + TTS + VAD 拼出"实时感"。这正是私有化部署该走的路——OpenAI Realtime 那种 WebSocket 音频流在客户内网里没有可用端点。

`49.9 MB` 的 HTML 说明它把所有资源内联了，这种形态没法当组件库用。**读它的 VAD 断句与打断逻辑，然后自己写。**

---

## 5. C 类：CCUS / 碳捕集 / 工业大屏 —— 空集

这一节是本报告最重要的一节，因为它推翻了你的一个前提。

### 5.1 搜索证据

| 查询 | 命中总数 | 最高星数 | 有许可证的 | 结论 |
|---|---|---|---|---|
| `carbon capture dashboard` | 19 | **1** | 少数 MIT | 全部为个人作业 |
| `carbon capture performance dashboard` | **0** | — | — | 无 |
| `co2 eor digital twin` | **0** | — | — | 无 |
| `carbon capture monitoring app` | 1 | 0 | Apache-2.0 | 空壳 monorepo |
| `ccus` | 2673 | — | — | **被 Homematic CCU（智能家居）污染**，与碳捕集无关 |
| `sci-fi hud dashboard` | 少量 | 24 | 多数 `None` | 个人主页/主题 |

`carbon capture dashboard` 的 19 个结果长这样：

```
1★  defnalk/mea-dashboard            MEA 碳捕集中试装置实时看板（无许可证）
0★  ideniaa/Carbon-Capture-Dashboard 公司内部看板（无许可证）
0★  efugh89/carbon-capture-dashboard 无描述
0★  Chris-pter/Global-Carbon-Capture-Storage-Dashboard   Power BI 可视化作业
0★  Ayush5474/...-using-Tableau      Tableau 作业
0★  jask89738/...-for-India          区域统计看板
```

### 5.2 为什么是空集（这不是偶然）

三个结构性原因：

1. **CCUS 是项目制 B 端生意。** 真实数据在业主（油田、水泥厂）的 DCS/ historian 里，脱敏都做不到，没人会开源。开源出来的必然是假数据。
2. **看得懂的人不写前端。** 会做 CCUS 工艺建模的人在做工艺包，不会去写 React。
3. **工业界面的"复用"本来就不发生在业务层。** 没人开源"碳捕集看板"，但所有人都在用 ECharts 画仪表盘。**真正该复用的是底座，不是业务面板。**

### 5.3 结论：C 类复用率 0%，但底座复用率 80%

C 类必须自建。但要建在下面这些底座上——这些是实测过的、许可证干净的、活跃的：

| 库 | 许可证 | ★ | 最后推送 | 在本产品里的职责 |
|---|---|---|---|---|
| `apache/echarts` | Apache-2.0 | 67608 | 2026-10-13 | 全部二维图表、时序、热力图 |
| `echarts-gl` (npm) | **MIT** | — | v2.1.0 @ 2026-05-28 | 地质剖面 3D surface（**比 three.js 省 10 倍工作量**） |
| `mrdoob/three.js` | MIT | 117010 | 2026-10-27 | 注入粒子、管网流光 |
| `@react-three/fiber` (npm) | MIT | — | v9.8.2 @ 2026-10-14 | three.js 的 React 声明式封装 |
| `xyflow/xyflow` | MIT | 38791 | 2026-10-26 | 管网拓扑图 / 节点图（比 jarvis-ui 的 JNodeGraph 成熟得多） |
| `shadcn-ui/ui` | MIT | 126638 | 2026-10-26 | 表单、弹窗、无障碍底座 |
| `antvis/G6` | MIT | 12353 | 2026-10-22 | 大规模图布局（xyflow 不够时用） |
| `visgl/deck.gl` | MIT | 14707 | 2026-10-27 | 井位/管线地理图层（比 Leaflet 更适合大数据量） |
| `CesiumGS/cesium` | Apache-2.0 | 15884 | 2026-10-27 | 3D 地球（**本产品不需要，地形瓦片要付费**） |
| `recharts` (npm) | MIT | — | v3.10.1 | 轻量图表（jarvis-ui 内部已用它） |
| `@tanstack/react-table` | MIT | — | — | 工程师模式的高密度数据表（**必须用，别自己写虚拟滚动**） |

### 5.4 两个"看起来能用、实际有毒"的平台

| 平台 | 许可证 | 为什么不能用 |
|---|---|---|
| `grafana/grafana` | **AGPL-3.0** | 77537★，工业看板的事实标准。但 AGPL：只要你把**修改版**以网络服务形式提供，就必须公开源码。私有化交付给客户 = 分发，改一行就得开源整个 Grafana。**除非你不改它、只做纯配置**——而你的产品不可能不改。 |
| `thingsboard/thingsboard` | **NOASSERTION** | 社区版 Apache-2.0，但仓库里混着商业版模块，许可证探测器判不出来。要逐文件读，风险高于收益。 |

**Grafana 这条值得单独强调**：它是这个领域最容易踩的坑。星数、生态、视觉都完美，唯独许可证把商业闭环堵死了。

---

## 6. Top 3 推荐组合

### 6.1 主方案（推荐）

> **A `jarvis-ui`（fork 后裁剪） + B `elevenlabs/ui` + C 自建（ECharts / React Flow / three.js / TanStack Table）**

```
┌─────────────────────────────────────────────────────────────┐
│  A  HUD 外壳层          jarvis-ui (fork, MIT)                │
│     JHudFrame / JHudCanvas / JHudBar / JStatCard             │
│     JRadialMenu / JStatusPill / JGaugeChart                  │
│     ✂ 删除 JDesktop / JWindow / JTaskbar / JStartMenu        │
│     ✂ 删除 @react-google-maps/api                            │
├─────────────────────────────────────────────────────────────┤
│  B  语音与对话层        elevenlabs/ui (MIT, registry add)     │
│     orb (状态环) / waveform / live-waveform                   │
│     conversation-bar / transcript-viewer / speech-input      │
│     ⚠ 组件是代码，TTS 服务换成本地（Piper/CosyVoice）          │
├─────────────────────────────────────────────────────────────┤
│  C  业务面板层          自建                                  │
│     报警面板      ← shadcn Card + TanStack Table             │
│     指标卡/时序   ← ECharts                                   │
│     地质剖面 3D   ← echarts-gl surface                        │
│     注入粒子      ← @react-three/fiber                        │
│     管网拓扑      ← xyflow (React Flow)                       │
│     审批状态机    ← shadcn Dialog + 自己写的状态机             │
├─────────────────────────────────────────────────────────────┤
│  D  底座              shadcn/ui + Tailwind + Radix (MIT)      │
└─────────────────────────────────────────────────────────────┘
```

**为什么这个组合成立：**

1. **同一生态，零胶水。** A 的 peer 依赖是 `react>=18`，B 建在 shadcn 上，C 的库全是 React 生态。四层可以叠在同一个 Vite/Next 项目里。
2. **许可证全绿。** MIT × 4 + Apache-2.0 × 1（ECharts），没有任何 GPL/AGPL。可以闭源卖给油企。
3. **每一层都能独立替换。** A 视觉过时了换 A，不影响 C；B 想换语音方案，只动 B。这是分层的意义。
4. **C 层自建不是浪费。** 你的业务面板要接的是 `ops/` 那套真实契约（`severity` 五级 + `indeterminate`、`provenance` 阈值出处、`[alarm:A-...]` 引用），市面上没有任何库能替你表达这些。**这恰恰是产品壁垒所在。**

### 6.2 备选 1：保守方案（如果工期紧）

> **放弃 HUD 外壳，纯 shadcn/ui + ECharts 自绘**

- 优点：一周能出可用界面；零 fork 维护成本；无障碍与暗色模式开箱即用；视觉靠 CSS 渐变 + 网格纹理也能做出"工业感"。
- 缺点：没有 `JHudFrame` 那种斜切角、扫描线、环形刻度，视觉记忆点弱。
- **适用**：先给客户看能不能用，再决定要不要"贾维斯"。

**这个方案值得认真考虑。** 工业软件的第一版通常不需要酷，需要准。

### 6.3 备选 2：激进方案（如果视觉是卖点）

> **jarvis-ui 全量 fork + 深度改造 + 自建地质 3D**

- 优点：最接近"贾维斯"，汇报大屏的冲击力最强。
- 缺点：69 个组件里要删一半、改一半；2 星项目的代码质量需要逐个审；工期至少翻倍。
- **适用**：投标/演示阶段，视觉是差异化武器时。

### 6.4 明确不推荐

| 方案 | 原因 |
|---|---|
| 基于 Grafana 二次开发 | AGPL，商业闭环堵死 |
| 引入 `kaustav991a/J.A.R.V.I.S` | GPL-3.0 + 形态不符 |
| 引入 `jarvis-amber-protocol-hud` | 无许可证 |
| 自研图表库 / 自研虚拟滚动表格 | 已有 Apache-2.0 / MIT 的成熟实现，重写是纯浪费 |
| Cesium 3D 地球 | 本产品看的是地下剖面，不是地球；地形瓦片要付费 |

---

## 7. 组件映射表

三档：**直接用**（装完就能上）/ **二次封装**（拿来改，改的是样式与契约适配）/ **从零写**（没有可复用的，或复用反而更慢）。

### 7.1 外壳与布局

| 模块 | 档位 | 来源 | 说明 |
|---|---|---|---|
| 三栏骨架 | 从零写 | — | CSS Grid，两行代码的事 |
| HUD 边框/斜切角 | 直接用 | `jarvis-ui` `JHudFrame` | fork 后改描边色 |
| 顶部状态条 | 二次封装 | `JHudBar` + `JHudClock` | 加数据新鲜度（`max(ts)` 而非系统时钟） |
| 命令面板（Ctrl+K） | 二次封装 | `JCommandPalette` | 改成"跳转到测点/告警" |
| 双模切换开关 | 从零写 | — | 本质是 CSS 变量切换 + `prefers-reduced-motion` |
| 主题令牌（深海蓝/工业灰） | 从零写 | Tailwind config | 客户换肤靠这个，见 8.3 |

### 7.2 指标与图表

| 模块 | 档位 | 来源 | 说明 |
|---|---|---|---|
| 关键指标卡（带阈值红线） | 二次封装 | `JStatCard` | **必须加**阈值刻度线与 `provenance` 悬浮说明 |
| 时序图 | 直接用 | ECharts `line` + `markLine` | 阈值线用 `markLine`，不要画在自己 canvas 上 |
| 环形/仪表 | 直接用 | `JArcMeter` / `JGaugeChart` | 捕集率、纯度用这个 |
| 热力图（MVA 多测点） | 直接用 | ECharts `heatmap` | 5 个 MVA 测点 × 时间 |
| 高密度数据表 | 直接用 | `@tanstack/react-table` + shadcn Table | 工程师模式主力，**别自己写虚拟滚动** |
| 3D 地质剖面 | 二次封装 | `echarts-gl` `surface` | 比 three.js 省 10 倍工作量；层位用 series 分色 |
| CO₂ 注入粒子 | 从零写 | `@react-three/fiber` | 只做状态指示，不遮挡数据 |
| 管网拓扑流光 | 二次封装 | `xyflow` + 自定义 edge | xyflow 的 edge 动画 API 够用 |

### 7.3 告警与归因（产品壁垒所在）

| 模块 | 档位 | 来源 | 对齐的真实字段 |
|---|---|---|---|
| 告警列表 | 二次封装 | `JAlert` / `JActivityFeed` | `alarm_id` `sensor` `severity` `severity_label` `ts` `value` `direction` |
| 五级 + 不可判定 色带 | 从零写 | — | `SEVERITIES` + `INDETERMINATE`，**6 个状态，不是 5 个** |
| 判定依据展开 | 从零写 | — | `rationale: list[str]`，逐条规则展示 |
| 阈值出处悬浮 | 从零写 | — | `provenance`（"DNV-RP-F104 §x" / "项目设计文件" / "运营方风险接受准则"） |
| 归因候选卡 | 二次封装 | shadcn Card | `cause` `confidence` `evidence` `ruled_out` `ruled_out_reason` |
| **已排除原因折叠区** | 从零写 | — | `ruled_out=true` 的条目必须可见，否则读者怀疑漏了什么 |
| 处置建议 | 直接用 | `JDataRow` | `actions: [{action, owner}]` |
| 盲区提示条 | 从零写 | — | `unconfigured` / `no_data` 两个列表，**必须与"正常"视觉区分** |

### 7.4 语音与对话

| 模块 | 档位 | 来源 |
|---|---|---|
| 语音状态环（待机/聆听/思考/回复） | 直接用 | `elevenlabs/ui` `orb` |
| 实时波形 | 直接用 | `waveform` / `live-waveform` |
| 对话气泡 | 直接用 | `conversation` `message` `response` |
| 输入条 + 麦克风选择 | 直接用 | `conversation-bar` `mic-selector` `speech-input` |
| 转写文本 | 直接用 | `transcript-viewer` |
| 语音意图 → 结构化动作 | 从零写 | — | **绝不由语音直接触发** |
| 高危操作二次确认 | 从零写 | shadcn `AlertDialog` | 见 8.4 |

### 7.5 报告与审批

| 模块 | 档位 | 来源 | 对齐的真实字段 |
|---|---|---|---|
| 报告预览（Markdown） | 直接用 | `react-markdown` | 正文含 `<!-- strict -->` 区块 |
| 引用角标 `[alarm:A-...]` | 从零写 | — | `Citation.render()` 输出，点击跳到告警 |
| 审批状态机按钮 | 从零写 | shadcn Button | `draft → submitted → approved / rejected → reopened` |
| 审批轨迹时间线 | 二次封装 | `JActivityFeed` | `trail: list[dict]`，只追加 |
| 引用检查失败提示 | 从零写 | — | `problems: list[str]`，非空时禁用提交按钮 |

---

## 8. 风险提示

### 8.1 许可证

| 风险 | 具体项 | 处置 |
|---|---|---|
| **AGPL 传染** | Grafana（77537★） | 不用。或仅作为**未修改的独立服务**通过 iframe 嵌入，且法务书面确认 |
| **GPL 传染** | `kaustav991a/J.A.R.V.I.S` | 不用 |
| **无许可证 = 保留所有权利** | `jarvis-amber-protocol-hud`(64★) | 不用。64 星不构成授权 |
| **探测失败≠无证** | `jarvis-ui` 的 LICENSE 在子目录 | 每个依赖**手动读 LICENSE 第一段**，不看徽章 |
| **混合许可证** | ThingsBoard（NOASSERTION） | 不用 |
| **字体授权** | DIN / DIN Next 是 Monotype 商业字体 | 用 `IBM Plex Mono`(OFL) / `JetBrains Mono`(OFL) / `Roboto Mono`(Apache-2.0) 替代。B612（空客座舱字体）也可商用，但需自行复核 OFL 条款 |
| **图标** | 各家图标包许可证不同 | 统一用 `lucide`（ISC），别混用 |

### 8.2 私有化部署（这是最容易翻车的一类）

| 风险 | 触发条件 | 处置 |
|---|---|---|
| **Google Maps 不可用** | `jarvis-ui` 的 `@react-google-maps/api` | fork 后删除该依赖 |
| **瓦片源是服务不是代码** | Leaflet/地图组件 | 自建瓦片 或 采购天地图/高德企业版 |
| **ElevenLabs 云 TTS 出不去** | 客户内网无公网出口 | 换 Piper / CosyVoice 本地推理；**组件可留，服务必换** |
| **CDN 字体/图标** | 默认走 Google Fonts / unpkg | 全部 vendor 到本地 `public/` |
| **模型权重下载** | 本地 TTS/STT 首次运行拉权重 | 预置离线包，容器构建期 bake 进去 |

### 8.3 性能

| 风险 | 量化 | 处置 |
|---|---|---|
| **three.js 无独显** | 工控机/一体机常见 Intel UHD，`r3f` 满屏粒子会掉到 20 fps | 粒子数上限、`dpr` 封顶 1.5、工程师模式强制 `frameloop="demand"` |
| **ECharts GL 内存** | 地质剖面 `surface` 网格 > 100×100 时内存涨得快 | 网格降到 60×60，用 `visualMap` 而非逐点材质 |
| **HUD 组件同时动画** | 25 个组件各带 SVG 动画 = 主线程抖动 | 统一 `prefers-reduced-motion`；工程师模式全局 `animation: none` |
| **50 MB 级单页** | `AIChat-UI` 的形态 | 不要学，走 Vite 分包 + 路由懒加载 |
| **实时数据频率** | 传感器 1 Hz × 16 测点 × 8 小时 = 46 万点 | 前端只拿聚合后数据，原始点按需拉取 |

### 8.4 高危操作（这条是产品红线，不是技术问题）

语音识别在工业现场的错误率远高于办公环境：压缩机 85 dB、安全帽、方言、术语（"井口压力" 会被识别成 "进口压力"）。

**因此：阀门、泵、泄压、联锁旁路——任何改变物理状态的操作，都不允许由语音直接执行。** 语音只能做到"打开确认面板"，落笔必须是人手点击 + 二次确认 + 记录操作人。这不是设计偏好，是安全底线。

### 8.5 需要客户额外采购的

| 项 | 谁出钱 | 说明 |
|---|---|---|
| 地图瓦片服务 | 客户 或 你 | 天地图企业版 / 自建 |
| 商业字体（若坚持用 DIN） | 你 | 每台终端授权，报价按 install 计 |
| GPU（若走 Cesium / 重度 3D） | 客户 | 本方案已避开，用 echarts-gl 降级 |
| 本地 TTS/STT 模型 | 你 | 开源权重免费，但推理服务器要算力 |

---

## 9. 待确认的三个决策

在进入视觉设计前，这三件事必须先定，否则画出来的稿子会返工：

**决策 1：React 还是 Vue？**
实测结论：**Vue 会让 B 类（语音组件）和 A 类（HUD 外壳）同时归零**——ElevenLabs UI 与 jarvis-ui 都是 React。选 Vue 等于这两层全部自研。建议 React。

**决策 2：语音是否进一期？**
如果一期只做"看板 + 告警 + 报告审批"，B 类可以整体延后，界面中央就不需要状态环，改放地质剖面。这会显著改变中央区的设计。

**决策 3：视觉激进到什么程度？**
三档：① 纯 shadcn 自绘（保守，1 周）② jarvis-ui 裁剪 fork（推荐，2–3 周）③ 全量 fork + 深度改造（激进，5 周+）。

---

*本报告所有星数、许可证、推送时间均为 2026-10-27 接口实测值。仓库状态会变，进入开发前建议重跑一次核查。*
