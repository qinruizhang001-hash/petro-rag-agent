# CCUS 全链运维控制台 · 前端

目标只有一个：**让六态语义（含"不可判定"）在界面上站得住**，并且中央那块屏
描述的是物理状态，不是氛围。

- **第 1 周**（已完成）：左栏告警卡 / 指标卡接 `/api/ops/alarms` 的真实响应。
- **第 2 周**（已完成）：中央地质剖面（three.js + SVG 降级）、双模、语音状态环。
- **第 2.5 周**（已完成）：右栏知识库（语料清单 / 缺口 / 带出处的检索 / 切片原文）。
- **第 3 周**（未开始）：右栏对话、对话正文里的引用角标、审批状态机。

## 跑起来

```bash
cd web
npm install
npm run dev          # http://localhost:5180
```

默认读 `src/fixtures/*.json`，**不需要后端**。

要接真实后端：

```bash
# 后端先起在 127.0.0.1:8000（见仓库根 README）
VITE_OPS_SOURCE=live npm run dev
```

`vite.config.ts` 里把 `/api` 代理到 `OPS_API_TARGET`（默认 `http://127.0.0.1:8000`），
所以不需要为本地调试去开 CORS。

### URL 参数

一个只能靠点两下才能到达的状态，评审时总会有人截错图。所以下面这些状态都能用
查询参数直达：

| 参数 | 取值 | 说明 |
|---|---|---|
| `mode` | `deck` / `engineer` | 指挥舱 / 工程师。默认工程师 |
| `tab` | `section` / `chain` | 中央区的两个视图 |
| `voice` | `idle` / `listening` / `thinking` | 语音状态环的三种形态 |
| `rail` | `kb` / `dialog` | 右栏两个 Tab。默认知识库（对话还是占位） |
| `kb` | `overview` / `search` | 知识库内部的概览 / 检索 |
| `q` | 任意问题 | 预填检索词。**只在 fixture 模式自动提交**，见下 |
| `perf` | `1` | 打开渲染开销面板，同时暴露 `window.__perf()` |
| `quality` | `high` / `medium` / `low` | 钉死渲染档位，关闭自适应降级 |
| `gpu` | `low` | 强制 `powerPreference: low-power`（尝试落到集显） |
| `ws` | `ws://…` | 指定推送地址，覆盖 `VITE_OPS_WS` |

```bash
npx vite preview --host 127.0.0.1 --port 4181
# http://127.0.0.1:4181/?mode=deck&rail=kb&kb=search&q=盖层完整性
```

`q` 只在 fixture 模式自动提交：live 模式下打开页面就跑一次 21 秒的检索，
会让"我刚点进来怎么卡住了"变成一个查不出原因的问题 —— 而它其实只是 URL 里带着参数。

## fixture 不是手写的

`src/fixtures/*.json` 由 `scripts/capture_ops_fixtures.py` 用 `TestClient`
向真实的 `/api/ops/*` 发请求抓下来的。重跑：

```bash
python scripts/capture_ops_fixtures.py           # 重新抓取
python scripts/capture_ops_fixtures.py --check   # 只比对（CI 用）
```

为什么值得这么麻烦：手写 fixture 会和契约悄悄分家。后端字段改名、枚举多一个值、
错误形状从 `{"detail": "..."}` 变成 `{"detail": {"message": ...}}` ——
这些都不会让任何东西变红，前端只是继续渲染那份"看起来还对"的旧数据。
`--check` 在 CI 里跑，接口一变就报出来。

场景是构造的，**阈值不是**：阈值直接取自 `config/ops_thresholds.example.json`，
一个数字都没改，`source` 里保留着 `【替换】` 前缀，界面上照原样显示
"这还不是你项目的真实阈值"。给示例阈值换一个像样的出处，是这套系统最不该做的事。

**羽流几何是示意值**，界面上同样照原样标注「示意（未接储层模拟）」。
它来自储层模拟，不是从压力读数反推的 —— 用 `reservoir_p` 去"推算"羽流半径，
得到的是一个看起来很像真数的编造值，而它会被截图放进汇报材料。

### 知识库 fixture 分两类，不能一视同仁

`src/fixtures/{overview,documents,search}.json` 由
`scripts/capture_kb_fixtures.py` 抓取，但它们的**可校验性不一样**：

| 文件 | 来源 | CI 校验 |
|---|---|---|
| `overview.json` / `documents.json` | 纯文件读（`ccus_docs/`、`lightrag_storage/`、`chroma.sqlite3`） | ✅ 每次 |
| `search.json` | 真跑 LightRAG + Ollama，本机约 21 秒 | ❌ 没有 Ollama |

脚本会把"未比对 search.json"打印出来。一个连检索都包含进去的绿灯，
等于替没做的检查背书 —— 而它读起来和真校验过一模一样。

`search.json` 有 97 KB，是**动态 import** 的（`dist/assets/search-*.js`）。
改成静态 import 不会报错，只会让首屏多下载 78 KB，而这个面板在首屏是看不见的
（默认停在概览页）。CI 的 `KB search fixture is still code-split` 盯着这件事。

## 知识库面板：它要回答的不是"有多少篇"

是**"这个知识库能信到什么程度"**。所以三件事必须在首屏：

1. **缺口**。盘上 24 篇、图谱里 22 篇，差的那 2 篇按名字列出来，并说明后果
   （"检索不到它们"）。一个"共 22 篇"的数字读起来是完整的，而它其实漏了 2 篇 ——
   没有任何地方会提示这件事。
2. **可引用性**。只有图谱切片带 `file_path` 血缘，能点回原文；Chroma 那 1,982 条
   的 metadata 只有 `{"source": "legacy_v2"}`，**没有文件级血缘**。
   这件事写在界面上，否则用户会以为每条结论都能溯源。
3. **检索的三种状态长得不一样**。见下面第 6 条约束。

引用角标的语义也在界面上标死了：`reference_id` 是 **LightRAG 在一次响应内**
给来源编的号，同一个 `[3]` 在两次查询里指向不同文献。它能当"本次检索的第 3 个来源"，
不能存下来当书签 —— 要长期指向一篇文献只能用 `file_path`。

## 目录

```
src/
├── lib/
│   ├── types.ts      # 响应类型，照着 fixture 反推（不是照文档抄）
│   ├── severity.ts   # 六级状态的唯一真相：排序 / 标签 / 余量 / Tailwind 类名
│   ├── stages.ts     # 四段链路标签，与后端 STAGE_LABELS 逐字一致
│   ├── section.ts    # 剖面几何的唯一真值：层序 / 测点位置 / 羽流半径函数
│   ├── webgl.ts      # WebGL 探测（单独一个文件，避免为它静态引入 three）
│   ├── kb-types.ts   # /api/kb/* 的响应类型
│   ├── kb.ts         # 知识库客户端；检索 fixture 走动态 import
│   ├── api.ts        # /api/ops/* 客户端，fixture 与 live 两种数据源
│   ├── push.ts       # WebSocket 推送通道（带退避重连），默认不启用
│   ├── perf.ts       # 帧耗时采样（中位数 / p95 / 超预算帧）
│   └── utils.ts      # cn()
├── three/            # 纯 three.js，不依赖 React；整体动态 import
│   ├── geoscene.ts   # 装配层：渲染器 / 渲染循环 / 模式 / 释放（~330 行）
│   ├── context.ts    # builder 的共享上下文（scene / data / anchors / track / uniforms）
│   ├── quality.ts    # 渲染档位与降档判定（纯函数，有单元测试）
│   ├── camera.ts     # 正交取景与像素换算
│   ├── strata.ts     # 地层板 / 地表网格 / 盖层底界
│   ├── wells.ts      # 井筒 / 井口 / 射孔段
│   ├── plume.ts      # 羽流轮廓 / 3D 包络体 / 等压环
│   ├── particles.ts  # 3000 个 CO₂ 点（单次 draw call）
│   ├── monitors.ts   # 测点球 / 外环 / 深度尺锚点
│   └── sceneRegistry.ts  # 当前活动场景的登记处（给性能探针用）
├── components/
│   ├── Panel.tsx              # 面板外框 + 12 px 角标（HUD 装饰只保留这一处）
│   ├── SeverityChip.tsx       # 六级标签，三重信号（颜色 / 边框 / 文字）
│   ├── AlarmCard.tsx          # 告警卡：判据逐条、出处、归因、已排除删除线、处置
│   ├── MetricCard.tsx         # 指标卡：带阈值余量，不是只有数值
│   ├── BlindSpotBanner.tsx    # 盲区提示："不可判定 ≠ 正常"
│   ├── CeilingCard.tsx        # 注入压力上限（法规判据，四态 + 比值刻度）
│   ├── LeftRail.tsx           # 左栏：盲区 + 上限校核 + 按严重度排序的单列表
│   ├── GeoSection.tsx         # 中央：剖面 / 全链两个 Tab
│   ├── GeoSectionCanvas.tsx   # 剖面画布 + HTML 标签叠加（投影跟随）
│   ├── GeoSectionFallback.tsx # WebGL 不可用时的 SVG 降级版
│   ├── VoiceRing.tsx          # 语音状态环（待机 / 聆听 / 思考），不可交互
│   ├── ChainPanel.tsx         # 全链状态矩阵（同一份数据的第二种读法）
│   ├── kb/                    # 右栏知识库，按页面拆开（单个文件最大 198 行）
│   │   ├── KnowledgePanel.tsx # 外壳：页签 + 概览数据加载
│   │   ├── KbOverview.tsx     # 规模 / 缺口 / 向量库 / 文档清单
│   │   ├── KbSearch.tsx       # 输入、提交、详情路由
│   │   ├── KbResultView.tsx   # 三种检索状态（ok / empty / unavailable）
│   │   ├── KbChunkCard.tsx    # 一条原文证据（一条一张卡，绝不合并）
│   │   ├── KbDocDetail.tsx    # 引用跳转落点（高亮命中切片）
│   │   ├── RefBadge.tsx       # 引用角标（被两个上层共用，所以独立）
│   │   └── format.ts          # 建图耗时的数值 / 单位分离
│   ├── RightRail.tsx          # 右栏：知识库 / 对话两个 Tab
│   ├── PerfOverlay.tsx        # `?perf=1` 的渲染开销面板
│   └── StatusBar.tsx          # 底栏：六态计数（含 0 的那些）+ 推送状态
├── fixtures/         # 真实响应快照，勿手改
├── App.tsx
└── index.css         # 令牌的唯一来源
```

## 八个必须守住的约束（外加一条前置原则）

### 前置 · 缺席不能被渲染成合格

这一条排在所有约束之前，因为它是这套界面的**唯一失败模式**：不是画错颜色，
是把"没查"画成"查过没问题"。同一个形状在三处出现过，三处都必须有独立样式：

- 测点没配阈值 → `indeterminate`（灰 + 虚线 + 破折号），不是 `normal`
- 注入压力上限缺读数 → `CeilingCard` 的 `indeterminate` 态，**不画进度条**
  （一条 0% 的条会被读成"压力很低"，而实际是"算不出来"）
- `ceiling` 字段整个缺失 → 显示"未执行"，并写明"未执行 ≠ 合规"

第三条尤其容易漏：字段缺失是 `undefined`，而 `undefined` 在 JSX 里渲染成空 ——
空白看起来像"这一栏不需要"。所以 `CeilingCard` 对 `null` / `undefined` 有独立分支。

同一原则的推论：**不可判定的原因不要在前端重新枚举**。`CeilingCard` 的折叠区
直接显示后端 `basis[0]`，展开区从 `basis[1]` 开始 —— 因为后端新增一种
"算不出来"的情形（例如"峰值时刻没有生效的破裂压力测定"）时，前端那句
手写的"缺井底压力或破裂压力读数"会**悄悄变成不完整的**，而它读起来
和之前一样确定。原因只有一个来源，展示层只负责选哪一行。

### 1. 六级，不是五级

`indeterminate` 不属于 `normal..critical` 那个序列 —— 它否定的是"能判定"这件事本身。
界面上是**三重信号**：灰 + 虚线边框 + 破折号。少任何一重，
色觉障碍用户或强光下的现场都会把它读成"正常"，而"把不知道渲染成没事"
是这套界面唯一不会报错的失败。

### 2. Tailwind 类名不能拼

`text-sev-${token}` 在运行时是对的，但 Tailwind 是**编译期扫描源码文本**生成 CSS 的。
它扫不到模板字符串，那批类名根本不会进产物 —— 症状是颜色全失效、文字继承父级，
而且没有任何报错。所以 `lib/severity.ts` 里有一张 `SEVERITY_CLASS` 字面量表。
加新状态时**两处都要改**，CI 还会 grep 产物确认六个类名都在。

### 3. 品牌色不进数据层

中石化红 `#D8232A` 与 critical `#B23A3A` 的色相只差 4°，同屏会把"品牌"读成"严重告警"。
品牌色只允许出现在页眉细线 / Logo / 客户联合品牌位 / 面板角标。
数据区永远是青、钢蓝与六级语义色。

### 4. 装饰不能吃掉数据

`html[data-mode="engineer"] .fx { display: none }` 这种规则，一旦标错元素，
删掉的就是数据而不是动效。**羽流本身是数据**（它表示 CO₂ 波及范围），
工程师模式关掉的是粒子与 3D 包络体，不是羽流。
剖面里唯一的判断标准是：这个图元如果消失，有没有一条结论会跟着消失。

### 5. 3D 必须保持懒加载

`three` 压缩后约 130 KB gzip，用动态 `import()` 拆成独立 chunk。
左栏的告警卡是"这个班次要不要叫人"的全部依据，它不该等一个 3D 库下载完才出现。
CI 的 `3D scene is still code-split` 一步会断言独立 chunk 存在、首屏 chunk < 320 KB ——
改成静态 import 时"左栏先亮"会静默消失，没有报错，只是工控机上多等一秒。

### 6. `empty` 与 `unavailable` 不能合并

检索有三种结果，不是两种：

| 状态 | 含义 | 界面 |
|---|---|---|
| `ok` | 检索到内容 | 证据卡 |
| `empty` | 检索跑完了，确实没有相关内容 | 「图谱里没有相关内容」 |
| `unavailable` | 检索没跑起来（Ollama 未启动 / 图谱未初始化） | 「检索未连接」+ 原因 |

把第三种渲染成前两种里的任何一个，操作员都会去补文献 —— 而真实原因只是
Ollama 没启动。所以后端对 `unavailable` 也返回 **200**（不是 503）：
它是需要被看见的**系统状态**，不是请求错误；返回 503 的话前端只会弹一个
通用报错，操作员学不到"是模型没起"还是"是库空了"。

这与 `indeterminate ≠ normal` 是同一条约束，只是发生在检索层。

### 7. 渲染开销有上限，超了要自己降档

"在集显上能跑 60 FPS"是一句必须能被证伪的话。没有测量，它就只是一句写在
文档里的承诺；而下一次有人把 3000 个粒子改成 30 万个、或者把 `pixelRatio`
上限从 2 改成 `devicePixelRatio` 时，没有任何东西会红。

集显是个很宽的区间（Intel UHD 620 与 Iris Xe 差三倍，插电与省电模式又差一倍），
所以**不能调到一个值，只能给一个能自己往下走的台阶**：

| 档位 | pixelRatio 上限 | 粒子 | 3D 包络体 | 数据图元 |
|---|---|---|---|---|
| `high` | 2 | 有 | 有 | 全部 |
| `medium` | 1.25 | 有 | 有 | 全部 |
| `low` | 1 | **无** | **无** | **全部** |

低档关掉的只有"让人看出它是个体"和"流动感"两件装饰 —— 剖面轮廓、等压环、
测点球、地层板全在。判断标准是：**这个图元消失后，有没有一条结论跟着消失。**

三条防它自己变成故障的约束：

1. **只降不升。** 升档要重新评估，而重新评估意味着又一次抖动窗口 ——
   在"刚好及格"的机器上，升 → 掉帧 → 降 会变成一个可见的循环。
2. **最多降两档。** 降到底就定住，没有这个终点"一直降"是永不结束的过程。
3. **判据是中位数，不是平均。** 一次 GC 尖峰会把平均值拉过阈值，
   于是在一台本来及格的机器上触发降级。

### 8. 依赖许可证必须允许闭源商用

规则与完整审查结果在 [`docs/licenses.md`](../docs/licenses.md)。
前端 143 个包（含全部传递依赖）零 GPL / AGPL / LGPL / SSPL。

两条经验：

- **必须扫传递依赖。** 直接依赖只有 13 个，而 GPL 的传染性判断看的是整棵树。
- **审查脚本本身要有自检。** 一张写错的正则表会让所有包都判为放行，
  然后 CI 亮着绿灯说"无 GPL"—— 一个替没做的检查背书的绿灯，
  比没有这道检查更危险。`--self-test` 断言名单本身有效。

## 性能：数字与取证

```bash
# 另开一个终端
npm run preview -- --host 127.0.0.1 --port 4173

npm run perf          # 指挥舱，实测帧率 + 逐项渲染开销
npm test              # 降档判定的单元测试（纯函数，不需要浏览器）
npm run licenses      # 依赖许可证审查
```

实测（1680×1000，RTX 5060 Laptop，`?mode=deck`）：

| 指标 | 值 |
|---|---|
| p50 / p95 帧耗时 | 3.3 / 3.4 ms（预算 16.7 ms） |
| draw calls | 30 |
| 三角形 | 5,164 |
| 粒子 | 3,000（**单次** draw call） |
| 几何体 / 纹理 / 着色器程序 | 30 / 5 / 6 |
| 首屏 JS chunk | 243 KB（gzip 80 KB） |
| three 独立 chunk | 508 KB（gzip 132 KB） |

工程师模式：28 draw calls / 1,964 三角形 / 0 粒子，**渲染循环停止**（只画一帧）。

### 一条实测出来的教训

三个透明网格原来都写了 `side: THREE.DoubleSide`。实测 `renderer.info`：

```
DoubleSide  → 33 draw calls, 8,508 triangles
FrontSide   → 30 draw calls, 5,164 triangles
```

多出的 3 次调用、3,344 个三角形，正好等于三个网格的三角形数之和
（128 + 16 + 3,200）。也就是说**每个 DoubleSide 透明网格都被画了两次**。
而这张剖面上没有任何东西需要看背面 —— 相机固定，只从 +Z 方向看。

> 结论：`renderer.info` 是唯一能证明"这帧很轻"的客观数据。
> 源码里的 `COUNT = 3000` 说明不了问题 —— 3000 个点画 1 次和画 3000 次
> 差三个数量级，而两种写法看起来都很"省"。

### 关于 FPS 这个数字

开发机上 rAF 常常**没有锁 vsync**（远程桌面、虚拟显示器、某些驱动的省电策略），
于是 `fps` 会跑到 300。这时它读作"每帧还剩多少余量"，**不读作"用户看到的流畅度"**。
判据永远是 `p50`：它才是这一帧的真实成本。`measure-fps.mjs` 在 fps > 120 时
会明确打印这条提醒。

## 数据源：三档，且组件不知道区别

| 档位 | 触发 | 说明 |
|---|---|---|
| `fixture` | 默认 | 读 `src/fixtures/*.json`，那是真实接口响应的快照 |
| `live` | `VITE_OPS_SOURCE=live` | 打到 `VITE_OPS_BASE`（默认同源 `/api`） |
| `push` | `VITE_OPS_WS` 或 `?ws=` | WebSocket 推送，带退避重连 |

**后门是接上的，不是留了个注释。** 未配置推送地址时不建立任何连接（不是
"假装连上"）；配置之后 `App` 订阅 `alarm` / `thresholds` 两类消息并更新状态，
底栏出现一个连接状态角标。组件始终只拿到 `AlarmReport` —— 它不知道这份数据
是拉来的还是推来的。

断开必须看得见：推送断了而界面照旧显示旧数字，是最难发现的一类故障。
`?ws=` 可以指到一个假服务端做联调：

```bash
node scripts/fake-ws-server.mjs --port 8787        # 零依赖的假推送服务
node scripts/cdp-eval.mjs --url "http://127.0.0.1:4173/?ws=ws://127.0.0.1:8787/" \
  --expr "document.body.innerText.includes('推送已连接')"
```

## 还没做

- 工程师模式的 16 测点全量表（虚拟滚动）：剖面之外，现场还需要一屏能扫完的纯表格
- 对话输入与流式回答、对话正文里的引用角标、审批状态机（第 3 周）
- 知识库的写入侧：现在只能看和查，入库仍走 `ingest_ccus_batch.py` 命令行
- 语音输入：状态环只有三个形态，麦克风与 ASR 都还没接
- shadcn/ui 组件库：目前是手写的 shadcn 风格原语（`cn()` + 令牌），
  之后 `npx shadcn add <comp>` 可以直接替换，不需要改令牌
