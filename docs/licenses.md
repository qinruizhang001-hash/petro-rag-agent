# 依赖许可证审查

> 目的只有一个：**这套东西以后能不能闭源卖。**
>
> 结论先说：
>
> - **前端（`web/`）：干净。** 143 个包（含全部传递依赖）零 GPL / AGPL / LGPL / SSPL。
> - **后端 Python：有一处阻断级依赖 —— `pymupdf` 是 AGPL-3.0 或 Artifex 商业授权。**
>   它只用在离线入库（PDF → 文本），但这条边界是否成立是法律判断，不是工程判断。
>   三个处置方案见 §3。
>
> 审查是**可重复执行的**，不是一次性的人工核对：
>
> ```bash
> cd web && node scripts/audit-licenses.mjs          # 人类可读
> cd web && node scripts/audit-licenses.mjs --json   # 机器可读
> cd web && node scripts/audit-licenses.mjs --self-test   # 校验阻断名单本身
> ```
>
> CI 的 `web` job 会跑前两条，命中 copyleft 直接失败。

---

## 1. 为什么必须扫传递依赖

只看 `web/package.json` 里那 13 个直接依赖是不够的。GPL 的传染性判断看的是
**整个依赖树** —— 一个 MIT 的库完全可以在自己的依赖里带一个 GPL 的解析器，
而那一层不会出现在任何人的 `package.json` 里。

实测：

| 口径 | 包数 |
|---|---|
| 直接依赖（`package.json`） | 13 |
| **含传递依赖（`node_modules` 全量）** | **143** |

漏掉的是 130 个。所以审计脚本递归遍历 `node_modules`（含嵌套的
`node_modules` 与 `@scope` 目录），读每个包的 `package.json` 的 `license` 字段。

## 2. 前端依赖分布（143 个包）

| 协议 | 数量 | 判定 |
|---|---|---|
| MIT | 127 | 放行 |
| ISC | 9 | 放行 |
| Apache-2.0 | 4 | 放行 |
| BSD-3-Clause | 2 | 放行 |
| CC-BY-4.0 | 1 | 允许商用，**需署名** |
| GPL / AGPL / LGPL / SSPL | **0** | — |

### 2.1 那个 CC-BY-4.0

`caniuse-lite` —— 浏览器兼容性数据库，被 `browserslist` 与 `autoprefixer` 使用。

- 它是**构建期数据**，不打进产物（`dist/` 里没有它的代码）。
- CC-BY-4.0 **允许商用**，条件是保留署名。
- 结论：可以用。如果将来要做开源合规声明，在 `NOTICE` 里带一句即可。

### 2.2 直接依赖逐项

| 包 | 协议 | 用途 |
|---|---|---|
| `react` / `react-dom` | MIT | UI |
| `three` | MIT | 3D 剖面 |
| `clsx` / `tailwind-merge` | MIT | 类名合并 |
| `vite` / `@vitejs/plugin-react` | MIT | 构建 |
| `tailwindcss` / `postcss` / `autoprefixer` | MIT | 样式 |
| `typescript` | **Apache-2.0** | 类型检查（含专利授权条款，对商用有利） |
| `@types/*` | MIT | 类型定义 |

全部允许闭源商用二次开发。

## 3. 后端 Python：一处阻断级依赖

从已安装包的元数据读取（`importlib.metadata`，不是看文档）：

| 包 | 版本 | 协议 | 判定 |
|---|---|---|---|
| **`pymupdf`** | 1.28.2 | **AGPL-3.0 / Artifex 商业** | **阻断** |
| `pypdf` | 6.20.1 | BSD-3-Clause | 放行 |
| `chromadb` | 1.5.9 | Apache-2.0 | 放行 |
| `sentence-transformers` | 6.1.0 | Apache-2.0 | 放行 |
| `lightrag-hku` | 1.5.7 | MIT | 放行 |
| `langgraph` | 1.2.15 | MIT | 放行 |
| `streamlit` | 1.65.0 | Apache-2.0 | 放行 |
| `plotly` | 7.1.0 | MIT | 放行 |
| `matplotlib` | 3.11.2 | PSF | 放行 |
| `fastapi` / `gunicorn` / `mcp` / `ollama` | — | MIT | 放行 |
| `openai` | 1.109.1 | Apache-2.0 | 放行 |
| `tiktoken` | 0.14.0 | MIT | 放行 |

### 3.1 `pymupdf` 为什么是问题

AGPL-3.0 第 13 条（Remote Network Interaction）：**如果用户通过网络与这个
程序交互，你必须向他们提供整个作品的对应源码。** 这比 GPL 更狠 ——
GPL 只在"分发"时触发，AGPL 在"提供网络服务"时就触发。

也就是说：只要产品是 SaaS，而 `pymupdf` 参与了这个服务，闭源就站不住。

`requirements.txt` 里的注释说明了它为什么在：

> 两个都装是有意的：PyMuPDF 是主抽取器（能正确还原单词间空格），pypdf 只作
> 兜底。实测 pypdf 会把某些 PDF 的整段文字抽成 `actiontostabilize...`，
> 而 PyMuPDF 在个别加密/畸形文件上会失败 —— 互为补充。

这是一个**真实的工程质量理由**，不是随手加的依赖。所以不能简单地删掉。

### 3.2 三个处置方案

| 方案 | 做法 | 代价 |
|---|---|---|
| **A. 换成宽松协议的主抽取器** | 用 `pdfplumber`（MIT，基于 `pdfminer.six`）作主抽取器，`pypdf` 兜底 | 需要重跑语料抽取 + `docs/corpus-audit.json` 全部数字会变；要验证"单词间空格"这个具体问题是否复现 |
| **B. 买商业授权** | Artifex 的商业许可 | 有成本；好处是抽取质量不变 |
| **C. 划清边界** | `pymupdf` 只出现在**离线一次性入库脚本**里，不进入任何被分发的服务 | 需要法务确认"离线预处理 + 只分发文本"构成独立程序；且必须保证没有任何在线路径能触发它 |

**当前状态：未处置。** 这是一个需要产品决策的项，不是一个可以在代码里
单方面决定的事 —— 方案 A 会改变语料抽取结果，方案 B 要花钱，
方案 C 要法务判断。**在选定之前，不要对外声称"全栈可闭源"。**

### 3.3 顺带一条：模型权重的协议不在这张表里

`requirements.txt` 管的是**代码**协议。`sentence-transformers` 拉下来的
嵌入模型、`ollama` 跑的 LLM，各有自己的权重协议（例如 Llama 系列有
自定义的社区许可，带商用门槛）。这类协议不在包管理器的元数据里，
**必须按实际使用的模型单独核对**。本项目当前用 `qwen2.5` 系列
（Apache-2.0）与 BGE 系列嵌入模型，但这条需要在使用新模型时重新确认。

## 4. 判定规则（写在代码里，可被审查）

`web/scripts/audit-licenses.mjs`：

- **阻断**：`AGPL*` / `GPL*` / `LGPL*` / `SSPL*` / `OSL*` / `EUPL*` / `CC-BY-NC*` / `CPAL*` / `RPL*`
- **需署名**：`CC-BY*`
- **放行**：其余（MIT / ISC / Apache-2.0 / BSD / 0BSD / Unlicense / PSF）

为什么 `LGPL` 也阻断：动态链接的 LGPL 通常可以闭源分发，但前端是
**打包**（bundle）的 —— 代码被内联进一个文件，"动态链接"这个前提不成立。
与其在法务上争这条线，不如直接不用。

为什么 `CC-BY-NC` 阻断：非商用条款与"以后卖钱"直接冲突。

### 4.1 守卫的守卫

一张正则表最容易出的问题是**它其实什么也没匹配到**：写错一个字，所有包
都判为放行，然后 CI 亮着绿灯说"无 GPL"—— 一个替没做的检查背书的绿灯，
比没有这道检查更危险。

所以 `--self-test` 对名单本身做断言：必须命中 6 种典型 copyleft，
必须**不**误伤 7 种常见宽松协议。CI 会跑它。

同理，如果扫描路径解析错了（在错误目录下运行），脚本会扫到 0 个包；
这种情况直接以退出码 2 失败，而不是报告"零阻断"。

## 5. 复现

```bash
cd web
npm ci
node scripts/audit-licenses.mjs --self-test   # 名单自检
node scripts/audit-licenses.mjs               # 全树扫描
node scripts/audit-licenses.mjs --json        # 给机器读
```

Python 侧：

```bash
python scripts/audit_python_licenses.py              # 读已安装包（本地，精确）
python scripts/audit_python_licenses.py --from-pypi  # 读 PyPI 元数据（CI，不装依赖）
```

两种数据源的分工：`requirements.txt` 里有 `sentence-transformers`（拖 torch），
装一遍要几个 GB，而 CI 需要的是"这棵依赖树里有没有 AGPL"，不是"能不能跑起来"。
所以 CI 用 `--from-pypi`，本地用默认模式。

### 5.1 例外登记

`docs/license-exceptions.json` 记录**已接受**的风险。CI 的判定：

| 情况 | 结果 |
|---|---|
| 命中阻断，已登记例外，未过期 | 通过（打印警告） |
| 命中阻断，未登记 | **失败** |
| 例外已过期（超过 `review_by`） | **失败** |
| 登记的例外已不再是阻断协议 | **失败**（条目要删，否则表会腐烂） |

为什么要这个机制而不是直接让 CI 红着：一个长期红的检查很快会被 `--no-verify`
绕过，那时它连"提醒"都不算了。而"我知道这件事、我决定暂时接受、我在某个日期前
会重新看"是一个**可以被机器检查**的陈述。

当前登记了 1 条：`pymupdf`，复审期限 2027-01-31。

### 5.2 一个真实的踩坑

第一版 Python 审查用的是锚定正则 `^AGPL`。而 `pymupdf` 的 `License` 字段写的是
`Dual Licensed - GNU AFFERO GPL 3.0 or Artifex Commercial License` ——
**锚定匹配不到**，于是脚本漏掉了唯一一个真正阻断的依赖，还反过来报
"例外已失效，请删除该条目"。

锚定只对规范字段成立，而审查的对象恰恰是不规范的字段。现在两边都改成
关键词判定（先排 AFFERO / LESSER，再排 GENERAL PUBLIC LICENSE，顺序有意义），
并且 `--self-test` / `npm run licenses:self-test` 会对名单本身做断言。
