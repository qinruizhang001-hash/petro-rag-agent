# 界面截图

五张图都由 `web/scripts/cdp-shot.mjs` 生成，**不是手工截的**。视口固定
1680×1000，等待条件由脚本内置（DOM 就绪 + 剖面画布建好 + 两帧）。

## 这五张图跑的是 `live`，不是 `fixture`

URL 里都带 `?source=live`，也就是**它们证明的是"界面连着一个活的后端"**，
不是"界面能读一份快照"。这个区别值得为截图多花一步：`fixture` 模式下后端
挂了三天，界面照样全绿 —— 而截图是唯一会被贴进文档、给不认识这个仓库的人
看的东西。如果它拍的是快照，那它什么也没证明。

代价是重拍必须先起后端和 Ollama：

```bash
# 1. 后端（127.0.0.1:8000）
cd /d/python-learn/petro-rag-agent
OPS_THRESHOLDS_PATH=config/ops_thresholds.example.json \
  /d/python-learn/.venv/Scripts/python.exe -m uvicorn api_server:api --host 127.0.0.1 --port 8000

# 2. Ollama（检索与 embedding 要它；不启动时 /health 会报 degraded）
ollama serve

# 3. 前端
cd web && npm run build && npm run preview      # 127.0.0.1:4173
```

确认后端真的通了，再拍：

```bash
cd web && npm run smoke          # 六组检查，全过再截图
```

## 五条重拍命令

```bash
cd web

node scripts/cdp-shot.mjs --out ../docs/ui/web/deck.png \
  --url "http://127.0.0.1:4173/?mode=deck&source=live"

node scripts/cdp-shot.mjs --out ../docs/ui/web/engineer.png \
  --url "http://127.0.0.1:4173/?mode=engineer&source=live"

node scripts/cdp-shot.mjs --out ../docs/ui/web/chain.png \
  --url "http://127.0.0.1:4173/?mode=deck&tab=chain&rail=dialog&source=live"

node scripts/cdp-shot.mjs --out ../docs/ui/web/kb-search.png \
  --url "http://127.0.0.1:4173/?mode=deck&source=live&kb=search" \
  --ready "!!document.querySelector('[aria-label=\"知识库检索\"]')" \
  --kb-search "CO2 泄漏的监测手段有哪些"

node scripts/cdp-shot.mjs --out ../docs/ui/web/dialog.png \
  --url "http://127.0.0.1:4173/?mode=deck&source=live&rail=dialog" \
  --ready "!!document.querySelector('[aria-label=\"向系统提问\"]')" \
  --ask "过去 4 小时井筒和管道有什么异常"

# 语音：按下去之后的状态（录音计时 + 引擎角标）。
# 没有麦克风的机器上要加两个 Chrome 开关，否则 getUserMedia 直接被拒 ——
# 那不是脚本坏了，是这台机器没有录音设备。
CDP_EXTRA_ARGS="--use-fake-device-for-media-stream --use-fake-ui-for-media-stream" \
node scripts/cdp-shot.mjs --out ../docs/ui/web/dialog-voice.png \
  --url "http://127.0.0.1:4173/?mode=deck&source=live&rail=dialog" \
  --ready "!!document.querySelector('[aria-label=\"向系统提问\"]')" \
  --voice
```

`--kb-search` / `--ask` 是这两个脚本参数，因为这两屏**只有先点一次才会出现**。
手点 + 手截的后果是这张图没法重拍：换个查询词就是另一批文献，图里的
`12 片 / 6 篇` 与任何一次新的检索都对不上。参数里带着问句，重拍就是同一次
检索。脚本等的是**状态**（按钮从「检索中」回到「检索」、正文之后「数字依据」
出现），不是固定时长 —— 检索 4~15 秒，取决于 LightRAG 是否预热过。

## 为什么要把命令写下来

这五张图曾经只有文件、没有出处：谁也不知道它们对应哪个 URL、哪个视口、
哪一版构建。后果是 `engineer.png` 里印着「捕集率 86.8 · 警告线 88 · 超 1.2」——
一个后来被判定为错误的说法（86.8 比 88 小，`low` 方向越限该说「低」）——
而它作为"界面长这样"的证据在仓库里待了很久。图不能自己说明它是什么时候
的，所以**能重新生成的图，就必须留下生成它的那行命令**。

同理，改完界面要顺手重拍；`git diff --stat docs/ui/web/` 为空而
`web/src/components/` 有改动，通常意味着图又过期了。

## 每张图对应什么

| 文件 | 看什么 |
|---|---|
| `deck.png` | 指挥舱：左栏告警卡（带阈值余量）、中央剖面、右栏知识库概览、底栏六态计数 |
| `engineer.png` | 工程师模式：卡片/表格切换、更密的读数、剖面角标关掉 |
| `chain.png` | 全链视图：捕集 → 运输 → 注入 → MVA 四段的测点与判据；右栏对话与授权边界 |
| `kb-search.png` | 知识库检索：一次真实检索的结果 —— `12 片 / 6 篇`、证据强度三维度、来源表、原文证据 |
| `dialog.png` | 对话：问「过去 4 小时井筒和管道有什么异常」，回答里每个测点都带采样数、分位、阈值出处，末尾是 `数字依据（10 条）` 与 `12 片 / 8 篇` 原文证据 |

`kb-search.png` 与 `dialog.png` 里的数字**来自拍图那一刻的后端**，不是写死的。
换一次语料、换一批读数，这两张图上的数字会跟着变 —— 这正是它们该有的样子：
它们展示的是"这条链路能跑通并给出可回代的数字"，不是"界面长这个样子"。
