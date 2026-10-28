# 界面截图

四张图都由 `web/scripts/cdp-shot.mjs` 生成，**不是手工截的**。视口固定
1680×1000，数据源是 `fixture`（默认产物），等待条件由脚本内置（DOM 就绪 +
剖面画布建好 + 两帧）。

重拍（后端不必启动，前端 preview 要先跑起来）：

```bash
cd web && npm run build && npm run preview      # 127.0.0.1:4173

node scripts/cdp-shot.mjs --out ../docs/ui/web/deck.png        --url "http://127.0.0.1:4173/?mode=deck"
node scripts/cdp-shot.mjs --out ../docs/ui/web/engineer.png    --url "http://127.0.0.1:4173/?mode=engineer"
node scripts/cdp-shot.mjs --out ../docs/ui/web/chain.png       --url "http://127.0.0.1:4173/?mode=deck&tab=chain&rail=dialog"
node scripts/cdp-shot.mjs --out ../docs/ui/web/kb-search.png   --url "http://127.0.0.1:4173/?mode=deck&rail=kb&kb=search"
```

## 为什么要把命令写下来

这四张图曾经只有文件、没有出处：谁也不知道它们对应哪个 URL、哪个视口、
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
| `kb-search.png` | 知识库检索：切片卡片、证据强度三维度、血缘徽标 |
