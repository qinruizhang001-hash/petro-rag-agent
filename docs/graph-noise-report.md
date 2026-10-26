# 图谱噪声报告

> 本文件由 `python normalize_graph.py` 生成，**不要手改**。
> 表里的数字全部取自当次运行的实测/预演结果；语料或规则一变，重新跑一次即可，手改的数字会静默过期。

## 1. 归一化前后

| 口径 | 节点 | 边 | 连通分量 | 最大分量 | 孤立节点 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 归一化前（实测） | 4988 | 5128 | 1226 | 3073 | 892 |
| 归一化后（预演） | 4415 | 4760 | 1085 | 2829 | 820 |

合并组 348 个，覆盖 755 个节点（净减 407）；待删脏节点 166 个。

## 2. 命中的噪声规则

| 原因 | 命中 | 占脏节点 | 这类节点是什么 | 样例 |
| --- | ---: | ---: | --- | --- |
| `type:person` | 5 | 3.0% | 人名。语料里没有需要保留的人物实体 | `Darin Damiani`、`Feld`、`Mark Ackiewicz` |
| `citation` | 32 | 19.3% | `et al.` 残片，是引文碎片不是实体 | `Borello et al.`、`Cheung et al. (2012)`、`Fleiter et al.` |
| `year` | 2 | 1.2% | 单独的四位年份 | `2021`、`2045` |
| `bare-quantity` | 10 | 6.0% | 只有数值加单位的碎片（`2500 m`、`15%`） | `1.29%`、`1.89%`、`2.12%` |
| `street-address` | 1 | 0.6% | 街道地址片段 | `Tescan Spectral Suite` |
| `affiliation` | 8 | 4.8% | 作者单位（大学 / 研究所 / 系 / 实验室） | `Fraunhofer Institute for Algorithms and Scientific Computing SCAI`、`Fraunhofer Institute for Systems- & Innovation Research ISI`、`Kassel University` |
| `journal` | 1 | 0.6% | 期刊名 | `Petroleum & Petrochemical Engineering Journal` |
| `funding-body` | 4 | 2.4% | 资助 / 行政机构（部委、研究联盟），连的是经费关系不是地质知识 | `Georgia Research Alliance`、`German Federal Ministry of Research, Technology, and Space`、`German Ministry of Economic Affairs and Climate Protection (BMWK)` |
| `geo-political` | 6 | 3.6% | 国家 / 地区 / 城市名，会变成跨论文的假枢纽 | `China`、`Europe`、`European Union` |
| `too-short` | 83 | 50.0% | ≤2 字符的拉丁字母名字（CJK 与化学式不受此限） | `4D`、`A`、`Ar` |
| `title-like` | 14 | 8.4% | 超过 10 个词，基本是论文标题而不是概念 | `Accepted Realizations Using Monitoring Well Data and Plume Location Seismic Data`、`Adjoint-state gradient of the data misfit with respect to the P-wave velocity`、`Conv, 32 filters of size 3×3×3, stride (1, 1, 1), padding (1, 1, 1)` |

## 3. 零命中的规则

下面这些规则**在本语料一次都没命中**。这不等于「已验证有效」，只等于「还没被验证」：它们是为已知的脏数据类型准备的防线，当前语料恰好没触发。

| 原因 | 命中 | 说明 |
| --- | ---: | --- |
| `empty` | 0 | 空字符串节点 |
| `type:content` | 0 | `content` 是模型无类型可归时的垃圾桶 |
| `type:publication` | 0 | 出版物条目，已被上面的期刊 / 出版社规则覆盖 |
| `doi` | 0 | DOI / URL / 网址，是引用标识不是概念 |
| `email` | 0 | 邮箱 |
| `date-label` | 0 | `Received date` 一类投稿流程标签 |
| `volume-issue` | 0 | 卷号 / 期号 |
| `section-label` | 0 | `Abstract` / `References` / `Acknowledg…` 一类章节标题 |
| `decade` | 0 | `1990s` 一类年代 |
| `postal-code` | 0 | 邮编 |
| `publisher` | 0 | 出版社名 |
| `meta-filler` | 0 | `this study` / `corresponding author` 一类元话语 |

## 4. 考虑过、但没有写成规则的模式

判据是**实测命中数**：写死一条零命中的规则只会扩大误杀面，不会减少噪声。下表每次运行都会重算，所以它不是记忆，是结论。

| 模式 | 命中 | 处理 | 样例 |
| --- | ---: | --- | --- |
| 会议 / 论文集名 | 0 | 零命中 → 不加规则 | — |
| 会议举办地式命名 | 0 | 零命中 → 不加规则 | — |
| 机构名（无 affiliation 关键词） | 4 | **有命中，需要人工看样例后再决定是否加规则** | `Georgia Research Alliance`、`German Federal Ministry of Research, Technology, and Space`、`German Ministry of Economic Affairs and Climate Protection (BMWK)`、`Ministry of Economic Affairs (The Netherlands)` |

已知的一处覆盖关系：机构名（大学 / 研究所 / 系 / 实验室）由 `affiliation` 规则按**名字**处理，而不是按 `organization` 类型整类删除 —— 因为 `IPCC`、`International Energy Agency` 这类政策主体也是 `organization`，整类删会连监管实体一起删掉。

## 5. 怎么复现

```bash
# 只预演，不改存储（本文件就是这样生成的）
python normalize_graph.py

# 真正执行
python normalize_graph.py --apply
```

预演用并查集在内存里把「合并 + 删除」应用一遍，因此「归一化后（预演）」这一行是**算出来的**，不是执行后的实测值；执行后本文件会多一行「实测」。
