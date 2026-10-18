"""
LightRAG 实体抽取的领域约束（CCUS）
===================================
把"只抽地质/工艺实体"这条约束做成数据，而不是散落在构造参数里。

为什么需要它
------------
LightRAG 默认的实体类型表（Person / Organization / Location / Content …）是
**通用百科**视角。喂 23 篇 CCUS 论文时，它会把 `University of Oslo`、
`AGH University of Krakow`、`2050`、`10100 Burnet Rd., Bldg. 130, Austin, TX`
全部抽成节点 —— 实测 765 个节点里有 43 个（5.6%）是作者单位和年份。

这些节点不只是"多占几行"：
- 它们让图谱看起来更连通（几乎所有论文都提到同一所大学），但连的是**元数据**
  而不是知识，会把跨论文的实体合并引到错误的方向；
- 它们会进入实体向量库，在语义检索里和真正的领域实体抢名额。

注入方式
--------
LightRAG 1.5.x 支持 ``addon_params["entity_types_guidance"]``，该字符串会被
渲染进 ``---Entity Types---`` 段（system prompt 和 user prompt 各一次）。
这是官方预留的覆盖点，不需要 monkey-patch，也不依赖 ``PROMPT_DIR`` 环境变量。

为什么不用 YAML profile
-----------------------
``ENTITY_TYPE_PROMPT_FILE`` 走的是 YAML profile 通道，但它有一个硬性要求：
JSON 抽取模式下必须同时提供 ``entity_extraction_json_examples``，否则构造
LightRAG 时直接抛 ValueError。我们只想改类型表、不想重写示例，用
``addon_params`` 更直接，也少一个需要随包分发的文件。
"""
from __future__ import annotations

# 领域与类型表。用英文书写，与 LightRAG 自带的英文指令同段渲染；
# 实体名本身仍按原文语言输出（中文文献产出中文节点），这一点在最后一节明确。
ENTITY_TYPES_GUIDANCE = """---Domain---
This corpus covers CO2 capture, utilisation and storage (CCUS), CO2 geological
storage, caprock and fault integrity, wellbore integrity, and multiphase flow in
porous media. Extract only entities that carry domain knowledge of this kind.

---Allowed Entity Types---
Classify every entity as exactly one of the types below. There is no `Other`:
if nothing fits, do not extract the entity.

- GeologicalBody: rock units and structural features (reservoir, caprock, saline
  aquifer, fault, fracture, seal, formation, coal seam, salt diapir, overburden)
- WellboreComponent: well, casing, cement sheath, tubing, packer, plug,
  abandoned well, perforation
- Material: chemical species, minerals, engineering materials (CO2, brine, CH4,
  H2, calcite, halite, Portland cement, corrosion-resistant alloy)
- Process: physical, chemical or operational processes (injection, dissolution,
  mineralisation, convection, diffusion, leakage, migration, hydraulic
  fracturing, salt precipitation)
- Technology: methods, models, instruments (seismic imaging, InSAR, DAS/DTS,
  digital twin, deep learning, reservoir simulation, history matching)
- Parameter: measurable quantities (pressure, temperature, permeability,
  porosity, saturation, injection rate, stress, plume extent)
- Risk: failure modes and hazards (caprock integrity failure, induced
  seismicity, well leakage, groundwater contamination)
- Regulation: laws, standards and policies (45Q, UIC Class VI, EU CCS Directive)

---Never Extract---
The following are formatting artefacts, not domain knowledge. Skip them even
when they appear prominently:
- Institutions: universities, institutes, laboratories, companies, funding
  bodies (e.g. "University of Oslo", "SINTEF Industry", "China University of
  Petroleum")
- People: authors, editors, reviewers, any personal name
- Time: years, decades, dates (e.g. "2050", "2040s", "September 24, 2025")
- Places of publication: street addresses, postal codes, cities named only as
  an affiliation (e.g. "10100 Burnet Rd., Bldg. 130, Austin, TX 78758, USA")
- Bibliographic data: journal names, publishers, volumes, DOIs, "et al."
  citations, "Received Date", "A R T I C L E I N F O"
- Document scaffolding: section headings, figure/table captions, paper titles
- Generic filler: "This study", "Current Models", "Empirical Results",
  "additional measurements", "unreliable results"
- Bare symbols or equation variables that carry no domain meaning

If a passage contains no allowed entity, return an empty entity list. Precision
matters more than recall here: a node that is not geological, operational or
regulatory knowledge can never be retrieved usefully, and it competes for
retrieval slots with the nodes that can.

---Canonical Naming---
Emit the same name for the same thing so that nodes merge across papers. The
graph merges entities by exact name, so a variant spelling silently creates a
second node:
- Always write English entity names in Title Case: `Saline Aquifer`, never
  `saline aquifer` or `Saline aquifer`. Do not emit the same concept twice in
  two different casings.
- Singular only: `Fault` not `Faults`, `Caprock` not `Caprocks`.
- One canonical form per concept, ignoring case: `CO2` (never `Carbon Dioxide`,
  `Carbon dioxide (CO2)` or `CO₂`); `CCS` (never `Carbon Capture and Storage
  (CCS)`); `CCUS`; `Caprock` (never `Cap Rock`); `Saline Aquifer`; `Wellbore
  Integrity`; `Caprock Fracturing`; `CO2 Leakage`; `CO2 Plume`; `CO2
  Saturation`; `Digital Twin`; `Deep Learning`; `Reinforcement Learning`.
- Never append a citation, year or author to a name: `Digital Shadow`, not
  `Digital Shadow (Gahlot, Orozco, et al. 2024)`.
- Drop parenthetical glosses and acronyms from the name: `Calcite`, not
  `Calcite (Calcium Carbonate)`. Keep the acronym only when it *is* the
  canonical form (`CO2`, `CCS`).
- Keep proper nouns in their original language. For Chinese text emit the
  Chinese term (`盖层`, `断层`, `储层`) and do not translate it.
"""


def entity_types_guidance() -> str:
    """返回注入 LightRAG 的实体类型指引。"""
    return ENTITY_TYPES_GUIDANCE


def addon_params() -> dict:
    """
    构造 LightRAG 的 ``addon_params``。

    只覆盖 ``entity_types_guidance`` 一项 —— 其余键（``language`` /
    ``entity_type_prompt_file`` / ``chunker``）由 LightRAG 的
    ``normalize_addon_params`` 从环境变量回填默认值。显式传入的 Mapping
    不会触发 default_factory，所以不要在这里自己补 language：那样会绕过
    ``SUMMARY_LANGUAGE`` 环境变量的语义。
    """
    return {"entity_types_guidance": ENTITY_TYPES_GUIDANCE}
