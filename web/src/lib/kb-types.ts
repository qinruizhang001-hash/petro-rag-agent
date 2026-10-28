/**
 * 知识库接口的响应类型（对应 `api_kb.py`）。
 *
 * 字段名与后端**逐字对应**，不做 camelCase 转换。理由与 `types.ts` 相同：
 * 转换层是第二份字段名，两份迟早不一致，而不一致的表现是某个数字永远显示 0。
 */

/** 语料 / 图谱 / 向量库三块状态。 */
export interface KbOverview {
  corpus: {
    /** 盘上的语料文件数。**不含** `manifest.json` 之类的元数据文件 */
    files: number;
    bytes: number;
    by_kind: Record<string, number>;
  };
  graph: {
    documents: number;
    chunks: number;
    chars: number;
    /** 建图总耗时（秒）。新增一篇 12 万字符的论文约 30 分钟，这个数字决定要不要排到夜里 */
    ingest_seconds: number;
    entities: number;
    relationships: number;
    graph_bytes: number;
  };
  vector: {
    available: boolean;
    active_collection: string;
    vectors: number;
    collections: Array<{ name: string; vectors: number }>;
    /**
     * 是否有文件级血缘。
     *
     * 当前是 `false` —— Chroma 侧的 metadata 只有 `{"source": "legacy_v2"}`。
     * 这意味着向量分支的召回**无法**回溯到某一篇文献，界面上必须照实显示，
     * 否则用户会以为每条结论都能点回原文。
     */
    lineage: boolean;
  };
  gaps: {
    /** 盘上有、图谱里没有 —— 界面上最该被看见的一组 */
    not_ingested: string[];
    unknown_source: string[];
    /** 图谱里有、盘上已删 */
    orphaned: string[];
    /** 状态不是 processed 的文档，会阻塞后续插入 */
    unfinished: string[];
  };
  readiness: { graph: string; vector: string };
  titles: { covered: number; missing: string[] };
}

export interface KbDocumentRow {
  name: string;
  bytes: number;
  kind: string;
  ingested: boolean;
  status: string;
  chunks: number;
  chars: number;
  /** 题录。手工放进去的 PDF 没有条目，这里是空串 —— 不拿文件名硬凑 */
  title: string;
  authors: string[];
  published: string;
  url: string;
  ingested_at: string;
  ingest_seconds: number;
}

export interface KbDocuments {
  total: number;
  documents: KbDocumentRow[];
}

/**
 * 原文切片卡 —— 唯一可引用的一类证据。
 *
 * 实体与关系是模型抽出来的中间产物，读者要核对"这句到底是不是这么写的"
 * 时只能看原文，所以引用角标只挂在切片上。
 */
export interface KbChunkCard {
  kind: "chunk";
  /** 响应内编号（"1"、"2"…）。跨查询不稳定，不能当文档 ID 存下来 */
  reference_id: string;
  chunk_id: string;
  source: string;
  content: string;
  chars: number;
  truncated: boolean;
  /**
   * 这条切片能不能读成句子。
   *
   * 存在的理由是一次实测：问「盖层完整性对 CO2 泄漏风险的影响」，检索器
   * 排第一的是一条 `𝑡+1 = 𝐶(𝑥𝑡)𝑥𝑡+1 + 𝐷(𝑥𝑡)𝑢𝑡` 的离散格式公式 ——
   * 满篇 CO2 / storage / pressure，向量相似度确实高，但它提供不了任何
   * 机理证据。
   *
   * **不重排**：引用编号必须等于检索名次，否则报告里的 `[1]` 回不到检索
   * 日志。所以只标注，顺序照旧。
   */
  text_quality: {
    /** 字母（CJK + 拉丁）占非空白字符的比例 */
    readable_ratio: number;
    /** 数学字母数字符号（U+1D400–U+1D7FF）占比 */
    math_ratio: number;
    /** 以公式为主 → 界面标"公式为主"，不改变排序 */
    low: boolean;
  };
}

export interface KbEntityCard {
  kind: "entity";
  name: string;
  entity_type: string;
  description: string;
  /** 多来源：合并实体必然来自多篇论文（实测最多 16 篇） */
  sources: string[];
  chunk_ids: string[];
  truncated: boolean;
}

export interface KbRelationCard {
  kind: "relationship";
  src: string;
  tgt: string;
  keywords: string[];
  description: string;
  sources: string[];
  truncated: boolean;
}

export interface KbEvidence {
  chunks: KbChunkCard[];
  entities: KbEntityCard[];
  relationships: KbRelationCard[];
  references: Array<{ reference_id: string; source: string }>;
  stats: {
    chunks: number;
    entities: number;
    relationships: number;
    sources: number;
    /** 说不出来源的条数。必须显示 —— 少一个文件名不构成任何提示 */
    unsourced: number;
    traceable: number;
    /** 以公式为主、难以作为机理性证据的条数。同样必须显示 */
    low_quality_chunks: number;
  };
}

/**
 * 检索结果的三种状态。
 *
 * `empty` 与 `unavailable` **不能合并**：前者是关于语料的事实陈述
 * （"库里没有这个内容"），后者是系统状态（"检索没跑起来"）。
 * 把后者渲染成前者，操作员会去补文献 —— 而真实原因只是 Ollama 没启动。
 */
export type KbSearchStatus = "ok" | "empty" | "unavailable";

export interface KbSearchResult {
  status: KbSearchStatus;
  query: string;
  mode: string;
  reason: string;
  evidence: KbEvidence;
  elapsed_ms: number;
}

export interface KbDocDetail {
  name: string;
  title: string;
  url: string;
  status: string;
  total_chunks: number;
  offset: number;
  /** 命中的切片 id（引用跳转时定位到这里） */
  focused: string | null;
  has_more: boolean;
  chunks: Array<{
    chunk_id: string;
    index: number;
    chars: number;
    content: string;
  }>;
}

/**
 * 从对话里的引用角标跳到知识库详情时的入参。
 *
 * ## 为什么带 `fallback`
 *
 * 跳转落点是 `/kb/documents/{name}`，而 fixture 模式下**没有**这个端点
 * （12 篇论文的全文切片是 MB 级，不为演示存一份）。但检索响应里
 * **已经有那一片原文** —— 把它带过去，落点就成立：读者看到的是真实的
 * 那一片，而不是一个"演示数据不可用"的错误页。
 *
 * 带 `fallback` 时详情页**不显示总片数**：那个数字没取到。
 * 显示成"共 1 片"会把"缺数据"说成"数据是 1"。
 */
export interface FocusDoc {
  name: string;
  /** 命中切片的 id，后端会换算成 `chunks[].chunk_id` 并高亮 */
  focus?: string;
  fallback?: {
    chunk_id: string;
    chars: number;
    content: string;
  };
}

/**
 * 缺口反馈（`POST /api/kb/gaps`）。
 *
 * ## 为什么种类是枚举而不是自由文本
 *
 * 队列是给**人**看的。自由文本的后果是同一个意思有十种写法，而分类的
 * 全部价值在于"一眼看出积压的是哪一类" —— 那需要能计数。
 *
 * ## 为什么有的种类必须带对象
 *
 * `not_in_graph` 不说哪一篇、`wrong_evidence` 不带当时的问题，处理这条
 * 反馈的人第一步就只能回来问反馈的人。**说不出对象的反馈不是反馈**，
 * 所以那几种缺字段时后端直接 400（见 `api_kb._REQUIRED_FIELD`）。
 */
export type KbGapKind =
  | "missing_document"
  | "not_in_graph"
  | "unanswered"
  | "wrong_evidence"
  | "other";

export interface KbGapInput {
  kind: KbGapKind;
  /** 相关文件名。只允许裸名字 —— 带路径会被后端拒（400） */
  target?: string;
  /** 当时问的问题。`wrong_evidence` 必填 */
  query?: string;
  /** 补充说明。与 target / query 至少有一个非空，否则后端 400 */
  note?: string;
}

export interface KbGapRecord extends KbGapInput {
  id: string;
  /** ISO 8601，UTC */
  recorded_at: string;
  /**
   * 提交后会发生什么。**由后端给**，前端不自己编 ——
   * 承诺分散在几个组件里时，总有一份会说成"将自动重建索引"。
   */
  next_step?: string;
}

export interface KbGapsPayload {
  /**
   * 队列文件是否读得出来。
   *
   * 与 `total` 分开是因为读不到时 `total` 也是 0 —— 而"没人反馈"和
   * "读不出来"在界面上必须长得不一样。把前者显示成后者是唯一
   * 不会报错的失败。
   */
  available: boolean;
  reason: string;
  total: number;
  /** 坏行的条数。不吞掉：全坏的时候界面上会像"队列是空的" */
  unreadable_lines: number;
  by_kind: Record<string, number>;
  /** 新的在前 */
  records: KbGapRecord[];
  next_step?: string;
}
