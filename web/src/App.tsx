import { useCallback, useEffect, useState } from "react";
import { GeoSection } from "./components/GeoSection";
import type { VoiceState } from "./components/GeoSectionCanvas";
import { LeftRail } from "./components/LeftRail";
import { PerfOverlay } from "./components/PerfOverlay";
import { RightRail } from "./components/RightRail";
import { StatusBar } from "./components/StatusBar";
import { dataSource, fetchAlarms, fetchThresholds } from "./lib/api";
import { HEALTH_INITIAL, probeBackend } from "./lib/health";
import type { BackendHealth } from "./lib/health";
import { createPushClient, pushUrl } from "./lib/push";
import type { PushStatus } from "./lib/push";
import type { AlarmReport, ThresholdStatus } from "./lib/types";
import { cn } from "./lib/utils";

type Mode = "deck" | "engineer";

/**
 * 性能探针开关（`?perf=1`）。
 *
 * 默认关闭。常驻的性能数字会变成背景噪声，最后谁都不看；而它真正的用途
 * 是**取证** —— 判断"这版还能不能上集显"时打开一次，看完关掉。
 * 同时它也是 `web/scripts/measure-fps.mjs` 的入口（通过 `window.__perf`）。
 */
function perfEnabled(): boolean {
  if (typeof window === "undefined") return false;
  return new URLSearchParams(window.location.search).get("perf") === "1";
}

/**
 * 推送通道地址。
 *
 * 优先 `?ws=`（供联调与自测：不用重新构建就能指到一个假服务端），
 * 其次 `VITE_OPS_WS`。两者都没有就返回 `null`，**不建立任何连接** ——
 * 一个静默重试的假连接会让人以为推送是通的。
 */
function resolvePushUrl(): string | null {
  if (typeof window !== "undefined") {
    const fromQuery = new URLSearchParams(window.location.search).get("ws");
    if (fromQuery) return fromQuery;
  }
  return pushUrl();
}

/**
 * 初始模式。
 *
 * 默认**工程师模式**，不是指挥舱。设计规范 §8 的建议是"先做工程师模式"：
 * 它没有 3D、没有动效、没有语音，但它是客户每天真正在用的那个；
 * 指挥舱是投标和汇报用的，做起来快、演示效果好，容易让人误以为进度很快。
 *
 * 支持 `?mode=deck` 是为了截图与评审 —— 一个只能在浏览器里点两下才能到达的
 * 状态，评审时就会有人截错图。
 */
function initialMode(): Mode {
  if (typeof window === "undefined") return "engineer";
  return new URLSearchParams(window.location.search).get("mode") === "deck"
    ? "deck"
    : "engineer";
}

/**
 * 双模不是开关，是**三件事**（设计规范 §4）：
 * 1. 中央区换视图（这里先只做字号与装饰的差异）；
 * 2. 字号整体 ×1.15 —— 由 `html[data-mode]` 上的 `--type-scale` 驱动；
 * 3. 关掉网格底与全部动效 —— 现场强光下，纹理和动画都是噪声。
 *
 * 所以模式写在 `<html>` 的 data 属性上而不是某个组件的 props 里：
 * 字号缩放要作用到所有元素，靠 React 逐层传参会漏。
 */
export default function App() {
  const [mode, setMode] = useState<Mode>(initialMode);
  /**
   * 语音状态。当前只有待机 —— 麦克风与 ASR 都还没接。
   *
   * 留着 `?voice=listening|thinking` 是为了截图与评审：状态环的三个形态
   * 如果只能靠改代码才能看到，评审时就会有人只看到待机，
   * 然后以为"这个环不会动"。
   */
  const [voice] = useState<VoiceState>(() => {
    if (typeof window === "undefined") return "idle";
    const v = new URLSearchParams(window.location.search).get("voice");
    return v === "listening" || v === "thinking" ? v : "idle";
  });
  const [report, setReport] = useState<AlarmReport | null>(null);
  const [thresholds, setThresholds] = useState<ThresholdStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [perf] = useState(perfEnabled);
  const [push, setPush] = useState<PushStatus>("idle");
  const [health, setHealth] = useState<BackendHealth>(HEALTH_INITIAL);

  /*
   * 后端可达性。
   *
   * 与数据加载**分开**：`Promise.all` 里任何一项失败都会让整屏进入
   * "数据加载失败"，而后端在不在跟这份快照能不能显示是两件事。
   * `probeBackend` 永不抛，所以这里不需要 catch。
   */
  const probe = useCallback(async () => {
    setHealth((h) => ({ ...h, state: "checking" }));
    setHealth(await probeBackend());
  }, []);

  useEffect(() => {
    void probe();
  }, [probe]);

  useEffect(() => {
    document.documentElement.dataset.mode = mode;
  }, [mode]);

  useEffect(() => {
    let alive = true;
    Promise.all([fetchAlarms(), fetchThresholds()])
      .then(([r, t]) => {
        if (!alive) return;
        setReport(r);
        setThresholds(t);
      })
      .catch((e: unknown) => {
        if (alive) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      alive = false;
    };
  }, []);

  /*
   * 实时推送。
   *
   * 未配置地址时**什么也不做** —— 这是默认路径，不是降级路径。
   * 配置之后，组件仍然只拿到 `AlarmReport`：它不知道这份数据是拉来的
   * 还是推来的。这就是"UI 与数据源解耦"的具体形态，而不是一句口号。
   */
  useEffect(() => {
    const url = resolvePushUrl();
    if (!url) return;

    const client = createPushClient(url);
    const offMessage = client.subscribe((m) => {
      if (m.type === "alarm") setReport(m.payload);
      else if (m.type === "thresholds") setThresholds(m.payload);
      // `ping` 只用于保活，没有可更新的状态
    });
    const offStatus = client.onStatus(setPush);

    return () => {
      offMessage();
      offStatus();
      client.close();
    };
  }, []);

  return (
    <div className="relative z-10 flex h-full flex-col bg-surface-0">
      <Header mode={mode} onModeChange={setMode} />

      {error && (
        <div className="border-b border-sev-critical/60 bg-sev-critical/10 px-3 py-2 text-xs text-sev-critical">
          数据加载失败：{error}
        </div>
      )}

      <main
        className={cn(
          "grid min-h-0 flex-1 gap-2.5 p-2.5",
          /*
           * 行高必须被**钉死**成容器高度，不能交给内容决定。
           *
           * 这条不是排版偏好，是修一个真实故障：右栏接上知识库之后，
           * 检索结果能有三屏高，而 grid 的隐式行是 `auto` —— 行高被内容
           * 撑到 3000 px，中央剖面（`h-full`）跟着变成 3000 px 高，
           * 相机按 3000×900 取景，于是**整个剖面被压到画布最底下**，
           * 看起来像"3D 挂了"。
           *
           * `minmax(0,1fr)` 是唯一能压住它的写法：`1fr` 单独用会被
           * `min-height:auto` 顶开，仍然按内容算。
           *
           * 三档布局：
           * - 窄屏（< 1024）：单列三行、行高交给内容、整页滚。剖面的高度
           *   靠 `min-h` 撑住，否则画布在 auto 行里会是 0 px 高。
           * - 平板（1024–1279）：两列，右栏落到第二行整宽（460 px）。
           * - 桌面（≥ 1280）：三列，行高钉死，各栏内部自己滚。
           */
          "grid-rows-none lg:grid-rows-[minmax(0,1fr)_460px] xl:grid-rows-[minmax(0,1fr)]",
          "overflow-auto lg:overflow-hidden",
          "grid-cols-1 lg:grid-cols-[minmax(300px,22rem)_minmax(0,1fr)] xl:grid-cols-[minmax(300px,22rem)_minmax(0,1fr)_minmax(280px,20rem)]",
        )}
      >
        {report && thresholds ? (
          <>
            <LeftRail report={report} thresholds={thresholds} />
            <GeoSection
              report={report}
              thresholds={thresholds}
              mode={mode}
              voice={voice}
              className="min-h-[440px] lg:min-h-0"
            />
            {/*
              `min-h-0` 不能省：包一层的 div 是 grid item，它的高度默认由内容
              决定，而 `h-full` 的 Panel 拿到的是一个被内容撑高的父级 ——
              右栏于是不再"填满这一格"，而是"和内容一样高"。

              窄屏不再是 `hidden`：藏掉右栏等于在手机上关掉知识库与对话，
              而这两样恰好是"这条结论出自哪篇"的唯一入口。
            */}
            <div className="min-h-[480px] min-w-0 lg:col-span-2 lg:min-h-0 xl:col-span-1">
              <RightRail />
            </div>
          </>
        ) : (
          !error && (
            <div className="col-span-full grid place-items-center text-xs text-ink-2">
              载入中…
            </div>
          )
        )}
      </main>

      {report && (
        <StatusBar
          counts={report.counts}
          source={dataSource}
          period={report.period}
          push={push}
          health={health}
          onProbe={() => void probe()}
        />
      )}

      {perf && <PerfOverlay />}
    </div>
  );
}

function Header({
  mode,
  onModeChange,
}: {
  mode: Mode;
  onModeChange: (m: Mode) => void;
}) {
  return (
    <header className="flex items-center gap-3 border-b border-line px-3 py-2">
      {/*
        品牌色只允许出现在这四处身份元素里：页眉细线、Logo、客户联合品牌位、面板角标。
        数据区永远是青 / 钢蓝 + 六级语义色 —— 中石化红与 critical 的色相只差 4°，
        同屏会把"品牌"读成"严重告警"。
      */}
      <div className="flex items-baseline gap-2">
        <span className="text-sm tracking-wide text-ink-0">CCUS 全链运维</span>
        <span className="text-2xs text-ink-2">捕集 · 运输 · 注入 · MVA</span>
      </div>

      <div className="ml-auto flex items-center gap-2">
        <div
          className="flex rounded-panel border border-line p-0.5"
          role="group"
          aria-label="显示模式"
        >
          {(
            [
              ["engineer", "工程师"],
              ["deck", "指挥舱"],
            ] as const
          ).map(([value, label]) => (
            <button
              key={value}
              type="button"
              onClick={() => onModeChange(value)}
              aria-pressed={mode === value}
              className={cn(
                "rounded-panel px-2 py-0.5 text-2xs",
                mode === value
                  ? "bg-acc/15 text-acc"
                  : "text-ink-2 hover:text-ink-1",
              )}
            >
              {label}
            </button>
          ))}
        </div>
      </div>
    </header>
  );
}
