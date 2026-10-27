import { useEffect, useMemo, useRef, useState } from "react";
import { SEVERITY_META, formatValue, isIndeterminate } from "../lib/severity";
import { RULER_DEPTHS } from "../lib/section";
import type { SectionData } from "../lib/section";
import type { GeoScene } from "../three/geoscene";
import { isQuality } from "../three/quality";
import type { Quality } from "../three/quality";
import { setActiveScene } from "../three/sceneRegistry";
import { cn } from "../lib/utils";
import { VoiceRing } from "./VoiceRing";

export type VoiceState = "idle" | "listening" | "thinking";

interface LabelDef {
  id: string;
  /** 主文本 */
  text: string;
  /** 第二行（读数 / 备注） */
  sub?: string;
  /** 主文本的语义色 class */
  tone?: string;
  /** 相对锚点的像素偏移 */
  dx: number;
  /**
   * 纵向偏移。地表测点（深度 0）必须往上挪，
   * 否则标签会压在剖面顶边那条线上 —— 而那条线正好是"地表"本身。
   */
  dy?: number;
  /** 对齐方式：决定标签从锚点往哪个方向展开 */
  align: "left" | "right" | "center";
  /** 层级：数字与测点标签要压过层名 */
  z: number;
}

interface Props {
  data: SectionData;
  mode: "deck" | "engineer";
  voice?: VoiceState;
}

/**
 * `?gpu=low` —— 强制 WebGL 落到集显。
 *
 * 混合显卡的机器上，指挥舱用 `high-performance` 会被路由到独显（产品行为正确），
 * 于是"集显能不能跑 60 FPS"这条承诺在这类机器上**永远量不到**：你量到的
 * 一直是独显。这个开关把 `powerPreference` 钉成 `low-power`，同一份代码、
 * 同样的几何、同样的 3000 个粒子，只是换一块 GPU。
 *
 * 和 `?perf=1` / `?mode=` / `?kb=search` 一样：一个只能靠改代码才能到达的
 * 状态，评审时总会有人说"你量一下集显"然后没人能量。
 */
function forceLowPower(): boolean {
  if (typeof window === "undefined") return false;
  return new URLSearchParams(window.location.search).get("gpu") === "low";
}

/**
 * `?quality=high|medium|low` —— 钉死渲染档位，关闭自适应降级。
 *
 * 存在的理由是**确定性**：截图、评审、比对三张图时，如果每台机器都按自己
 * 的帧耗时自动降档，同一份代码会产出三种画质，而它们看起来都"正常"。
 */
function pinnedQuality(): Quality | undefined {
  if (typeof window === "undefined") return undefined;
  const v = new URLSearchParams(window.location.search).get("quality");
  return isQuality(v) ? v : undefined;
}

/**
 * 3D 地质剖面（three.js 画布 + HTML 标签叠加）。
 *
 * ## 标签为什么在 DOM 里
 *
 * WebGL 里写字要么用字形图集，要么用 SDF 字体。中文常用字 3500 个，
 * 图集轻松几 MB，而这块屏上只出现 15 个标签。HTML 标签直接复用系统字体、
 * 能选中、能跟随 `prefers-reduced-motion`，代价只是每帧投影一次坐标。
 *
 * ## 为什么动态 import three
 *
 * `three` 压缩后 ~160 KB gzip。左栏的告警卡是"这个班次要不要叫人"的全部依据，
 * 它不该等一个 3D 库下载完才出现。把 three 拆成独立 chunk，
 * 左栏先亮、剖面随后补上，是这块屏上唯一正确的加载顺序。
 */
export default function GeoSectionCanvas({ data, mode, voice = "idle" }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sceneRef = useRef<GeoScene | null>(null);
  const labelRefs = useRef(new Map<string, HTMLElement>());
  const [ready, setReady] = useState(false);
  const [failed, setFailed] = useState<string | null>(null);

  // 场景构造只跑一次：数据变化走 updateData，模式变化走 applyMode。
  // 重建场景会把相机、锚点、纹理全部丢掉，视觉上是一次闪白。
  const dataRef = useRef(data);
  const modeRef = useRef(mode);
  dataRef.current = data;
  modeRef.current = mode;

  const labels = useMemo<LabelDef[]>(() => {
    const out: LabelDef[] = [];

    for (const m of data.monitors) {
      const sev = m.severity;
      const unknown = sev === null || isIndeterminate(sev);
      // 显式收窄而不是 `unknown ? A : B`：`unknown` 是个布尔变量，
      // TS 不会从它反推出 `sev` 非空，靠断言会把空值检查挪到编译器看不见的地方。
      let tone = "text-sev-unknown";
      if (sev !== null && !isIndeterminate(sev)) tone = SEVERITY_META[sev].token;

      // 靠右的测点标签向左展开，否则会顶出画布
      const align: LabelDef["align"] = m.x > 320 ? "right" : "left";
      out.push({
        id: `monitor:${m.id}`,
        text: m.label,
        sub: unknown ? m.caveat || "不可判定" : `${formatValue(m.value)} ${m.unit}`,
        tone,
        dx: align === "right" ? -14 : 14,
        dy: m.dy ?? (m.depth <= 0 ? -12 : 0),
        align,
        z: 30,
      });
    }

    for (const s of data.strata) {
      out.push({
        id: `stratum:${s.key}`,
        text: s.label,
        tone: s.seal ? "text-ink-0" : "text-ink-2",
        dx: 10,
        align: "left",
        z: 10,
      });
    }

    for (const d of RULER_DEPTHS) {
      out.push({
        id: `ruler:${d}`,
        text: `${d} m`,
        tone: "text-ink-2",
        dx: -8,
        align: "right",
        z: 10,
      });
    }

    out.push({
      id: "perforation",
      text: "射孔段",
      sub: `${data.plume.base} m`,
      tone: "text-acc",
      dx: 14,
      align: "left",
      z: 20,
    });

    return out;
  }, [data]);

  useEffect(() => {
    let cancelled = false;
    let instance: GeoScene | null = null;
    let ro: ResizeObserver | null = null;

    import("../three/geoscene")
      .then(({ GeoScene }) => {
        if (cancelled) return;
        const canvas = canvasRef.current;
        const wrap = wrapRef.current;
        if (!canvas || !wrap) return;

        const scene = new GeoScene(canvas, dataRef.current, modeRef.current, {
          lowPower: forceLowPower(),
          quality: pinnedQuality(),
        });
        instance = scene;
        sceneRef.current = scene;
        // 登记给性能探针（`?perf=1`）。失败不影响主流程 ——
        // 探针读不到场景时会显示 "—"，那本身就是正确的读数。
        setActiveScene(scene);

        // 相机每帧都在漂移（指挥舱），标签必须跟着投影，
        // 否则数字会慢慢从圆点上滑走 —— 而且不会报错
        scene.onAfterRender = () => {
          for (const a of scene.screenAnchors()) {
            const el = labelRefs.current.get(a.id);
            if (!el) continue;
            el.style.opacity = a.visible ? "1" : "0";
            el.style.transform = `translate3d(${a.x}px, ${a.y}px, 0)`;
          }
        };

        const apply = () =>
          scene.resize(wrap.clientWidth, wrap.clientHeight, window.devicePixelRatio || 1);
        apply();

        ro = new ResizeObserver(apply);
        ro.observe(wrap);

        // 首次 resize 已经渲染过一帧，标签要立刻到位（否则会先闪在左上角）
        scene.onAfterRender();
        setReady(true);
      })
      .catch((e: unknown) => {
        // WebGL 上下文创建失败在工控机上是真实会发生的事，不是理论风险
        if (!cancelled) setFailed(e instanceof Error ? e.message : String(e));
      });

    return () => {
      cancelled = true;
      // 观察器必须在这里断开。写在 `.then()` 的返回值里没用 ——
      // 那是 Promise 的返回值，没人会调它，于是每换一次数据就多一个泄漏的 observer
      ro?.disconnect();
      instance?.dispose();
      sceneRef.current = null;
      // 不清空的话，切到"全链"Tab 之后探针仍会读到上一帧的 stats()，
      // 显示出一组看起来正常、但已经不属于任何活着的场景的数字。
      setActiveScene(null);
    };
  }, []);

  useEffect(() => {
    sceneRef.current?.updateData(data);
  }, [data]);

  useEffect(() => {
    sceneRef.current?.applyMode(mode);
  }, [mode]);

  if (failed) {
    return (
      <div className="grid h-full place-items-center p-4 text-center">
        <div className="space-y-1">
          <p className="text-xs text-sev-critical">3D 视图不可用：{failed}</p>
          <p className="text-2xs text-ink-2">
            这台机器没能创建 WebGL 上下文。左侧告警与中央链路状态不受影响。
          </p>
        </div>
      </div>
    );
  }

  return (
    <div ref={wrapRef} className="relative h-full w-full overflow-hidden">
      <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />

      {/* 标签层：全部 pointer-events-none，避免挡住画布 */}
      <div className="pointer-events-none absolute inset-0">
        {labels.map((l) => (
          <div
            key={l.id}
            ref={(el) => {
              if (el) labelRefs.current.set(l.id, el);
              else labelRefs.current.delete(l.id);
            }}
            className="absolute left-0 top-0 whitespace-nowrap opacity-0 transition-opacity duration-300"
            style={{ zIndex: l.z }}
          >
            <div
              style={{
                transform:
                  l.align === "right"
                    ? "translate(-100%, -50%)"
                    : l.align === "center"
                      ? "translate(-50%, -50%)"
                      : "translate(0, -50%)",
                marginLeft: l.dx,
                marginTop: l.dy ?? 0,
              }}
            >
              <div className={cn("text-2xs leading-tight", l.tone ?? "text-ink-1")}>
                {l.text}
              </div>
              {l.sub && (
                <div className="tnum text-2xs leading-tight text-ink-2">{l.sub}</div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* 盖层余量标注：这张图上最该被读到的一个数 */}
      <SealClearanceBadge data={data} mode={mode} />

      <div className="pointer-events-none absolute inset-x-0 bottom-2 flex justify-center">
        <VoiceRing state={voice} mode={mode} />
      </div>

      {!ready && (
        <div className="absolute inset-0 grid place-items-center text-2xs text-ink-2">
          正在构建剖面…
        </div>
      )}
    </div>
  );
}

/**
 * 羽流顶部到盖层底界的距离。
 *
 * 单独做成一块浮标，是因为它无法从"看"得出：羽流顶和盖层底在图上贴得很近，
 * 差 6 m 和差 60 m 在屏幕上几乎一样。而这个数决定的是"CO₂ 有没有顶到封闭层"，
 * 是整套系统里唯一一个失效后没有补救手段的环节。
 */
function SealClearanceBadge({ data, mode }: { data: SectionData; mode: "deck" | "engineer" }) {
  const gap = data.sealClearance;
  // 门槛本身是政策选择：30 m 以内就该进提示带。写出来，不要让它看起来像标准。
  const tight = gap <= 30;

  return (
    <div className="pointer-events-none absolute bottom-2 right-2 max-w-[13rem] rounded-panel border border-line bg-surface-0/80 px-2 py-1.5 backdrop-blur-sm">
      <div className="text-2xs text-ink-2">羽流顶距盖层底界</div>
      <div
        className={cn(
          "tnum text-base leading-tight",
          tight ? "text-sev-notice" : "text-ink-0",
        )}
      >
        {formatValue(gap, 0)} <span className="text-2xs text-ink-2">m</span>
      </div>
      <div className="text-2xs leading-tight text-ink-2">
        {tight ? "已贴近封闭层" : "仍在储层内"}
        {mode === "engineer" && " · 30 m 为项目阈值"}
      </div>
    </div>
  );
}
