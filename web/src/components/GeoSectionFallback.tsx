import { SEVERITY_META, formatValue, isIndeterminate } from "../lib/severity";
import { DEPTH_MAX, HALF_WIDTH_M, RULER_DEPTHS, SEAL_BASE, plumeRadiusM } from "../lib/section";
import { withUnit } from "../lib/units";
import type { SectionData } from "../lib/section";

/**
 * 地质剖面的 SVG 降级版。
 *
 * ## 它为什么存在
 *
 * 不是"没做 3D 所以画个 2D 凑数"。它存在的理由只有一个：**WebGL 会失败**。
 * 老工控机的 Intel 集显驱动、远程桌面会话、被策略禁用的硬件加速，
 * 这三样在油田现场都不罕见。3D 起不来时，这块屏不能变成一句报错 ——
 * 剖面上有 5 个测点、羽流范围和盖层间距，它们是数据，不是特效。
 *
 * ## 它和 3D 版的数字来自同一份 `section.ts`
 *
 * 坐标、层序、羽流半径函数都复用，所以两个版本不可能画出两个不同的剖面。
 * 差异只在表现力：没有厚度、没有粒子、没有透视。
 *
 * 尺寸按 `viewBox` 归一化，容器多大都能撑满。
 */

const W = 900;
const H = 700;
const PAD_L = 96;
const PAD_R = 132;

const xToPx = (xM: number) => PAD_L + ((xM + HALF_WIDTH_M) / (HALF_WIDTH_M * 2)) * (W - PAD_L - PAD_R);
const dToPx = (dM: number) => (dM / DEPTH_MAX) * H;

export function GeoSectionFallback({ data }: { data: SectionData }) {
  const p = data.plume;
  const topY = dToPx(p.apex);
  const botY = dToPx(p.base);

  const right: string[] = [];
  const left: string[] = [];
  for (let i = 0; i <= 40; i++) {
    const t = i / 40;
    const y = botY + (topY - botY) * t;
    right.push(`${xToPx(p.cx + plumeRadiusM(p, t))},${y}`);
    left.push(`${xToPx(p.cx - plumeRadiusM(p, t))},${y}`);
  }
  const plumePath = `M ${right.join(" L ")} L ${left.reverse().join(" L ")} Z`;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="xMidYMid meet"
      className="h-full w-full"
      role="img"
      aria-label="地下地质剖面（SVG 降级视图）"
    >
      <defs>
        <pattern id="fb-hatch" width="10" height="10" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="10" stroke="rgba(200,230,240,0.18)" strokeWidth="1.4" />
        </pattern>
        <pattern id="fb-dots" width="16" height="16" patternUnits="userSpaceOnUse">
          <circle cx="3" cy="4" r="1.3" fill="rgba(200,230,240,0.16)" />
          <circle cx="11" cy="12" r="1" fill="rgba(200,230,240,0.14)" />
        </pattern>
        <linearGradient id="fb-plume" x1="0" y1="1" x2="0" y2="0">
          <stop offset="0%" stopColor="rgb(var(--acc))" stopOpacity="0.12" />
          <stop offset="100%" stopColor="rgb(var(--acc))" stopOpacity="0.34" />
        </linearGradient>
      </defs>

      {data.strata.map((s) => {
        const y = dToPx(s.top);
        const h = dToPx(s.bottom) - y;
        return (
          <g key={s.key}>
            <rect
              x={PAD_L}
              y={y}
              width={W - PAD_L - PAD_R}
              height={h}
              fill={`rgb(var(${s.colorVar}))`}
              opacity={s.seal ? 0.55 : 0.4}
            />
            <rect
              x={PAD_L}
              y={y}
              width={W - PAD_L - PAD_R}
              height={h}
              fill={s.texture === "hatch" ? "url(#fb-hatch)" : s.texture === "dots" ? "url(#fb-dots)" : "none"}
            />
            <line
              x1={PAD_L}
              y1={y}
              x2={W - PAD_R}
              y2={y}
              stroke="rgba(122,178,204,0.28)"
              strokeWidth={s.seal ? 1.2 : 0.8}
            />
            <text
              x={PAD_L + 10}
              y={y + 14}
              className="fill-ink-2"
              fontSize="11"
              fill={s.seal ? "rgb(var(--txt-0))" : "rgb(var(--txt-2))"}
            >
              {s.label}
            </text>
          </g>
        );
      })}

      {/* 盖层底界 */}
      <line
        x1={PAD_L}
        y1={dToPx(SEAL_BASE)}
        x2={W - PAD_R}
        y2={dToPx(SEAL_BASE)}
        stroke="rgb(var(--acc-2))"
        strokeWidth="1"
        strokeOpacity="0.55"
      />

      {/* 羽流：数据，不是装饰 */}
      <path d={plumePath} fill="url(#fb-plume)" stroke="rgb(var(--acc))" strokeWidth="1.2" strokeOpacity="0.8" />

      {/* 等压虚线：压力影响范围比 CO₂ 范围更大 */}
      {[1.18, 1.5].map((f, i) => (
        <ellipse
          key={f}
          cx={xToPx(p.cx)}
          cy={botY + (topY - botY) * 0.78}
          rx={((p.radius * f) / (HALF_WIDTH_M * 2)) * (W - PAD_L - PAD_R)}
          ry={((p.radius * f * 0.34) / DEPTH_MAX) * H}
          fill="none"
          stroke="rgb(var(--acc-2))"
          strokeOpacity={0.3 - i * 0.08}
          strokeDasharray="5 4"
        />
      ))}

      {/* 注入井 */}
      <line x1={xToPx(0)} y1={0} x2={xToPx(0)} y2={dToPx(p.base)} stroke="rgb(var(--acc))" strokeOpacity="0.5" />
      <line
        x1={xToPx(0) - 16}
        y1={dToPx(p.base)}
        x2={xToPx(0) + 16}
        y2={dToPx(p.base)}
        stroke="rgb(var(--acc))"
        strokeOpacity="0.9"
        strokeWidth="2"
      />

      {/* 测点 */}
      {data.monitors.map((m) => {
        const unknown = m.severity === null || isIndeterminate(m.severity);
        let token = "unknown";
        if (m.severity !== null && !isIndeterminate(m.severity)) {
          token = SEVERITY_META[m.severity].token;
        }
        const cx = xToPx(m.x);
        const cy = dToPx(m.depth);
        const alignRight = m.x > 320;
        return (
          <g key={m.id}>
            <circle
              cx={cx}
              cy={cy}
              r={unknown ? 5.5 : 6.5}
              fill={unknown ? "none" : `rgb(var(--sev-${token}))`}
              stroke={`rgb(var(--sev-${token}))`}
              strokeWidth={unknown ? 1.4 : 0}
              strokeDasharray={unknown ? "3 2.5" : undefined}
            />
            <text
              x={alignRight ? cx - 14 : cx + 14}
              y={cy - 3}
              textAnchor={alignRight ? "end" : "start"}
              fontSize="11"
              fill={`rgb(var(--sev-${token}))`}
            >
              {m.label}
            </text>
            <text
              x={alignRight ? cx - 14 : cx + 14}
              y={cy + 10}
              textAnchor={alignRight ? "end" : "start"}
              fontSize="11"
              fill="rgb(var(--txt-2))"
            >
              {unknown ? m.caveat || "不可判定" : withUnit(formatValue(m.value), m.unit)}
            </text>
          </g>
        );
      })}

      {/* 深度尺 */}
      {RULER_DEPTHS.map((d) => (
        <g key={d}>
          <line
            x1={W - PAD_R + 6}
            y1={dToPx(d)}
            x2={W - PAD_R + 16}
            y2={dToPx(d)}
            stroke="rgba(122,178,204,0.35)"
          />
          <text x={W - PAD_R + 22} y={dToPx(d) + 4} fontSize="10" fill="rgb(var(--txt-2))">
            {d} m
          </text>
        </g>
      ))}

      <text x={PAD_L + 10} y={dToPx(p.apex) - 6} fontSize="10" fill="rgb(var(--acc))">
        羽流顶 {p.apex} m · 距盖层 {data.sealClearance} m
      </text>
    </svg>
  );
}
