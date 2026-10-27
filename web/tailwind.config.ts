import type { Config } from "tailwindcss";

/**
 * 中文注释：令牌只在 src/index.css 里定义一次，这里做的是**映射**而不是
 * 重新声明一遍。两处各写一份色值，改一处就会有一半的界面还停在旧颜色上 ——
 * 而这种不一致在深色界面里几乎看不出来，只会让人觉得"哪里怪"。
 *
 * 颜色值走 `rgb(var(--x) / <alpha-value>)`，这样 `bg-sev-critical/10`
 * 才真的能变淡。直接写 `var(--x)` 会让透明度修饰符静默失效。
 */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        surface: {
          0: "rgb(var(--bg-0) / <alpha-value>)",
          1: "rgb(var(--bg-1) / <alpha-value>)",
          2: "rgb(var(--bg-2) / <alpha-value>)",
          3: "rgb(var(--bg-3) / <alpha-value>)",
        },
        ink: {
          0: "rgb(var(--txt-0) / <alpha-value>)",
          1: "rgb(var(--txt-1) / <alpha-value>)",
          2: "rgb(var(--txt-2) / <alpha-value>)",
        },
        // 中文注释：描边色本身已带 alpha，不走修饰符。
        line: {
          DEFAULT: "var(--line)",
          strong: "var(--line-2)",
        },
        acc: {
          DEFAULT: "rgb(var(--acc) / <alpha-value>)",
          alt: "rgb(var(--acc-2) / <alpha-value>)",
        },
        // 中文注释：六级语义色。**品牌色不在这张表里** —— 中石化红与
        // critical 的色相只差 4°，同屏会把"品牌"读成"严重告警"。
        sev: {
          normal: "rgb(var(--sev-normal) / <alpha-value>)",
          notice: "rgb(var(--sev-notice) / <alpha-value>)",
          warn: "rgb(var(--sev-warn) / <alpha-value>)",
          alert: "rgb(var(--sev-alert) / <alpha-value>)",
          critical: "rgb(var(--sev-critical) / <alpha-value>)",
          unknown: "rgb(var(--sev-unknown) / <alpha-value>)",
        },
      },
      fontFamily: {
        cjk: "var(--font-cjk)",
        num: "var(--font-num)",
        mono: "var(--font-mono)",
      },
      borderRadius: {
        // 中文注释：工业界面用直角，圆角只用于柔化，不做胶囊。
        panel: "2px",
      },
      fontSize: {
        // 中文注释：字号阶来自设计规范 §2.3。工程师模式整体 ×1.15，
        // 由 html[data-mode="engineer"] 里的 --type-scale 控制。
        "2xs": "calc(10.5px * var(--type-scale))",
        xs: "calc(11.5px * var(--type-scale))",
        sm: "calc(13px * var(--type-scale))",
        base: "calc(15px * var(--type-scale))",
        lg: "calc(17px * var(--type-scale))",
        xl: "calc(25px * var(--type-scale))",
        "2xl": "calc(30px * var(--type-scale))",
      },
    },
  },
  plugins: [],
} satisfies Config;
