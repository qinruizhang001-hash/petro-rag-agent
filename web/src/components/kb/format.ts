/**
 * 建图耗时：秒 → 数值 + 单位分开返回。
 *
 * 不拼成一个字符串再塞进 `font-num`：那会把"小时"两个字也交给 DIN 渲染，
 * 靠字体回退显示。数字用数字字体、汉字用汉字字体，是这套界面里
 * 唯一不能靠回退糊过去的地方 —— 回退在不同机器上会换字形。
 */
export function formatHours(seconds: number): { value: string; unit: string } {
  if (!seconds) return { value: "—", unit: "" };
  if (seconds < 3600) return { value: String(Math.round(seconds / 60)), unit: "分" };
  return { value: (seconds / 3600).toFixed(1), unit: "小时" };
}
