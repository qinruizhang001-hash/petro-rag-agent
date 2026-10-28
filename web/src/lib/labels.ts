/**
 * 标签避让。
 *
 * ## 为什么需要它
 *
 * 测点标签的屏幕位置是锚点投影出来的：`(x, y)` 直接由测点的世界坐标决定。
 * 而"盖层完整性 / 地表形变 / 近地表土壤通量 / 浅层地下水"这四个测点在
 * 深度上挨得很近（0–100 m），画布一矮就叠成一团 —— 每个字都还在，
 * 但没有一个能读。这不是"缩小了"，这是**信息在但不可读**。
 *
 * ## 为什么只推测点标签
 *
 * 层名标签（"盖层（泥岩封闭）"）的语义是"贴在这条边界线上"，推离边界
 * 之后它就在标注一条错误的线 —— 那比重叠更糟。深度尺同理（刻度必须
 * 对准自己的高度）。所以层名与深度尺只作为**障碍物**参与判断，
 * 不会被移动；只有测点标签会被推开。
 *
 * ## 为什么是纯函数
 *
 * 控制器"没接上"和"接上了但算错"在截图上长得一样。`placeLabels` 不碰
 * DOM、不读时间，给定输入必然给同一输出，可以直接断言。
 */

export interface Anchor {
  id: string;
  x: number;
  y: number;
  visible: boolean;
}

export interface LabelBox {
  /** 相对锚点的横向偏移（同 `LabelDef.dx`） */
  dx: number;
  /** 相对锚点的纵向偏移 */
  dy: number;
  /** 标签从锚点往哪个方向展开 */
  align: "left" | "right" | "center";
  w: number;
  h: number;
  /** 是否允许被推开。层名与深度尺为 `false` —— 它们必须对准自己那条线 */
  movable: boolean;
}

export interface Rect {
  left: number;
  right: number;
  top: number;
  bottom: number;
}

function rectAt(a: Anchor, box: LabelBox, centerY: number): Rect {
  const left =
    box.align === "right"
      ? a.x + box.dx - box.w
      : box.align === "center"
        ? a.x + box.dx - box.w / 2
        : a.x + box.dx;
  return { left, right: left + box.w, top: centerY - box.h / 2, bottom: centerY + box.h / 2 };
}

function overlaps(p: Rect, q: Rect): boolean {
  // 严格不等：贴边不算重叠，否则相邻两行永远互相推
  return p.left < q.right && q.left < p.right && p.top < q.bottom && q.top < p.bottom;
}

/**
 * 把重叠的标签沿垂直方向推开。
 *
 * 返回**新的**锚点数组（不修改入参）。被推开的标签只改 `y`，
 * `x` 永远不动 —— 横向是测点在图上的真实位置，动了就是在说谎。
 *
 * @param gap       推开后两标签之间的最小间隙（px）
 * @param maxShift  单个标签最多推多远。超过就放弃（保持原位）：
 *                  推出画布外面，等于让这个测点从图上消失。
 */
export function placeLabels(
  anchors: Anchor[],
  boxes: Map<string, LabelBox>,
  { gap = 2, maxShift = 56 }: { gap?: number; maxShift?: number } = {},
): Anchor[] {
  const out = anchors.map((a) => ({ ...a }));

  // 先按原始垂直中心排序，让"谁在上"稳定；同 y 时保持输入顺序。
  // 不排序的话，推开顺序取决于锚点的插入顺序（Map 迭代序），
  // 而那个顺序与画面无关，会让同一份数据在两次渲染里推出不同结果。
  //
  // **不可移动的先处理**：它们的 y 是权威的（层名要贴在自己那条线上），
  // 让它们先占位，可移动的才有东西可避。反过来（固定的排在后面）会留下
  // 一个消不掉的重叠 —— 固定的不推别人，而它自己也不会动。
  const order = anchors
    .map((a, i) => ({ a, i }))
    .filter((it) => it.a.visible && boxes.has(it.a.id))
    .sort((p, q) => {
      const bp = boxes.get(p.a.id)!;
      const bq = boxes.get(q.a.id)!;
      if (bp.movable !== bq.movable) return bp.movable ? 1 : -1;
      return p.a.y + bp.dy - (q.a.y + bq.dy) || p.i - q.i;
    });

  const taken: Rect[] = [];

  for (const { a, i } of order) {
    const box = boxes.get(a.id)!;
    const baseCenter = a.y + box.dy;
    let center = baseCenter;
    let rect = rectAt(a, box, center);

    for (let tries = 0; tries < 6; tries++) {
      const hit = taken.filter((r) => overlaps(r, rect));
      if (hit.length === 0) break;
      // 下移到所有冲突矩形中最低的那条底边之下
      const bottom = Math.max(...hit.map((r) => r.bottom));
      center = bottom + gap + box.h / 2;
      rect = rectAt(a, box, center);
    }

    const shift = center - baseCenter;
    if (!box.movable || Math.abs(shift) <= maxShift) {
      const target = out[i];
      // `noUncheckedIndexedAccess` 下 out[i] 可能为 undefined；i 来自
      // anchors 的下标，必然存在，但断言比"索引安全"更难维护
      if (box.movable && target) target.y = a.y + shift;
      taken.push(rect);
    } else {
      // 推不动就让它重叠：飞出画布比重叠更糟（重叠至少还读得到一半）
      taken.push(rectAt(a, box, baseCenter));
    }
  }

  return out;
}
