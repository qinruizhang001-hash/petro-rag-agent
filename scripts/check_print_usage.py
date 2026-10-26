"""守卫：``chart_tools.py`` 里的 ``print()`` 只许出现在命令行入口里。

为什么需要它
------------
项目规则是"运行链路一律走 ``utils.logging_setup``"：``print`` 没有级别、
不落盘、没有模块归属，线上出问题既不能按级别过滤也不能事后回溯。

``chart_tools.py`` 是个特例：它既有**运行链路**（``generate_pressure_curve``
被 aggregate 节点调用），又有**命令行入口**（``_main``，输出本身就是产品）。
所以它不能像 ``nodes/`` 那样一刀切禁 print，只能按"在哪个函数里"区分。

这个区分一开始只写在注释里，结果真的漏了：自动清理的日志用了 ``print``，
而那行是在**出图过程中**执行的 —— 用户跑一次评估，清理记录只出现在
stdout 里，``app.log`` 里一个字都没有。注释挡不住这种回归，AST 能。

用法::

    python scripts/check_print_usage.py            # 默认检查 chart_tools.py
    python scripts/check_print_usage.py a.py b.py

退出码：0 = 合规；1 = 有 print 跑出了白名单函数。
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 允许使用 print 的函数名（命令行入口）。新加 CLI 函数时在这里登记，
# 并顺手问一句"它真的只在命令行里被调用吗"。
CLI_FUNCTIONS = frozenset({"_main", "main"})


def _print_calls(tree: ast.AST) -> list[tuple[int, str]]:
    """返回 ``[(行号, 所在函数名或 <module>)]``。"""
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == "print"):
            continue
        out.append((node.lineno, "<module>"))
    return out


def _function_spans(tree: ast.Module) -> list[tuple[int, int, str]]:
    """``[(起始行, 结束行, 函数名)]``，用于把行号映射回函数。"""
    spans: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno
            end = max(
                (getattr(n, "lineno", start) for n in ast.walk(node)),
                default=start,
            )
            spans.append((start, end, node.name))
    return spans


def check(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    spans = _function_spans(tree)
    problems: list[str] = []
    for lineno, _ in _print_calls(tree):
        # 取**最内层**的包裹函数：嵌套时起始行更晚的那个就是内层
        owner, best_start = "<module>", -1
        for start, end, name in spans:
            if start <= lineno <= end and start > best_start:
                owner, best_start = name, start
        if owner not in CLI_FUNCTIONS:
            problems.append(f"{path.name}:{lineno} print() 在 {owner}() 里")
    return problems


def main(argv: list[str]) -> int:
    targets = [Path(a) for a in argv[1:]] or [ROOT / "chart_tools.py"]
    problems: list[str] = []
    for path in targets:
        if not path.exists():
            problems.append(f"{path}: 文件不存在")
            continue
        problems.extend(check(path))

    if problems:
        print("::error::print() 跑到了运行链路上（应改用 utils.logging_setup）：")
        for p in problems:
            print("  - " + p)
        print(
            "允许的位置：" + "、".join(f"{f}()" for f in sorted(CLI_FUNCTIONS))
            + "。命令行入口的输出本身就是产品，其余地方一律用日志。"
        )
        return 1
    print(f"print() 用法合规：{', '.join(p.name for p in targets)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
