"""
图测试替身的**契约**测试：替身必须覆盖被替模块的全部导入名
==========================================================
零依赖、零 IO、零模型。

为什么需要这个文件
------------------
``tests/_graph_stubs.py`` 用 ``types.ModuleType`` 把叶子模块换成替身，
再让 ``workflow.py`` 与 ``nodes/`` 真实加载。这样闸门逻辑能在不装 torch
的机器上被验证，代价是**替身成了一个隐式契约**：``aggregate_node`` 从
``tools.physics_tools`` import 哪几个名字，替身里就必须有哪几个。

2026-10-25 踩过：给 ``tools.physics_tools`` 加了一个新函数
（``injection_risk_sensitivity``）并让 ``aggregate_node`` 导入它，本地
``unittest discover`` 全绿 —— 因为 ``tests/test_degradation.py`` 字母序在前，
先把真实的 ``tools/physics_tools.py`` 塞进了 ``sys.modules``，于是
``_install`` 里的 ``if name in sys.modules: return`` 直接跳过，替身根本没生效。
CI 的 "gate + whole-graph negative case" 只跑那两个图测试，替身生效，立刻红了::

    ImportError: cannot import name 'injection_risk_sensitivity'
    from 'tools.physics_tools' (unknown location)

``(unknown location)`` 说的是替身没有 ``__file__``，很容易被读成打包问题，
而真实原因是替身少了一个名字。这类失败**只在装了桩的那条路径上出现**，
所以本地全绿没有任何意义 —— 必须有一条不依赖环境的断言。

做法
----
用 ``ast`` 静态读两边：从 ``_graph_stubs.py`` 里取出每个 ``_install(name, **attrs)``
的模块名与属性名，从 ``workflow.py`` + ``nodes/*.py`` 里取出所有
``from <stubbed> import ...`` 的名字，断言后者是前者的子集。
纯静态解析，不导入任何东西，因此在 unit-core 里秒级跑完。

运行::

    python -m unittest -v tests.test_graph_stubs
"""
from __future__ import annotations

import ast
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

STUB_FILE = os.path.join(ROOT, "tests", "_graph_stubs.py")


def _installed_stubs() -> dict[str, set[str]]:
    """
    从 ``_graph_stubs.py`` 的源码里取出 ``{模块名: {属性名, ...}}``。

    只认 ``_install("name", key=value, ...)`` 这种写法 —— 如果哪天改成动态
    拼名字，这个测试会拿到空集合而不是静默通过（见下面的"非空"断言）。
    """
    with open(STUB_FILE, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=STUB_FILE)

    stubs: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == "_install"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        name = str(node.args[0].value)
        stubs[name] = {kw.arg for kw in node.keywords if kw.arg}
    return stubs


def _runtime_imports() -> list[tuple[str, str, str]]:
    """扫描运行链路里的 ``from X import a, b``，返回 ``[(模块, 名字, 文件)]``。"""
    targets = [os.path.join(ROOT, "workflow.py")]
    nodes_dir = os.path.join(ROOT, "nodes")
    targets += [
        os.path.join(nodes_dir, f) for f in sorted(os.listdir(nodes_dir))
        if f.endswith(".py")
    ]

    found: list[tuple[str, str, str]] = []
    for path in targets:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            for alias in node.names:
                found.append((node.module, alias.name, os.path.basename(path)))
    return found


class TestStubsCoverTheRuntimeImports(unittest.TestCase):
    """替身缺名字 → workflow import 阶段就崩，而本地开发看不到。"""

    @classmethod
    def setUpClass(cls):
        cls.stubs = _installed_stubs()
        cls.imports = _runtime_imports()

    def test_stub_file_was_parsed(self):
        # 解析失败时下面几条会"因为没有被替模块而全部通过"，所以先钉住非空
        self.assertGreaterEqual(len(self.stubs), 5, self.stubs)
        self.assertIn("tools.physics_tools", self.stubs)

    def test_every_imported_name_exists_in_the_stub(self):
        missing: list[str] = []
        for module, name, where in self.imports:
            provided = self.stubs.get(module)
            if provided is None:
                continue            # 不是被替模块，交给真实 import
            if name not in provided:
                missing.append(f"{where}: from {module} import {name}")
        self.assertEqual(
            missing, [],
            "以下导入在替身里没有对应属性，workflow 会在 import 阶段崩：\n"
            + "\n".join(missing)
            + "\n修法：在 tests/_graph_stubs.py 的 _install(...) 里补上该名字。",
        )

    def test_physics_tools_stub_covers_the_aggregate_imports(self):
        """把最容易被漏的那一处单独钉住（新增函数时第一个撞的就是它）。"""
        self.assertLessEqual(
            {"co2_density_uq", "injection_risk_uq", "injection_risk_sensitivity"},
            self.stubs["tools.physics_tools"],
        )

    def test_stub_names_are_exactly_what_the_graph_imports(self):
        """
        反向检查：替身里不该留没人 import 的名字。

        这条**必须真的断言**。初版写成了 ``self.assertTrue(True, msg)`` ——
        它永远通过，而且那句 msg 只有失败时才显示，等于既没检查也没告知，
        正是"伪装通过"。多出来的名字本身不会让 CI 变红，但它会让下一个读
        代码的人以为某个替身属性是必需的，于是把它当成契约维护。

        扫描范围就是 ``workflow.py`` + ``nodes/*.py``：这是装了替身之后
        唯一会被真实 import 的集合。把它扩大反而会引入假失败 ——
        ``api_server.py`` 从 ``chart_tools`` import ``CHART_DIR``、
        ``memory/memory_tools.py`` 从 ``memory.memory_manager`` import
        ``retrieve_memory``，这两个文件都不在这条链路上（它们自己也被替身
        或压根没被加载），要求替身提供它们只是无谓的耦合。
        """
        used: dict[str, set[str]] = {}
        for module, name, _where in self.imports:
            used.setdefault(module, set()).add(name)

        problems: list[str] = []
        for module, provided in sorted(self.stubs.items()):
            imported = used.get(module, set())
            for name in sorted(provided - imported):
                problems.append(f"{module}: 替身提供了 {name}，但这条链路没人 import 它")
            for name in sorted(imported - provided):
                problems.append(f"{module}: 这条链路 import 了 {name}，但替身没有")

        self.assertEqual(
            problems, [],
            "替身与运行链路的导入集合不一致：\n"
            + "\n".join(problems)
            + "\n修法：少名字 → 在 _install(...) 里补；多名字 → 从 _install(...) 里删。",
        )


# ─── 桩泄漏到别的测试模块 ──────────────────────────────
# 这两个文件就是**装桩的人**：它们消费替身是设计意图，不是事故。
_STUB_CONSUMERS = {"test_quality_gate.py", "test_negative_gate.py"}


def _released_modules(path: str) -> set[str]:
    """取出文件里所有 ``_graph_stubs.release("x")`` 的 ``x``。"""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)

    out: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "release"):
            continue
        if not (isinstance(func.value, ast.Name) and func.value.id == "_graph_stubs"):
            continue
        if node.args and isinstance(node.args[0], ast.Constant):
            out.add(str(node.args[0].value))
    return out


def _top_level_stub_imports() -> list[tuple[str, str, str]]:
    """
    扫描 ``tests/*.py`` 里对**被替模块**的顶层 import，返回 ``[(文件, 模块, 名字)]``。

    只看模块顶层（``tree.body``）而不是 ``ast.walk``：写在测试方法内部的
    import 发生在运行期，那时桩早已装好但也不会再影响别处；真正会被污染的是
    **顶层** import —— 它在 ``unittest discover`` 的收集阶段执行，
    而收集阶段会把所有测试模块先导一遍。
    """
    tests_dir = os.path.join(ROOT, "tests")
    found: list[tuple[str, str, str]] = []
    for fname in sorted(os.listdir(tests_dir)):
        if not fname.endswith(".py") or fname.startswith("_"):
            continue
        if fname in _STUB_CONSUMERS or fname == "test_graph_stubs.py":
            continue
        path = os.path.join(tests_dir, fname)
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=path)
        for node in tree.body:
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            for alias in node.names:
                found.append((fname, node.module, alias.name))
    return found


class TestStubLeakageIntoOtherTestModules(unittest.TestCase):
    """
    谁在顶层 import 被替模块，谁就必须先声明 ``_graph_stubs.release(...)``。

    ``install_leaf_stubs()`` 写在两个图测试的**模块顶层**（桩必须早于
    ``import workflow``），而 discover 是"先导入全部测试模块，再跑"，
    于是桩的存活期覆盖了所有测试模块的导入期。

    2026-10 踩过：手工按 ``tests.test_negative_gate`` → ``tests.test_optimize``
    的顺序跑，``injection_risk_uq`` 变成 ``_fake_injection_uq``，它返回的
    dict 没有 ``counts``，4 项契约测试当场 ``KeyError: 'counts'``。
    全量 discover 不复现，只是因为 ``test_optimize`` 恰好先被加载 ——
    那是**碰巧**，不是设计。所以必须有一条不依赖加载顺序的断言。
    """

    def test_every_top_level_stub_import_is_released_first(self):
        stubs = _installed_stubs()
        problems: list[str] = []
        for fname, module, name in _top_level_stub_imports():
            if module not in stubs:
                continue            # 没被替，顶层 import 拿到的一直是真实模块
            if module in _released_modules(os.path.join(ROOT, "tests", fname)):
                continue
            problems.append(f"{fname}: from {module} import {name}")

        self.assertEqual(
            problems, [],
            "以下测试模块在顶层 import 了会被整图测试替换成替身的模块，"
            "却没有先把它要回真实实现：\n"
            + "\n".join(problems)
            + "\n修法：在 import 之前加一行 "
              '_graph_stubs.release("<模块名>")。\n'
              "理由：模块加载顺序不该决定拿到的是真实现还是替身。",
        )

    def test_the_scan_actually_sees_the_two_known_imports(self):
        """
        解析失败会让上面那条"因为没有命中而通过"，所以先钉住扫描确实有输出。

        这两个模块就是当前唯一需要 ``release()`` 的地方；如果哪天它们改名或
        删掉了 import，这条会红，提醒去更新上面的白名单判断。
        """
        stubs = _installed_stubs()
        hits = {
            (fname, module)
            for fname, module, _name in _top_level_stub_imports()
            if module in stubs
        }
        self.assertIn(("test_optimize.py", "tools.physics_tools"), hits)
        self.assertIn(("test_chart_tools.py", "chart_tools"), hits)


if __name__ == "__main__":
    unittest.main(verbosity=2)
