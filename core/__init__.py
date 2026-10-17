"""
core —— 配置、状态与共享单例。

这个包的 __init__ 承担两件"必须最先执行"的事，所以它们不能挪到别处：

1. 设置 ``HF_HUB_OFFLINE=1``。``huggingface_hub`` 在 **import 时**就把这个
   变量读进常量，之后再 setdefault 完全无效。历史上这个 bug 让每次流程末尾
   卡 5 次指数退避重试（每个模型文件各一次），端到端耗时 593s → 修复后 93s。
2. 把项目根目录加入 ``sys.path``。``memory/``、``tools/`` 都是按根目录为基准
   导入的包，而 api_server 可能从任意 CWD 启动。

放在这里是因为任何 ``import core.*`` / ``import nodes.*`` / ``import utils.*``
都会先执行本文件，天然保证顺序正确 —— 而把这两行写在某个普通模块里，
就只能靠"调用方记得先 import 它"来保证，迟早会破。
"""
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
