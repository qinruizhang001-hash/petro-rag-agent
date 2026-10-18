"""
物理计算工具（供 Function Calling 调用）
========================================
与 ``knowledge_tools``（联网检索）并列：这里是"算出来的事实"，
那里是"查回来的事实"，两者都不该由语言模型口算/编造。

``calculate_co2_density`` 用的是简化状态方程 ``ρ = 1.9·P / T``，
不是 Span–Wagner。它演示的是**工具调用契约**（数值必须来自代码而非模型），
不是储层工程计算器 —— 需要工程精度时替换本函数即可，调用方无需改动。
"""
from __future__ import annotations

from utils.logging_setup import get_logger

log = get_logger(__name__)

def calculate_co2_density(pressure_bar, temperature_celsius):
    """计算给定压力、温度下 CO2 密度的物理工具（简化状态方程）。"""
    log.info(f"  [Tool 调用] 正在计算压力 {pressure_bar} bar, 温度 {temperature_celsius}℃ 下的CO2密度...")
    density = 1.9 * pressure_bar / (temperature_celsius + 273.15)
    return f"{density:.2f} kg/m³"
