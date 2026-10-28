"""pyworld 在 Windows 上经常编译失败。CosyVoice 只在训练数据处理里用到它，
推理用不到，但加载模型配置时会 import。装不上时放一个空壳模块即可。"""
import site
from pathlib import Path

sp = Path(site.getsitepackages()[-1])
(sp / "pyworld.py").write_text(
    '"""占位模块：推理不需要 pyworld（只用于训练时提取 F0）。"""\n'
    "def __getattr__(name):\n"
    "    raise ImportError('pyworld 未安装（推理不需要）')\n", encoding="utf-8")
print(f"已写入占位 pyworld：{sp / 'pyworld.py'}")
