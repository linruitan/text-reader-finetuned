"""生成《妙法莲华经》全部 30 品（CosyVoice3 + GPU + 漏读检查）。
已生成的品会自动跳过；想重做某一品，删掉 法华经 文件夹里对应的 mp3 再运行。"""
import subprocess
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
cmd = [sys.executable, str(here / "local_tts.py"), str(here / "法华经" / "妙法莲华经.txt"),
       "--engine", "cosyvoice",
       "--instruct", "请用舒缓、平和、庄重的语速朗读。",
       "--verify", "--fp16",
       "-o", str(here / "法华经")]
cmd += sys.argv[1:]          # 例如：run_fahua.bat --chapters 23
sys.exit(subprocess.call(cmd))
