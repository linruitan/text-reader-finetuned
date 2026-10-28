"""下载 CosyVoice3 模型和漏读检查用的 Paraformer 模型到模型缓存目录。
缓存目录：环境变量 TTS_MODEL_CACHE，没有就用本文件夹下的 ModelCache。"""
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

root = Path(os.environ.get("TTS_MODEL_CACHE") or Path(__file__).resolve().parent / "ModelCache")
root.mkdir(parents=True, exist_ok=True)
print(f"模型缓存目录：{root}")

cv = root / "Fun-CosyVoice3-0.5B"
if (cv / "cosyvoice3.yaml").exists():
    print("CosyVoice3 模型已存在，跳过")
else:
    print("下载 Fun-CosyVoice3-0.5B（约 9GB，需要一段时间）…")
    from huggingface_hub import snapshot_download
    snapshot_download("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", local_dir=str(cv))

asr = root / "asr"
name = "sherpa-onnx-paraformer-zh-int8-2025-10-07"
if (asr / name / "model.int8.onnx").exists():
    print("Paraformer 模型已存在，跳过")
else:
    asr.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{name}.tar.bz2"
    tmp = asr / f"{name}.tar.bz2"
    print(f"下载漏读检查模型 {name}（约 230MB）…")
    urllib.request.urlretrieve(url, tmp)
    with tarfile.open(tmp, "r:bz2") as t:
        t.extractall(asr)
    tmp.unlink()
print("模型准备完成")
