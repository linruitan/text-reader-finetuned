# Text reader finetuned

把中文长文本（txt / pdf / docx）按章节朗读成 mp3 的本地工具，专门为古文、佛经调过读音。
用开源语音模型 [Fun-CosyVoice3](https://github.com/FunAudioLLM/CosyVoice) 在自己的电脑上合成，不上传任何文字。

**示例：《妙法莲华经》全经诵读** → https://linruitan.github.io/text-reader-finetuned/lotus-sutra/
（经文与语音逐句同步，点哪一句就从哪一句听起，语速 0.5×–2×）

## 它做了什么

- **按章节切分**：自动识别「第X章」「XX品第X」「序」等标题，每章一个 mp3
- **读音词典**：`读音词典.txt` 修正多音字、生僻字和佛经传统读法，下次生成读法一致
- **漏读检查**：每句合成后用本地识别模型听一遍，少字、多字就自动重新合成
- **只改句子**：发现个别句子读错时，`patch_sentences.py` 只重新合成那几句，替换进原来的 mp3，不用整章重来

## 读音词典

`读音词典.txt` 里的每一条都用三个识别模型（Paraformer、SenseVoice、Zipformer）对整部《法华经》的录音逐字比对，再人工试听确认过，例如：

| 词 | 读法 | 说明 |
|---|---|---|
| 阿耨多罗三藐三菩提 | nòu | |
| 阿僧祇、祇夜 | qí | |
| 摩侯罗伽 | jiā | 八部众；其余「伽」读 qié |
| 乐 | lè / yuè | 快乐、喜乐义读 lè；伎乐、天乐等音乐义读 yuè |
| 逮 | dǎi | 逮得己利、逮大神通 |
| 偏袒 | piān tǎn | |
| 闇、躄、呰、柅、隤、婇、赍、蚖 | àn、bì、zǐ、nǐ、tuí、cǎi、jī、wán | 生僻字 |

词典有三种写法（拼音、同音字、正则），文件开头有说明。拼音最准，但少数拼音 CosyVoice 读不出来（运行时会提示），这时改用同音字。

## 安装（Windows + NVIDIA 显卡）

需要 [Anaconda](https://www.anaconda.com/download) 和较新的 NVIDIA 驱动。

1. 下载本仓库，再把 CosyVoice 代码放进来：
   ```
   git clone https://github.com/linruitan/text-reader-finetuned.git
   cd text-reader-finetuned
   git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
   ```
2. 双击 `setup.bat`：建 conda 环境 `fahua`、装 PyTorch（CUDA 12.8，支持 RTX 50 系）、下载约 9GB 的模型到 `ModelCache\`。最后一行是 `SETUP_OK` 就成功了。

Mac 也能跑（CPU，慢很多），依赖见 `requirements-mac.txt`。

## 使用

朗读任意文本：

```
conda activate fahua
python local_tts.py 我的书.txt --engine cosyvoice --verify --fp16 -o 输出文件夹
```

常用选项：`--chapters 3-5` 只做部分章节；`--instruct "请用舒缓、平和的语速朗读。"` 控制语气；`--speed 0.9` 调语速；`--dry-run` 只打印替换读音后的文本，用来检查词典。已经生成的章节会自动跳过。

重做《法华经》示例：双击 `run_fahua.bat`（原文在 `法华经/妙法莲华经.txt`）。

## 只重做读错的句子

1. 在 `读音词典.txt` 里加上正确读法
2. 写一个修改清单 `校对/修改清单.json`，列出章节和句子编号（编号和 `--dry-run` 打印的顺序一致，标题算第 1 句）：
   ```json
   [{"idx": 2, "stem": "002_序品第一",
     "pieces": [{"n": 54, "text": "独处闲静，乐诵经典。又见菩萨，勇猛精进，", "why": ["乐"]}]}]
   ```
3. 运行 `修改句子.bat`（或 `python patch_sentences.py --text 我的书.txt --audio-dir 输出文件夹`）

原 mp3 会先备份到 `_修改前备份`。脚本靠句间固定长度的停顿找到每一句的位置，所以要用本工具生成的 mp3。

## 文件

| 文件 | 作用 |
|---|---|
| `local_tts.py` | 主程序：读取、分章、分句、套用词典、合成、漏读检查、导出 mp3 |
| `读音词典.txt` | 读音修正（自动加载） |
| `patch_sentences.py`、`修改句子.bat` | 只重新合成指定句子 |
| `setup.bat`、`_setup_steps.bat`、`download_models.py`、`make_stub.py` | 安装 |
| `run_fahua.bat`、`run_fahua.py` | 《法华经》示例 |
| `lotus-sutra/` | 《法华经》诵读网页和 30 个 mp3 |

## 致谢

语音模型：[FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice)（Apache-2.0）。识别模型来自 [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)。经文为姚秦鸠摩罗什译《妙法莲华经》。
