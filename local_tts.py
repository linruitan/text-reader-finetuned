#!/usr/bin/env python3
"""
本地 AI 有声书生成：txt / pdf / docx -> 按章节输出 mp3（Mac / Windows / Linux，CUDA 可加速）

两种引擎：
  kokoro    （默认）轻量、安装简单、速度快，内置多个中文音色
  cosyvoice  更自然，支持用一段参考音频模仿音色（需要单独装 CosyVoice 仓库）

读音词典：
  TTS 按普通话常用读音读字，佛经、古文里的特殊读音（一乘 shèng、般若 bō rě…）
  容易读错。脚本在合成前会把词替换成同音字（只影响发音，不改原文、不改文件名）。
  内置了一批常见佛教词（“乘”保持 chéng 的读法）；自己的词写进文本文件，用 --lexicon 加载：

      # 读音词典.txt  （# 开头是注释；每行：原词=替换）
      阙失=缺失
      一乘=一剩                   # 想读 shèng 就自己加
      re:(?<=是)乘=剩             # re: 开头表示正则
      佛=fo2                      # 等号右边是拼音时直接改读音（仅 Kokoro；1–4 声调数字或带调字母）
      南无=na1 mo2                # 词组：拼音个数要和字数一致

  脚本同目录下的「读音词典.txt」会自动加载。

模型缓存：
  所有模型（Kokoro、CosyVoice、HuggingFace / ModelScope / torch 的下载）统一放进
  --cache-dir 指定的目录；不指定时依次看环境变量 TTS_MODEL_CACHE，再默认用
  脚本同目录下的 ModelCache/。

用法示例：
  python local_tts.py 书.txt
  python local_tts.py 书.pdf --voice zm_yunxi --speed 1.1
  python local_tts.py 书.txt --lexicon 我的词典.txt
  python local_tts.py --text "见种种乘皆是一乘。" -o test_out     # 直接合成一段文字
  python local_tts.py 书.txt --cache-dir /Volumes/SSD/ModelCache   # 指定模型缓存目录
  python local_tts.py 书.txt --dry-run --chapters 1   # 只打印替换后要读的文本，不合成
  python local_tts.py 书.txt --list-chapters          # 只看章节切分结果
  python local_tts.py 书.docx --engine cosyvoice \
      --ref-audio 参考.wav --ref-text "参考音频里说的那句话" \
      --instruct "请用舒缓、平和的语速朗读。"

CosyVoice：
  默认模型 <缓存目录>/Fun-CosyVoice3-0.5B，仓库在脚本同目录 CosyVoice/。
  不给 --ref-audio 时用仓库自带的示例音色。词典里的拼音条目（佛=fo2）会变成
  CosyVoice 的拼音热修复 token（[f][ó]），内置的“佛”修正只对 Kokoro 生效
  （那是 pypinyin 的问题，CosyVoice 自己一般读得对）。
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONUTF8", "1")
# Windows 控制台默认 GBK，打印中文/符号会报错，统一改成 UTF-8
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def setup_model_cache(cache_dir) -> Path:
    """把各类模型下载都指到同一个目录。必须在 import kokoro / torch / cosyvoice 之前调用。"""
    root = Path(cache_dir or os.environ.get("TTS_MODEL_CACHE")
                or Path(__file__).resolve().parent / "ModelCache").expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    env = {
        "HF_HOME": root / "huggingface",            # Kokoro 权重、音色
        "HF_HUB_CACHE": root / "huggingface" / "hub",
        "MODELSCOPE_CACHE": root / "modelscope",    # CosyVoice 走 ModelScope 下载时
        "TORCH_HOME": root / "torch",
        "XDG_CACHE_HOME": root / "cache",           # spacy / jieba 等其他缓存
        "NLTK_DATA": root / "nltk_data",
    }
    for k, v in env.items():
        os.environ[k] = str(v)
    return root

# ----------------------------------------------------------------------------
# 0. 读音词典
# ----------------------------------------------------------------------------

# 内置条目：原词 -> 同音替换。只收读音争议较小的；有异议直接用 --no-default-lexicon 关掉，
# 或在自己的词典里覆盖（自己的词典优先）。
DEFAULT_LEXICON = {
    # 般若 bō rě
    "般若": "波惹",
    # 伽 qié
    "僧伽": "僧茄", "伽蓝": "茄蓝", "伽陀": "茄陀", "伽耶": "茄耶", "楞伽": "楞茄",
    # 刹 chà
    "佛刹": "佛诧", "刹土": "诧土", "国刹": "国诧", "宝刹": "宝诧", "刹那": "诧那",
    # 塞 sè
    "优婆塞": "优婆色",
    # 阙 quē（缺）
    "阙失": "缺失", "阙漏": "缺漏",
    # 迦叶 jiā shè
    "迦叶": "迦摄",
    # 阇 shé（耆阇崛、阿阇世、阇梨、波阇波提）
    "阇": "蛇",
    # 其他
    "阿閦": "阿处",          # 阿閦佛 ā chù
    "三昧": "三妹",          # sān mèi
    "揭谛": "揭帝",          # jiē dì
    "菩提萨埵": "菩提萨朵",  # sà duǒ
}


# 拼音级覆盖（只对 Kokoro 生效：直接改 pypinyin 的字典，比同音字更准）。
# pypinyin 把单字“佛”默认标成 fú，这里改回 fó；“仿佛”等词仍按词典读 fú。
DEFAULT_PINYIN = {
    "佛": "fo2",
    "南无": "na1 mo2",
}

# CosyVoice 默认的拼音热修复（“佛”CosyVoice 一般能读对，不在这里强改）
DEFAULT_HOTFIX = {
    "南无": "na1 mo2",
}

PINYIN_RE = re.compile(r"^(?:[a-zü:āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜńňǹ]+[1-5]?)(?:\s+[a-zü:āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜńňǹ]+[1-5]?)*$", re.I)


def is_pinyin(key: str, val: str) -> bool:
    return bool(PINYIN_RE.match(val)) and len(val.split()) == len(key) \
        and re.fullmatch(r"[\u4e00-\u9fff]+", key) is not None


def apply_pinyin_overrides(table: dict):
    """把 {汉字/词: 拼音} 写进 pypinyin 的字典（单字改默认读音，多字作为词组）。"""
    if not table:
        return
    from pypinyin import load_single_dict, load_phrases_dict
    from pypinyin.contrib.tone_convert import to_tone
    single, phrases = {}, {}
    for k, v in table.items():
        syl = [to_tone(x.lower().replace("v", "ü")) for x in v.split()]
        if len(k) == 1:
            single[ord(k)] = syl[0]
        else:
            phrases[k] = [[x] for x in syl]
    if single:
        load_single_dict(single)
    if phrases:
        load_phrases_dict(phrases)
        try:  # 让 jieba 把这些词切成一个整体，词组读音才会生效
            import jieba
            for k in phrases:
                jieba.add_word(k, freq=10 ** 6)
        except ImportError:
            pass


def pinyin_to_hotfix(py: str, tokens=None):
    """'na1 mo2' -> '[n][ā][m][ó]'；tokens 给定时校验是否在 CosyVoice 词表里。"""
    from pypinyin.contrib.tone_convert import to_initials, to_finals_tone
    out = []
    for syl in py.lower().split():
        syl = syl.replace("v", "ü")
        ini, fin = to_initials(syl, strict=True), to_finals_tone(syl, strict=True)
        parts = ([f"[{ini}]"] if ini else []) + [f"[{fin}]"]
        if tokens is not None:
            fixed = []
            for t in parts:
                if t not in tokens and t.replace("ü", "v") in tokens:
                    t = t.replace("ü", "v")
                if t not in tokens:
                    return None
                fixed.append(t)
            parts = fixed
        out.append("".join(parts))
    return "".join(out)


def load_lexicon_file(path: Path):
    """读取词典文件，返回 (普通替换 dict, 正则替换 list)。"""
    plain, regex = {}, []
    for n, line in enumerate(read_txt(path).splitlines(), 1):
        line = re.sub(r"\s+#.*$", "", line).strip()   # 行尾注释
        if not line or line.startswith("#"):
            continue
        # 按最后一个 = 分割（正则里可能有 (?<= …) 之类的等号）
        m = re.match(r"^(.+)\s*(?:=>|=|\t)\s*([^=\t]*)$", line)
        if not m:
            print(f"⚠️ 词典 {path.name} 第 {n} 行格式不对，已忽略：{line}")
            continue
        k, v = m.group(1).strip(), m.group(2).strip()
        if k.startswith("re:"):
            try:
                regex.append((re.compile(k[3:]), v))
            except re.error as e:
                print(f"⚠️ 词典 {path.name} 第 {n} 行正则有误，已忽略：{e}")
        else:
            plain[k] = v
    return plain, regex


class Lexicon:
    def __init__(self, files, use_default=True):
        self.plain = dict(DEFAULT_LEXICON) if use_default else {}
        self.pinyin = dict(DEFAULT_PINYIN) if use_default else {}
        self.user_pinyin = {}     # 词典文件里写的拼音条目：Kokoro 和 CosyVoice 都用
        self.regex = []
        for f in files:
            p, r = load_lexicon_file(f)
            for k, v in p.items():
                if is_pinyin(k, v):          # 例：佛=fo2  南无=na1 mo2
                    self.pinyin[k] = v
                    self.user_pinyin[k] = v
                    self.plain.pop(k, None)
                else:
                    self.plain[k] = v
                    self.pinyin.pop(k, None)
            self.regex += r
        self.plain = {k: v for k, v in self.plain.items() if k and k != v}
        if self.plain:
            keys = sorted(self.plain, key=len, reverse=True)   # 长词优先
            self.pattern = re.compile("|".join(map(re.escape, keys)))
        else:
            self.pattern = None

    def __len__(self):
        return len(self.plain) + len(self.regex) + len(self.pinyin)

    def hotfix_table(self, tokens=None, include_default=False):
        """CosyVoice 的拼音热修复：把词替换成 [f][ó] 这样的发音 token。"""
        table = dict(DEFAULT_HOTFIX) if include_default else {}
        table.update(self.user_pinyin)
        out = {}
        for k, v in table.items():
            h = pinyin_to_hotfix(v, tokens)
            if h is None:
                print(f"⚠️ CosyVoice 不支持「{k}={v}」的拼音 token，该条只对 Kokoro 生效")
            else:
                out[k] = h
        return out

    def apply(self, text: str) -> str:
        for rx, rep in self.regex:
            text = rx.sub(rep, text)
        if self.pattern:
            text = self.pattern.sub(lambda m: self.plain[m.group(0)], text)
        return text

# ----------------------------------------------------------------------------
# 1. 读取文本
# ----------------------------------------------------------------------------

def read_txt(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "gb18030", "big5", "utf-16"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    import fitz  # pymupdf
    doc = fitz.open(path)
    pages = []
    for page in doc:
        lines = page.get_text("text").splitlines()
        # 去掉单独成行的页码
        lines = [l for l in lines if not re.fullmatch(r"\s*[-—–]?\s*\d{1,4}\s*[-—–]?\s*", l)]
        pages.append("\n".join(lines))
    return join_broken_lines("\n".join(pages))


def read_docx(path: Path):
    """返回 (纯文本, 基于标题样式的章节列表或 None)"""
    import docx
    d = docx.Document(path)
    chapters, cur_title, cur_body, all_text = [], None, [], []
    for p in d.paragraphs:
        t = p.text.strip()
        if not t:
            continue
        all_text.append(t)
        style = (p.style.name or "").lower()
        if style.startswith("heading 1") or style.startswith("heading 2") or style == "title" \
                or style.startswith("标题 1") or style.startswith("标题 2"):
            if cur_title is not None or cur_body:
                chapters.append((cur_title or "前言", "\n".join(cur_body)))
            cur_title, cur_body = t, []
        else:
            cur_body.append(t)
    if cur_title is not None or cur_body:
        chapters.append((cur_title or "正文", "\n".join(cur_body)))
    has_headings = sum(1 for c in chapters if c[0] not in ("前言", "正文")) >= 2
    return "\n".join(all_text), (chapters if has_headings else None)


def join_broken_lines(text: str) -> str:
    """PDF 常把一句话断成多行：行尾不是句末标点就接到下一行。"""
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            out.append("")
            continue
        if out and out[-1] and not re.search(r"[。！？!?；;：:…”」』）)\]]$", out[-1]) \
                and not is_heading(s) and not is_heading(out[-1]):
            sep = " " if re.search(r"[A-Za-z0-9]$", out[-1]) and re.match(r"[A-Za-z0-9]", s) else ""
            out[-1] += sep + s
        else:
            out.append(s)
    return "\n".join(out)

# ----------------------------------------------------------------------------
# 2. 章节切分
# ----------------------------------------------------------------------------

CN_NUM = r"[0-9０-９零〇一二两三四五六七八九十百千万]+"
CHAPTER_RE = re.compile(
    rf"^\s*("
    rf"第\s*{CN_NUM}\s*[章回节卷部篇集幕品]"
    rf"|(?:chapter|part|book)\s+(?:\d+|[ivxlcdm]+)\b"
    rf"|[\u4e00-\u9fff]{{1,12}}品第[一二三四五六七八九十百]+"      # 佛经：序品第一、方便品第二…
    rf"|(?:序章|序言|序|前言|引子|楔子|引言|自序|弘传序|后序|后记|尾声|终章|番外|开经偈|回向偈)(?=$|[\s:：·、.—\-（(])"
    rf")[^\n]{{0,40}}$",
    re.IGNORECASE,
)


def is_heading(line: str) -> bool:
    s = line.strip()
    # 标题不会以句末标点结尾，也不会太长
    return bool(s) and len(s) <= 40 and CHAPTER_RE.match(s) is not None \
        and not re.search(r"[。！？!?，,；;.]$", s)


def split_chapters(text: str, fallback_chars: int = 6000):
    lines = [l.rstrip() for l in text.splitlines()]
    chapters, title, body = [], None, []
    for line in lines:
        s = line.strip()
        if is_heading(s):
            if body and any(b.strip() for b in body):
                chapters.append((title or "开篇", "\n".join(body)))
            title, body = s, []
        else:
            body.append(line)
    if body and any(b.strip() for b in body):
        chapters.append((title or "开篇", "\n".join(body)))

    if len(chapters) >= 2:
        return chapters

    # 没识别到章节：按段落累积到约 fallback_chars 字切一段
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    parts, buf, n = [], [], 0
    for p in paras:
        buf.append(p)
        n += len(p)
        if n >= fallback_chars:
            parts.append("\n".join(buf))
            buf, n = [], 0
    if buf:
        parts.append("\n".join(buf))
    if len(parts) == 1:
        return [("正文", parts[0])]
    return [(f"第{i + 1:03d}部分", t) for i, t in enumerate(parts)]

# ----------------------------------------------------------------------------
# 3. 文本清洗 & 分句
# ----------------------------------------------------------------------------

def clean_text(t: str) -> str:
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"[ \t　]+", " ", t)
    t = re.sub(r"[﻿​]", "", t)
    t = re.sub(r"[*#=_~]{3,}", "", t)       # 分隔线
    t = re.sub(r"\[\d+\]|【\d+】", "", t)    # 注释角标
    return t.strip()


def split_sentences(text: str, max_len: int):
    """按句末标点切句，再把短句拼到不超过 max_len 字。返回 [(片段, 是否段落结尾)]"""
    chunks = []
    for para in text.split("\n"):
        para = clean_text(para)
        if not para:
            continue
        sents = re.findall(r"[^。！？!?；;…]+(?:[。！？!?；;…]+[”」』’\"）)]*)?", para) or [para]
        # 过长的句子再按逗号切
        pieces = []
        for s in sents:
            s = s.strip()
            if not s:
                continue
            if len(s) <= max_len:
                pieces.append(s)
            else:
                sub = re.findall(r"[^，,、：:]+[，,、：:]?", s)
                cur = ""
                for x in sub:
                    if len(cur) + len(x) > max_len and cur:
                        pieces.append(cur)
                        cur = x
                    else:
                        cur += x
                if cur:
                    pieces.append(cur)
        # 合并短句
        merged, cur = [], ""
        for p in pieces:
            if len(cur) + len(p) <= max_len:
                cur += p
            else:
                if cur:
                    merged.append(cur)
                cur = p
        if cur:
            merged.append(cur)
        for i, m in enumerate(merged):
            if re.search(r"[\w一-鿿]", m):  # 跳过纯标点
                chunks.append((m, i == len(merged) - 1))
    return chunks

# ----------------------------------------------------------------------------
# 4. TTS 引擎
# ----------------------------------------------------------------------------

class KokoroEngine:
    max_len = 150

    def __init__(self, voice, speed, device, repo):
        from kokoro import KPipeline
        kwargs = {"lang_code": "z"}
        if repo:
            kwargs["repo_id"] = repo
        if device:
            kwargs["device"] = device
        self.pipe = KPipeline(**kwargs)
        self.voice, self.speed = voice, speed
        self.sr = 24000

    def synth(self, text):
        out = []
        for _, _, audio in self.pipe(text, voice=self.voice, speed=self.speed, split_pattern=None):
            if audio is not None:
                a = audio.cpu().numpy() if hasattr(audio, "cpu") else np.asarray(audio)
                out.append(a.astype(np.float32))
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


class CosyVoiceEngine:
    """CosyVoice2 / Fun-CosyVoice3（AutoModel 自动识别）。Mac 上跑在 CPU。"""
    max_len = 80
    DEFAULT_REF_TEXT = "希望你以后能够做的比我还好呦。"   # 仓库自带 asset/zero_shot_prompt.wav

    def __init__(self, model_dir, ref_audio, ref_text, speed, repo, instruct, fp16=False):
        repo = Path(repo).expanduser().resolve()
        if not (repo / "cosyvoice").is_dir():
            sys.exit(f"找不到 CosyVoice 仓库：{repo}\n"
                     f"先运行：git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git {repo}")
        sys.path.insert(0, str(repo))
        sys.path.insert(0, str(repo / "third_party" / "Matcha-TTS"))
        from cosyvoice.cli.cosyvoice import AutoModel

        if not ref_audio:
            ref_audio = repo / "asset" / "zero_shot_prompt.wav"
            ref_text = ref_text or self.DEFAULT_REF_TEXT
            print(f"未指定 --ref-audio，使用仓库自带示例音色：{ref_audio.name}")
        self.ref_audio = str(Path(ref_audio).expanduser())
        import torch
        kw = {}
        if torch.cuda.is_available():
            kw["fp16"] = fp16
            print(f"使用 GPU：{torch.cuda.get_device_name(0)}" + ("（fp16）" if fp16 else ""))
        else:
            print("未检测到 CUDA，使用 CPU")
        self.model = AutoModel(model_dir=str(model_dir), **kw)
        self.v3 = type(self.model).__name__ == "CosyVoice3"
        sys_prefix = "You are a helpful assistant." if self.v3 else ""
        self.instruct = None
        if instruct:
            self.instruct = (f"{sys_prefix} {instruct}" if self.v3 else instruct).strip() + "<|endofprompt|>"
        self.ref_text = (sys_prefix + "<|endofprompt|>" if self.v3 else "") + (ref_text or "") if ref_text else None
        self.speed = speed
        self.sr = self.model.sample_rate
        print(f"CosyVoice 模型：{type(self.model).__name__}，采样率 {self.sr}，"
              f"模式：{'instruct' if self.instruct else ('zero-shot' if self.ref_text else 'cross-lingual')}")

    def special_tokens(self):
        try:
            import inspect
            import cosyvoice.tokenizer.tokenizer as T
            return set(re.findall(r'"(\[[^\]"]+\])"', inspect.getsource(T)))
        except Exception:
            return None

    def synth(self, text):
        m = self.model
        kw = dict(stream=False, speed=self.speed)
        if self.instruct:
            gen = m.inference_instruct2(text, self.instruct, self.ref_audio, **kw)
        elif self.ref_text:
            gen = m.inference_zero_shot(text, self.ref_text, self.ref_audio, **kw)
        else:  # 没有参考文本：只模仿音色
            prefix = "You are a helpful assistant.<|endofprompt|>" if self.v3 else ""
            gen = m.inference_cross_lingual(prefix + text, self.ref_audio, **kw)
        out = [r["tts_speech"].squeeze(0).cpu().numpy().astype(np.float32) for r in gen]
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

class OmissionChecker:
    """每句合成后用本地 Paraformer 听一遍，按拼音对齐原句，统计漏掉的字数。
    CosyVoice 偶尔会吞掉几个字（采样有随机性），重新合成一般就好了。"""

    def __init__(self, model_dir):
        import sherpa_onnx
        d = Path(model_dir)
        if not (d / "model.int8.onnx").exists():
            sys.exit(f"找不到漏读检查用的 Paraformer 模型：{d}")
        self.rec = sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=str(d / "model.int8.onnx"), tokens=str(d / "tokens.txt"), num_threads=2)

    def check(self, text, audio, sr):
        """返回 (漏掉的字数, 多出的字数, 识别结果)。只看拼音，不管写成哪个字。"""
        import difflib
        from pypinyin import lazy_pinyin
        st = self.rec.create_stream()
        st.accept_waveform(sr, audio)
        self.rec.decode_stream(st)
        hyp = st.result.text
        han = re.compile(r"[\u4e00-\u9fff]")
        rp = lazy_pinyin("".join(han.findall(text)))
        hp = lazy_pinyin("".join(han.findall(hyp)))
        missing = extra = 0
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, rp, hp, autojunk=False).get_opcodes():
            d = (i2 - i1) - (j2 - j1)
            if tag in ("delete", "replace") and d > 0:
                missing += d
            elif tag in ("insert", "replace") and d < 0:
                extra -= d
        return missing, extra, hyp

# ----------------------------------------------------------------------------
# 5. 音频输出
# ----------------------------------------------------------------------------

def write_wav(path: Path, audio: np.ndarray, sr: int):
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def to_mp3(wav_path: Path, mp3_path: Path, title: str, album: str, track: int, bitrate: str):
    # -nostdin + stdin=DEVNULL：后台运行（nohup … &）时 ffmpeg 不去读终端，否则进程会被挂起
    cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(wav_path),
           "-codec:a", "libmp3lame", "-b:a", bitrate,
           "-metadata", f"title={title}", "-metadata", f"album={album}",
           "-metadata", f"track={track}", str(mp3_path)]
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)
    wav_path.unlink()


def safe_name(s: str) -> str:
    s = re.sub(r'[\\/:*?"<>|\n\r\t]', " ", s).strip()
    return re.sub(r"\s+", " ", s)[:60] or "untitled"


def parse_ranges(spec: str):
    sel = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            sel.update(range(int(a), int(b) + 1))
        else:
            sel.add(int(part))
    return sel

# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="本地 AI 有声书：txt/pdf/docx → 按章节 mp3")
    ap.add_argument("input", nargs="?", help="输入文件（.txt / .pdf / .docx）")
    ap.add_argument("--text", help="直接合成一段文字（代替输入文件）")
    ap.add_argument("-o", "--outdir", help="输出目录（默认：书名_audio）")
    ap.add_argument("--engine", choices=["kokoro", "cosyvoice"], default="kokoro")
    ap.add_argument("--voice", default="zf_xiaoxiao",
                    help="Kokoro 音色：zf_xiaobei zf_xiaoni zf_xiaoxiao zf_xiaoyi "
                         "zm_yunjian zm_yunxi zm_yunxia zm_yunyang")
    ap.add_argument("--kokoro-repo", default=None,
                    help="可选：hexgrad/Kokoro-82M-v1.1-zh（中文专项版，音色名为 zf_001 … zm_100）")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="语速倍率，如 0.6；句间停顿会随之等比例拉长")
    ap.add_argument("--device", default=None, help="cpu 或 mps（默认自动）")
    ap.add_argument("--ref-audio", help="CosyVoice 参考音频 wav")
    ap.add_argument("--ref-text", help="参考音频对应的文字（填了效果更好）")
    ap.add_argument("--cosyvoice-model", default=None,
                    help="CosyVoice 模型目录（默认：<缓存目录>/Fun-CosyVoice3-0.5B）")
    ap.add_argument("--instruct", default=None,
                    help="CosyVoice 风格指令，如「请用舒缓、平和的语速朗读。」（用 instruct 模式，仍用参考音频的音色）")
    ap.add_argument("--fp16", action="store_true", help="CUDA 上用半精度，更快、更省显存")
    ap.add_argument("--verify", action="store_true",
                    help="每句合成后用本地识别模型检查漏读/多读，有问题就重新合成")
    ap.add_argument("--verify-model", default=None,
                    help="漏读检查用的 Paraformer 模型目录（默认：<缓存目录>/asr/sherpa-onnx-paraformer-zh-int8-2025-10-07）")
    ap.add_argument("--max-retry", type=int, default=3, help="漏读时最多重新合成几次（默认 3）")
    ap.add_argument("--no-hotfix", action="store_true",
                    help="CosyVoice 不使用拼音热修复（词典里的拼音条目不生效）")
    ap.add_argument("--cache-dir", default=None,
                    help="模型缓存目录（默认：$TTS_MODEL_CACHE 或 脚本同目录/ModelCache）")
    ap.add_argument("--cosyvoice-repo", default=None,
                    help="CosyVoice 仓库路径（默认：脚本同目录/CosyVoice）")
    ap.add_argument("--lexicon", action="append", default=[],
                    help="读音词典文件，可多次指定；脚本同目录的「读音词典.txt」会自动加载")
    ap.add_argument("--no-default-lexicon", action="store_true", help="不使用内置读音条目")
    ap.add_argument("--sentence-pause", type=float, default=0.25, help="句间停顿（秒）")
    ap.add_argument("--para-pause", type=float, default=0.6, help="段间停顿（秒）")
    ap.add_argument("--bitrate", default="96k")
    ap.add_argument("--no-title", action="store_true", help="不朗读章节标题")
    ap.add_argument("--chapters", help="只合成部分章节，如 1-5,8")
    ap.add_argument("--list-chapters", action="store_true", help="只打印章节切分结果")
    ap.add_argument("--dry-run", action="store_true", help="打印替换读音后实际要读的文本，不合成")
    args = ap.parse_args()

    # ---- 读入 ----
    docx_chapters = None
    if args.text:
        src_stem = "text"
        chapters = [("片段", args.text)]
        default_out = Path("tts_output")
        args.no_title = True
    else:
        if not args.input:
            ap.error("需要输入文件，或用 --text 直接给文字")
        src = Path(args.input).expanduser()
        if not src.exists():
            sys.exit(f"找不到文件：{src}")
        ext = src.suffix.lower()
        if ext == ".txt":
            text = read_txt(src)
        elif ext == ".pdf":
            text = read_pdf(src)
        elif ext == ".docx":
            text, docx_chapters = read_docx(src)
        else:
            sys.exit("只支持 .txt / .pdf / .docx")
        chapters = docx_chapters or split_chapters(text)
        src_stem = src.stem
        default_out = src.with_name(src.stem + "_audio")
    chapters = [(t, b) for t, b in chapters if b.strip()]

    if args.list_chapters:
        for i, (t, b) in enumerate(chapters, 1):
            print(f"{i:4d}  {t}  （{len(b)} 字）")
        return

    selected = parse_ranges(args.chapters) if args.chapters else set(range(1, len(chapters) + 1))

    # ---- 读音词典 ----
    lex_files = [Path(p).expanduser() for p in args.lexicon]
    auto = Path(__file__).resolve().with_name("读音词典.txt")
    if auto.exists() and auto not in [f.resolve() for f in lex_files if f.exists()]:
        lex_files.insert(0, auto)
    for f in lex_files:
        if not f.exists():
            sys.exit(f"找不到词典文件：{f}")
    lexicon = Lexicon(lex_files, use_default=not args.no_default_lexicon)
    print(f"读音词典：{len(lexicon)} 条" + (f"（{', '.join(f.name for f in lex_files)}）" if lex_files else ""))

    max_len = CosyVoiceEngine.max_len if args.engine == "cosyvoice" else KokoroEngine.max_len

    hotfix = {}
    if args.engine == "cosyvoice" and not args.no_hotfix:
        hotfix = lexicon.hotfix_table(include_default=not args.no_default_lexicon)   # 加载模型后再按词表校验
    hotfix_rx = [None]

    def apply_hotfix(t):
        if not hotfix:
            return t
        if hotfix_rx[0] is None:
            keys = sorted(hotfix, key=len, reverse=True)
            hotfix_rx[0] = re.compile("|".join(map(re.escape, keys)))
        return hotfix_rx[0].sub(lambda m: hotfix[m.group(0)], t)

    def pieces_of(title, body):
        ps = split_sentences(body, max_len)
        if not args.no_title:
            ps = [(clean_text(title), True)] + ps
        return [(apply_hotfix(lexicon.apply(s)), end, s) for s, end in ps]

    if args.dry_run:
        for idx, (title, body) in enumerate(chapters, 1):
            if idx not in selected:
                continue
            print(f"\n=== [{idx}] {title} ===")
            for s, end, _ in pieces_of(title, body):
                print(s + ("  ¶" if end else ""))
        return

    # ---- 合成 ----
    outdir = Path(args.outdir) if args.outdir else default_out
    outdir.mkdir(parents=True, exist_ok=True)
    has_ffmpeg = shutil.which("ffmpeg") is not None
    if not has_ffmpeg:
        hint = "winget install ffmpeg" if sys.platform == "win32" else "brew install ffmpeg"
        print(f"⚠️ 没找到 ffmpeg，将输出 wav。安装：{hint}")

    cache_root = setup_model_cache(args.cache_dir)
    print(f"模型缓存：{cache_root}")
    print(f"共 {len(chapters)} 章，加载 {args.engine} 模型…")
    if args.engine == "kokoro":
        engine = KokoroEngine(args.voice, args.speed, args.device, args.kokoro_repo)
    else:
        model_dir = Path(args.cosyvoice_model or cache_root / "Fun-CosyVoice3-0.5B")
        if not model_dir.exists():
            sys.exit(f"找不到 CosyVoice 模型：{model_dir}\n下载：\n"
                     f"  python -c \"from huggingface_hub import snapshot_download; "
                     f"snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='{model_dir}')\"")
        repo = args.cosyvoice_repo or Path(__file__).resolve().parent / "CosyVoice"
        engine = CosyVoiceEngine(model_dir, args.ref_audio, args.ref_text,
                                 args.speed, repo, args.instruct, fp16=args.fp16)
        if hotfix:
            toks = engine.special_tokens()
            if toks:
                hotfix.clear()
                hotfix.update(lexicon.hotfix_table(toks, include_default=not args.no_default_lexicon))
                hotfix_rx[0] = None
            print(f"拼音热修复：{len(hotfix)} 条")

    if args.engine == "kokoro":
        apply_pinyin_overrides(lexicon.pinyin)

    checker = None
    if args.verify:
        vdir = args.verify_model or cache_root / "asr" / "sherpa-onnx-paraformer-zh-int8-2025-10-07"
        checker = OmissionChecker(vdir)
        print(f"漏读检查：开启（最多重试 {args.max_retry} 次）")
    vlog = (outdir / "漏读检查.txt") if checker else None

    sr = engine.sr
    # 语速放慢时，句间/段间停顿同比例拉长，听起来更自然
    pause_scale = 1.0 / max(args.speed, 0.3)
    s_pause = np.zeros(int(sr * args.sentence_pause * pause_scale), dtype=np.float32)
    p_pause = np.zeros(int(sr * args.para_pause * pause_scale), dtype=np.float32)
    width = max(3, len(str(len(chapters))))

    for idx, (title, body) in enumerate(chapters, 1):
        if idx not in selected:
            continue
        stem = f"{idx:0{width}d}_{safe_name(title)}"
        final = outdir / (stem + (".mp3" if has_ffmpeg else ".wav"))
        if final.exists():
            print(f"[{idx}/{len(chapters)}] 已存在，跳过：{final.name}")
            continue

        pieces = pieces_of(title, body)
        print(f"[{idx}/{len(chapters)}] {title}：{len(pieces)} 段")

        audio = []
        retried = 0
        for j, (seg, para_end, orig) in enumerate(pieces, 1):
            best = None
            for attempt in range(1 + (args.max_retry if checker else 0)):
                try:
                    a = engine.synth(seg)
                except Exception as e:  # 单句失败不影响整章
                    print(f"   ⚠️ 第 {j} 段合成失败：{e}")
                    continue
                if not checker:
                    best = (0, 0, a, "")
                    break
                miss, extra, heard = checker.check(orig, a, sr)
                if best is None or (miss, extra) < best[:2]:
                    best = (miss, extra, a, heard)
                if miss < 2 and extra < 3:      # 单字差异多半是识别误差，不算
                    break
                retried += 1
                print(f"   ↻ 第 {j} 段疑似{'漏读 %d 字' % miss if miss >= 2 else ''}{'多读 %d 字' % extra if extra >= 3 else ''}，重新合成（第 {attempt + 1} 次）")
            if best is None:
                print(f"   ⚠️ 第 {j} 段多次合成失败，已跳过")
                continue
            miss, extra, a, heard = best
            if checker and (miss >= 2 or extra >= 3):
                with open(vlog, "a", encoding="utf-8") as f:
                    f.write(f"{stem}\t第{j}段\t漏{miss}\t多{extra}\t原文：{orig}\t听到：{heard}\n")
            audio += [a, p_pause if para_end else s_pause]
            if j % 20 == 0:
                print(f"   {j}/{len(pieces)}")

        if not audio:
            continue
        wav_path = outdir / (stem + ".wav")
        write_wav(wav_path, np.concatenate(audio), sr)
        if has_ffmpeg:
            to_mp3(wav_path, final, title, src_stem, idx, args.bitrate)
        print(f"   ✅ {final.name}" + (f"（重新合成 {retried} 次）" if checker else ""))

    print(f"完成，输出在：{outdir.resolve()}")


if __name__ == "__main__":
    main()
