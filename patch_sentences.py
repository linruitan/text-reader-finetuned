"""按 校对\\修改清单.json 只重新合成指定的句子，替换进原来的 mp3。
原 mp3 先备份到 法华经\\_修改前备份\\。已完成的章节记录在 校对\\修改结果.json，中断后重跑会跳过。"""
import json, subprocess, sys, time
from pathlib import Path
import numpy as np
here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
import local_tts as L

INSTRUCT = "请用舒缓、平和、庄重的语速朗读。"
MAX_RETRY = 3

def decode(mp3, sr):
    raw = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(mp3), "-f", "f32le", "-ac", "1",
                          "-ar", str(sr), "-"], capture_output=True, check=True, stdin=subprocess.DEVNULL).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()

def silences(x, sr, thr=1e-5, minlen=0.15):
    q = np.abs(x) < thr
    d = np.diff(np.concatenate([[0], q.astype(np.int8), [0]]))
    st = np.where(d == 1)[0]; en = np.where(d == -1)[0]
    return [(int(a), int(b)) for a, b in zip(st, en) if b - a >= minlen * sr]

def match_bounds(x, sr, flags):
    s = silences(x, sr); d = [(b - a) / sr for a, b in s]
    n, m = len(flags), len(s)
    if m < n: return None, None
    INF = float("inf")
    def cost(j, k):
        if (d[k] > 0.45) != flags[j]: return 50.0
        return abs(d[k] - (0.6 if flags[j] else 0.25))
    D = [[INF] * (m + 1) for _ in range(n + 1)]; P = [[0] * (m + 1) for _ in range(n + 1)]
    for k in range(m + 1): D[0][k] = float(k)
    for j in range(1, n + 1):
        for k in range(j, m + 1):
            a = D[j][k - 1] + 1.0
            b = D[j - 1][k - 1] + cost(j - 1, k - 1)
            D[j][k], P[j][k] = (b, 1) if b <= a else (a, 0)
    j, k, pick = n, m, []
    while j > 0:
        if P[j][k] == 1: pick.append(k - 1); j -= 1; k -= 1
        else: k -= 1
    pick.reverse()
    return [s[k] for k in pick], D[n][m]

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="修改清单.json")
    ap.add_argument("--result", default="修改结果.json")
    ap.add_argument("--from-current", action="store_true", help="在当前 mp3 上继续改（第二轮），而不是从备份开始")
    ap.add_argument("--text", default=str(Path("法华经") / "妙法莲华经.txt"), help="原文 txt（相对本文件夹）")
    ap.add_argument("--audio-dir", default="法华经", help="mp3 所在文件夹（相对本文件夹）")
    ap.add_argument("--instruct", default=INSTRUCT, help="CosyVoice 风格指令，要和最初生成时一致")
    args = ap.parse_args()
    plan = json.load(open(here / "校对" / args.plan, encoding="utf-8"))
    done_path = here / "校对" / args.result
    done = json.load(open(done_path, encoding="utf-8")) if done_path.exists() else {}
    cache_root = L.setup_model_cache(None)
    lexicon = L.Lexicon([here / "读音词典.txt"])
    print(f"读音词典：{len(lexicon)} 条")
    engine = L.CosyVoiceEngine(cache_root / "Fun-CosyVoice3-0.5B", None, None, 1.0, here / "CosyVoice", args.instruct, fp16=True)
    hotfix = lexicon.hotfix_table(engine.special_tokens(), include_default=True)
    import re
    hrx = re.compile("|".join(map(re.escape, sorted(hotfix, key=len, reverse=True))))
    spoken = lambda s: hrx.sub(lambda m: hotfix[m.group(0)], lexicon.apply(s))
    print(f"拼音热修复：{len(hotfix)} 条")
    checker = L.OmissionChecker(cache_root / "asr" / "sherpa-onnx-paraformer-zh-int8-2025-10-07")
    text = L.read_txt(here / args.text)
    chapters = [(t, b) for t, b in L.split_chapters(text) if b.strip()]
    sr = engine.sr
    backup = here / args.audio_dir / "_修改前备份"; backup.mkdir(exist_ok=True)
    total = sum(len(c["pieces"]) for c in plan); n_done = 0; t0 = time.time()
    for ch in plan:
        idx, stem = ch["idx"], ch["stem"]
        if stem in done:
            n_done += len(ch["pieces"]); print(f"[{idx}] 已完成，跳过"); continue
        title, body = chapters[idx - 1]
        pieces = [(L.clean_text(title), True)] + L.split_sentences(body, engine.max_len)
        mp3 = here / args.audio_dir / f"{stem}.mp3"
        src = backup / mp3.name
        if not src.exists():
            src.write_bytes(mp3.read_bytes())
        x = decode(mp3 if args.from_current else src, sr)
        sil, c = match_bounds(x, sr, [p[1] for p in pieces])
        if sil is None or c > 20:
            print(f"[{idx}] ⚠️ 句子边界对不上（{c}），跳过这一品"); continue
        starts = [0] + [b for _, b in sil[:-1]]
        new = {}
        for p in ch["pieces"]:
            j = p["n"]; orig = pieces[j - 1][0]
            if orig != p["text"]:
                print(f"[{idx}] ⚠️ 第 {j} 句文字对不上，跳过：{orig[:20]}"); continue
            best = None
            for attempt in range(1 + MAX_RETRY):
                a = engine.synth(spoken(orig))
                miss, extra, heard = checker.check(orig, a, sr)
                if best is None or (miss, extra) < best[:2]: best = (miss, extra, a, heard)
                if miss < 2 and extra < 3: break
                print(f"   ↻ 第 {j} 句疑似漏/多读，重试")
            new[j] = best[2]
            n_done += 1
            el = time.time() - t0
            print(f"[{idx}] 第 {j} 句 {''.join(p['why'])}  ({n_done}/{total}，已用 {el/60:.1f} 分钟)", flush=True)
        out, spans, pos = [], {}, 0
        for j in range(1, len(pieces) + 1):
            a = new.get(j, x[starts[j - 1]:sil[j - 1][0]])
            if j in new: spans[j] = [round(pos / sr, 2), round((pos + len(a)) / sr, 2)]
            out += [a, x[sil[j - 1][0]:sil[j - 1][1]]]
            pos += len(a) + (sil[j - 1][1] - sil[j - 1][0])
        out.append(x[sil[-1][1]:])
        wav = mp3.with_suffix(".wav")
        L.write_wav(wav, np.concatenate(out), sr)
        L.to_mp3(wav, mp3, title, "妙法莲华经", idx, "96k")
        wav.unlink(missing_ok=True)
        done[stem] = spans
        json.dump(done, open(done_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"   ✅ {mp3.name}：替换 {len(new)} 句")
    print("全部完成 PATCH_DONE")

if __name__ == "__main__":
    main()
