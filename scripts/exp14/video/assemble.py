"""Join the rendered explainer scenes into one MP4, merge the caption logs, and cut review stills.

Outputs: report/assets/video/critical_windows_explainer.mp4, results/exp14/anim/captions_all.json, and stills
(one every 4 s plus the last frame of each scene) in /n/netscratch/olveczky_lab/Lab/charleszhang/exp14_video/stills/video.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess

import imageio_ffmpeg

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
WORK = "/n/netscratch/olveczky_lab/Lab/charleszhang/exp14_video"
SCENES = ["S01Question", "S02Toy", "S03Measure", "S04Control", "S05Transport", "S06Finding1", "S07Finding2", "S08Finding3",
          "S09Finding4", "S10Recommend"]  # fmt: skip
FF = imageio_ffmpeg.get_ffmpeg_exe()
OUT = os.path.join(ROOT, "report/assets/video/critical_windows_explainer.mp4")


def duration(p):
    r = subprocess.run([FF, "-i", p], capture_output=True, text=True).stderr
    hms = r.split("Duration: ")[1].split(",")[0]
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def main():
    files = [os.path.join(WORK, f"media_{s}/videos/explainer/720p30/{s}.mp4") for s in SCENES]  # one media dir per scene (parallel renders)
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        print("missing scenes, not assembling:", missing)
        return
    lst = os.path.join(WORK, "concat.txt")
    with open(lst, "w") as fh:
        fh.writelines(f"file '{f}'\n" for f in files)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    subprocess.run([FF, "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24",
                    "-preset", "slow", "-tune", "animation", "-movflags", "+faststart", "-an", OUT], check=True, capture_output=True)  # fmt: skip
    # captions with absolute times
    t0, allc, bounds = 0.0, [], []
    for s, f in zip(SCENES, files):
        d = duration(f)
        cj = os.path.join(ROOT, "results/exp14/anim/captions", f"{s}.json")
        caps = json.load(open(cj))["captions"] if os.path.exists(cj) else []
        allc += [{"scene": s, "t": round(t0 + c["t"], 1), "text": c["text"]} for c in caps]
        bounds.append((s, t0, t0 + d))
        t0 += d
    json.dump({"duration": t0, "scenes": bounds, "captions": allc}, open(os.path.join(ROOT, "results/exp14/anim/captions_all.json"), "w"), indent=1)
    # stills
    sd = os.path.join(WORK, "stills/video")
    os.makedirs(sd, exist_ok=True)
    for f in glob.glob(os.path.join(sd, "*.png")):
        os.remove(f)
    times = sorted({round(x, 1) for x in [i * 4.0 + 2.0 for i in range(int(t0 // 4))]} | {round(b[2] - 0.6, 1) for b in bounds})
    for t in times:
        subprocess.run([FF, "-y", "-ss", f"{t}", "-i", OUT, "-frames:v", "1", "-vf", "scale=960:-1", os.path.join(sd, f"t{t:06.1f}.png")],
                       check=True, capture_output=True)  # fmt: skip
    with open(os.path.join(sd, "captions.txt"), "w") as fh:
        for c in allc:
            fh.write(f"[{c['t']:6.1f} s] {c['text']}\n")
    print(OUT, f"{os.path.getsize(OUT) / 1e6:.1f} MB", f"{t0:.0f} s", len(times), "stills")


if __name__ == "__main__":
    main()
