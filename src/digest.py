import argparse
import json
from pathlib import Path

import numpy as np

NOISE = ["gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise"]


def ms(v):
    return [round(100 * float(np.mean(v)), 1), round(100 * float(np.std(v)), 1)]


def digest(r):
    # Compact per-condition numbers for the report: mean and std over seeds, in percent.
    row = {k: ms(r[k]) for k in ("clean", "seen_mean", "unseen_mean")}
    cm, n = r["corruption_mean"], len(r["seeds"])
    row["noise"] = ms([np.mean([cm[c][i] for c in NOISE]) for i in range(n)])
    row["contrast"] = ms(cm["contrast"])
    probes = r.get("probe", [])
    if probes:
        for s in ("1", "2", "3"):
            row[f"bias{s}"] = ms([p["hybrid"][s]["bias"] for p in probes])
        row["low2"] = ms([p["lowpass"]["2"] for p in probes])
        row["high1"] = ms([p["highpass"]["1"] for p in probes])
        row["fgsm2"] = ms([p["fgsm"]["2/255"] for p in probes])
        row["pgd1"] = ms([p["pgd10"]["1/255"] for p in probes])
        row["pgd2"] = ms([p["pgd10"]["2/255"] for p in probes])
    h = np.array(r["arm_hist"])
    share = h / h.sum(2, keepdims=True)
    row["worst"] = np.round(share[:, :, 15].mean(0).reshape(5, -1).mean(1), 2).tolist()
    row["blur"] = np.round((share * (np.arange(16) // 4)).sum(2).mean(0).reshape(5, -1).mean(1), 2).tolist()
    row["n"] = n
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="/content/drive/MyDrive/auto-curriculum/results")
    a = p.parse_args()
    out = {}
    for b in ("short", "full"):
        for c, r in json.loads((Path(a.results) / f"{b}_summary.json").read_text()).items():
            out[f"{b}:{c}"] = digest(r)
    print(json.dumps(out, separators=(",", ":")))


if __name__ == "__main__":
    main()
