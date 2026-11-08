import argparse
import json
from pathlib import Path

import numpy as np

SIGMAS = np.array([0.0, 0.75, 1.5, 3.0])
CONTRASTS = np.array([1.0, 0.6, 0.35, 0.2])


def main():
    # Per-epoch mean blur sigma (px) and contrast of the chosen arms, averaged over seeds.
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="/content/drive/MyDrive/auto-curriculum/results")
    a = p.parse_args()
    out = {}
    for b in ("short", "full"):
        for c, r in json.loads((Path(a.results) / f"{b}_summary.json").read_text()).items():
            h = np.array(r["arm_hist"], dtype=float)
            share = h / h.sum(2, keepdims=True)
            sig = (share * SIGMAS[np.arange(16) // 4]).sum(2).mean(0)
            con = (share * CONTRASTS[np.arange(16) % 4]).sum(2).mean(0)
            out[f"{b}:{c}"] = {"s": np.round(sig, 2).tolist(), "c": np.round(con, 2).tolist()}
    print(json.dumps(out, separators=(",", ":")))


if __name__ == "__main__":
    main()
