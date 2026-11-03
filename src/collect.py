import argparse
import json
from pathlib import Path

import numpy as np

CONDITIONS = ["sharp", "staircase", "reverse", "random", "shuffled", "teacher_pg", "teacher_spg"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="results/<budget> directory")
    p.add_argument("--epochs", type=int, required=True)
    a = p.parse_args()
    root = Path(a.results)
    out = {}
    for c in CONDITIONS:
        runs = sorted(d for d in (root / c).glob("*") if (d / "eval.json").exists())
        evals = [json.loads((d / "eval.json").read_text()) for d in runs]
        row = {"seeds": [d.name for d in runs]}
        for k in ("clean", "seen_mean", "unseen_mean", "all_mean"):
            row[k] = [e[k] for e in evals]
        row["corruption_mean"] = {n: [e["corruption_mean"][n] for e in evals] for n in evals[0]["corruption_mean"]} if evals else {}
        # Arm counts per epoch, per seed: [seed][epoch][arm]
        row["arm_hist"] = []
        for d in runs:
            arms = np.load(d / "arms.npy")
            spe = len(arms) // a.epochs
            row["arm_hist"].append([np.bincount(arms[e * spe:(e + 1) * spe], minlength=16).tolist() for e in range(a.epochs)])
        out[c] = row
    print(json.dumps(out, separators=(",", ":")))


if __name__ == "__main__":
    main()
