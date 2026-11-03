import json
from pathlib import Path

BUDGETS = {"short": 30, "full": 100}
SEEDS = [0, 1, 2]
RESULTS = "/content/drive/MyDrive/auto-curriculum/results"


def condition_configs(budget, seed, results=RESULTS):
    base = {"seed": seed, "epochs": BUDGETS[budget]}
    return {
        "teacher_pg": base | {"condition": "teacher", "signal": "pg"},
        "teacher_spg": base | {"condition": "teacher", "signal": "spg"},
        "sharp": base | {"condition": "sharp"},
        "random": base | {"condition": "random"},
        "shuffled": base | {"condition": "shuffled", "arms_path": f"{results}/{budget}/teacher_pg/{seed}/arms.npy"},
        "staircase": base | {"condition": "staircase"},
        "reverse": base | {"condition": "reverse"},
    }


def main(out=Path("configs")):
    for budget in BUDGETS:
        (out / budget).mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            for name, cfg in condition_configs(budget, seed).items():
                (out / budget / f"{name}_s{seed}.json").write_text(json.dumps(cfg) + "\n")


if __name__ == "__main__":
    main()
