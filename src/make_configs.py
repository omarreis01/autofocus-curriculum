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
        # extended experiments (docs/plans/genis-deneyler.md)
        "teacher_rpg": base | {"condition": "teacher", "signal": "rpg"},
        "teacher_pg_eta0.003": base | {"condition": "teacher", "signal": "pg", "eta": 0.003},
        "teacher_pg_eta0.03": base | {"condition": "teacher", "signal": "pg", "eta": 0.03},
        "teacher_pg_noiw": base | {"condition": "teacher", "signal": "pg", "importance": False},
        "teacher_alp": base | {"condition": "alp"},
        "random_image": base | {"condition": "random_image"},
        "random_blur": base | {"condition": "random_blur"},
        "random_contrast": base | {"condition": "random_contrast"},
        "sharp_gn": base | {"condition": "sharp", "norm": "gn"},
        "random_gn": base | {"condition": "random", "norm": "gn"},
        "teacher_pg_gn": base | {"condition": "teacher", "signal": "pg", "norm": "gn"},
        "staircase_f025": base | {"condition": "staircase", "staircase_frac": 0.25},
        "staircase_f075": base | {"condition": "staircase", "staircase_frac": 0.75},
    }


def main(out=Path("configs")):
    for budget in BUDGETS:
        (out / budget).mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            for name, cfg in condition_configs(budget, seed).items():
                (out / budget / f"{name}_s{seed}.json").write_text(json.dumps(cfg) + "\n")


if __name__ == "__main__":
    main()
