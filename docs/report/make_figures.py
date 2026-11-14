"""Figures and tables for the CMPE 492 report.

Reads the per-condition summary `D` and the per-epoch curves `CV` from
docs/reports/pilot-short-s0.html (both made from the Drive result summaries)
and writes vector PDFs to figures/ and LaTeX table bodies to tables/.

Run: .venv/bin/python docs/report/make_figures.py
"""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "docs" / "reports" / "pilot-short-s0.html"
FIG = HERE / "figures"
TAB = HERE / "tables"

COLOR = {
    "sharp": "#222222",
    "staircase": "#2a9d4b",
    "staircase_f025": "#2a9d4b",
    "staircase_f075": "#2a9d4b",
    "reverse": "#d1342f",
    "random": "#1f6fd1",
    "shuffled": "#7fb2ee",
    "teacher_pg": "#7b3fa0",
    "teacher_spg": "#e68a00",
    "teacher_rpg": "#0f9e9e",
    "teacher_alp": "#8c6d31",
    "teacher_pg_eta0.003": "#b48ad0",
    "teacher_pg_eta0.03": "#4a1f66",
    "teacher_pg_noiw": "#c06bd6",
    "teacher_pg_gn": "#7b3fa0",
}
DASH = {"shuffled": (0, (5, 3)), "staircase_f025": (0, (2, 2)), "staircase_f075": (0, (6, 2)),
        "teacher_pg_gn": (0, (1, 2)), "teacher_pg_noiw": (0, (3, 1, 1, 1))}
UNIFORM_SIGMA = (0 + 0.75 + 1.5 + 3) / 4
UNIFORM_C = (1 + 0.6 + 0.35 + 0.2) / 4
MAIN7 = ["sharp", "staircase", "reverse", "random", "shuffled", "teacher_pg", "teacher_spg"]


def js_to_json(js):
    def expand(m, spread):
        n, v = int(m.group(1)), m.group(2)
        items = ",".join([v] * n)
        return items if spread else "[" + items + "]"

    js = re.sub(r"\.\.\.Array\((\d+)\)\.fill\(([^)]*)\)", lambda m: expand(m, True), js)
    js = re.sub(r"Array\((\d+)\)\.fill\(([^)]*)\)", lambda m: expand(m, False), js)
    js = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', js)
    js = re.sub(r"(?<=[\[,:])(-?)\.(\d)", r"\g<1>0.\2", js)
    return json.loads(js)


def load():
    s = SRC.read_text()
    d = s[s.index("const D=") + len("const D="):]
    d = d[: d.index("};") + 1]
    cv = s[s.index("const CV=") + len("const CV="):]
    cv = cv[: cv.index("}}};") + 3]
    return json.loads(d), js_to_json(cv)


def style():
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.6,
    })


def line(ax, y, name, lw=1.6):
    x = range(1, len(y) + 1)
    ax.plot(x, y, color=COLOR[name], ls=DASH.get(name, "-"), lw=lw, label=name)


def curves(cv, budget, groups, out):
    fig, axes = plt.subplots(len(groups), 2, figsize=(6.5, 2.3 * len(groups)), sharex=True)
    for row, (title, names) in enumerate(groups):
        for col, (key, ylab, ref, ylim) in enumerate([("s", r"mean blur $\sigma$ (px)", UNIFORM_SIGMA, (-0.1, 3.15)),
                                                      ("c", "mean contrast $c$", UNIFORM_C, (0.15, 1.05))]):
            ax = axes[row, col]
            ax.axhline(ref, color="#999999", ls=":", lw=1)
            for n in names:
                line(ax, cv[budget][n][key], n, lw=2.2 if n in ("random", "teacher_pg") else 1.5)
            ax.set_ylim(*ylim)
            ax.set_ylabel(ylab)
            ax.set_title(f"{title}: {'blur' if key == 's' else 'contrast'}")
            if row == len(groups) - 1:
                ax.set_xlabel("epoch")
        axes[row, 1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / out)
    plt.close(fig)


def curves_rows(cv, rows, out):
    """Compact report: one row per (title, budget, names); fixed orders are drawn once (30 epochs)."""
    fig, axes = plt.subplots(len(rows), 2, figsize=(6.5, 1.85 * len(rows)))
    for row, (title, budget, names) in enumerate(rows):
        for col, (key, ylab, ref, ylim) in enumerate([("s", r"mean blur $\sigma$ (px)", UNIFORM_SIGMA, (-0.1, 3.15)),
                                                      ("c", "mean contrast $c$", UNIFORM_C, (0.15, 1.05))]):
            ax = axes[row, col]
            ax.axhline(ref, color="#999999", ls=":", lw=1)
            for n in names:
                line(ax, cv[budget][n][key], n, lw=2.2 if n in ("random", "teacher_pg") else 1.5)
            ax.set_ylim(*ylim)
            ax.set_ylabel(ylab)
            ax.set_title(f"{title}: {'blur' if key == 's' else 'contrast'}")
            ax.set_xlabel("epoch")
        axes[row, 1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / out)
    plt.close(fig)


LABEL_POS = {  # (dx, dy, ha) so labels in the 85-87 % cluster do not overlap
    ("seen_mean", "reverse"): (-0.4, 0.15, "right"), ("seen_mean", "shuffled"): (-0.4, -0.45, "right"),
    ("seen_mean", "teacher_pg"): (0.4, -0.5, "left"), ("seen_mean", "teacher_spg"): (0.4, 0.25, "left"),
    ("unseen_mean", "shuffled"): (-0.4, -0.45, "right"), ("unseen_mean", "teacher_pg"): (0.4, -0.5, "left"),
    ("unseen_mean", "teacher_spg"): (0.4, 0.25, "left"), ("seen_mean", "random"): (-0.4, 0.3, "right"),
    ("unseen_mean", "random"): (-0.4, 0.3, "right"),
}


def main_scatter(D, out):
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9), sharey=True)
    for ax, key, lab in [(axes[0], "seen_mean", "CIFAR-10-C seen (%)"), (axes[1], "unseen_mean", "CIFAR-10-C unseen (%)")]:
        for n in MAIN7:
            r = D["full:" + n]
            ax.errorbar(r[key][0], r["clean"][0], xerr=r[key][1], yerr=r["clean"][1], fmt="o", ms=6,
                        color=COLOR[n], mfc=COLOR[n] if n != "shuffled" else "white", mew=1.5, capsize=2)
            dx, dy, ha = LABEL_POS.get((key, n), (0.35, 0.3, "left"))
            ax.annotate(n, (r[key][0] + dx, r["clean"][0] + dy), fontsize=7.5, color=COLOR[n], ha=ha)
        ax.set_xlabel(lab)
        ax.set_xlim(70, 87.5)
        ax.set_xticks([70, 75, 80, 85])
        ax.set_ylim(84.5, 95.5)
    axes[0].set_ylabel("clean accuracy (%)")
    fig.tight_layout()
    fig.savefig(FIG / out)
    plt.close(fig)


def worst_share(D, out):
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.4), sharey=True, gridspec_kw={"width_ratios": [1.5, 1]})
    x = range(1, 6)
    for ax, budget, names, title in [
        (axes[0], "short", ["teacher_pg", "teacher_spg", "teacher_rpg", "teacher_alp", "teacher_pg_eta0.003",
                            "teacher_pg_eta0.03", "teacher_pg_noiw", "teacher_pg_gn"], "30 epochs"),
        (axes[1], "full", ["teacher_pg", "teacher_spg", "shuffled"], "100 epochs"),
    ]:
        ax.axhline(1 / 16, color="#999999", ls=":", lw=1)
        for n in names:
            ax.plot(x, D[f"{budget}:{n}"]["worst"], marker="o", ms=3.5, color=COLOR[n], ls=DASH.get(n, "-"),
                    lw=2 if n == "teacher_pg" else 1.4, label=n)
        ax.set_xticks(list(x))
        ax.set_xlabel("fifth of training")
        ax.set_title(title)
        ax.set_ylim(-0.02, 0.9)
        ax.legend(frameon=False, ncol=2 if budget == "short" else 1, loc="upper center", bbox_to_anchor=(0.5, -0.24),
                  fontsize=6.8)
    axes[0].set_ylabel(r"share of batches on ($\sigma$=3, $c$=0.2)")
    fig.tight_layout()
    fig.savefig(FIG / out)
    plt.close(fig)


def shape_bias(D, out, height=2.6):
    fig, ax = plt.subplots(figsize=(6.3, height))
    w = 0.38
    for i, n in enumerate(MAIN7):
        for j, (b, alpha) in enumerate([("short", 0.55), ("full", 1.0)]):
            m, s = D[f"{b}:{n}"]["bias2"]
            ax.bar(i + (j - 0.5) * w, m, w, yerr=s, color=COLOR[n], alpha=alpha, capsize=2,
                   edgecolor=COLOR[n], hatch="//" if n == "shuffled" else None)
            ax.text(i + (j - 0.5) * w, m + s + 1.2, f"{m:.0f}", ha="center", fontsize=6.8)
    ax.set_xticks(range(len(MAIN7)))
    ax.set_xticklabels(MAIN7, rotation=0, fontsize=7.5)
    ax.set_ylabel(r"shape bias, $\sigma$=2 (%)")
    ax.set_ylim(0, 78)
    ax.text(0.01, 0.95, "left bar: 30 epochs (light)   right bar: 100 epochs (dark)", transform=ax.transAxes,
            fontsize=7.5, va="top")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / out)
    plt.close(fig)


# ---------- tables ----------
COLS = [("clean", "Clean"), ("seen_mean", "C-seen"), ("unseen_mean", "C-unseen"), ("noise", "Noise"),
        ("contrast", "C-contr."), ("bias2", "Shape"), ("high1", "HP $\\sigma$1"), ("pgd1", "PGD")]


def tex_name(n):
    return "\\texttt{" + n.replace("_", "\\_") + "}"


def cell(v):
    return f"${v[0]:.1f}{{\\scriptstyle\\,\\pm{v[1]:.1f}}}$"


def table(D, rows, cols, out, extra=None):
    """rows: list of (budget, name) or a string for a \\midrule group label."""
    spec = "l" + ("c" if any(isinstance(r, tuple) and r[2] for r in rows if isinstance(r, tuple)) else "")
    show_ep = any(isinstance(r, tuple) and r[2] for r in rows)
    head = ["Condition"] + (["Ep."] if show_ep else []) + [c[1] for c in cols] + ([extra[0]] if extra else [])
    lines = ["\\begin{tabular}{" + spec + "c" * (len(cols) + (1 if extra else 0)) + "}", "\\hline",
             " & ".join(head) + " \\\\", "\\hline"]
    for r in rows:
        if isinstance(r, str):
            lines.append(f"\\multicolumn{{{len(head)}}}{{l}}{{\\textit{{{r}}}}} \\\\")
            continue
        b, n, ep = r
        d = D[f"{b}:{n}"]
        vals = [tex_name(n)] + ([str(30 if b == "short" else 100)] if show_ep else []) + [cell(d[k]) for k, _ in cols]
        if extra:
            vals.append(extra[1](d))
        lines.append(" & ".join(vals) + " \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    (TAB / out).write_text("\n".join(lines) + "\n")


def tables(D):
    c_all = COLS
    orders = []
    for b in ("short", "full"):
        for n in ("sharp", "staircase", "reverse", "random"):
            orders.append((b, n, True))
    table(D, orders, c_all, "orders.tex")

    late = ("Hard arm 3/5", lambda d: f"{d['worst'][2]:.2f} / {d['worst'][4]:.2f}")
    teach = [("short", n, False) for n in ["random", "teacher_pg", "teacher_spg", "teacher_rpg", "teacher_pg_eta0.003",
                                           "teacher_pg_eta0.03", "teacher_pg_noiw", "teacher_alp", "shuffled"]]
    table(D, teach, [c for c in c_all if c[0] in ("clean", "seen_mean", "unseen_mean", "bias2", "pgd1")],
          "teachers.tex", extra=late)

    regime = ["Per batch vs per image (BatchNorm)"] + [("short", n, False) for n in ["random", "random_image"]] + \
             ["One axis only"] + [("short", n, False) for n in ["random_blur", "random_contrast"]] + \
             ["GroupNorm vs BatchNorm"] + [("short", n, False) for n in ["sharp", "sharp_gn", "random_gn", "teacher_pg",
                                                                          "teacher_pg_gn"]] + \
             ["Staircase length"] + [("short", n, False) for n in ["staircase_f025", "staircase", "staircase_f075"]]
    table(D, regime, [c for c in c_all if c[0] in ("clean", "seen_mean", "unseen_mean", "noise", "contrast", "pgd1")],
          "regime.tex")

    pgd = []
    for n in MAIN7:
        pgd += [("short", n, True), ("full", n, True)]
    table(D, pgd, [c for c in c_all if c[0] in ("clean", "bias2", "high1", "pgd1")], "pgd.tex")

    order30 = ["sharp", "staircase", "reverse", "random", "shuffled", "teacher_pg", "teacher_spg", "teacher_rpg",
               "teacher_pg_eta0.003", "teacher_pg_eta0.03", "teacher_pg_noiw", "teacher_alp", "random_image",
               "random_blur", "random_contrast", "sharp_gn", "random_gn", "teacher_pg_gn", "staircase_f025",
               "staircase_f075"]
    assert sorted(order30) == sorted(k[6:] for k in D if k.startswith("short:")), "missing a 30-epoch condition"
    table(D, [("short", n, False) for n in order30], c_all, "full30.tex")
    table(D, [("full", n, False) for n in MAIN7], c_all, "full100.tex")


def compact_figures(D, CV):
    shape_bias(D, "shape_bias_compact.pdf", height=2.0)
    curves_rows(CV, [("Orders, 30 ep", "short", ["sharp", "staircase", "reverse", "random"]),
                     ("Teachers, 30 ep", "short", ["teacher_pg", "teacher_spg", "teacher_rpg", "teacher_alp",
                                                   "shuffled"]),
                     ("Teachers, 100 ep", "full", ["teacher_pg", "teacher_spg", "shuffled"])],
                "curves_compact.pdf")


def main():
    FIG.mkdir(exist_ok=True)
    TAB.mkdir(exist_ok=True)
    D, CV = load()
    style()
    curves(CV, "short", [("Orders", ["sharp", "staircase", "reverse", "random"]),
                         ("Teachers", ["teacher_pg", "teacher_spg", "teacher_rpg", "teacher_alp", "shuffled"])],
           "curves_30ep.pdf")
    curves(CV, "full", [("Orders", ["sharp", "staircase", "reverse", "random"]),
                        ("Teachers", ["teacher_pg", "teacher_spg", "shuffled"])], "curves_100ep.pdf")
    compact_figures(D, CV)
    main_scatter(D, "main_100ep.pdf")
    worst_share(D, "worst_share.pdf")
    shape_bias(D, "shape_bias.pdf")
    tables(D)
    print("figures:", sorted(p.name for p in FIG.glob("*.pdf")))
    print("tables:", sorted(p.name for p in TAB.glob("*.tex")))


if __name__ == "__main__":
    main()
