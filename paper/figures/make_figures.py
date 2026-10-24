#!/usr/bin/env python3
"""Make the paper's four figures (vector PDF) from the committed CSVs.

Inputs : ../../data/*.csv
Outputs: fig1_same_seed_pair.pdf, fig2_pareto_valid.pdf, fig3_pareto_test.pdf,
         fig4_screens.pdf      (written next to this script)

Every plotted number is read from the CSVs; nothing is retyped. Valid and test
never share a figure or an axis. Run: python make_figures.py
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
CSV = HERE.parent.parent / "data"

# Start from matplotlib's shipped defaults (ignores any site matplotlibrc, e.g. one that swaps the font),
# then change only what the paper needs.
matplotlib.style.use("default")
plt.rcParams.update({
    "font.size": 8,
    "pdf.fonttype": 42,   # embed TrueType so text stays text in the PDF
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
})
WIDE = (6.5, 2.8)
GRID = (6.5, 4.5)

TAB = {name: f"tab:{name}" for name in
       ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]}

# One colour + one marker per arm family (ids as in the Pareto CSVs' `id` prefix).
FAMILY = {
    # references
    "cot_lr3e-4":       dict(label="CoT bar (tuned)",      color="black",        marker="*"),
    "cot_lr3e-5":       dict(label="CoT bar (tuned)",      color="black",        marker="*"),
    "no_cot":           dict(label="no_cot (124M)",        color=TAB["gray"],    marker="X"),
    "plain_depth_355m": dict(label="plain depth 355M",     color=TAB["gray"],    marker="P"),
    "pause":            dict(label="pause",                color=TAB["cyan"],    marker="p"),
    # latent arms
    "coconut":          dict(label="coconut",              color=TAB["blue"],    marker="o"),
    "c1b_k8":           dict(label="C1b k=8",              color=TAB["orange"],  marker="s"),
    "c1b_k4":           dict(label="C1b k=4",              color=TAB["green"],   marker="D"),
    "c2":               dict(label="C2",                   color=TAB["red"],     marker="^"),
    "c2_shuffled":      dict(label="C2-shuffled",          color=TAB["purple"],  marker="v"),
    "c3":               dict(label="C3",                   color=TAB["brown"],   marker="<"),
    "c3_foreign":       dict(label="C3-foreign",           color=TAB["pink"],    marker=">"),
    "c2c3":             dict(label="C2+C3",                color=TAB["olive"],   marker="h"),
}
REF_FAMILIES = ["cot", "no_cot", "plain_depth_355m", "pause"]
TASKS = [("gsm50k", "GSM-50k"), ("prosqa_hard", "ProsQA-hard")]


def save(fig, name: str) -> None:
    out = HERE / name
    fig.savefig(out, format="pdf", bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    print(f"wrote {out} ({out.stat().st_size / 1024:.1f} KB)")


def panel_letter(ax, letter: str) -> None:
    ax.text(0.01, 0.98, f"({letter})", transform=ax.transAxes, ha="left", va="top",
            fontweight="bold")


# --------------------------------------------------------------------------- fig 1
def fig1() -> None:
    curves = pd.read_csv(CSV / "figure1_same_seed_pair_curves.csv")
    pair = pd.read_csv(CSV / "table1b_same_seed_pair.csv")
    colors = [TAB["blue"], TAB["orange"], TAB["green"]]
    markers = ["o", "s", "D"]

    sel_epochs = sorted(int(e) for e in pair["selected_epoch"])

    def label_side(sel: int):
        """Neighbouring selections (within 6 epochs) get their labels pushed apart."""
        i = sel_epochs.index(sel)
        if i + 1 < len(sel_epochs) and sel_epochs[i + 1] - sel <= 6:
            return "right", -3
        if i > 0 and sel - sel_epochs[i - 1] <= 6:
            return "left", 3
        return "center", 0

    fig, ax = plt.subplots(figsize=WIDE)
    for (_, row), c, m in zip(pair.iterrows(), colors, markers):
        col = f"{row['run']}_val_acc"
        y = curves[col] * 100.0  # CSV holds fractions of valid 300
        ax.plot(curves["epoch"], y, color=c, lw=1.1, label=f"{row['member']} ({row['run']})")
        sel = int(row["selected_epoch"])
        ysel = float(y[curves["epoch"] == sel].iloc[0])
        # The selected epoch is the all-epoch best on valid for every member.
        assert math.isclose(ysel, y.max()), (row["run"], sel, ysel, y.max())
        ax.plot([sel], [ysel], marker=m, ms=6, color=c, mec="black", mew=0.6, ls="none", zorder=5)
        ha, dx = label_side(sel)
        ax.annotate(f"epoch {sel}", (sel, ysel), xytext=(dx, 6), textcoords="offset points",
                    ha=ha, va="bottom", color=c, fontsize=7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Valid accuracy (%)")
    ax.set_xlim(curves["epoch"].min() - 0.5, curves["epoch"].max() + 0.5)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 2.0)
    ax.grid(True, lw=0.3, alpha=0.5)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], ls="none", marker="o", color="0.4", mec="black", mew=0.6))
    labels.append("selected epoch (best valid)")
    ax.legend(handles, labels, loc="lower right", frameon=False)
    save(fig, "fig1_same_seed_pair.pdf")


# --------------------------------------------------------------------------- fig 2 / 3
def family_of(point_id: str) -> str:
    return point_id.split(":")[0]


def role_of(point_id: str) -> str:
    role = point_id.split(":")[1]
    if role.startswith("sweep"):
        return "sweep"
    assert role in ("selection", "earlier"), point_id
    return role


def x_flops(df: pd.DataFrame):
    """Mean GFLOPs; a 'log-mean range' point sits at the middle of [lo, hi] with the range as x error."""
    mid = (df["gflops_mean_lo"] + df["gflops_mean_hi"]) / 2.0
    err = [mid - df["gflops_mean_lo"], df["gflops_mean_hi"] - mid]
    return mid, err


def pareto(csv_name: str, split_word: str, out: str) -> None:
    df = pd.read_csv(CSV / csv_name)
    assert df["split"].nunique() == 1  # one split per figure
    df["family"] = df["id"].map(family_of)
    df["role"] = df["id"].map(role_of)
    unknown = set(df["family"]) - set(FAMILY)
    assert not unknown, unknown
    # Draw the acc_band interval only if every non-sweep point carries one. On valid the bar, the
    # references and the earlier-stage points have no per-example record (no band), so drawing the
    # bands that exist would make those points look exact; on test every point has one.
    main_pts = df[df["role"] != "sweep"]
    show_bands = bool(main_pts[["acc_band_lo", "acc_band_hi"]].notna().all(axis=None))

    fig, axes = plt.subplots(2, 2, figsize=GRID)
    letters = iter("abcd")
    for r, (task, task_name) in enumerate(TASKS):
        d = df[df["task"] == task]
        for c, xkind in enumerate(["tokens", "flops"]):
            ax = axes[r, c]
            for fam, g in d.groupby("family", sort=False):
                st = FAMILY[fam]
                for role, gg in g.groupby("role", sort=False):
                    if xkind == "tokens":
                        x, xerr = gg["emitted_mean_E"], None
                    else:
                        x, xerr = x_flops(gg)
                    y = gg["accuracy"]
                    band = gg[["acc_band_lo", "acc_band_hi"]].notna().all(axis=1)
                    if show_bands and role != "sweep" and band.any():
                        yb = gg[band]
                        xb = x[band]
                        ax.errorbar(xb, yb["accuracy"],
                                    yerr=[yb["accuracy"] - yb["acc_band_lo"],
                                          yb["acc_band_hi"] - yb["accuracy"]],
                                    fmt="none", ecolor=st["color"], elinewidth=0.5,
                                    alpha=0.45, capsize=0, zorder=2)
                    if xerr is not None and role != "sweep":
                        ax.errorbar(x, y, xerr=xerr, fmt="none", ecolor=st["color"],
                                    elinewidth=0.6, alpha=0.6, capsize=0, zorder=2)
                    is_star = st["marker"] == "*"
                    kw = dict(marker=st["marker"], ls="none", zorder=4)
                    if role == "selection":
                        kw.update(ms=9 if is_star else 5.5, color=st["color"], mec="black", mew=0.4)
                    elif role == "earlier":
                        kw.update(ms=5.5, mfc="white", mec=st["color"], mew=1.0)
                    else:  # step sweep of the final checkpoint (valid only)
                        kw.update(ms=3, color=st["color"], alpha=0.45, mew=0, zorder=3)
                    ax.plot(x, y, **kw)
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            if xkind == "tokens":
                ax.set_xticks([t for t in (1, 2, 5, 10, 20, 50, 100)
                               if d["emitted_mean_E"].min() / 1.3 <= t <= d["emitted_mean_E"].max() * 1.3])
            else:
                lo_f, hi_f = d["gflops_mean_lo"].min(), d["gflops_mean_hi"].max()
                cands = (10, 12, 15, 20, 30, 40) if hi_f < 100 else (100, 120, 150, 200, 300, 400)
                ax.set_xticks([t for t in cands if lo_f / 1.1 <= t <= hi_f * 1.1])
            ax.grid(True, which="major", lw=0.3, alpha=0.5)
            if r == 1:
                ax.set_xlabel("Mean emitted tokens per example, $E'$ (log scale)" if xkind == "tokens"
                              else "Mean FLOPs per example (GFLOPs, log scale)")
            if c == 0:
                ax.set_ylabel(f"{task_name}\n{split_word} accuracy (%)")
            panel_letter(ax, next(letters))

    # Legend: families present (colour + marker), then the role key.
    present = list(dict.fromkeys(df["family"]))
    order = [f for f in FAMILY if f in present]
    fam_handles, seen = [], set()
    for f in order:
        lab = FAMILY[f]["label"]
        if lab in seen:
            continue
        seen.add(lab)
        st = FAMILY[f]
        fam_handles.append(Line2D([], [], ls="none", marker=st["marker"], color=st["color"],
                                  mec="black", mew=0.4, ms=8 if st["marker"] == "*" else 5,
                                  label=lab))
    roles = set(df["role"])
    role_handles = [Line2D([], [], ls="none", marker="o", color="0.35", mec="black", mew=0.4,
                           ms=5, label="final selection")]
    if "earlier" in roles:
        role_handles.append(Line2D([], [], ls="none", marker="o", mfc="white", mec="0.35",
                                   mew=1.0, ms=5, label="earlier-stage checkpoint"))
    if "sweep" in roles:
        role_handles.append(Line2D([], [], ls="none", marker="o", color="0.35", alpha=0.45,
                                   mew=0, ms=3, label="step sweep, n = 0, 1, 2, 4, 8"))
    if show_bands:
        role_handles.append(Line2D([], [], color="0.35", lw=0.5, alpha=0.6,
                                   label="accuracy interval (per-example record)"))
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.legend(handles=fam_handles + role_handles, loc="lower center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, 0.0), handletextpad=0.3, columnspacing=1.0)
    save(fig, out)


# --------------------------------------------------------------------------- fig 4
def sigma_screen() -> float:
    """sigma_screen = max(1.0, binomial SE at the bar's valid accuracy, ProsQA cot 3-seed SD) (draft 3.4)."""
    k1 = pd.read_csv(CSV / "table1_k1_reproduction.csv")
    sd = float(k1[(k1["task"] == "ProsQA") & (k1["arm"] == "cot")]["sd"].iloc[0])
    pts = pd.read_csv(CSV / "figure2a_pareto_points_valid.csv")
    ses = []
    for task, _ in TASKS:
        bar = pts[(pts["task"] == task) & pts["id"].str.startswith("cot_") & (pts["id"].str.endswith(":selection"))]
        p, n = float(bar["accuracy"].iloc[0]) / 100.0, int(bar["n"].iloc[0])
        ses.append(100.0 * math.sqrt(p * (1 - p) / n))
    return max([1.0, sd] + ses)


def ref_kind(arm: str) -> str | None:
    if arm.startswith("cot ") and "(bar)" in arm:
        return "bar"
    if arm == "no_cot":
        return "no_cot"
    if arm.startswith("plain depth"):
        return "plain_depth"
    if arm == "pause":
        return "pause"
    return None


def short_arm(arm: str) -> str:
    if " (" in arm:
        head, tail = arm.split(" (", 1)
        return f"{head} (control)" if "control" in tail else head
    return arm


def fig4() -> None:
    sig = sigma_screen()
    refstyle = {
        "no_cot":      dict(color=TAB["gray"], ls="--", lw=0.8, label="no_cot"),
        "plain_depth": dict(color=TAB["gray"], ls="-.", lw=0.8, label="plain depth 355M no_cot"),
        "pause":       dict(color=TAB["cyan"], ls=":",  lw=1.1, label="pause"),
    }
    fig, axes = plt.subplots(1, 2, figsize=WIDE)
    for ax, (fname, task_name), letter in zip(
            axes, [("table2_gsm50k_screen.csv", "GSM-50k"), ("table3_prosqa_hard_screen.csv", "ProsQA-hard")],
            "ab"):
        t = pd.read_csv(CSV / fname, dtype={"valid_acc": str})
        t["ref"] = t["arm"].map(ref_kind)
        refs = {k: float(v) for k, v in zip(t["ref"], t["valid_acc"]) if isinstance(k, str)}
        assert set(refs) == {"bar", "no_cot", "plain_depth", "pause"}, refs
        arms = t[t["ref"].isna()].reset_index(drop=True)
        ypos = list(range(len(arms)))[::-1]  # first CSV row at the top
        bar = refs["bar"]
        ax.axvspan(bar - 2 * sig, bar + 2 * sig, color="0.92", lw=0, zorder=0)
        ax.axvspan(bar - sig, bar + sig, color="0.80", lw=0, zorder=0)
        ax.axvline(bar, color="black", lw=1.2, zorder=3)
        for k, st in refstyle.items():
            ax.axvline(refs[k], zorder=3, **{kk: vv for kk, vv in st.items() if kk != "label"})
        xmax = bar + 2 * sig
        for y, (_, row) in zip(ypos, arms.iterrows()):
            txt = row["valid_acc"]
            try:
                v = float(txt)
            except ValueError:  # e.g. "UNTESTED": no bar, the CSV's word printed instead
                ax.text(0.5, y, txt, va="center", ha="left", color="0.35", fontstyle="italic", fontsize=7)
                continue
            ax.barh(y, v, height=0.62, color=TAB["blue"], zorder=2)
            ax.text(v + 0.012 * xmax, y, txt, va="center", ha="left", fontsize=7, zorder=4,
                    bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.85))
        ax.set_yticks(ypos)
        ax.set_yticklabels([short_arm(a) for a in arms["arm"]])
        ax.set_ylim(min(ypos) - 0.6, max(ypos) + 0.6)
        ax.set_xlim(0, xmax * 1.04)
        ax.set_xlabel(f"{task_name} valid accuracy (%)")
        ax.tick_params(axis="y", length=0)
        ax.text(0.0, 1.01, f"({letter})", transform=ax.transAxes, ha="left", va="bottom",
                fontweight="bold")
        ax.text(bar, max(ypos) + 0.6, f"bar {refs['bar']:.1f}", ha="center", va="bottom", fontsize=7)
    handles = [Patch(color=TAB["blue"], label="arm (selected checkpoint)"),
               Line2D([], [], color="black", lw=1.2, label="CoT bar"),
               Patch(color="0.80", label=f"bar ± σ_screen ({sig:.2f})"),
               Patch(color="0.92", label="bar ± 2σ_screen")]
    handles += [Line2D([], [], **st) for st in refstyle.values()]
    fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=1.5)
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.0), handletextpad=0.4, columnspacing=1.2)
    save(fig, "fig4_screens.pdf")


def main() -> None:
    fig1()
    pareto("figure2a_pareto_points_valid.csv", "Valid", "fig2_pareto_valid.pdf")
    pareto("figure2b_pareto_points_test.csv", "Test", "fig3_pareto_test.pdf")
    fig4()


if __name__ == "__main__":
    main()
