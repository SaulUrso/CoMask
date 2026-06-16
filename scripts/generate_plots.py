"""Generate the comparison table + the extended (4-dataset) RQ2/RQ3 figures as files.

Robust, re-runnable alternative to executing plots.ipynb. Writes everything to out/paper_plots/.
Data + plotting logic are pulled from plots.ipynb cells (single source of truth) and executed
here; figures are saved to disk. As a bonus the figures are also embedded back into the notebook.

Run:  uv run --env-file .env -- python scripts/generate_plots.py
Caches W&B history to runs_history-prune.pkl / runs_history_and_results_rq2.pkl / wandb_dir01_cache.pkl
so re-runs are fast.
"""
import io
import os
import json
import base64
import pickle
import contextlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
NB = "plots.ipynb"
OUTDIR = os.path.join("out", "paper_plots")
os.makedirs(OUTDIR, exist_ok=True)

nb = json.load(open(NB))
byid = {c.get("id"): c for c in nb["cells"]}
SRC = lambda cid: "".join(byid[cid]["source"])

ns = {}
exec("import matplotlib\nmatplotlib.use('Agg')", ns)
exec(SRC("bceb0695"), ns)               # maps + imports (plt, np, wandb)
ns["api"] = wandb.Api(timeout=180)
exec(SRC("dir01helpers"), ns)
print("[setup] done", flush=True)


def _loadcache(path, key):
    try:
        if os.path.getsize(path) > 0:
            return pickle.load(open(path, "rb"))[key]
    except (OSError, EOFError, KeyError, pickle.UnpicklingError):
        pass
    return {}


def exec_quiet(cid):
    with contextlib.redirect_stdout(io.StringIO()):
        exec(SRC(cid), ns)


def exec_plot_and_save(cid, fname, embed=True):
    """Exec a plot cell, save the produced figure(s) to OUTDIR, optionally embed in the notebook."""
    plt.close("all")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(SRC(cid), ns)
    nums = plt.get_fignums()
    saved, outs = [], []
    if buf.getvalue().strip():
        outs.append({"output_type": "stream", "name": "stdout",
                     "text": buf.getvalue().splitlines(keepends=True)})
    for i, num in enumerate(nums):
        fig = plt.figure(num)
        suffix = "" if len(nums) == 1 else f"_{i}"
        path = os.path.join(OUTDIR, f"{fname}{suffix}.png")
        fig.savefig(path, format="png", bbox_inches="tight", dpi=130)
        saved.append(path)
        b = io.BytesIO()
        fig.savefig(b, format="png", bbox_inches="tight", dpi=110)
        outs.append({"output_type": "display_data",
                     "data": {"image/png": base64.b64encode(b.getvalue()).decode(),
                              "text/plain": ["<Figure>"]}, "metadata": {}})
    plt.close("all")
    if embed:
        byid[cid]["outputs"] = outs
        byid[cid]["execution_count"] = None
    print(f"[plot] {cid} -> {saved}", flush=True)
    return saved


# ============================= TABLE =============================
print("[table] building ...", flush=True)
ns["display"] = lambda *_a, **_k: None
exec_quiet("comptablecode")
df = ns["comparison_df"]
df.to_csv(os.path.join(OUTDIR, "comparison_table.csv"))
with open(os.path.join(OUTDIR, "comparison_table.tex"), "w") as f:
    f.write(df.to_latex(caption="Personalized / Global test accuracy (\\%) at 500 rounds.",
                        label="tab:comparison"))
# table as a PNG
fig, ax = plt.subplots(figsize=(13, 2.6))
ax.axis("off")
tbl = ax.table(cellText=df.values,
               rowLabels=df.index,
               colLabels=[f"{a}\n{b}" for a, b in df.columns],
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 1.6)
ax.set_title("Personalized / Global test accuracy (%) @ 500 rounds", pad=14)
fig.savefig(os.path.join(OUTDIR, "comparison_table.png"), bbox_inches="tight", dpi=140)
plt.close("all")
print("[table] saved csv/tex/png\n" + df.to_string(), flush=True)

# ============================= RQ3 (prune) =============================
print("[rq3] fetching ICDCS (cell 12) ...", flush=True)
ns["runs_history_cache"] = _loadcache("runs_history-prune.pkl", "runs_history_cache")
print(f"[rq3] preloaded {len(ns['runs_history_cache'])} cache keys", flush=True)
exec_quiet("36755a7c")                  # heavy ICDCS RQ3 fetch (fast if cached)
pickle.dump({"runs_history_cache": ns["runs_history_cache"]},
            open("runs_history-prune.pkl", "wb"))
print("[rq3] ICDCS done:", list(ns["results_by_dataset"].keys()), flush=True)
exec_quiet("dir01rq3fetch")
print("[rq3] dir0.1 runs:", len(ns["results_by_dataset"]["CIFAR10_dir0.1"]), flush=True)
exec_plot_and_save("966eb8c2", "rq3_accuracy_cost_vs_prune")
exec_plot_and_save("92ee5c3a", "rq3_ratio_vs_prune")

# ============================= RQ2 (consensus) =============================
print("[rq2] fetching ICDCS (cell 24) ...", flush=True)
ns["runs_history_cache"] = _loadcache("runs_history_and_results_rq2.pkl", "runs_history_cache")
ns["results_by_dataset_rq2"] = {"FEMNIST": {}, "CIFAR10": {}, "HAR": {}}
print(f"[rq2] preloaded {len(ns['runs_history_cache'])} cache keys", flush=True)
exec_quiet("907b864a")                  # heavy ICDCS RQ2 fetch
pickle.dump({"runs_history_cache": ns["runs_history_cache"],
             "results_by_dataset_rq2": ns["results_by_dataset_rq2"]},
            open("runs_history_and_results_rq2.pkl", "wb"))
print("[rq2] ICDCS done:", list(ns["results_by_dataset_rq2"].keys()), flush=True)
exec_quiet("dir01rq2fetch")
print("[rq2] dir0.1 runs:", len(ns["results_by_dataset_rq2"]["CIFAR10_dir0.1"]), flush=True)
exec_plot_and_save("7e77a977", "rq2_accuracy_cost_vs_consensus")
exec_plot_and_save("0e19c2d3", "rq2_pruning_round_vs_consensus")
exec_plot_and_save("d8e93172", "rq2_ratio_vs_consensus")

# ============================= embed into notebook (best effort) =============================
try:
    # also embed the table output
    outs = [{"output_type": "execute_result", "execution_count": None,
             "data": {"text/plain": repr(df).splitlines(keepends=True),
                      "text/html": df._repr_html_().splitlines(keepends=True)}, "metadata": {}}]
    byid["comptablecode"]["outputs"] = outs
    byid["comptablecode"]["execution_count"] = None
    json.dump(nb, open(NB, "w"), indent=1, ensure_ascii=False)
    json.load(open(NB))
    print("[notebook] figures embedded into", NB, flush=True)
except Exception as e:  # noqa
    print("[notebook] embed skipped:", repr(e), flush=True)

print("\nDONE. All outputs in", OUTDIR, flush=True)
print("\n".join(sorted(os.listdir(OUTDIR))), flush=True)
