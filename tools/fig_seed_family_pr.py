"""Three-seed r2dtr1c family P/R profile (spec §15.73 read-out action).

P/R plane per split for s0 (board), s1 (new clean retrain), s2 (new retrain,
filled from attribution.json when ready). Shows the P/R swing mechanism of
ledger 15.26. Generates /tmp/rna_figs/seed_family_pr.png on the cluster.
"""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in ['/System/Library/Fonts/Supplemental/Songti.ttc',
          '/System/Library/Fonts/PingFang.tsc',
          '/System/Library/Fonts/Hiragino Sans GB.ttc']:
    try:
        font_manager.fontManager.addfont(f)
    except Exception:
        pass
plt.rcParams['font.family'] = ['Hiragino Sans GB', 'Songti SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
TEAL = '#0E7C86'; RED = '#C4534E'; INK = '#1F2937'; GRID = '#D8DEE6'; FAINT = '#7B8494'; AMBER = '#D9950B'

E = "/root/rna_jepa_data"
SPLITS = ["bprna_ts0", "bprna_new", "ref_pdb_ts1", "ref_pdb_ts2", "ref_pdb_ts3",
          "ref_pdb_ts_hard"]
LABELS = ["TS0", "new", "TS1", "TS2", "TS3", "TS-hard"]

SEEDS = {
    "s0 (板值)": (f"{E}/eval_decision_ref/ow_rinalmo_r2dtr1c_b4_s0_step20000", f"{E}/eval_decision_ref/ow_rinalmo_r2dtr1c_b4_s0_step20000", TEAL, 'o'),
    "s1 新训": (f"{E}/eval_decision/ow_rinalmo_r2dtr1c_b4_s1_step20000", f"{E}/eval_decision_ref/ow_rinalmo_r2dtr1c_b4_s1_step20000", AMBER, 's'),
    "s2 新训": (f"{E}/eval_decision/ow_rinalmo_r2dtr1c_b4_s2_step20000", f"{E}/eval_decision_ref/ow_rinalmo_r2dtr1c_b4_s2_step20000", RED, '^'),
}

def get_pr(base_new, base_old, sp):
    # new layout: nested; old ref layout: flat with _sp suffix
    for f in [f"{base_new}/{sp}/result.json", f"{base_old}_{sp}/result.json"]:
        if os.path.exists(f):
            m = json.load(open(f))["pair_level"]["micro"]
            return m["precision"], m["recall"], m["f1"]
    return None

fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.0), dpi=200, sharex=True, sharey=True)
for ax, sp, lab in zip(axes.flat, SPLITS, LABELS):
    for name, (bn, bo, col, mk) in SEEDS.items():
        got = get_pr(bn, bo, sp)
        if got is None:
            continue
        p, r, f1 = got
        ax.scatter(r, p, s=120, color=col, marker=mk, edgecolor=INK, zorder=3, label=f"{name} F1={f1:.3f}")
        ax.annotate(name.split()[0], (r, p), textcoords="offset points", xytext=(8, -4),
                    fontsize=8.5, color=col)
    ax.plot([0, 1], [0, 1], ls=':', color=FAINT, lw=0.8, alpha=0.6)
    ax.set_title(lab, fontsize=11, color=INK)
    ax.set_xlim(0.4, 1.0); ax.set_ylim(0.4, 1.0)
    ax.grid(color=GRID, lw=0.5, alpha=0.6); ax.set_axisbelow(True)
    for sp_ in ('top', 'right'):
        ax.spines[sp_].set_visible(False)
    ax.legend(fontsize=7, loc='lower left', frameon=True, framealpha=0.9, edgecolor=GRID)
for ax in axes[-1]:
    ax.set_xlabel('召回 R', fontsize=9.5, color=INK)
for ax in axes[:, 0]:
    ax.set_ylabel('精确率 P', fontsize=9.5, color=INK)
fig.suptitle('r2dtr1c 三种子家族 P/R 剖面：种子在 P/R 平面上摆动（§15.26 机制）', fontsize=12, color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.95])
os.makedirs('/root/rna_jepa_data/eval_decision/figs', exist_ok=True)
fig.savefig('/root/rna_jepa_data/eval_decision/figs/seed_family_pr.png', dpi=200, facecolor='white')
print('saved seed_family_pr.png')
