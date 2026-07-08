import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.patches as mpatches, numpy as np, json
from pathlib import Path
plt.rcParams.update({'font.family':'serif','font.size':10,'axes.labelsize':11,'axes.titlesize':12,
 'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,'figure.dpi':300,'savefig.dpi':300,
 'savefig.bbox':'tight','savefig.pad_inches':0.05})
BLUE='#4A90D9'; GREEN='#50B86C'; ORANGE='#F5A623'; RED='#D94A4A'; PURPLE='#8B5CF6'; GRAY='#9CA3AF'; DARKGRAY='#4B5563'
LIGHTBLUE='#DBEAFE'; LIGHTGREEN='#D1FAE5'; LIGHTORANGE='#FEF3C7'; LIGHTPURPLE='#EDE9FE'; LIGHTRED='#FEE2E2'
D=json.load(open(Path(__file__).parent/'paper_data.json'))

OUT = Path(__file__).parent
WRITTEN = []
SKIPPED = []


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{name}.{ext}')
    plt.close(fig)
    WRITTEN.append(name)


# ----------------------------------------------------------------------------
# 1. fig_mechanism : hero schematic
# ----------------------------------------------------------------------------
def fig_mechanism():
    from matplotlib.patches import FancyBboxPatch, Circle
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax in (axL, axR):
        ax.set_xlim(0, 10.4); ax.set_ylim(0, 10); ax.axis('off')

    # A concrete all-fail group: 4 rollouts of an edit-a-config agent. Every
    # rollout ends in task failure (outcome 0), but the trajectories differ in
    # HOW they behaved, so a verifiable process potential separates them.
    #   credit  (+1): a fulfilled precondition -- read the file before writing it
    #   penalty (-1): an irreversible action    -- rm -rf before the task is done
    # step kind in {'credit','task','penalty'}
    trajs = [
        ('$g_1$', [('read', 'credit'), ('write', 'task')], +1),
        ('$g_2$', [('write', 'task')], 0),
        ('$g_3$', [('read', 'credit'), ('diff', 'credit'), ('write', 'task')], +2),
        ('$g_4$', [('rm -rf', 'penalty'), ('write', 'task')], -1),
    ]
    rows_y = [7.55, 6.25, 4.95, 3.65]
    step_w, step_h, gap = 1.42, 0.72, 0.34
    start_x = 1.15
    col_O = 7.25   # outcome marker
    col_S = 8.55   # score column (O on left panel, Phi on right)
    col_A = 9.75   # advantage column

    def draw_x(ax, xc, yc, r, color, lw=1.9, alpha=1.0):
        ax.add_patch(Circle((xc, yc), r, facecolor=LIGHTRED, edgecolor=color,
                            linewidth=1.5, alpha=alpha))
        d = r * 0.5
        ax.plot([xc - d, xc + d], [yc + d, yc - d], color=color, lw=lw, alpha=alpha)
        ax.plot([xc - d, xc + d], [yc - d, yc + d], color=color, lw=lw, alpha=alpha)

    def panel(ax, title, title_color):
        box = FancyBboxPatch((0.25, 0.4), 9.9, 9.3, boxstyle='round,pad=0.1,rounding_size=0.3',
                             linewidth=1.4, edgecolor=DARKGRAY, facecolor='white')
        ax.add_patch(box)
        ax.text(5.2, 9.28, title, ha='center', va='center', fontsize=13,
                fontweight='bold', color=title_color)

    def sgn(v):
        return ('$+%g$' % v) if v > 0 else ('$%g$' % v) if v < 0 else '$0$'

    def acol(v):
        return GREEN if v > 0 else RED if v < 0 else GRAY

    def draw_row(ax, label, steps, y, process):
        ax.text(0.62, y, label, ha='center', va='center', fontsize=11.5,
                fontweight='bold', color=DARKGRAY)
        for i, (name, kind) in enumerate(steps):
            x0 = start_x + i * (step_w + gap)
            xc = x0 + step_w / 2
            if not process:
                fc, ec, tc = '#F4F5F7', '#C9CDD4', '#9AA0A8'   # de-emphasised
            elif kind == 'credit':
                fc, ec, tc = LIGHTGREEN, GREEN, DARKGRAY
            elif kind == 'penalty':
                fc, ec, tc = LIGHTRED, RED, DARKGRAY
            else:
                fc, ec, tc = '#EEF2F8', DARKGRAY, DARKGRAY
            ax.add_patch(FancyBboxPatch((x0, y - step_h / 2), step_w, step_h,
                         boxstyle='round,pad=0.02,rounding_size=0.1',
                         linewidth=1.3, edgecolor=ec, facecolor=fc))
            ax.text(xc, y, name, ha='center', va='center', fontsize=9.5,
                    color=tc, family='monospace')
            if process and kind == 'credit':
                ax.text(xc, y + step_h / 2 + 0.26, '$+1$', ha='center', va='center',
                        fontsize=9.5, fontweight='bold', color=GREEN)
            elif process and kind == 'penalty':
                ax.text(xc, y + step_h / 2 + 0.26, '$-1$', ha='center', va='center',
                        fontsize=9.5, fontweight='bold', color=RED)
            if i < len(steps) - 1:
                ax.annotate('', xy=(x0 + step_w + gap, y), xytext=(x0 + step_w, y),
                            arrowprops=dict(arrowstyle='-|>', color='#B6BBC2', lw=1.1))
        # trailing arrow to the outcome
        xend = start_x + len(steps) * (step_w + gap) - gap
        ax.annotate('', xy=(col_O - 0.55, y), xytext=(xend, y),
                    arrowprops=dict(arrowstyle='-|>', color='#B6BBC2', lw=1.1))
        draw_x(ax, col_O, y, 0.34, RED)   # task failed: outcome = 0 everywhere

    # ===================== LEFT: outcome-only GRPO =====================
    panel(axL, 'Outcome-only GRPO', RED)
    axL.text(col_O, 8.55, 'outcome', ha='center', fontsize=9, style='italic', color=DARKGRAY)
    axL.text(col_S, 8.55, '$R_i$', ha='center', fontsize=10, color=DARKGRAY)
    axL.text(col_A, 8.55, '$A_i$', ha='center', fontsize=10, color=DARKGRAY)
    for (label, steps, _phi), y in zip(trajs, rows_y):
        draw_row(axL, label, steps, y, process=False)
        axL.text(col_S, y, '$0$', ha='center', va='center', fontsize=11, color=DARKGRAY)
        axL.text(col_A, y, '$0$', ha='center', va='center', fontsize=11,
                 fontweight='bold', color=GRAY)
    axL.plot([col_S - 0.5, col_A + 0.5], [2.85, 2.85], color='#D5D8DD', lw=1.0)
    axL.text((col_S + col_A) / 2, 2.5, r'$\bar R=0$', ha='center', fontsize=9.5, color=DARKGRAY)
    band = FancyBboxPatch((0.7, 0.95), 9.0, 1.1, boxstyle='round,pad=0.05,rounding_size=0.2',
                          linewidth=1.2, edgecolor='#D5D8DD', facecolor='#F4F5F7')
    axL.add_patch(band)
    axL.text(5.2, 1.72, r'rewards identical $\Rightarrow\ \mathrm{Var}_G(R)=0\ \Rightarrow\ A_i\equiv 0$',
             ha='center', va='center', fontsize=10.5, color=DARKGRAY)
    axL.text(5.2, 1.28, 'DEAD UPDATE — nothing to learn from the group',
             ha='center', va='center', fontsize=11, fontweight='bold', color=RED)

    # ===================== RIGHT: RLVP (verifiable process potential) ====
    panel(axR, r'RLVP  (process potential $\Phi$)', BLUE)
    axR.text(col_O, 8.55, 'outcome', ha='center', fontsize=9, style='italic', color=DARKGRAY)
    axR.text(col_S, 8.55, r'$\Phi_i$', ha='center', fontsize=10, color=DARKGRAY)
    axR.text(col_A, 8.55, '$A_i$', ha='center', fontsize=10, color=DARKGRAY)
    phis = [t[2] for t in trajs]
    mean_phi = sum(phis) / len(phis)   # = 0.5
    for (label, steps, phi), y in zip(trajs, rows_y):
        draw_row(axR, label, steps, y, process=True)
        axR.text(col_S, y, sgn(phi), ha='center', va='center', fontsize=11,
                 fontweight='bold', color=acol(phi))
        a = phi - mean_phi
        axR.text(col_A, y, ('$+%.1f$' % a) if a > 0 else ('$%.1f$' % a),
                 ha='center', va='center', fontsize=11, fontweight='bold', color=acol(a))
    axR.plot([col_S - 0.5, col_A + 0.5], [2.85, 2.85], color='#D5D8DD', lw=1.0)
    axR.text((col_S + col_A) / 2, 2.5, r'$\bar\Phi=0.5$', ha='center', fontsize=9.5, color=DARKGRAY)
    # key: what credit / penalty mean (placed in the free band below the rows)
    axR.add_patch(FancyBboxPatch((0.95, 2.78), 0.55, 0.34, boxstyle='round,pad=0.02,rounding_size=0.08',
                  linewidth=1.1, edgecolor=GREEN, facecolor=LIGHTGREEN))
    axR.text(1.225, 2.95, '$+1$', ha='center', va='center', fontsize=8.5, fontweight='bold', color=GREEN)
    axR.text(1.65, 2.95, r'credit: precondition fulfilled (read before write)',
             ha='left', va='center', fontsize=8.6, color=DARKGRAY)
    axR.add_patch(FancyBboxPatch((0.95, 2.28), 0.55, 0.34, boxstyle='round,pad=0.02,rounding_size=0.08',
                  linewidth=1.1, edgecolor=RED, facecolor=LIGHTRED))
    axR.text(1.225, 2.45, '$-1$', ha='center', va='center', fontsize=8.5, fontweight='bold', color=RED)
    axR.text(1.65, 2.45, r'penalty: irreversible action (rm -rf before the task is done)',
             ha='left', va='center', fontsize=8.6, color=DARKGRAY)
    band2 = FancyBboxPatch((0.7, 0.95), 9.0, 1.1, boxstyle='round,pad=0.05,rounding_size=0.2',
                           linewidth=1.2, edgecolor=GREEN, facecolor=LIGHTGREEN)
    axR.add_patch(band2)
    axR.text(5.2, 1.72, r'$\Phi$ varies across failures $\Rightarrow\ \mathrm{Var}_G(\Phi)>0\ \Rightarrow\ A_i\neq 0$',
             ha='center', va='center', fontsize=10.5, color=DARKGRAY)
    axR.text(5.2, 1.28, r'GRADIENT — reinforce read/diff ($g_3$), suppress rm -rf ($g_4$)',
             ha='center', va='center', fontsize=10.5, fontweight='bold', color=GREEN)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.06)
    save(fig, 'fig_mechanism')


# ----------------------------------------------------------------------------
# 2. fig_efficiency : success vs episodes generated
# ----------------------------------------------------------------------------
def fig_efficiency():
    fig, ax = plt.subplots(figsize=(6, 4))
    specs = [
        ('flag_rlvp', 'RLVP', BLUE, '-', 2.6, 10),
        ('flag_outcome', 'GRPO', RED, '--', 1.8, 5),
        ('flag_dapo', 'DAPO', ORANGE, ':', 1.8, 5),
        ('flag_gigpo', 'GiGPO', GRAY, '-', 1.2, 3),
        ('flag_steptool', 'StepTool', GREEN, '-', 1.2, 3),
    ]
    for key, label, color, ls, lw, z in specs:
        run = D['chain4'].get(key)
        if run is None or run.get('eval_curve') is None or run.get('gen_curve') is None:
            SKIPPED.append(f'chain4/{key}'); continue
        gen = run['gen_curve']
        xs, ys = [], []
        for it, succ in run['eval_curve']:
            idx = it - 1
            if 0 <= idx < len(gen):
                xs.append(gen[idx] / 1000.0)
                ys.append(succ)
        ax.plot(xs, ys, color=color, linestyle=ls, linewidth=lw, marker='o',
                markersize=3.5, label=label, zorder=z)
    ax.axhline(0.5, color=DARKGRAY, linewidth=0.9, linestyle=(0, (3, 3)), alpha=0.7, zorder=1)
    ax.text(ax.get_xlim()[1], 0.5, ' 50%', va='center', ha='left', fontsize=8, color=DARKGRAY)
    ax.set_xlabel('episodes generated (thousands)')
    ax.set_ylabel('held-out success')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(loc='lower right', frameon=True, framealpha=0.9)
    fig.tight_layout()
    save(fig, 'fig_efficiency')


# ----------------------------------------------------------------------------
# 3. fig_seeds : grouped bars with error bars
# ----------------------------------------------------------------------------
def fig_seeds():
    s = D['seeds']
    methods = ['GRPO', 'DAPO', 'RLVP']
    colors = [RED, ORANGE, BLUE]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.6))

    # (a) episodes to 50%
    means = [s[m]['eps50_mean'] for m in methods]
    stds = [s[m]['eps50_std'] for m in methods]
    x = np.arange(len(methods))
    bars = ax1.bar(x, means, yerr=stds, color=colors, edgecolor='white', linewidth=0.8,
                   width=0.62, capsize=5, error_kw={'elinewidth': 1.3, 'ecolor': DARKGRAY})
    ax1.set_xticks(x); ax1.set_xticklabels(methods)
    ax1.set_ylabel('episodes to 50% success  (lower = better)')
    ax1.grid(True, axis='y', alpha=0.25, linewidth=0.6)
    for xi, m, sd in zip(x, means, stds):
        ax1.text(xi, m + sd + 60, f'{int(round(m))}', ha='center', va='bottom', fontsize=9)
    ax1.annotate('zero variance\nacross seeds', xy=(2, means[2]), xytext=(1.55, 1450),
                 fontsize=8.5, color=BLUE, ha='center',
                 arrowprops=dict(arrowstyle='->', color=BLUE, linewidth=1.1))
    ax1.set_ylim(0, max(m + sd for m, sd in zip(means, stds)) * 1.18)

    # (b) dead iterations + DAPO oversample
    dead = [s[m]['dead_mean'] for m in methods]
    over = [s[m]['oversample_mean'] for m in methods]
    bars2 = ax2.bar(x, dead, color=colors, edgecolor='white', linewidth=0.8, width=0.62)
    ax2.axhline(60, color=DARKGRAY, linewidth=0.8, linestyle=':', alpha=0.7)
    ax2.text(2.45, 60, '60 iters', va='center', ha='left', fontsize=8, color=DARKGRAY)
    ax2.set_xticks(x); ax2.set_xticklabels(methods)
    ax2.set_ylabel('dead iterations  (of 60)')
    ax2.grid(True, axis='y', alpha=0.25, linewidth=0.6)
    ax2.set_ylim(0, 66)
    for xi, dv, ov in zip(x, dead, over):
        ax2.text(xi, dv + 1.2, f'{dv:.0f}', ha='center', va='bottom', fontsize=9)
        if ov > 1.05:
            ax2.text(xi, dv + 6, f'{ov:.1f}x\noversample', ha='center', va='bottom',
                     fontsize=7.5, color=ORANGE, fontweight='bold')
    fig.tight_layout()
    save(fig, 'fig_seeds')


# ----------------------------------------------------------------------------
# 4. fig_paired_dead : 2-bar dead-update comparison
# ----------------------------------------------------------------------------
def fig_paired_dead():
    # Dead-update RATE across two task families and three credit schemes.
    # The point is generalisation across tasks -- and that on the compliance
    # task even a process reward goes dead unless it is the admissible one.
    s = D['seeds']; g = D['gated']
    N4 = 60                       # chain4 iterations (matches fig_seeds)
    Ng = g['base']['n']           # 80 iterations on the gated task
    methods = ['GRPO', 'DAPO', 'RLVP']
    colors = {'GRPO': RED, 'DAPO': ORANGE, 'RLVP': BLUE}
    # (label, {method: (dead_fraction, oversample_x)})
    tasks = [
        ('chained task\n(bottleneck is reachable)', {
            'GRPO': (s['GRPO']['dead_mean'] / N4, s['GRPO']['oversample_mean']),
            'DAPO': (s['DAPO']['dead_mean'] / N4, s['DAPO']['oversample_mean']),
            'RLVP': (s['RLVP']['dead_mean'] / N4, s['RLVP']['oversample_mean']),
        }),
        ('silent precondition gate\n(pivotal action never sampled)', {
            'GRPO': (g['outcome']['dead'] / Ng, 1.0),
            'DAPO': (g['dapo']['dead'] / Ng, g['dapo']['oversample']),
            'RLVP': (g['clean_rlvp']['dead'] / Ng, 1.0),   # reward-only: behaviour unreachable
        }),
    ]

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    gw = 0.26                      # bar width
    centers = [0.0, 1.2]           # group centers
    offs = {'GRPO': -gw, 'DAPO': 0.0, 'RLVP': gw}
    for c, (tlabel, md) in zip(centers, tasks):
        for m in methods:
            frac, ov = md[m]
            x = c + offs[m]
            # on the gate, the RLVP bar is reward-only (the behaviour is unreachable): hatch it
            hatched = (tlabel.startswith('silent') and m == 'RLVP')
            ax.bar(x, frac * 100, width=gw, color=colors[m], edgecolor='white',
                   linewidth=0.8, hatch='//' if hatched else None)
            ax.text(x, frac * 100 + 1.4, f'{frac*100:.0f}%', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color=DARKGRAY)
            if ov > 1.05:
                ax.text(x, frac * 100 - 9, f'{ov:.1f}$\\times$\noversample', ha='center',
                        va='top', fontsize=7.3, color='white', fontweight='bold')
    # on the gate, no reward-only method escapes: only a demonstration seed breaks the wall
    xr = centers[1] + offs['RLVP']
    ax.text(xr + 0.17, 48, 'reward alone\nstays dead;\nonly a demo\nbreaks the wall',
            ha='left', va='center', fontsize=8, color=GREEN, fontweight='bold')
    ax.annotate('', xy=(xr + 0.15, 60), xytext=(xr + 0.05, 72),
                arrowprops=dict(arrowstyle='-|>', color=GREEN, lw=1.5))

    ax.set_xticks(centers)
    ax.set_xticklabels([t[0] for t in tasks], fontsize=9.5)
    ax.set_xlim(-0.5, centers[1] + 0.9)
    ax.set_ylabel('dead updates  (\\% of training iterations)')
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(True, axis='y', alpha=0.25, linewidth=0.6)
    ax.axhline(100, color=DARKGRAY, linewidth=0.7, linestyle=':', alpha=0.6)

    # legend
    handles = [mpatches.Patch(color=colors[m], label=m) for m in methods]
    handles.append(mpatches.Patch(facecolor=BLUE, hatch='//', edgecolor='white',
                                  label='RLVP, reward-only'))
    ax.legend(handles=handles, loc='upper center', ncol=4, frameon=False, fontsize=8,
              handletextpad=0.4, columnspacing=1.0, bbox_to_anchor=(0.5, 1.10))
    fig.tight_layout()
    save(fig, 'fig_paired_dead')


# ----------------------------------------------------------------------------
# 5. fig_ablation : horizontal bars of eps50, annotate final success
# ----------------------------------------------------------------------------
def fig_ablation():
    abl = D['ablation']
    items = list(abl.items())
    # order by eps50 ascending (best at top after invert)
    items.sort(key=lambda kv: kv[1]['eps50'])
    names = [k for k, _ in items]
    eps = [v['eps50'] for _, v in items]
    finals = [v['final'] for _, v in items]

    best_eps = min(eps)
    worst_final = min(finals)
    colors = []
    for n, e, f in zip(names, eps, finals):
        if n.startswith('clean'):
            colors.append(GREEN)
        elif f <= 0.7:
            colors.append(RED)
        elif e >= 600:
            colors.append(RED)
        else:
            colors.append(ORANGE)

    fig, ax = plt.subplots(figsize=(7, 3.4))
    y = np.arange(len(names))[::-1]
    ax.barh(y, eps, color=colors, edgecolor='white', linewidth=0.8, height=0.62)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('episodes to 50% success  (lower = better)')
    ax.set_xlim(0, max(eps) * 1.28)
    ax.grid(True, axis='x', alpha=0.25, linewidth=0.6)
    for yi, e, f in zip(y, eps, finals):
        ax.text(e + max(eps) * 0.015, yi, f'{e}  (final {f:.2f})', va='center', ha='left',
                fontsize=8.5, color=DARKGRAY)
    fig.tight_layout()
    save(fig, 'fig_ablation')


# ----------------------------------------------------------------------------
# 6. fig_gated_ceiling : two panels
# ----------------------------------------------------------------------------
def fig_gated_ceiling():
    g = D['gated']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))

    # (a) final gated success bars
    specs = [
        ('outcome', 'GRPO', g['outcome'].get('final')),
        ('dapo', 'DAPO', g['dapo'].get('final')),
        ('clean_rlvp', 'clean RLVP', g['clean_rlvp'].get('final')),
        ('prompted', 'prompted', g['prompted'].get('final')),
        ('rlvp_mix', 'RLVP + mixing', g['rlvp_mix'].get('final')),
    ]
    labels, vals, colors = [], [], []
    for key, lab, val in specs:
        if val is None:
            SKIPPED.append(f'gated/{key}/final'); continue
        labels.append(lab); vals.append(val)
        colors.append(GREEN if val >= 0.5 else RED)
    x = np.arange(len(labels))
    ax1.bar(x, vals, color=colors, edgecolor='white', linewidth=0.8, width=0.66)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=25, ha='right')
    ax1.set_ylabel('final gated success')
    ax1.set_ylim(0, 1.08)
    ax1.grid(True, axis='y', alpha=0.25, linewidth=0.6)
    for xi, v in zip(x, vals):
        ax1.text(xi, v + 0.02, f'{v:.1f}', ha='center', va='bottom', fontsize=9)

    # (b) phase-transition curves
    mix = g['rlvp_mix']['curve']
    clean = g['clean_rlvp']['curve']
    mx, my = zip(*mix)
    cx, cy = zip(*clean)
    ax2.plot(mx, my, color=GREEN, linewidth=2.4, marker='o', markersize=4, label='RLVP + mixing')
    ax2.plot(cx, cy, color=RED, linewidth=1.8, linestyle='--', marker='s', markersize=3.5,
             label='clean RLVP')
    ax2.set_xlabel('iteration')
    ax2.set_ylabel('held-out success')
    ax2.set_ylim(-0.05, 1.08)
    ax2.grid(True, alpha=0.25, linewidth=0.6)
    ax2.legend(loc='center left', frameon=True, framealpha=0.9)
    ax2.annotate('exploration wall broken\nby demonstration', xy=(40, 1.0), xytext=(20, 0.62),
                 fontsize=8.5, color=GREEN, ha='center',
                 arrowprops=dict(arrowstyle='->', color=GREEN, linewidth=1.1))
    fig.tight_layout()
    save(fig, 'fig_gated_ceiling')


# ----------------------------------------------------------------------------
# 7. fig_boundary : conceptual when-RLVP-helps diagram
# ----------------------------------------------------------------------------
def fig_boundary():
    from matplotlib.patches import FancyBboxPatch
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9, 4.6))
    for ax in (axL, axR):
        ax.set_xlim(0, 10); ax.set_ylim(0, 12); ax.axis('off')

    def col(ax, header, hcolor, facecolor, lines, verdict, vcolor):
        box = FancyBboxPatch((0.4, 0.5), 9.2, 11.0, boxstyle='round,pad=0.1,rounding_size=0.3',
                             linewidth=1.5, edgecolor=hcolor, facecolor=facecolor)
        ax.add_patch(box)
        ax.text(5, 10.85, header, ha='center', va='center', fontsize=11.5,
                fontweight='bold', color=hcolor)
        y = 9.5
        for ln in lines:
            ax.text(5, y, ln, ha='center', va='center', fontsize=9.5, color=DARKGRAY)
            y -= 0.78
        vb = FancyBboxPatch((1.0, 0.95), 8.0, 1.25, boxstyle='round,pad=0.05,rounding_size=0.2',
                            linewidth=1.3, edgecolor=vcolor, facecolor='white')
        ax.add_patch(vb)
        ax.text(5, 1.57, verdict, ha='center', va='center', fontsize=11,
                fontweight='bold', color=vcolor)

    c4 = D['chain4']
    gated = D['gated']
    rlvp_eps = c4['flag_rlvp']['eps50']
    grpo_eps = c4['flag_outcome']['eps50']

    col(axL, 'Rules outcome-INSTRUMENTAL', GREEN, LIGHTGREEN,
        ['Following the rules is the path',
         'to task success.',
         '',
         f'chains:  RLVP {rlvp_eps} vs GRPO {grpo_eps}',
         f'    episodes to 50%  ({grpo_eps/rlvp_eps:.1f}x faster)',
         '',
         f'gated:  RLVP+mix {gated["rlvp_mix"]["final"]:.0f}  vs',
         f'    outcome-only {gated["outcome"]["final"]:.0f}  (wall broken)'],
        'RLVP wins', GREEN)

    t2 = D['tau2']
    col(axR, 'Rules ORTHOGONAL to outcome', RED, LIGHTRED,
        ['Compliance does not advance',
         'the task objective.',
         '',
         f"$\\tau$2:  outcome reward {t2['outcome']['reward']:.2f}",
         f"    RLVP reward {t2['rlvp']['reward']:.2f}",
         '',
         'process reward pulls policy into a',
         'compliant-but-useless attractor'],
        'RLVP collapses', RED)

    # center divider arrow with question
    fig.text(0.5, 0.965, 'Is rule-following instrumental to the outcome?',
             ha='center', va='top', fontsize=10.5, fontweight='bold', color=DARKGRAY)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, 'fig_boundary')


# ----------------------------------------------------------------------------
# 8. fig_tau2_collapse : train reward + violations
# ----------------------------------------------------------------------------
def fig_tau2_collapse():
    t2 = D['tau2']
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4),
                                   gridspec_kw={'width_ratios': [1.55, 1]})

    # --- Left: four training trajectories ---
    ot = t2['outcome_train']; rt = t2['rlvp_train']
    at = t2.get('aligned_train', []); se = t2.get('semantic_train', [])
    it = np.arange(1, len(ot) + 1)
    axL.plot(it, ot, color=BLUE, linewidth=2.2, marker='o', markersize=3,
             label='outcome-only GRPO')
    axL.plot(it[:len(rt)], rt, color=RED, linewidth=2.2, marker='s', markersize=3,
             label='RLVP: generic (orthogonal) rules')
    if at:
        axL.plot(np.arange(1, len(at) + 1), at, color=ORANGE, linewidth=2.0,
                 marker='^', markersize=3, label='RLVP: aligned procedural rules')
    if se:
        axL.plot(np.arange(1, len(se) + 1), se, color=GREEN, linewidth=2.2,
                 marker='D', markersize=3, label='RLVP: aligned semantic rules')
    axL.set_xlabel('iteration'); axL.set_ylabel('training task reward')
    axL.set_ylim(-0.03, 0.78); axL.grid(True, alpha=0.25, linewidth=0.6)
    axL.annotate('compliance-only attractor\n(orthogonal rules: reward $\\rightarrow$ 0)',
                 xy=(len(rt), rt[-1]), xytext=(13, 0.55), fontsize=8, color=RED, ha='center',
                 arrowprops=dict(arrowstyle='->', color=RED, linewidth=1.0))
    axL.legend(loc='upper left', frameon=True, framealpha=0.92, fontsize=7.8)

    # --- Right: the coverage gradient (training reward; mean bar + peak tick) ---
    tiers = t2['tiers_train']; peaks = t2.get('tiers_peak', {})
    labels = list(tiers.keys()); vals = [tiers[k] for k in labels]
    cols = [BLUE, RED, ORANGE, GREEN]
    y = np.arange(len(labels))[::-1]
    axR.barh(y, vals, color=cols, alpha=0.9, height=0.62, edgecolor=DARKGRAY)
    for yi, k, v in zip(y, labels, vals):
        axR.text(v + 0.015, yi, f'{v:.2f}', va='center', fontsize=9.5, fontweight='bold')
        if k in peaks:  # peak as a thin marker
            axR.plot([peaks[k]], [yi], marker='|', markersize=14, color=DARKGRAY, mew=1.6)
    axR.set_yticks(y); axR.set_yticklabels(labels, fontsize=8)
    axR.set_xlim(0, 0.82); axR.set_xlabel('training reward (bar: mean; tick: peak)')
    axR.grid(True, axis='x', alpha=0.25, linewidth=0.6)
    axR.text(0.80, y[1], 'HARM', va='center', ha='right', fontsize=7.5, color=RED, fontweight='bold')
    axR.text(0.80, y[2], 'no harm', va='center', ha='right', fontsize=7, color='#B45309')
    axR.text(0.80, y[3], 'no harm', va='center', ha='right', fontsize=7, color='#15803D')
    fig.tight_layout()
    save(fig, 'fig_tau2_collapse')


# ----------------------------------------------------------------------------
# 9. fig_selfcritique_2x2 : when does a rule vs a learned self-critic work?
#    (schematic; headline numbers from the self-critique ablation / SELFCRITIC.md)
# ----------------------------------------------------------------------------
def fig_selfcritique_2x2():
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')

    # numbers from the ablation (paper_data.json['selfcritique'])
    sc = D.get('selfcritique', {})
    _off = sc.get('tau2_offline', {})
    scf1 = _off.get('selfcritic_F1', {}).get('mean')
    semf1 = _off.get('semantic_F1', {}).get('mean')
    llm_late = sc.get('tau2_train', {}).get('llmcritic', {}).get('late')
    nseed = sc.get('multiseed_csops', {}).get('rule', {}).get('viol', {}).get('n', 3)
    c_l1 = (f'detects offline (F1 {scf1:.2f} vs {semf1:.2f})'
            if scf1 is not None and semf1 is not None else 'detects offline (F1 .63 vs .23)')
    c_l2 = (f'but collapses as a reward ({llm_late:.1f})'
            if llm_late is not None else 'but collapses as a reward (0.0)')
    b_l2 = f'rule decisive, critic inert ({nseed} seeds)'

    def cell(x, y, w, h, tag, edge, face, title, lines, verdict, vcol):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle='round,pad=0.08,rounding_size=0.25',
                     linewidth=1.6, edgecolor=edge, facecolor=face))
        ax.text(x + 0.30, y + h - 0.28, tag, ha='left', va='top',
                fontsize=13, fontweight='bold', color=edge)
        ax.text(x + w / 2, y + h - 0.95, title, ha='center', va='center',
                fontsize=10.2, fontweight='bold', color=DARKGRAY)
        yy = y + h - 1.65
        for ln in lines:
            ax.text(x + w / 2, yy, ln, ha='center', va='top', fontsize=8.6,
                    color=DARKGRAY)
            yy -= 0.55
        ax.add_patch(FancyBboxPatch((x + w / 2 - 1.6, y + 0.30), 3.2, 0.72,
                     boxstyle='round,pad=0.04,rounding_size=0.16',
                     linewidth=1.2, edgecolor=vcol, facecolor='white'))
        ax.text(x + w / 2, y + 0.66, verdict, ha='center', va='center',
                fontsize=10, fontweight='bold', color=vcol)

    xL, xR, w, h = 1.45, 5.55, 3.9, 3.7
    yT, yB = 4.6, 0.6
    cell(xL, yT, w, h, 'A', BLUE, LIGHTBLUE, 'surface-ordering norms',
         ['both detect it — rule is the', 'cleaner reward (critic = noisy proxy)'],
         'use the rule', BLUE)
    cell(xR, yT, w, h, 'B', GREEN, LIGHTGREEN, 'stateful-bookkeeping norms',
         ['critic blind even when told;', b_l2],
         'RULES WIN', GREEN)
    cell(xL, yB, w, h, 'C', ORANGE, LIGHTORANGE, r'task intent ($\tau$2)',
         [c_l1, c_l2],
         'DIAGNOSE ONLY', '#B45309')
    cell(xR, yB, w, h, 'D', GRAY, '#F3F4F6', 'hidden / masked failures',
         ['neither channel has signal;', 'needs outcome or a demo'],
         'open', DARKGRAY)

    ax.text(5.5, 9.6, 'On-policy self-critic detects the violation?',
            ha='center', va='center', fontsize=11, fontweight='bold', color='black')
    ax.text(xL + w / 2, 8.5, 'DETECTS from trajectory', ha='center', va='bottom',
            fontsize=9.5, fontweight='bold', color=DARKGRAY)
    ax.text(xR + w / 2, 8.5, 'BLIND', ha='center', va='bottom',
            fontsize=9.5, fontweight='bold', color=DARKGRAY)
    ax.text(0.95, 4.65, 'Verifiable rule cheaply specifiable?', ha='center',
            va='center', rotation=90, fontsize=11, fontweight='bold', color='black')
    ax.text(1.30, yT + h / 2, 'YES', ha='center', va='center', rotation=90,
            fontsize=9.5, fontweight='bold', color=DARKGRAY)
    ax.text(1.30, yB + h / 2, 'NO', ha='center', va='center', rotation=90,
            fontsize=9.5, fontweight='bold', color=DARKGRAY)
    save(fig, 'fig_selfcritique_2x2')


def main():
    figs = [fig_mechanism, fig_efficiency, fig_seeds, fig_paired_dead,
            fig_ablation, fig_gated_ceiling, fig_boundary, fig_tau2_collapse,
            fig_selfcritique_2x2]
    for fn in figs:
        fn()
    print(f'Wrote {len(WRITTEN)} figures:')
    for name in WRITTEN:
        print(f'  {name}.pdf / .png')
    if SKIPPED:
        print('Skipped (None) series:')
        for s in SKIPPED:
            print(f'  {s}')
    else:
        print('No series skipped (all data present).')


if __name__ == '__main__':
    main()
