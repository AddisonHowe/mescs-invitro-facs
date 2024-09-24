"""Script 1b: Signal Plots

Plot the signal profile for each experimental condition.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from constants import CONDITIONS, CONDITION_SIGNALS
from constants import CHIR_COLOR, FGF_COLOR, PD_COLOR, LFGF_COLOR, WHITE

sf = 1/2.54  # scale factor from [cm] to inches

OUTDIR = f"out/1b_signal_plots"
FIGSIZE = (5 * sf, 1.5 * sf)

TIMEPOINTS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

TP_MARKER_COLOR = 'g'
TP_MARKER_LINESTYLE = '-'
TP_MARKER_LINEWIDTH = 2


def plot_condition(
        cond_name, 
        signals, 
        is_ref=None, 
        is_training=None, 
        is_validation=None,
        rect_height=1,
        tp_marker=None,
        tp_marker_color=None,
        tp_marker_linestyle=None,
        tp_marker_linewidth=None,
):    
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, layout='constrained')
    ytick_info = {}
    for sig_info in signals:
        sig_name, interval, y, color = sig_info
        t0, t1 = interval
        rect = patches.Rectangle(
            (t0, y), t1 - t0, 
            rect_height, 
            facecolor=color,
        )
        ax.add_patch(rect)
        ytick_info[y + rect_height/2] = sig_name
    yticks = np.sort(list(ytick_info.keys()))
    ytick_labels = [ytick_info[y] for y in yticks]
    ax.set_yticks(yticks, labels=ytick_labels)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 2)
    # ax.set_xlabel("Time")
    ax.set_title(cond_name)
    if tp_marker:
        ax.axvline(
            tp_marker,
            color=tp_marker_color,
            linestyle=tp_marker_linestyle,
            linewidth=tp_marker_linewidth,
        )
    return ax


def plot_effective_condition(
        cond_name, 
        signals, 
        is_ref=None, 
        is_training=None, 
        is_validation=None,
        rect_height=1,
        tp_marker=None,
        tp_marker_color=None,
        tp_marker_linestyle=None,
        tp_marker_linewidth=None,
        xlims=[0,5],
        ylims=[0,2],
        ax=None,
):    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, layout='constrained')
    ytick_info = {}
    rect = patches.Rectangle(
        (0, 0), 5, 
        rect_height, 
        facecolor=LFGF_COLOR,
    )
    ax.add_patch(rect)
    for sig_info in signals:
        sig_name, interval, y, color = sig_info
        t0, t1 = interval
        rect = patches.Rectangle(
            (t0, y), t1 - t0, 
            rect_height, 
            facecolor=color
        )
        ax.add_patch(rect)
        ytick_info[y + rect_height/2] = sig_name
    yticks = np.sort(list(ytick_info.keys()))
    ytick_labels = [ytick_info[y] for y in yticks]
    ax.set_yticks(yticks, labels=ytick_labels)
    if xlims is not None:
        ax.set_xlim(*xlims)
    if ylims is not None:
        ax.set_ylim(*ylims)
    # ax.set_xlabel("Time")
    if tp_marker:
        ax.axvline(
            tp_marker,
            color=tp_marker_color,
            linestyle=tp_marker_linestyle,
            linewidth=tp_marker_linewidth,
        )
    ax.set_title(cond_name)
    return ax


def main():
    plt.style.use('styles/fig_signal_plots.mplstyle')
    
    os.makedirs(OUTDIR, exist_ok=True)
    for cond_idx, cond_name in CONDITIONS.items():        
        cond_sigs = CONDITION_SIGNALS[cond_idx]
        chir_sig, fgf_sig, pd_sig = cond_sigs
        
        # Signal Plots
        cond_sigs = [
            ["CHIR", chir_sig, 1.0, CHIR_COLOR],
            ["FGF/PD",  fgf_sig,  0.0, FGF_COLOR],
            ["FGF/PD",   pd_sig,   0.0, PD_COLOR],
        ]
        _ = plot_condition(
            cond_name, 
            cond_sigs,
        )
        plt.savefig(
            f"{OUTDIR}/signal_cond_{cond_name}.pdf", 
            bbox_inches="tight"
        )
        plt.close()

        subdir = f"{OUTDIR}/signal_cond_tmark_{cond_name}"
        os.makedirs(subdir, exist_ok=True)
        for t in TIMEPOINTS:
            _ = plot_condition(
                cond_name, 
                cond_sigs,
                tp_marker=t,
                tp_marker_color=TP_MARKER_COLOR,
                tp_marker_linestyle=TP_MARKER_LINESTYLE,
                tp_marker_linewidth=TP_MARKER_LINEWIDTH,
            )

            plt.savefig(
                f"{subdir}/signal_mark_tp_{t}.pdf", 
                bbox_inches="tight"
            )
            plt.close()

        # Effective Signal Plots
        eff_cond_sigs = [
            ["CHIR", chir_sig, 1.0, CHIR_COLOR],
            ["FGF",  fgf_sig,  0.0, FGF_COLOR],
            ["FGF",   pd_sig,  0.0, WHITE],
        ]
        _ = plot_effective_condition(
            cond_name, 
            eff_cond_sigs,
        )
        plt.savefig(
            f"{OUTDIR}/eff_signal_cond_{cond_name}.pdf", 
            bbox_inches="tight"
        )
        plt.close()

        subdir = f"{OUTDIR}/effsignal_cond_tmark_{cond_name}"
        os.makedirs(subdir, exist_ok=True)
        for t in TIMEPOINTS:
            _ = plot_condition(
                cond_name, 
                cond_sigs,
                tp_marker=t,
                tp_marker_color=TP_MARKER_COLOR,
                tp_marker_linestyle=TP_MARKER_LINESTYLE,
                tp_marker_linewidth=TP_MARKER_LINEWIDTH,
            )

            plt.savefig(
                f"{subdir}/signal_mark_tp_{t}.pdf", 
                bbox_inches="tight"
            )
            plt.close()



if __name__ == "__main__":
    main()
