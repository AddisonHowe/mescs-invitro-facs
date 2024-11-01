"""Script 3b: Isolation of the second binary decision

From the experimental data, exclude the data at times D2 and D2.5. Then remove 
all remaining EPI and AN cells.

Loads the main and meta dataframes. Generates and saves a dataframe of the same 
form as the main dataframe.
"""

import argparse
import os
import time
import warnings
warnings.filterwarnings(
    "ignore", category=UserWarning, 
    message="Ignoring `palette` because no `hue` variable has been assigned."
)

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy

from constants import CONDITIONS, CONDITION_FILES, GENE_TO_IDX
from constants import CONDITION_SIGNALS
from script2_clustering import NAMEMAP, COLORMAP
from script2_clustering import CTYPE_ORDER, CTYPE_TO_IDX, TIMEPOINT_CLUSTERS
from script1b_signal_plots import plot_condition, plot_effective_condition
from script1b_signal_plots import CHIR_COLOR, FGF_COLOR, PD_COLOR, WHITE


timestart = time.time()

TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 10

sf = 1/2.54  # scale factor from [cm] to inches

FIGSIZE_HISTOGRAMS = (5*sf, 4*sf)

TP_MARKER_COLOR = 'y'
TP_MARKER_LINESTYLE = '-'
TP_MARKER_LINEWIDTH = 2

########################
##  Argument Parsing  ##
########################

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=32123)
args = parser.parse_args()

seed = args.seed
rng = np.random.default_rng(seed=seed)
print(f"USING SEED: {seed}")


TIMEPOINTS_INCLUDE = [3.0, 3.5, 4.0, 4.5, 5.0]
CTYPES_INCLUDE = ['CE', 'PN', 'M', 'UT']
CTYPES_EXCLUDE = ['EPI', 'AN', 'Tr']

OUTDIR = f"out/3b_isolate2"
IMGDIR = f"{OUTDIR}/images"
GMMDIR = "out/2_clustering/gmms"

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(IMGDIR, exist_ok=True)

#####################################
##  Load Main and Meta dataframes  ##
#####################################
    
df_meta_fpath = "out/1_preprocessing/df_meta.csv"
df_main_fpath = "out/1_preprocessing/df_main.csv"
df_clusters_fpath = "out/2_clustering/cluster_assignment_corrected.csv"

DF_META = pd.read_csv(df_meta_fpath)
DF_MAIN = pd.read_csv(df_main_fpath)

DF_CLUSTER_ASSIGNMENT = pd.read_csv(df_clusters_fpath, dtype=int)
DF_MAIN['cluster_assignment_corrected'] = DF_CLUSTER_ASSIGNMENT["cluster_assignment_corrected"]

timepoints = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

##########################
##  Plotting Functions  ##
##########################

def plot_condition_histograms(
        df_main, imgdir, figsize, 
        assignment_key='cluster_post_replacement',
        saveas_basename="clustering_decision1",
        skip_no_chir=True,
):

    plot_order = np.flip(CTYPE_ORDER)  # flip order so EPI is on bottom
    ts = np.sort(df_main['timepoint'].unique())
    mindiff = np.diff(ts).min()
    bins = [t - mindiff/2 for t in ts] + [ts[-1] + mindiff/2]
    
    for cond_idx, cond_name in CONDITIONS.items():
        if cond_idx == 0 and skip_no_chir:
            # Skip NO CHIR condition
            continue
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Subset the dataframe to capture only the condition of interest.
        df = df_main[df_main['filename'].isin(CONDITION_FILES[cond_idx])]
        
        print(cond_name)
        for tp in ts:
            assignments = df[df['timepoint'] == tp][assignment_key].values
            counts = Counter(assignments)
            total = len(assignments)
            freqs = {k: counts[k]/total for k in counts.keys()}
            print(
                f"  {tp}\t", 
                [f"{k}: {freqs[k]:.4f}" for k in np.sort(list(freqs.keys()))]
            )

        hplot = sns.histplot(
            data=df,
            x="timepoint",
            hue=assignment_key,
            hue_order=plot_order,  
            palette=COLORMAP,
            multiple="fill",
            legend=True,
            bins=bins,
            shrink=0.9,
            common_norm=False,
            ax=ax,
            linewidth=0.5,
        )

        ax.set_xlabel("")
        ax.set_ylabel("")

        handles = ax.get_legend().legend_handles
        legend_labels = [NAMEMAP[i] for i in plot_order]
        handles = [
            handles[i] for i in range(len(handles)) 
            if plot_order[i] in df[assignment_key].unique()
        ]
        legend_labels = [
            legend_labels[i] for i in range(len(legend_labels)) 
            if plot_order[i] in df[assignment_key].unique()
        ]
        ax.get_legend().remove()

        # Add proportions as text annotations
        barheights = np.array(
            [[b.get_height() for b in bars] for bars in hplot.containers]
        )
        totals = np.sum(barheights, axis=0)
        for i, bars in enumerate(hplot.containers):
            heights = [b.get_height() for b in bars]
            tot = totals[i]
            labels = [f'{h/tot:.2f}' if h/tot > 0.05 else '' for h in heights]
            ax.bar_label(bars, labels=labels, label_type='center', fontsize=4)

        ax.set_title(cond_name)
        ax.set_xlim(1.775, ax.get_xlim()[1])

        plt.tight_layout()
        plt.savefig(
            f"{imgdir}/{saveas_basename}_{cond_name}.pdf", 
            bbox_inches='tight'
        )
        plt.close()


def plot_gene_expression_3d(
        df_main, 
        gene_sets,
        conditions,
        assignment_key,
        figsize=None,
        alpha=0.6,
        contour_alpha=1.0,
        contour_levels=[],
        elev=None,
        azim=None,
        roll=None,
        titlefontsize=None,
        labelfontsize=None,
        imgdir=None,

):
    for cond_idx, cond_name in conditions.items():
        if cond_name == 'NO CHIR':
            continue
        print(cond_name)
        # Subset the dataframe to capture only the condition of interest.
        df = df_main[df_main['filename'].isin(CONDITION_FILES[cond_idx])]

        for tp in timepoints:
            # Plot the signal
            cond_sigs = CONDITION_SIGNALS[cond_idx]
            chir_sig, fgf_sig, pd_sig = cond_sigs
            # Signal Plots
            cond_sigs = [
                ["CHIR", chir_sig, 1.0, CHIR_COLOR],
                ["FGF/PD",  fgf_sig,  0.0, FGF_COLOR],
                ["FGF/PD",   pd_sig,   0.0, PD_COLOR],
            ]
            with plt.style.context('styles/fig_signal_plots.mplstyle'):
                _ = plot_condition(
                    cond_name, 
                    cond_sigs,
                    tp_marker=tp,
                    tp_marker_color=TP_MARKER_COLOR,
                    tp_marker_linestyle=TP_MARKER_LINESTYLE,
                    tp_marker_linewidth=TP_MARKER_LINEWIDTH,
                )
            if imgdir:
                subdir = f"{imgdir}/{tp}"
                os.makedirs(subdir, exist_ok=True)
                fname = f"signal_{cond_name}"
                plt.savefig(f"{subdir}/{fname}.pdf")
                plt.close()

            eff_cond_sigs = [
                ["CHIR", chir_sig, 1.0, CHIR_COLOR],
                ["FGF",  fgf_sig,  0.0, FGF_COLOR],
                ["FGF",   pd_sig,  0.0, WHITE],
            ]
            with plt.style.context('styles/fig_signal_plots.mplstyle'):
                _ = plot_effective_condition(
                    cond_name, 
                    eff_cond_sigs,
                    tp_marker=tp,
                    tp_marker_color=TP_MARKER_COLOR,
                    tp_marker_linestyle=TP_MARKER_LINESTYLE,
                    tp_marker_linewidth=TP_MARKER_LINEWIDTH,
                )
            if imgdir:
                fname = f"effsignal_{cond_name}"
                plt.savefig(f"{subdir}/{fname}.pdf")
                plt.close()


            df_tp = df[df['timepoint'] == tp]
            assignments = df[df['timepoint'] == tp][assignment_key].values
            # assert df_tp.shape == (7200, 10), "Expected df_tp.shape (7200, 10)"
            for gene_list in gene_sets:
                gexp = df_tp[gene_list].to_numpy()
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(projection='3d')
                ax.view_init(elev=elev, azim=azim, roll=roll)
                
                colors = np.array([COLORMAP[a] for a in assignments])
                sc = ax.scatter(
                    *gexp.T,
                    rasterized=True,
                    s=1,
                    c=colors,
                    alpha=alpha,
                )
                # ax.set_title(f"{cond_name}, t={tp}", fontsize=titlefontsize)
                ax.set_xlabel(gene_list[0], fontsize=labelfontsize)
                ax.set_ylabel(gene_list[1], fontsize=labelfontsize)
                ax.set_zlabel(gene_list[2], fontsize=labelfontsize, rotation=90)

                glims = [[df[gene].min()-1, df[gene].max()+1] for gene in gene_list]

                ax.set_xlim(*glims[0])
                ax.set_ylim(*glims[1])
                ax.set_zlim(*glims[2])

                # Get the GMM components, subsetting to the components of interest
                gene_idxs = np.array([GENE_TO_IDX[g] for g in gene_list], dtype=int)
                comp_means = np.load(f"{GMMDIR}/means_{tp}.npy")
                comp_covs = np.load(f"{GMMDIR}/covs_{tp}.npy")
                ncomps = comp_means.shape[0]
                assert comp_covs.shape[0] == ncomps, \
                    "comp_means and comp_covs should have same first dim"
                
                
                # Plot xy marginal
                proj_idxs_marg_idx = [
                    ([0, 1], 2),
                    ([0, 2], 1),
                    ([1, 2], 0),
                ]

                for proj_idxs, marg_idx in proj_idxs_marg_idx:
                    idx1, idx2 = proj_idxs
                    gidx_pair = [gene_idxs[idx1], gene_idxs[idx2]]
                    means = comp_means[:,gidx_pair]
                    covs = comp_covs[:,gidx_pair][:,:,gidx_pair]
                    assert means.shape == (ncomps, 2)
                    assert covs.shape == (ncomps, 2, 2)
                    mvns = [scipy.stats.multivariate_normal(means[i], covs[i]) 
                            for i in range(ncomps)]

                    v1arr, v2arr = [np.linspace(*glims[i], 100) for i in proj_idxs]
                    v1, v2 = np.meshgrid(v1arr, v2arr)
                    v3 = np.zeros(v1.shape)

                    if marg_idx == 2:
                        zdir = 'z'
                        offset = glims[2][0]
                        order = [v1, v2, v3]  # x, y | z
                    elif marg_idx == 1:
                        zdir = 'y'
                        offset = glims[1][1]
                        order = [v1, v3, v2]  # x, z | y
                    elif marg_idx == 0:
                        zdir = 'x'
                        offset = glims[0][0]
                        order = [v3, v1, v2]  # y, z | x
                    else:
                        raise RuntimeError(f"Bad marg_idx: {marg_idx}")
                    
                    pos = np.dstack([v1, v2])

                    for mvnidx, mvn in enumerate(mvns):
                        v3[:] = mvn.pdf(pos)
                        clust_idx = TIMEPOINT_CLUSTERS[tp][mvnidx]
                        # plot contour curves only for the retained cell types
                        if clust_idx in [CTYPE_TO_IDX[i] for i in CTYPES_EXCLUDE]:
                            continue 
                        col = COLORMAP[clust_idx]
                        alpha_levels = np.array(contour_levels)
                        c = -2*np.log(1 - alpha_levels)
                        levels = np.exp(-c/2) / (2*np.pi*np.sqrt(np.linalg.det(mvn.cov)))
                        ax.contour(
                            *order, 
                            levels=np.sort(levels),
                            zdir=zdir,
                            offset=offset,
                            colors=[col],
                            alpha=contour_alpha,
                        )

                gene_string = "_".join(gene_list)
                fname = f"gexp_{gene_string}_{cond_name}"
                if imgdir:
                    subdir = f"{imgdir}/{tp}"
                    os.makedirs(subdir, exist_ok=True)
                    plt.savefig(f"{subdir}/{fname}.pdf")
                plt.close()


#################################
##  Remove D2 and D2.5 cells   ##
#################################

DF_MAIN = DF_MAIN[DF_MAIN['timepoint'].isin(TIMEPOINTS_INCLUDE)]


##########################################
##  Remove remaining EPI and AN cells   ##
##########################################

ctype_in_idxs = [CTYPE_TO_IDX[ct] for ct in CTYPES_INCLUDE]
ctype_ex_idxs = [CTYPE_TO_IDX[ct] for ct in CTYPES_EXCLUDE]

DF_MAIN = DF_MAIN[DF_MAIN['cluster_assignment_corrected'].isin(ctype_in_idxs)]

DF_MAIN['cluster_post_replacement'] = DF_MAIN['cluster_assignment_corrected'].copy()

#####################################################
##  Plot condition histograms, including NO CHIR   ##
#####################################################

with plt.style.context('styles/fig_clustering.mplstyle'):
    plot_condition_histograms(
        DF_MAIN, IMGDIR, FIGSIZE_HISTOGRAMS,
        assignment_key='cluster_post_replacement',
        saveas_basename="clustering_decision2", 
        skip_no_chir=False,
    )

#################################
##  Remove NO CHIR condition   ##
#################################

assert CONDITIONS[0] == "NO CHIR", "Expected CONDITIONS[0] to be 'NO CHIR'!"
DF_MAIN = DF_MAIN[
    ~DF_MAIN['filename'].isin(CONDITION_FILES[0])
]

################################
##  Save resulting dataframe  ##
################################
        
DF_MAIN.to_csv(f"{OUTDIR}/df_main_with_replacement.csv")

################
##  Plotting  ##
################

GENE_SETS = [
    ['SOX1', 'SOX2', 'CDX2'], 
    ['BRA',  'SOX1', 'TBX6'],
]
sf = 1/2.54  # cm to inches multiplier
FIGSIZE = (8 * sf, 6.5 * sf)
TITLE_FONTSIZE = 10
LABEL_FONTSIZE = 8
ALPHA = 0.6 
CONTOUR_LEVELS = [0.95]
CONTOUR_ALPHA = 1.0

plot_gene_expression_3d(
    DF_MAIN,
    GENE_SETS,
    CONDITIONS,
    assignment_key='cluster_post_replacement',
    figsize=FIGSIZE,
    contour_levels=CONTOUR_LEVELS,
    alpha=ALPHA,
    contour_alpha=CONTOUR_ALPHA,
    titlefontsize=TITLE_FONTSIZE,
    labelfontsize=LABEL_FONTSIZE,
    imgdir=f"{IMGDIR}/gexp3d",
)

###########################
###########################
    
print(f"Finished in {time.time() - timestart:.4f} sec")
