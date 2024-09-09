"""Script 3a: Isolation of the first binary decision

From the experimental data, replace all PN and M cells with pseudo-cells 
sampled from the GMM component corresponding to the CE state.

Loads the main and meta dataframes. Generates and saves a dataframe of the same 
shape as the main dataframe, with PN and M cells replaced
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

from constants import GENES, CONDITIONS, CONDITION_FILES, GENE_TO_IDX
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
parser.add_argument('--seed', type=int, default=12321)
args = parser.parse_args()

seed = args.seed
rng = np.random.default_rng(seed=seed)
print(f"USING SEED: {seed}")


CTYPES_INCLUDE = ['EPI', 'Tr', 'CE', 'AN', 'UT']
CTYPES_EXCLUDE = ['PN', 'M']
REPLACEMENT_CTYPE = 'CE'

OUTDIR = f"out/3a_isolate1"
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

################################################################
##  Subset the data by the cell types to include and exclude  ##
################################################################

ctype_in_idxs = [CTYPE_TO_IDX[ct] for ct in CTYPES_INCLUDE]
ctype_ex_idxs = [CTYPE_TO_IDX[ct] for ct in CTYPES_EXCLUDE]


###########################################################################
##  Replace the PN and M cells at each timepoint with sampled CE cells   ##
###########################################################################

DF_MAIN['cluster_post_replacement'] = DF_MAIN['cluster_assignment_corrected'].copy()
DF_MAIN['synthetic'] = False
print(DF_MAIN.head())
for tp in timepoints:
    df_tp = DF_MAIN[DF_MAIN['timepoint'] == tp]
    df_subset_in = df_tp.loc[
        df_tp['cluster_assignment_corrected'].isin(ctype_in_idxs)
    ]
    df_subset_ex = df_tp.loc[
        df_tp['cluster_assignment_corrected'].isin(ctype_ex_idxs)
    ]
    # For the given timepoint, get the CE components of the GMM.
    means_fpath = f"{GMMDIR}/means_{tp}.npy"
    covs_fpath = f"{GMMDIR}/covs_{tp}.npy"
    weights_fpath = f"{GMMDIR}/weights_{tp}.npy"
    means = np.load(means_fpath)
    covs = np.load(covs_fpath)
    weights = np.load(weights_fpath)
    ce_comp_idxs = np.array(TIMEPOINT_CLUSTERS[tp]) == CTYPE_TO_IDX['CE']
    ce_comp_idxs = ce_comp_idxs.nonzero()[0]
    means_ce = means[ce_comp_idxs]
    covs_ce = covs[ce_comp_idxs]
    weights_ce = weights[ce_comp_idxs]
    weights_ce /= weights_ce.sum()

    nreplace = len(df_subset_ex)
    count_replaced = 0
    for i in range(len(means_ce)):
        if i == len(means_ce) - 1:
            nreplace_comp = nreplace - count_replaced  # avoid rounding issues
        else:
            nreplace_comp = int(np.round(nreplace * weights_ce[i]))
        sample = rng.multivariate_normal(
            mean=means_ce[i],
            cov=covs_ce[i],
            size=nreplace_comp,
        )
        while np.any(sample < 0):
            sample[np.any(sample < 0, axis=1),:] = rng.multivariate_normal(
                mean=means_ce[i],
                cov=covs_ce[i],
                size=np.count_nonzero(np.any(sample < 0, axis=1)),
            )
        idxs_replace = df_subset_ex.index[count_replaced:count_replaced+nreplace_comp]
        DF_MAIN.loc[idxs_replace, 'cluster_post_replacement'] = CTYPE_TO_IDX['CE']
        DF_MAIN.loc[idxs_replace, GENES] = sample
        DF_MAIN.loc[idxs_replace, 'synthetic'] = True
        count_replaced += nreplace_comp

################################
##  Save resulting dataframe  ##
################################
        
DF_MAIN.to_csv(f"{OUTDIR}/df_main_with_replacement.csv")

################
##  Plotting  ##
################

def plot_condition_histograms(
        df_main, imgdir, figsize, 
        assignment_key='cluster_post_replacement',
        saveas_basename="clustering_decision1"
):

    plot_order = np.flip(CTYPE_ORDER)  # flip order so EPI is on bottom
    ts = np.sort(df_main['timepoint'].unique())
    mindiff = np.diff(ts).min()
    bins = [t - mindiff/2 for t in ts] + [ts[-1] + mindiff/2]
    
    for cond_idx, cond_name in CONDITIONS.items():
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
                

with plt.style.context('styles/fig_clustering.mplstyle'):
    plot_condition_histograms(
        DF_MAIN, IMGDIR, FIGSIZE_HISTOGRAMS,
        assignment_key='cluster_post_replacement',
        saveas_basename="clustering_decision1", 
    )


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
