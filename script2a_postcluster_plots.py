"""Script 2a: Generate plots post clustering.

Generate plots detailing the change in cell types across time and experimental
condition.
"""

import os
import time
import warnings
warnings.filterwarnings(
    "ignore", category=UserWarning, 
    message="Ignoring `palette` because no `hue` variable has been assigned."
)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal

from constants import CONDITIONS, CONDITION_FILES, GENE_TO_IDX
from constants import CONDITION_SIGNALS
from script2_clustering import COLORMAP, TIMEPOINT_CLUSTERS
from script1b_signal_plots import plot_condition, plot_effective_condition
from script1b_signal_plots import CHIR_COLOR, FGF_COLOR, PD_COLOR, WHITE


timestart = time.time()

OUTDIR = f"out/2a_postcluster_plots"
IMGDIR = f"{OUTDIR}/images"

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(IMGDIR, exist_ok=True)

sf = 1/2.54  # cm to inches multiplier
FIGSIZE = (8 * sf, 6.5 * sf)
ALPHA = 0.2  # contour plot alpha 
TITLESIZE = 10
FONTSIZE = 8

CONTOUR_ALPHA = 1.0
CONTOUR_LEVELS = [0.95]  # plot level sets of GMMs at alpha level

ELEV = None
AZIM = None
ROLL = None

TP_MARKER_COLOR = 'y'
TP_MARKER_LINESTYLE = '-'
TP_MARKER_LINEWIDTH = 2

#####################################
##  Load Main and Meta dataframes  ##
#####################################

df_meta_fpath = "out/1_preprocessing/df_meta.csv"
df_main_fpath = "out/1_preprocessing/df_main.csv"
df_clusters_fpath = "out/2_clustering/cluster_assignment_corrected.csv"

GMMDIR = "out/2_clustering/gmms"

DF_META = pd.read_csv(df_meta_fpath)
DF_MAIN = pd.read_csv(df_main_fpath)

ASSIGNMENT_KEY = 'cluster_assignment_corrected'
DF_CLUSTER_ASSIGNMENT = pd.read_csv(df_clusters_fpath, dtype=int)
DF_MAIN[ASSIGNMENT_KEY] = DF_CLUSTER_ASSIGNMENT[ASSIGNMENT_KEY]

timepoints = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

######################
##  Generate Plots  ##
######################

"""
Loop over each condition and timepoint, and make 3d scatter plots of the cells
in a subset of gene expression space. Color cells according to their nominal
type. Also show the projection of the GMM in that space.
"""

GENE_SETS = [
    ['SOX1', 'SOX2', 'CDX2'], 
    ['BRA',  'SOX1', 'TBX6'],
]

for cond_idx, cond_name in CONDITIONS.items():
    print(cond_name)
    # Subset the dataframe to capture only the condition of interest.
    df = DF_MAIN[DF_MAIN['filename'].isin(CONDITION_FILES[cond_idx])]

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
            subdir = f"{IMGDIR}/{tp}"
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
            fname = f"effsignal_{cond_name}"
            plt.savefig(f"{subdir}/{fname}.pdf")
            plt.close()

        df_tp = df[df['timepoint'] == tp]
        assignments = df[df['timepoint'] == tp][ASSIGNMENT_KEY].values
        assert df_tp.shape == (7200, 10), "Expected df_tp.shape (7200, 10)"
        for gene_list in GENE_SETS:
            gexp = df_tp[gene_list].to_numpy()
            assert gexp.shape == (7200, 3), "Expected gexp.shape (7200, 3)"
            fig = plt.figure(figsize=FIGSIZE)
            ax = fig.add_subplot(projection='3d')
            ax.view_init(elev=ELEV, azim=AZIM, roll=ROLL)
            
            colors = np.array([COLORMAP[a] for a in assignments])
            sc = ax.scatter(
                *gexp.T,
                rasterized=True,
                s=1,
                c=colors,
                alpha=ALPHA,
            )
            # ax.set_title(f"{cond_name}, t={tp}", fontsize=TITLESIZE)
            ax.set_xlabel(gene_list[0], fontsize=FONTSIZE)
            ax.set_ylabel(gene_list[1], fontsize=FONTSIZE)
            ax.set_zlabel(gene_list[2], fontsize=FONTSIZE, rotation=90)

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
                mvns = [multivariate_normal(means[i], covs[i]) 
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
                    col = COLORMAP[TIMEPOINT_CLUSTERS[tp][mvnidx]]
                    alpha_levels = np.array(CONTOUR_LEVELS)
                    c = -2*np.log(1 - alpha_levels)
                    levels = np.exp(-c/2) / (2*np.pi*np.sqrt(np.linalg.det(mvn.cov)))
                    ax.contour(
                        *order, 
                        levels=np.sort(levels),
                        zdir=zdir,
                        offset=offset,
                        colors=[col],
                        alpha=CONTOUR_ALPHA
                    )

            gene_string = "_".join(gene_list)
            fname = f"gexp_{gene_string}_{cond_name}"
            subdir = f"{IMGDIR}/{tp}"
            os.makedirs(subdir, exist_ok=True)
            # plt.tight_layout()
            plt.savefig(f"{subdir}/{fname}.pdf")
            plt.close()
