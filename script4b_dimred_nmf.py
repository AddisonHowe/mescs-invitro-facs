"""Script 4b: Dimensionality Reduction (NMF)

Apply Nonnegative Matrix Factorization to the FACS data.
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
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import NMF

from helpers import get_signal_params_from_condition, get_hist2d
from constants import GENES, CONDITIONS, CONDITION_FILES
from constants import TRAINING_CONDITIONS, VALIDATION_CONDITIONS
from script2_clustering import CTYPE_TO_IDX, CTYPE_ORDER, NAMEMAP, COLORMAP


timestart = time.time()

SIGRATE = 1000

TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 10

MAKEPLOTS = {1, 2, 3}

DENSITY_CMAP = 'Greys_r'

########################
##  Argument Parsing  ##
########################

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--decision', type=int, default=1, choices=[1, 2])
parser.add_argument('--fit_on_subset', action="store_true")
args = parser.parse_args()

TRANSITION_IDX = args.decision
FIT_ON_SUBSET = args.fit_on_subset

if TRANSITION_IDX == 1:
    ctypes = ['EPI', 'Tr', 'CE', 'AN']
    training_conditions = TRAINING_CONDITIONS.copy()
    validation_conditions = VALIDATION_CONDITIONS.copy()
elif TRANSITION_IDX == 2:
    ctypes = ['CE', 'PN', 'M']
    training_conditions = TRAINING_CONDITIONS.copy()
    training_conditions.pop(0)  # Remove NO CHIR condition
    validation_conditions = VALIDATION_CONDITIONS.copy()
ctypes_str = "_".join([s.lower() for s in ctypes])

if FIT_ON_SUBSET:
    if TRANSITION_IDX == 1:
        TP0 = 2.0
        TP1 = 3.5
    elif TRANSITION_IDX == 2:
        TP0 = 3.0
        TP1 = 5.0

OUTDIR = f"out/4b_dimred_nmf/dec{TRANSITION_IDX}"
if FIT_ON_SUBSET:
    OUTDIR += "_fitonsubset"
    
IMGDIR = f"{OUTDIR}/images"

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(IMGDIR, exist_ok=True)

#####################################
##  Load Main and Meta dataframes  ##
#####################################
    
df_meta_fpath = "out/1_preprocessing/df_meta.csv"
if TRANSITION_IDX == 1:
    df_main_fpath = "out/3a_isolate1/df_main_with_replacement.csv"
elif TRANSITION_IDX == 2:
    df_main_fpath = "out/3b_isolate2/df_main_with_replacement.csv"

DF_META = pd.read_csv(df_meta_fpath)
DF_MAIN = pd.read_csv(df_main_fpath)

CLUSTER_KEY = "cluster_post_replacement"

timepoints = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

#####################################
##  Subset the data and normalize  ##
#####################################

ctype_idxs = [CTYPE_TO_IDX[ct] for ct in ctypes]

# Get the subset of the main dataframe with only the cell types of interest.
DF_SUBSET = DF_MAIN.loc[DF_MAIN[CLUSTER_KEY].isin(ctype_idxs)].copy()

# Get gene expression data for that subset, and normalize it
x = DF_SUBSET[GENES].to_numpy()

# Check for negative values
nneg = np.count_nonzero(np.any(x < 0, axis=1))
ntot = len(x)
print(f"{nneg}/{ntot}={nneg/ntot:.4f} cells contain negative gene expression")
x = np.where(x <= 0, 0., x)

# x = x - np.mean(x, 0)  # mean center
# if NORMALIZE_VARIANCE:
#     x = x / np.std(x, 0)  # normalize variance

assert len(GENES) == 5, f"Expected 5 genes. Got {len(GENES)}: {GENES}."
nmf = NMF(n_components=len(GENES), max_iter=300)

if FIT_ON_SUBSET:
    # Fit NMF on initial/terminal times, then apply to all cells of interest.
    screen = DF_SUBSET['timepoint'].isin([TP0, TP1]).values
    x_subset = x[screen,:]
    nmf.fit(x_subset)  # fit on subset
    res = nmf.transform(x)  # transform all cells
else:
    # Fit NMF on all of the cells of interest.
    res = nmf.fit_transform(x)

# Copy PC results to the dataframe
for i in range(res.shape[1]):
    DF_SUBSET.loc[:,f"pc{i+1}"] = res[:,i]


############################
##  Save the NMF results  ##
############################
    
pc_sets = [
    [1, 2],
    [1, 2, 3]
]

for pc_set in pc_sets:
    pc_str = "pc" + "".join([str(i) for i in pc_set])

    outdir = f"{OUTDIR}/transition{TRANSITION_IDX}_subset_{ctypes_str}_{pc_str}"
    os.makedirs(outdir, exist_ok=True)

    for cond_set, cond_set_name in zip([training_conditions, 
                                        validation_conditions], 
                                       ['training', 'validation']):
        for i, cond_idx in enumerate(cond_set):
            # Each condition constitutes a simulation
            simdir = f"{outdir}/{cond_set_name}/sim{i}"
            os.makedirs(simdir, exist_ok=True)
            # Subset the data to get the cells in the particular condition
            df = DF_SUBSET[DF_SUBSET['filename'].isin(CONDITION_FILES[cond_idx])]
            
            # Get saved timepoints of the experiment
            ts = np.sort(df['timepoint'].unique())

            # Save data in a list with elements corresponding to each timepoint
            xs = []
            for t in ts:
                data = df[df['timepoint'] == t]
                data = data[[f"pc{i}" for i in pc_set]].to_numpy()
                xs.append(data)
            ncells = [len(x) for x in xs]
            min_cells = min(ncells)

            # Determine signal parameters corresponding to the condition.
            sigparams = np.nan * np.ones([2, 4], dtype=np.float64)
            # Start times at 0.
            t0 = ts.min()
            ts = ts - t0
            sigparams = get_signal_params_from_condition(cond_idx, r=SIGRATE, t0=t0)

            np.save(f"{simdir}/xs.npy", np.array(xs, dtype=object), 
                    allow_pickle=True)
            np.save(f"{simdir}/ts.npy", ts)
            np.save(f"{simdir}/sigparams.npy", sigparams)
        
        nsims = len(cond_set)
        np.savetxt(f"{outdir}/{cond_set_name}/nsims.txt", [nsims], fmt='%d')


############################
##  Examing NMF Loadings  ##
############################

print(f"Genes: {GENES}")
for i in range(len(nmf.components_)):
    print(f"  {nmf.components_[i]}")


################
##  Plotting  ##
################

if 1 in MAKEPLOTS:

    time0 = time.time()
    print("Plotting 1...")

    fig1, axes1 = plt.subplots(7, 4, figsize=(16, 20))
    fig2, axes2 = plt.subplots(7, 4, figsize=(16, 20))
    fig3, axes3 = plt.subplots(7, 4, figsize=(16, 20))

    ctype_fig_axes = []
    for i in range(len(ctypes)):
        ctype_fig_axes.append(plt.subplots(7, 4, figsize=(16, 20)))

    pcs_plot = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
    ]

    # Loop over timepoints, i.e. rows of the figure grid.
    df_subset_scatter = DF_SUBSET.sample(n=10000, random_state=42)
    for i, tp in enumerate(timepoints):
        df_subset_tp = DF_SUBSET[DF_SUBSET['timepoint'] == tp]
        no_data = len(df_subset_tp) == 0

        # Loop over PC pairs, i.e. columns of the figure grid.
        for j, pcidxs in enumerate(pcs_plot):
            ax1 = axes1[i][j]
            ax2 = axes2[i][j]
            ax3 = axes3[i][j]
            axesc = [item[1][i][j] for item in ctype_fig_axes]

            pc1 = f"pc{pcidxs[0] + 1}"
            pc2 = f"pc{pcidxs[1] + 1}"

            #~~~  Figure 1  ~~~#
            ax1.plot(
                df_subset_scatter[pc1], df_subset_scatter[pc2], '.',
                color='grey',
                alpha=0.2,
                markersize=1,
                zorder=-1,
                rasterized=True,
            )
            sns.scatterplot(
                df_subset_tp, x=pc1, y=pc2, 
                hue=CLUSTER_KEY, 
                hue_order=CTYPE_ORDER,
                palette=COLORMAP, 
                s=5,
                ax=ax1,
                legend=(j == len(pcs_plot) - 1),
                rasterized=True,
            )

            #~~~  Figure 2  ~~~#
            edges_x = np.linspace(*ax1.get_xlim(), 50)
            edges_y = np.linspace(*ax1.get_ylim(), 50)
            hist2d = get_hist2d(
                df_subset_tp[[pc1, pc2]].to_numpy(), 
                edges_x, edges_y,
            )
            if not no_data:
                ax2.imshow(
                    hist2d, origin='lower', aspect='auto', 
                    extent=[edges_x[0], edges_x[-1], edges_y[0], edges_y[-1]],
                    cmap=DENSITY_CMAP,
                    norm='log',
                )
                ax2.set_xlim(*ax1.get_xlim())
                ax2.set_ylim(*ax1.get_ylim())


            #~~~  Figure 3  ~~~#
            if not no_data:
                # NOTE: This is the bottleneck
                sns.kdeplot(
                    df_subset_tp, 
                    x=pc1, 
                    y=pc2, 
                    hue=CLUSTER_KEY, 
                    hue_order=CTYPE_ORDER,
                    palette=COLORMAP,
                    common_norm=False,
                    # thresh=0.02,
                    ax=ax3,
                    legend=(j == len(pcs_plot) - 1),
                    warn_singular=False,
                )
                ax3.set_xlim(*ax1.get_xlim())
                ax3.set_ylim(*ax1.get_ylim())

            #~~~  Individual Cell Type Figures  ~~~#
            for cidx, ax in enumerate(axesc):
                hist2d = get_hist2d(
                    df_subset_tp[
                        df_subset_tp[CLUSTER_KEY] == \
                            CTYPE_TO_IDX[ctypes[cidx]]
                    ][[pc1, pc2]].to_numpy(), 
                    edges_x, edges_y,
                )
                if hist2d.sum() > 0:
                    ax.imshow(
                        hist2d, origin='lower', aspect='auto', 
                        extent=[edges_x[0], edges_x[-1], 
                                edges_y[0], edges_y[-1]],
                        cmap=DENSITY_CMAP,
                        norm='log',
                    )

            # Label appropriate axes
            for ax in [ax1, ax2, ax3] + axesc:
                if j == 0:
                    ax.set_ylabel(f"t={tp}\n{pc2.upper()}", 
                                  fontsize=LABEL_FONTSIZE)
                else:
                    ax.set_ylabel(pc2.upper(), fontsize=LABEL_FONTSIZE)
                if i == len(timepoints) - 1:
                    ax.set_xlabel(pc1.upper(), fontsize=LABEL_FONTSIZE)

        # Handle legends after inner loop, for only last column.
        for ax in [ax3, ax1]:
            handles, _ = ax1.get_legend_handles_labels()
            legend_labels = [NAMEMAP[i] for i in CTYPE_ORDER]
            handles = [
                handles[i] for i in range(len(handles)) 
                if CTYPE_ORDER[i] in df_subset_tp[CLUSTER_KEY].unique()
            ]
            legend_labels = [
                legend_labels[i] for i in range(len(legend_labels)) 
                if CTYPE_ORDER[i] in df_subset_tp[CLUSTER_KEY].unique()
            ]
            lgnd = ax.legend(
                handles, 
                legend_labels, 
                loc="upper left", 
                bbox_to_anchor=(1.01, 1), 
                title="Cell Type",
                scatterpoints=1,
                numpoints=1,
            )
            for i in range(len(lgnd.legend_handles)):
                lgnd.legend_handles[i].set_markersize(4)

    for fig in [fig1, fig2, fig3]:
        fig.suptitle(
            f"Decision {TRANSITION_IDX} (All Conditions)", 
            y=1.0, fontsize=TITLE_FONTSIZE,
        )

    for i, (figc, _) in enumerate(ctype_fig_axes):
        figc.suptitle(
            f"Decision {TRANSITION_IDX} (All Conditions, {ctypes[i]})", 
            y=1.0, fontsize=TITLE_FONTSIZE,
        )

    plt.figure(fig1)
    plt.tight_layout()
    plt.savefig(
        f"{IMGDIR}/nmf_decision_{TRANSITION_IDX}_all_conditions_scatter", 
        bbox_inches='tight'
    )
    plt.close()

    plt.figure(fig2)
    plt.tight_layout()
    plt.savefig(
        f"{IMGDIR}/nmf_decision_{TRANSITION_IDX}_all_conditions_density", 
        bbox_inches='tight'
    )
    plt.close()

    plt.figure(fig3)
    plt.tight_layout()
    plt.savefig(
        f"{IMGDIR}/nmf_decision_{TRANSITION_IDX}_all_conditions_kde", 
        bbox_inches='tight'
    )
    plt.close()

    for ctypeidx, ctype in enumerate(ctypes):
        fig, _ = ctype_fig_axes[ctypeidx]
        plt.figure(fig)
        plt.tight_layout()
        plt.savefig(
            f"{IMGDIR}/nmf_decision_{TRANSITION_IDX}_all_conditions_ctype_{ctype}", 
            bbox_inches='tight'
        )
        plt.close()

    time1 = time.time()
    print(f"Plot 1 completed in {time1-time0:.4f} sec")


##################################################
##  Temporal evolution of cells per conditions  ##
##################################################

if 2 in MAKEPLOTS:
    time0 = time.time()
    print("Plotting 2...")

    pcs_plot = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
    ]

    for cond_idx, cond_name in CONDITIONS.items():
        if TRANSITION_IDX == 2 and cond_idx == 0:
            continue

        # Subset the dataframe to capture only the condition of interest.
        df = DF_SUBSET[DF_SUBSET['filename'].isin(CONDITION_FILES[cond_idx])]

        fig1, axes1 = plt.subplots(7, 4, figsize=(16, 20))
        fig2, axes2 = plt.subplots(7, 4, figsize=(16, 20))
        fig3, axes3 = plt.subplots(7, 4, figsize=(16, 20))

        # Loop over timepoints, i.e. rows of the figure grid.
        for i, tp in enumerate(timepoints):
            print(cond_name, tp)
            df_tp = df[df['timepoint'] == tp]
            no_data = len(df_tp) == 0

            # Loop over PC pairs, i.e. columns of the figure grid.
            for j, pcidxs in enumerate(pcs_plot):
                ax1 = axes1[i][j]
                ax2 = axes2[i][j]
                ax3 = axes3[i][j]

                pc1 = f"pc{pcidxs[0] + 1}"
                pc2 = f"pc{pcidxs[1] + 1}"

                #~~~  Figure 1  ~~~#
                ax1.plot(
                    df[pc1], df[pc2], '.',
                    color='grey',
                    alpha=0.2,
                    markersize=1,
                    zorder=-1,
                    rasterized=True,
                )
                sns.scatterplot(
                    df_tp, x=pc1, y=pc2, 
                    hue=CLUSTER_KEY, 
                    hue_order=CTYPE_ORDER,
                    palette=COLORMAP, 
                    s=5,
                    ax=ax1,
                    legend=(j == len(pcs_plot) - 1),
                    rasterized=True,
                )

                if not no_data:
                    # NOTE: This is the bottleneck
                    sns.kdeplot(
                        df_tp, 
                        x=pc1, 
                        y=pc2, 
                        hue=CLUSTER_KEY, 
                        hue_order=CTYPE_ORDER,
                        palette=COLORMAP,
                        common_norm=False,
                        # thresh=0.02,
                        ax=ax1,
                        legend=False,
                        warn_singular=False,
                    )
                
                #~~~  Figure 2  ~~~#
                edges_x = np.linspace(*ax1.get_xlim(), 50)
                edges_y = np.linspace(*ax1.get_ylim(), 50)
                hist2d = get_hist2d(
                    df_tp[[pc1, pc2]].to_numpy(), 
                    edges_x, edges_y,
                )
                if not no_data:
                    ax2.imshow(
                        hist2d, origin='lower', aspect='auto', 
                        extent=[edges_x[0], edges_x[-1], 
                                edges_y[0], edges_y[-1]],
                        cmap=DENSITY_CMAP,
                        norm='log',
                    )

                #~~~  Figure 3  ~~~#
                if not no_data:
                    # NOTE: This is the bottleneck
                    sns.kdeplot(
                        df_tp, 
                        x=pc1, 
                        y=pc2, 
                        hue=CLUSTER_KEY, 
                        hue_order=CTYPE_ORDER,
                        palette=COLORMAP,
                        common_norm=False,
                        # thresh=0.02,
                        ax=ax3,
                        legend=(j == len(pcs_plot) - 1),
                        warn_singular=False,
                    )

                # Label appropriate axes
                for ax in [ax1, ax2, ax3]:
                    if j == 0:
                        ax.set_ylabel(f"t={tp}\n{pc2.upper()}", 
                                      fontsize=LABEL_FONTSIZE)
                    else:
                        ax.set_ylabel(pc2.upper(), fontsize=LABEL_FONTSIZE)
                    if i == len(timepoints) - 1:
                        ax.set_xlabel(pc1.upper(), fontsize=LABEL_FONTSIZE)

            # Handle legends after inner loop, for only last column.
            for ax in [ax3, ax1]:  # need ax1 last
                handles, _ = ax1.get_legend_handles_labels()
                legend_labels = [NAMEMAP[i] for i in CTYPE_ORDER]
                handles = [
                    handles[i] for i in range(len(handles)) 
                    if CTYPE_ORDER[i] in df_tp[CLUSTER_KEY].unique()
                ]
                legend_labels = [
                    legend_labels[i] for i in range(len(legend_labels)) 
                    if CTYPE_ORDER[i] in df_tp[CLUSTER_KEY].unique()
                ]
                lgnd = ax.legend(
                    handles, 
                    legend_labels, 
                    loc="upper left", 
                    bbox_to_anchor=(1.01, 1), 
                    title="Cell Type",
                    scatterpoints=1,
                    numpoints=1,
                )
                for i in range(len(lgnd.legend_handles)):
                    lgnd.legend_handles[i].set_markersize(4)

        for fig in [fig1, fig2, fig3]:
            fig.suptitle(
                f"Decision {TRANSITION_IDX}: {cond_name}", 
                y=1.0, fontsize=TITLE_FONTSIZE,
            )

        plt.figure(fig1)
        plt.tight_layout()
        plt.savefig(
            f"{IMGDIR}/nmf_decision_{TRANSITION_IDX}_{cond_name}_scatter.png", 
            bbox_inches='tight'
        )
        plt.close()
        
        plt.figure(fig2)
        plt.tight_layout()
        plt.savefig(
            f"{IMGDIR}/nmf_decision_{TRANSITION_IDX}_{cond_name}_density.png", 
            bbox_inches='tight'
        )
        plt.close()

        plt.figure(fig3)
        plt.tight_layout()
        plt.savefig(
            f"{IMGDIR}/nmf_decision_{TRANSITION_IDX}_{cond_name}_kde.png", 
            bbox_inches='tight'
        )
        plt.close()

    time1 = time.time()
    print(f"Plot 2 completed in {time1-time0:.4f} sec")


###########################
##  Histogram over time  ##
###########################

if 3 in MAKEPLOTS:
    time0 = time.time()
    print("Plotting 3...")
    
    plot_order = np.flip(CTYPE_ORDER)
    mindiff = np.diff(timepoints).min()
    bins = [t - mindiff/2 for t in timepoints] + [timepoints[-1] + mindiff/2]

    for cond_idx, cond_name in CONDITIONS.items():
        if TRANSITION_IDX == 2 and cond_idx == 0:
            continue
        fig, ax = plt.subplots(1, 1)

        # Subset the dataframe to capture only the condition of interest.
        df = DF_SUBSET[DF_SUBSET['filename'].isin(CONDITION_FILES[cond_idx])]

        hplot = sns.histplot(
            data=df,
            x="timepoint",
            hue=CLUSTER_KEY,
            hue_order=plot_order,
            palette=COLORMAP,
            multiple="stack",
            legend=True,
            bins=bins,
            shrink=0.9,
            ax=ax,
        )

        ax.set_xlabel("Time")

        handles = ax.get_legend().legend_handles
        legend_labels = [NAMEMAP[i] for i in plot_order]
        handles = [
            handles[i] for i in range(len(handles)) 
            if plot_order[i] in df[CLUSTER_KEY].unique()
        ]
        legend_labels = [
            legend_labels[i] for i in range(len(legend_labels)) 
            if plot_order[i] in df[CLUSTER_KEY].unique()
        ]
        lgnd = ax.legend(
            handles, 
            legend_labels, 
            loc="upper left", 
            bbox_to_anchor=(1.01, 1), 
            title="Cell Type",
            scatterpoints=1,
            numpoints=1,
        )

        ax.set_title(f"Decision {TRANSITION_IDX}: {cond_name}", 
                    fontsize=TITLE_FONTSIZE)

        plt.tight_layout()
        plt.savefig(
            f"{IMGDIR}/nmf_decision_{TRANSITION_IDX}_{cond_name}_histograms.png", 
            bbox_inches='tight'
        )
        plt.close()

    time1 = time.time()
    print(f"Plot 3 completed in {time1-time0:.4f} sec")


################################################################
##  (Single Axis) Temporal evolution of cells per conditions  ##
################################################################

FIGSIZE = (4, 4)
LABEL_FONTSIZE = 10

if 4 in MAKEPLOTS:
    time0 = time.time()
    print("Plotting 4...")

    pcs_plot = [
        [0, 1],
    ]

    for cond_idx, cond_name in CONDITIONS.items():
        if TRANSITION_IDX == 2 and cond_idx == 0:
            continue

        # Subset the dataframe to capture only the condition of interest.
        df = DF_SUBSET[DF_SUBSET['filename'].isin(CONDITION_FILES[cond_idx])]

        # Loop over timepoints, i.e. rows of the figure grid.
        for i, tp in enumerate(timepoints):
            print(cond_name, tp)
            df_tp = df[df['timepoint'] == tp]
            no_data = len(df_tp) == 0

            # Loop over PC pairs, i.e. columns of the figure grid.
            for j, pcidxs in enumerate(pcs_plot):
                pc1 = f"pc{pcidxs[0] + 1}"
                pc2 = f"pc{pcidxs[1] + 1}"

                subdir = f"{IMGDIR}/{pc1}{pc2}/{tp}"
                os.makedirs(subdir, exist_ok=True)
                
                fig1, ax1 = plt.subplots(1, 1, figsize=FIGSIZE)
                fig2, ax2 = plt.subplots(1, 1, figsize=FIGSIZE)
                fig3, ax3 = plt.subplots(1, 1, figsize=FIGSIZE)
                

                #~~~  Figure 1  ~~~#
                ax1.plot(
                    df[pc1], df[pc2], '.',
                    color='grey',
                    alpha=0.2,
                    markersize=1,
                    zorder=-1,
                    rasterized=True,
                )
                sns.scatterplot(
                    df_tp, x=pc1, y=pc2, 
                    hue=CLUSTER_KEY, 
                    hue_order=CTYPE_ORDER,
                    palette=COLORMAP, 
                    s=5,
                    ax=ax1,
                    legend=(j == len(pcs_plot) - 1),
                    rasterized=True,
                )
                
                if not no_data:
                    # NOTE: This is the bottleneck
                    sns.kdeplot(
                        df_tp, 
                        x=pc1, 
                        y=pc2, 
                        hue=CLUSTER_KEY, 
                        hue_order=CTYPE_ORDER,
                        palette=COLORMAP,
                        common_norm=False,
                        # thresh=0.02,
                        ax=ax1,
                        legend=False,
                        warn_singular=False,
                    )
                
                #~~~  Figure 2  ~~~#
                edges_x = np.linspace(*ax1.get_xlim(), 50)
                edges_y = np.linspace(*ax1.get_ylim(), 50)
                hist2d = get_hist2d(
                    df_tp[[pc1, pc2]].to_numpy(), 
                    edges_x, edges_y,
                )
                if not no_data:
                    ax2.imshow(
                        hist2d, origin='lower', aspect='auto', 
                        extent=[edges_x[0], edges_x[-1], 
                                edges_y[0], edges_y[-1]],
                        cmap=DENSITY_CMAP,
                        norm='log',
                    )
                    ax2.set_xlim(*ax1.get_xlim())
                    ax2.set_ylim(*ax1.get_ylim())

                #~~~  Figure 3  ~~~#
                if not no_data:
                    # NOTE: This is the bottleneck
                    sns.kdeplot(
                        df_tp, 
                        x=pc1, 
                        y=pc2, 
                        hue=CLUSTER_KEY, 
                        hue_order=CTYPE_ORDER,
                        palette=COLORMAP,
                        common_norm=False,
                        # thresh=0.02,
                        ax=ax3,
                        legend=(j == len(pcs_plot) - 1),
                        warn_singular=False,
                    )
                    ax3.set_xlim(*ax1.get_xlim())
                    ax3.set_ylim(*ax1.get_ylim())

                # Label appropriate axes
                for ax in [ax1, ax2, ax3]:
                    ax.set_xlabel(pc1.upper(), fontsize=LABEL_FONTSIZE)
                    ax.set_ylabel(pc2.upper(), fontsize=LABEL_FONTSIZE)

                # Handle legends after inner loop, for only last column.
                for ax in [ax3, ax1]:  # need ax1 last
                    handles, _ = ax1.get_legend_handles_labels()
                    legend_labels = [NAMEMAP[i] for i in CTYPE_ORDER]
                    handles = [
                        handles[i] for i in range(len(handles)) 
                        if CTYPE_ORDER[i] in df_tp[CLUSTER_KEY].unique()
                    ]
                    legend_labels = [
                        legend_labels[i] for i in range(len(legend_labels)) 
                        if CTYPE_ORDER[i] in df_tp[CLUSTER_KEY].unique()
                    ]
                    lgnd = ax.legend(
                        handles, 
                        legend_labels, 
                        loc="upper right", 
                        # bbox_to_anchor=(1.01, 1), 
                        title="Cell Type",
                        scatterpoints=1,
                        numpoints=1,
                    )
                    for i in range(len(lgnd.legend_handles)):
                        lgnd.legend_handles[i].set_markersize(4)

                plt.figure(fig1)
                plt.tight_layout()
                plt.savefig(
                    f"{subdir}/dec{TRANSITION_IDX}_scatter_{cond_name}.pdf",
                )
                plt.close()
                
                plt.figure(fig2)
                plt.tight_layout()
                plt.savefig(
                    f"{subdir}/dec{TRANSITION_IDX}_density_{cond_name}.pdf",
                )
                plt.close()

                plt.figure(fig3)
                plt.tight_layout()
                plt.savefig(
                    f"{subdir}/dec{TRANSITION_IDX}_kde_{cond_name}.pdf",
                )
                plt.close()

    time1 = time.time()
    print(f"Plot 4 completed in {time1-time0:.4f} sec")


###########################
###########################
    
print(f"Finished in {time.time() - timestart:.4f} sec")
