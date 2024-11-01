"""Script 4a: Dimensionality Reduction (PCA)

Apply PCA to the FACS data.
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
plt.style.use("styles/fig6.mplstyle")
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from flowutils.transforms import logicle

from helpers import get_signal_params_from_condition, get_hist2d
from constants import NORMALIZATION_CONSTANT
from constants import GENES, CONDITIONS, CONDITION_FILES, CONDITION_SIGNALS
from script1b_signal_plots import plot_condition, plot_effective_condition
from script1b_signal_plots import CHIR_COLOR, FGF_COLOR, PD_COLOR, WHITE
from script2_clustering import CTYPE_TO_IDX, CTYPE_ORDER, NAMEMAP, COLORMAP


timestart = time.time()

TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 10

MAKEPLOTS = {1, 2, 3, 4, 5, 6}

DENSITY_CMAP = 'Greys_r'

########################
##  Argument Parsing  ##
########################

parser = argparse.ArgumentParser()
parser.add_argument('-k', '--key', type=str, default='facs_v1')
parser.add_argument('-d', '--decision', type=int, default=1, choices=[1, 2])
parser.add_argument('-nv', '--normalize_variance', action="store_true")
parser.add_argument('--log_normalize', action="store_true")
parser.add_argument('--logicle', action="store_true")
parser.add_argument('--fit_on_subset', action="store_true")
parser.add_argument('-p', '--make_plots', type=int, nargs='+', default=None)
parser.add_argument('-t', '--time_shift', type=float, default=None)
parser.add_argument('--style', type=str, default=None)
args = parser.parse_args()

KEY = args.key
TRANSITION_IDX = args.decision
NORMALIZE_VARIANCE = args.normalize_variance
LOG_NORMALIZE = args.log_normalize
LOGICLE_NORMALIZE = args.logicle
FIT_ON_SUBSET = args.fit_on_subset
TIMESHIFT = args.time_shift
STYLE = args.style

if STYLE:
    plt.style.use(f"styles/{STYLE}.mplstyle")

if args.make_plots is not None:
    MAKEPLOTS = set(args.make_plots)

# Rate of transition to use for the signal profile.
KEY_TO_SIGRATE = {
    'facs_v1'  : 10,
    'facs_v2' : 1000,
    'facs_v3' : 1000,
    'facs_v4' : 1000,
    'facs_v5' : 1000,
}

SIGRATE = KEY_TO_SIGRATE[KEY]

# Cell types to retain in each decision.
KEY_TO_CTYPES1 = {
    'facs_v1' : ['EPI', 'Tr', 'CE', 'AN'],
    'facs_v2' : ['EPI', 'Tr', 'CE', 'AN'],
    'facs_v3' : ['EPI', 'Tr', 'CE', 'AN'],
    'facs_v4' : ['EPI', 'Tr', 'CE', 'AN'],
    'facs_v5' : ['EPI', 'Tr', 'CE', 'AN'],
}

KEY_TO_CTYPES2 = {
    'facs_v1' : ['CE', 'PN', 'M'],
    'facs_v2' : ['CE', 'PN', 'M'],
    'facs_v3' : ['CE', 'PN', 'M'],
    'facs_v4' : ['CE', 'PN', 'M'],
    'facs_v5' : ['CE', 'PN', 'M'],
}

# Train/Validate/Test split on the conditions for decisions 1 and 2.
# Indices correspond to the condition indices in constants.py
KEY_TO_CONDITION_SPLIT1 = {
    'facs_v1' : {
        'training'   : [0, 2, 4, 5, 6, 8, 10],
        'validation' : [1, 3, 7, 9],
        'testing'    : [],
    },
    'facs_v2' : {
        'training'   : [0, 2, 4, 5, 6, 8, 10],
        'validation' : [1, 3, 7, 9],
        'testing'    : [],
    },
    'facs_v3' : {
        'training'   : [0, 2, 5, 6, 10],
        'validation' : [1, 3, 7, 9],
        'testing'    : [4, 8],
    },
    'facs_v4' : {
        'training'   : [0, 2, 5, 6, 10],
        'validation' : [1, 3, 7, 9],
        'testing'    : [4, 8],
    },
    'facs_v5' : {
        'training'   : [0, 2, 5, 6, 10],
        'validation' : [1, 3, 7, 9],
        'testing'    : [4, 8],
    },
}

KEY_TO_CONDITION_SPLIT2 = {
    'facs_v1'    : {
        'training'   : [2, 4, 5, 6, 8, 10],
        'validation' : [1, 3, 7, 9],
        'testing'    : [],
    },
    'facs_v2' : {
        'training'   : [2, 4, 5, 6, 8, 10],
        'validation' : [1, 3, 7, 9],
        'testing'    : [],
    },
    'facs_v3' : {
        'training'   : [2, 5, 6, 10],
        'validation' : [1, 3, 7, 9],
        'testing'    : [4, 8],
    },
    'facs_v4' : {
        'training'   : [2, 5, 6, 10],
        'validation' : [1, 3, 7, 9],
        'testing'    : [4, 8],
    },
    'facs_v5' : {
        'training'   : [2, 5, 6, 10],
        'validation' : [1, 3, 7, 9],
        'testing'    : [4, 8],
    },
}

# The timepoints to use as the initial and final snapshot, on which the 
# dimensionality reduction will be performed.
KEY_TO_TPS_1 = {
    'facs_v1' : (2.0, 3.5),
    'facs_v2' : (2.0, 3.5),
    'facs_v3' : (2.0, 3.5),
    'facs_v4' : (2.0, 3.5),
    'facs_v5' : (2.0, 3.5),
}

KEY_TO_TPS_2 = {
    'facs_v1' : (3.0, 5.0),
    'facs_v2' : (3.0, 5.0),
    'facs_v3' : (3.0, 5.0),
    'facs_v4' : (3.0, 5.0),
    'facs_v5' : (3.0, 5.0),
}


if TRANSITION_IDX == 1:
    ctypes = KEY_TO_CTYPES1[KEY]
    training_conditions = KEY_TO_CONDITION_SPLIT1[KEY]['training']
    validation_conditions = KEY_TO_CONDITION_SPLIT1[KEY]['validation']
    testing_conditions = KEY_TO_CONDITION_SPLIT1[KEY]['testing']
elif TRANSITION_IDX == 2:
    ctypes = KEY_TO_CTYPES2[KEY]
    training_conditions = KEY_TO_CONDITION_SPLIT2[KEY]['training']
    validation_conditions = KEY_TO_CONDITION_SPLIT2[KEY]['validation']
    testing_conditions = KEY_TO_CONDITION_SPLIT2[KEY]['testing']

ctypes_str = "_".join([s.lower() for s in ctypes])

if FIT_ON_SUBSET:
    if TRANSITION_IDX == 1:
        TP0, TP1 = KEY_TO_TPS_1[KEY]
    elif TRANSITION_IDX == 2:
        TP0, TP1 = KEY_TO_TPS_2[KEY]

OUTDIR = f"out/4a_dimred_pca/{KEY}/dec{TRANSITION_IDX}"
if NORMALIZE_VARIANCE:
    OUTDIR += "_varnorm"
if FIT_ON_SUBSET:
    OUTDIR += "_fitonsubset"
if LOG_NORMALIZE:
    OUTDIR += "_lognorm"
elif LOGICLE_NORMALIZE:
    OUTDIR += "_logicle"
    
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

if TRANSITION_IDX == 1:
    TIMEPOINTS = {
        'facs_v1' : np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
        'facs_v2' : np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
        'facs_v3' : np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
        'facs_v4' : np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
        'facs_v5' : np.array([2.0, 2.5, 3.0, 3.5]),
    }[KEY]
elif TRANSITION_IDX == 2:
    TIMEPOINTS = {
        'facs_v1' : np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
        'facs_v2' : np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
        'facs_v3' : np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
        'facs_v4' : np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]),
        'facs_v5' : np.array([3.0, 3.5, 4.0, 4.5, 5.0]),
    }[KEY]

#####################################
##  Subset the data and normalize  ##
#####################################

ctype_idxs = [CTYPE_TO_IDX[ct] for ct in ctypes]

# Get the subset of the main dataframe with only the cell types of interest.
DF_SUBSET = DF_MAIN.loc[DF_MAIN[CLUSTER_KEY].isin(ctype_idxs)].copy()

# Get gene expression data for that subset, and normalize it
x = DF_SUBSET[GENES].to_numpy()

if LOG_NORMALIZE:
    x = np.log10(1 + x)
elif LOGICLE_NORMALIZE:
    xfacs = x * NORMALIZATION_CONSTANT  # rescale the data
    x = logicle(xfacs, None)

x = x - np.mean(x, 0)  # mean center
if NORMALIZE_VARIANCE:
    x = x / np.std(x, 0)  # normalize variance

assert len(GENES) == 5, f"Expected 5 genes. Got {len(GENES)}: {GENES}."
pca = PCA(n_components=len(GENES), svd_solver='full')

if FIT_ON_SUBSET:
    # Fit PCA on initial/terminal times, then apply to all cells of interest.
    screen = DF_SUBSET['timepoint'].isin([TP0, TP1]).values
    x_subset = x[screen,:]
    pca.fit(x_subset)  # fit on subset
    res = pca.transform(x)  # transform all cells
else:
    # Fit PCA on all of the cells of interest.
    res = pca.fit_transform(x)

# Copy PC results to the dataframe
for i in range(res.shape[1]):
    DF_SUBSET.loc[:,f"pc{i+1}"] = res[:,i]


############################
##  Save the PCA results  ##
############################
    
pc_sets = [
    [1, 2],
]

for pc_set in pc_sets:
    pc_str = "pc" + "".join([str(i) for i in pc_set])

    outdir = f"{OUTDIR}/transition{TRANSITION_IDX}_subset_{ctypes_str}_{pc_str}"
    os.makedirs(outdir, exist_ok=True)

    np.save(f"{outdir}/components.npy", pca.components_)
    np.save(f"{outdir}/exp_variance.npy", pca.explained_variance_)
    np.save(f"{outdir}/exp_variance_ratio.npy", pca.explained_variance_ratio_)
    np.savetxt(f"{outdir}/genes.txt", GENES, fmt='%s')

    for cond_set, cond_set_name in zip([training_conditions, 
                                        validation_conditions,
                                        testing_conditions], 
                                       ['training', 'validation', 'testing']):    
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
            sigparam_timeshift = t0 if TIMESHIFT is None else TIMESHIFT
            print("Time shift for signal parameters:", sigparam_timeshift)
            sigparams = get_signal_params_from_condition(
                cond_idx, 
                r=SIGRATE, 
                t0=sigparam_timeshift
            )
            print(sigparams)

            np.save(f"{simdir}/xs.npy", np.array(xs, dtype=object), 
                    allow_pickle=True)
            np.save(f"{simdir}/ts.npy", ts)
            np.save(f"{simdir}/sigparams.npy", sigparams)
            np.savetxt(f"{simdir}/condition.txt", [CONDITIONS[cond_idx]], fmt='%s')
        
        nsims = len(cond_set)
        if nsims > 0:
            np.savetxt(f"{outdir}/{cond_set_name}/nsims.txt", [nsims], fmt='%d')


###########################
##  Examing PC Loadings  ##
###########################

print(f"Genes: {GENES}")
for i in range(len(pca.components_)):
    print(f"PC{i+1}: {100*pca.explained_variance_ratio_[i]:.5f}%")
    print(f"  {pca.components_[i]}")


################
##  Plotting  ##
################
    
PCS_PLOT = [
    [0, 1],
    [0, 2],                            
    [0, 3],
    [0, 4],
]

if 1 in MAKEPLOTS:

    time0 = time.time()
    print("Plotting 1...")

    fig1, axes1 = plt.subplots(7, 4, figsize=(16, 20))
    fig2, axes2 = plt.subplots(7, 4, figsize=(16, 20))
    fig3, axes3 = plt.subplots(7, 4, figsize=(16, 20))

    ctype_fig_axes = []
    for i in range(len(ctypes)):
        ctype_fig_axes.append(plt.subplots(7, 4, figsize=(16, 20)))

    

    # Loop over timepoints, i.e. rows of the figure grid.
    df_subset_scatter = DF_SUBSET.sample(n=10000, random_state=42)
    for i, tp in enumerate(TIMEPOINTS):
        df_subset_tp = DF_SUBSET[DF_SUBSET['timepoint'] == tp]
        no_data = len(df_subset_tp) == 0

        # Loop over PC pairs, i.e. columns of the figure grid.
        for j, pcidxs in enumerate(PCS_PLOT):
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
                rasterized=True,
                zorder=-1,
            )
            sns.scatterplot(
                df_subset_tp, x=pc1, y=pc2, 
                hue=CLUSTER_KEY, 
                hue_order=CTYPE_ORDER,
                palette=COLORMAP, 
                s=5,
                ax=ax1,
                legend=(j == len(PCS_PLOT) - 1),
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
                    legend=(j == len(PCS_PLOT) - 1),
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
                if i == len(TIMEPOINTS) - 1:
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
        f"{IMGDIR}/pca_decision_{TRANSITION_IDX}_all_conditions_scatter", 
        bbox_inches='tight'
    )
    plt.close()

    plt.figure(fig2)
    plt.tight_layout()
    plt.savefig(
        f"{IMGDIR}/pca_decision_{TRANSITION_IDX}_all_conditions_density", 
        bbox_inches='tight'
    )
    plt.close()

    plt.figure(fig3)
    plt.tight_layout()
    plt.savefig(
        f"{IMGDIR}/pca_decision_{TRANSITION_IDX}_all_conditions_kde", 
        bbox_inches='tight'
    )
    plt.close()

    for ctypeidx, ctype in enumerate(ctypes):
        fig, _ = ctype_fig_axes[ctypeidx]
        plt.figure(fig)
        plt.tight_layout()
        plt.savefig(
            f"{IMGDIR}/pca_decision_{TRANSITION_IDX}_all_conditions_ctype_{ctype}", 
            bbox_inches='tight'
        )
        plt.close()

    time1 = time.time()
    print(f"Plot 1 completed in {time1-time0:.4f} sec")


##################################################
##  Temporal evolution of cells per conditions  ##
##################################################

PCS_PLOT = [
    [0, 1],
    [0, 2],
    [0, 3],
    [0, 4],
]

if 2 in MAKEPLOTS:
    time0 = time.time()
    print("Plotting 2...")

    for cond_idx, cond_name in CONDITIONS.items():
        if TRANSITION_IDX == 2 and cond_idx == 0:
            continue

        # Subset the dataframe to capture only the condition of interest.
        df = DF_SUBSET[DF_SUBSET['filename'].isin(CONDITION_FILES[cond_idx])]

        fig1, axes1 = plt.subplots(7, 4, figsize=(16, 20))
        fig2, axes2 = plt.subplots(7, 4, figsize=(16, 20))
        fig3, axes3 = plt.subplots(7, 4, figsize=(16, 20))

        # Loop over timepoints, i.e. rows of the figure grid.
        for i, tp in enumerate(TIMEPOINTS):
            print(cond_name, tp)
            df_tp = df[df['timepoint'] == tp]
            no_data = len(df_tp) == 0

            # Loop over PC pairs, i.e. columns of the figure grid.
            for j, pcidxs in enumerate(PCS_PLOT):
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
                    legend=(j == len(PCS_PLOT) - 1),
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
                        legend=(j == len(PCS_PLOT) - 1),
                        warn_singular=False,
                    )

                # Label appropriate axes
                for ax in [ax1, ax2, ax3]:
                    if j == 0:
                        ax.set_ylabel(f"t={tp}\n{pc2.upper()}", 
                                      fontsize=LABEL_FONTSIZE)
                    else:
                        ax.set_ylabel(pc2.upper(), fontsize=LABEL_FONTSIZE)
                    if i == len(TIMEPOINTS) - 1:
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
            f"{IMGDIR}/pca_decision_{TRANSITION_IDX}_{cond_name}_scatter.pdf", 
            bbox_inches='tight'
        )
        plt.close()
        
        plt.figure(fig2)
        plt.tight_layout()
        plt.savefig(
            f"{IMGDIR}/pca_decision_{TRANSITION_IDX}_{cond_name}_density.pdf", 
            bbox_inches='tight'
        )
        plt.close()

        plt.figure(fig3)
        plt.tight_layout()
        plt.savefig(
            f"{IMGDIR}/pca_decision_{TRANSITION_IDX}_{cond_name}_kde.pdf", 
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
    mindiff = np.diff(TIMEPOINTS).min()
    bins = [t - mindiff/2 for t in TIMEPOINTS] + [TIMEPOINTS[-1] + mindiff/2]

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
            f"{IMGDIR}/pca_decision_{TRANSITION_IDX}_{cond_name}_histograms.pdf", 
            bbox_inches='tight'
        )
        plt.close()

    time1 = time.time()
    print(f"Plot 3 completed in {time1-time0:.4f} sec")


################################################################
##  (Single Axis) Temporal evolution of cells per conditions  ##
################################################################
sf = 1/2.54
FIGSIZE = (2.6*sf, 2.6*sf)
PCS_PLOT = [
    [0, 1],
]
BUFFER = 0.025
LEGEND = False
AXIS_LABELS = False
LABEL_FONTSIZE = None
NSAMP = 800

if 4 in MAKEPLOTS:
    time0 = time.time()
    print("Plotting 4...")

    # Determine bounds across conditions and timepoints
    pc_bounds = []
    for j, pcidxs in enumerate(PCS_PLOT):
        pc1 = f"pc{pcidxs[0] + 1}"
        pc2 = f"pc{pcidxs[1] + 1}"
        v1 = DF_SUBSET[pc1] 
        v2 = DF_SUBSET[pc2]
        v1min, v1max = v1.min(), v1.max()
        v2min, v2max = v2.min(), v2.max()
        v1buff = BUFFER * (v1max - v1min)
        v2buff = BUFFER * (v2max - v2min)
        pc_bounds.append([
            (v1min-v1buff, v1max + v1buff), 
            (v2min-v2buff, v2max + v2buff)
        ])
    print("PC Bounds:\n", pc_bounds)

    for cond_idx, cond_name in CONDITIONS.items():
        if TRANSITION_IDX == 2 and cond_idx == 0:
            continue

        # Subset the dataframe to capture only the condition of interest.
        df = DF_SUBSET[DF_SUBSET['filename'].isin(CONDITION_FILES[cond_idx])]

        # Loop over timepoints, i.e. rows of the figure grid.
        for i, tp in enumerate(TIMEPOINTS):
            print(cond_name, tp)
            df_tp = df[df['timepoint'] == tp]
            no_data = len(df_tp) == 0

            # Loop over PC pairs, i.e. columns of the figure grid.
            for j, pcidxs in enumerate(PCS_PLOT):
                pc1 = f"pc{pcidxs[0] + 1}"
                pc2 = f"pc{pcidxs[1] + 1}"

                subdir = f"{IMGDIR}/{pc1}{pc2}/{tp}"
                os.makedirs(subdir, exist_ok=True)
                
                fig1, ax1 = plt.subplots(1, 1, figsize=FIGSIZE, layout="constrained")
                fig2, ax2 = plt.subplots(1, 1, figsize=FIGSIZE, layout="constrained")
                fig3, ax3 = plt.subplots(1, 1, figsize=FIGSIZE, layout="constrained")
                

                #~~~  Figure 1  ~~~#
                ax1.plot(
                    df[pc1], df[pc2], '.',
                    color='grey',
                    alpha=0.2,
                    markersize=1,
                    zorder=-1,
                    rasterized=True,
                )
                if not no_data:
                    sns.scatterplot(
                        df_tp.sample(NSAMP), x=pc1, y=pc2, 
                        hue=CLUSTER_KEY, 
                        hue_order=CTYPE_ORDER,
                        palette=COLORMAP, 
                        s=3,
                        linewidth=0,
                        alpha=0.5,
                        ax=ax1,
                        legend=LEGEND,
                        rasterized=False,
                    )
                ax1.set_xlim(*pc_bounds[j][0])
                ax1.set_ylim(*pc_bounds[j][1])
                ax1.set_xticks([])
                ax1.set_yticks([])
                
                if not no_data:
                    # NOTE: This is the bottleneck
                    # sns.kdeplot(
                    #     df_tp, 
                    #     x=pc1, 
                    #     y=pc2, 
                    #     hue=CLUSTER_KEY, 
                    #     hue_order=CTYPE_ORDER,
                    #     palette=COLORMAP,
                    #     common_norm=False,
                    #     # thresh=0.02,
                    #     ax=ax1,
                    #     legend=False,
                    #     warn_singular=False,
                    # )
                    pass
                
                #~~~  Figure 2  ~~~#
                edges_x = np.linspace(*ax1.get_xlim(), 50)
                edges_y = np.linspace(*ax1.get_ylim(), 50)
                hist2d = get_hist2d(
                    df_tp[[pc1, pc2]].to_numpy(), 
                    edges_x, edges_y,
                )
                if not no_data:
                    ax2.set_xlim(*ax1.get_xlim())
                    ax2.set_ylim(*ax1.get_ylim())
                    ax2.imshow(
                        hist2d, origin='lower', aspect='auto', 
                        extent=[edges_x[0], edges_x[-1], 
                                edges_y[0], edges_y[-1]],
                        cmap=DENSITY_CMAP,
                        norm='log',
                        interpolation='none'
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
                        linewidths=1,
                        # thresh=0.02,
                        ax=ax3,
                        legend=LEGEND,
                        warn_singular=False,
                    )
                    ax3.set_xlim(*ax1.get_xlim())
                    ax3.set_ylim(*ax1.get_ylim())

                # Label appropriate axes
                for ax in [ax1, ax2, ax3]:
                    if AXIS_LABELS:
                        ax.set_xlabel(pc1.upper(), fontsize=LABEL_FONTSIZE)
                        ax.set_ylabel(pc2.upper(), fontsize=LABEL_FONTSIZE)
                    else:
                        ax.set_xlabel("")
                        ax.set_ylabel("")
                    ax.set_xticks([])
                    ax.set_yticks([])

                # Handle legends
                if LEGEND:
                    type_counts = {i: len(df_tp[df_tp[CLUSTER_KEY] == i]) 
                                for i in CTYPE_ORDER}
                    for ax in [ax3, ax1]:  # need ax1 last
                        handles, _ = ax1.get_legend_handles_labels()
                        legend_labels = [NAMEMAP[i] + f" ({type_counts[i]})" 
                                        for i in CTYPE_ORDER]
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
                            # loc="upper right", 
                            # bbox_to_anchor=(1.01, 1), 
                            title="",
                            scatterpoints=1,
                            numpoints=1,
                            fontsize=6,
                        )
                        for i in range(len(lgnd.legend_handles)):
                            lgnd.legend_handles[i].set_markersize(4)

                plt.figure(fig1)
                # plt.tight_layout()
                plt.savefig(
                    f"{subdir}/dec{TRANSITION_IDX}_scatter_{cond_name}.pdf",
                    transparent=True,
                )
                plt.close()
                
                plt.figure(fig2)
                # plt.tight_layout()
                plt.savefig(
                    f"{subdir}/dec{TRANSITION_IDX}_density_{cond_name}.pdf",
                    transparent=True,
                )
                plt.close()

                plt.figure(fig3)
                # plt.tight_layout()
                plt.savefig(
                    f"{subdir}/dec{TRANSITION_IDX}_kde_{cond_name}.pdf",
                    transparent=True,
                )
                plt.close()

    time1 = time.time()
    print(f"Plot 4 completed in {time1-time0:.4f} sec")


##############################
##  Effective Signal Plots  ##
##############################
    
TP_MARKER_COLOR = 'y'
TP_MARKER_LINESTYLE = '-'
TP_MARKER_LINEWIDTH = 2
PCS_PLOT = [
    [0, 1],
]

if 5 in MAKEPLOTS:
    time0 = time.time()
    print("Plotting 5...")

    for cond_idx, cond_name in CONDITIONS.items():
        for i, tp in enumerate(TIMEPOINTS):
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
                fname = f"signal_{cond_name}"
                for j, pcidxs in enumerate(PCS_PLOT):
                    pc1 = f"pc{pcidxs[0] + 1}"
                    pc2 = f"pc{pcidxs[1] + 1}"
                    subdir = f"{IMGDIR}/{pc1}{pc2}/{tp}"
                    os.makedirs(subdir, exist_ok=True)
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
                for j, pcidxs in enumerate(PCS_PLOT):
                    pc1 = f"pc{pcidxs[0] + 1}"
                    pc2 = f"pc{pcidxs[1] + 1}"
                    subdir = f"{IMGDIR}/{pc1}{pc2}/{tp}"
                    plt.savefig(f"{subdir}/{fname}.pdf")
                plt.close()



############################################################
##  (Single Axis) Combined histogram and signal function  ##
############################################################
                
FIGSIZE = (7.25*sf, 4*sf)

if 6 in MAKEPLOTS:
    time0 = time.time()
    print("Plotting 6...")
    
    plot_order = np.flip(CTYPE_ORDER)
    mindiff = np.diff(TIMEPOINTS).min()
    bins = [t - mindiff/2 for t in TIMEPOINTS] + [TIMEPOINTS[-1] + mindiff/2]

    for cond_idx, cond_name in CONDITIONS.items():
        if TRANSITION_IDX == 2 and cond_idx == 0:
            continue

        fig, [ax1, ax2] = plt.subplots(
            2, 1, sharex=True,
            gridspec_kw={'height_ratios': [1,5]},
            figsize=FIGSIZE,

        )
        fig.subplots_adjust(hspace=0)

        # Subset the dataframe to capture only the condition of interest.
        df = DF_SUBSET[DF_SUBSET['filename'].isin(CONDITION_FILES[cond_idx])]

        hplot = sns.histplot(
            data=df,
            x="timepoint",
            hue=CLUSTER_KEY,
            hue_order=plot_order,
            palette=COLORMAP,
            multiple="stack",
            legend=False,
            bins=bins,
            shrink=0.9,
            ax=ax2,
        )

        # ax2.set_xlabel("Time")

        # handles = ax2.get_legend().legend_handles
        # legend_labels = [NAMEMAP[i] for i in plot_order]
        # handles = [
        #     handles[i] for i in range(len(handles)) 
        #     if plot_order[i] in df[CLUSTER_KEY].unique()
        # ]
        # legend_labels = [
        #     legend_labels[i] for i in range(len(legend_labels)) 
        #     if plot_order[i] in df[CLUSTER_KEY].unique()
        # ]
        # lgnd = ax2.legend(
        #     handles, 
        #     legend_labels, 
        #     loc="upper left", 
        #     bbox_to_anchor=(1.01, 1), 
        #     title="Cell Type",
        #     scatterpoints=1,
        #     numpoints=1,
        # )

        cond_sigs = CONDITION_SIGNALS[cond_idx]
        chir_sig, fgf_sig, pd_sig = cond_sigs
        eff_cond_sigs = [
            ["CHIR", chir_sig, 1.0, CHIR_COLOR],
            ["FGF",  fgf_sig,  0.0, FGF_COLOR],
            ["FGF",   pd_sig,  0.0, WHITE],
        ]
        # with plt.style.context('styles/fig_signal_plots.mplstyle'):
        _ = plot_effective_condition(
            cond_name, 
            eff_cond_sigs,
            tp_marker=None,
            tp_marker_color=TP_MARKER_COLOR,
            tp_marker_linestyle=TP_MARKER_LINESTYLE,
            tp_marker_linewidth=TP_MARKER_LINEWIDTH,
            xlims=None,
            ax=ax1,
        )

        # ax1.set_ylabel("Signal")
        ax2.set_ylabel("")
        ax2.set_xlabel("")

        plt.savefig(
            f"{IMGDIR}/pca_dec{TRANSITION_IDX}_{cond_name}_sig_hist.pdf", 
            transparent=True, bbox_inches='tight'
        )
        plt.close()

    time1 = time.time()
    print(f"Plot 6 completed in {time1-time0:.4f} sec")

###########################
###########################
    
print(f"Finished in {time.time() - timestart:.4f} sec")
