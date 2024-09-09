"""Script 2: Clustering.

Recapitulate the clustering algorithm performed by Sáez et al. and save the 
identified cluster of each cell.
"""

import os
import csv
import numpy as np
import pandas as pd
from collections import Counter
from scipy.io import loadmat
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.mixture import GaussianMixture

from constants import CONDITIONS, CONDITION_FILES, REFERENCE_CONDITIONS
from constants import GENES, NORMALIZATION_CONSTANT

sf = 1/2.54  # scale factor from [cm] to inches

DO_NEURAL_CORRECTION = False

LEGEND_SIZE = (2*sf, 1*sf)
FIGSIZE_HISTOGRAMS = (5*sf, 4*sf)

IDX_TO_CTYPE = {
    0: "EPI",
    1: "Tr",
    2: "CE",
    3: "AN", 
    4: "M",
    5: "PN",
    6: "UT",
}

CTYPE_TO_IDX = {
    "EPI" : 0,
    "Tr" : 1,
    "CE" : 2,
    "AN" : 3, 
    "M" : 4,
    "PN" : 5,
    "UT" : 6,
}

NAMEMAP = {
    0: "EPI",
    1: "Tr",
    2: "CE",
    3: "AN", 
    4: "M",
    5: "PN",
    6: "UT",
}

CTYPE_ORDER = [0, 1, 2, 3, 4, 5, 6]

#####################################
##  Define Cluster Initialization  ##
#####################################

# Number of clusters prescribed a priori for each timepoint
NCLUSTERS = {
    2.0: 1,
    2.5: 3,
    3.0: 4,
    3.5: 6,
    4.0: 6,
    4.5: 6,
    5.0: 5,
}

# Mean initialization scheme for the GMM components, for each timepoint.
MEANS_INITS = {
    2.0: 1.e3 / NORMALIZATION_CONSTANT * np.array([
        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000],  # EPI
    ]), 
    2.5: 1.e4 / NORMALIZATION_CONSTANT * np.array([
        [0.6239, 0.1866, 1.4312, 2.4941, 0.7893],  # EPI
        [0.9270, 2.5010, 3.1685, 1.1339, 1.5521],  # CE
        [0.7985, 0.5745, 2.3621, 0.8851, 1.3625],  # Tr
    ]),
    3.0: 1.e4 / NORMALIZATION_CONSTANT * np.array([
        [1.3829, 3.0865, 2.4710, 0.4354, 1.0642],  # M
        [0.8000, 0.2571, 1.6965, 2.7079, 1.3414],  # EPI
        [0.6928, 2.7153, 3.4347, 0.7795, 1.0185],  # CE
        [0.5940, 0.5410, 2.9854, 0.5219, 0.9152],  # CE
    ]),
    3.5: 1.e4 / NORMALIZATION_CONSTANT * np.array([
        [0.5675, 0.7155, 3.7952, 0.5681, 0.9598],  # CE
        [2.7966, 2.1712, 2.6225, 0.4343, 1.0251],  # M
        [0.8033, 2.5784, 3.7396, 0.8723, 1.1950],  # CE
        [0.7548, 0.2768, 1.4120, 4.8587, 1.2933],  # AN
        [0.7207, 0.4182, 2.9935, 1.6278, 1.8508],  # PN
        [0.8327, 0.4954, 1.4238, 0.4059, 1.2653],  # PN
    ]),
    4.0: 1.e4 / NORMALIZATION_CONSTANT * np.array([
        [0.1000, 0.1000, 0.1000, 0.1000, 0.1000],  # M
        [0.5425, 0.4116, 2.1754, 0.4045, 0.9633],  # CE
        [2.5610, 1.1754, 1.9153, 0.3671, 1.0011],  # M
        [0.5773, 0.2510, 0.9504, 1.8299, 2.3301],  # PN
        [0.5675, 0.2077, 0.8476, 4.9253, 1.0792],  # AN
        [0.4689, 0.2506, 1.9770, 2.0563, 1.8045],  # PN
    ]),
    4.5: 1.e4 / NORMALIZATION_CONSTANT * np.array([
        [2.3261, 0.7451, 1.5068, 0.3375, 0.9711],  # M
        [0.7238, 0.3254, 1.3005, 3.5918, 2.1064],  # PN
        [0.5782, 0.8986, 2.9840, 0.8401, 0.9549],  # CE
        [0.5780, 0.2670, 0.8282, 4.4704, 1.3627],  # AN
        [0.5071, 0.2635, 0.7301, 1.7883, 2.0980],  # PN
        [0.2000, 0.2000, 0.2000, 0.2000, 0.2000],  # M
    ]),
    5.0: 1.e4 / NORMALIZATION_CONSTANT * np.array([
        [0.5431, 0.2685, 0.7876, 0.2622, 0.8531],  # M
        [0.4899, 0.4752, 2.2509, 0.9558, 0.9039],  # CE
        [0.4817, 0.2181, 0.5982, 3.0215, 1.5508],  # PN
        [0.7954, 0.3091, 1.1974, 4.1025, 2.2023],  # AN
        [1.8517, 0.3901, 1.0812, 0.2761, 0.8592],  # M
    ]),
}

# Covariance/precision init scheme for the GMM components, for each timepoint.
PRECS_INITS = {}
for t, v in MEANS_INITS.items():
    cov = 1.e3 / NORMALIZATION_CONSTANT**2 * np.eye(v.shape[1])
    precs = np.array([np.linalg.inv(cov) for _ in range(v.shape[0])])
    PRECS_INITS[t] = precs

TIMEPOINT_CLUSTERS = {
    2.0: [0,6],            # 1 cluster   EPI
    2.5: [0,2,1,6],        # 3 clusters  EPI, CE, Tr
    3.0: [4,0,2,2,6],      # 4 clusters  M, EPI, CE, CE
    3.5: [2,4,2,3,5,5,6],  # 6 clusters  CE, M CE, AN, PN, PN
    4.0: [4,2,4,5,3,5,6],  # 6 clusters  M, CE, M, PN, AN, PN
    4.5: [4,5,2,3,5,4,6],  # 6 clusters  M, PN, CE, AN, PN, M
    5.0: [4,2,5,3,4,6],    # 5 clusters  M, CE, PN, AN, M
}

COLORMAP = {
    0: (0.36, 0.40, 0.64),  # EPI
    1: (0.90, 0.62, 0.00),  # Tr
    2: (0.29, 0.15, 0.07),  # CE
    3: (0.33, 0.52, 0.40),  # AN
    4: (0.34, 0.71, 0.91),  # M
    5: (0.65, 0.50, 0.61),  # PN
    6: (1.00, 1.00, 0.00),  # UT
}


def get_colors_at_timepoint(tp):
    return [COLORMAP[i] for i in TIMEPOINT_CLUSTERS[tp]]


def get_names_at_timepoint(tp):
    return [NAMEMAP[i] for i in TIMEPOINT_CLUSTERS[tp]]


def load_matlab_data(timepoints, nclusters, outdir, normalization):
    data = {}
    for tp in timepoints:
        nc = nclusters[tp]
        means_ml = loadmat(f"{outdir}/matlab_data/means_{tp}.mat")['means']
        covs_ml = loadmat(f"{outdir}/matlab_data/covs_{tp}.mat")['covs']
        weights_ml = loadmat(f"{outdir}/matlab_data/weights_{tp}.mat")['weights']
        if np.ndim(covs_ml) == 2:
            covs_ml = covs_ml[:,:,None]
        covs_ml = covs_ml.transpose(2, 0, 1)

        assert means_ml.shape == (nc, 5), f"Bad means shape: {means_ml.shape}"
        assert covs_ml.shape == (nc, 5, 5), f"Bad covs shape: {covs_ml.shape}"
        assert weights_ml.shape == (1, nc), f"Bad weights shape: {weights_ml.shape}"

        means_ml /= normalization
        covs_ml /= normalization**2
                
        prec_chol_ml = np.linalg.cholesky(np.linalg.inv(covs_ml)).transpose((0, 2, 1))
        data[tp] = {
            'means': means_ml,
            'covs': covs_ml,
            'weights': weights_ml,
            'precs_cholesky': prec_chol_ml,
        }
    return data


def fit_gmm(
        x, n_components, 
        means_inits=None, 
        precs_inits=None, 
        weights_init=None,
):
    gmm = GaussianMixture(
        n_components=n_components, 
        means_init=means_inits,
        precisions_init=precs_inits,
        weights_init=weights_init,
        max_iter=300,
        tol=1e-3,
        n_init=1,
    )
    assignments = gmm.fit_predict(x)
    probs = gmm.predict_proba(x)
    return assignments, probs, gmm, gmm.means_, gmm.covariances_


def assign_cluster(
        data, 
        gmm: GaussianMixture,
        pstar=0.65,
        unclustered_idx=-1,
):        
    probs = gmm.predict_proba(data)
    cluster_assignment = unclustered_idx * np.ones(probs.shape[0], dtype=int)
    pargmax = np.argmax(probs, 1)
    pmax = np.max(probs, 1)
    screen = pmax > pstar
    cluster_assignment[screen] = pargmax[screen]
    return cluster_assignment, probs


def correct_neural_labels():
    pass


def cluster_reference_data(
            reference_datasets, pstar, 
            genes=GENES, 
            timepoint_nclusters=NCLUSTERS, 
            means_inits=MEANS_INITS,
            precs_inits=PRECS_INITS,
            unclustered_idx=-1,
    ):
        cluster_predictions = {}
        cluster_assignments = {}
        cluster_means = {}
        cluster_covs = {}
        cluster_gmms = {}

        for timepoint, df in reference_datasets.items():
            gexp = df[genes].to_numpy()
            nclusters = timepoint_nclusters[timepoint]
            preds, probs, gmm, means, covs = fit_gmm(
                gexp, nclusters, 
                means_inits=means_inits[timepoint],
                precs_inits=precs_inits[timepoint],
                weights_init=np.ones(nclusters, dtype=float) / nclusters,
            )
            assert preds.shape == (len(gexp),)
            assert probs.shape == (len(gexp), nclusters)
            cluster_assignment = unclustered_idx * np.ones(preds.shape, dtype=int)
            pargmax = np.argmax(probs, 1)
            pmax = np.max(probs, 1)
            screen = pmax > pstar
            cluster_assignment[screen] = pargmax[screen]
            cluster_assignments[timepoint] = cluster_assignment
            cluster_predictions[timepoint] = preds
            cluster_means[timepoint] = means
            cluster_covs[timepoint] = covs
            cluster_gmms[timepoint] = gmm

        return cluster_assignments, cluster_predictions, cluster_gmms, \
            cluster_means, cluster_covs


def correct_clustering_an_vs_pn(assignments, max_an, probs_an):
    idx_an = CTYPE_TO_IDX['AN']
    idx_pn = CTYPE_TO_IDX['PN']
    print("idx_an:", idx_an)
    print("idx_pn:", idx_pn)
    

    num_an0 = np.count_nonzero(assignments == idx_an)
    num_pn0 = np.count_nonzero(assignments == idx_pn)
    num_neural0 = num_an0 + num_pn0
    num_an1 = min(max_an, num_neural0)
    num_pn1 = max(num_neural0 - max_an, 0)
    print("num_an0:", num_an0)
    print("num_pn0:", num_pn0)
    print("num_neural0:", num_neural0)
    print("num_an1:", num_an1)
    print("num_pn1:", num_pn1)

    neural_screen = (assignments == idx_an) | (assignments == idx_pn)
    print("neural_screen.shape:", neural_screen.shape)

    idxs_neural = neural_screen.nonzero()[0]
    assignments_corrected = assignments.copy()
    assignments_corrected[neural_screen] = -100
    
    neural_cell_prob_an = probs_an[neural_screen]

    partitioned_probs_an = np.argpartition(neural_cell_prob_an, -num_an1)
    idxs_of_top_an_cells = partitioned_probs_an[-num_an1:]
    idxs_of_bottom_an_cells = partitioned_probs_an[:num_pn1]

    print(len(idxs_of_top_an_cells))
    print(len(idxs_of_bottom_an_cells))

    assignments_corrected[idxs_neural[idxs_of_top_an_cells]] = idx_an
    assignments_corrected[idxs_neural[idxs_of_bottom_an_cells]] = idx_pn
    print("assigned CE:", np.count_nonzero(assignments == CTYPE_TO_IDX['CE']))
    print("assigned AN:", np.count_nonzero(assignments == idx_an)) 
    print("assigned PN:", np.count_nonzero(assignments == idx_pn))
    print("assigned CE:", np.count_nonzero(assignments_corrected == CTYPE_TO_IDX['CE']))
    print("assigned AN:", np.count_nonzero(assignments_corrected == idx_an))
    print("assigned PN:", np.count_nonzero(assignments_corrected == idx_pn))
    assert np.count_nonzero(assignments_corrected == idx_an) == num_an1
    assert np.count_nonzero(assignments_corrected == idx_pn) == num_pn1
    assert np.count_nonzero(assignments_corrected == -100) == 0
    return assignments_corrected

#####################
##  Main Function  ##
#####################

def main():

    plt.style.use('styles/fig_clustering.mplstyle')

    OUTDIR = "out/2_clustering"
    IMGDIR = "out/2_clustering/images"

    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(IMGDIR, exist_ok=True)

    #####################################
    ##  Load Main and Meta dataframes  ##
    #####################################
        
    df_meta_fpath = "out/1_preprocessing/df_meta.csv"
    df_main_fpath = "out/1_preprocessing/df_main.csv"

    DF_META = pd.read_csv(df_meta_fpath)
    DF_MAIN = pd.read_csv(df_main_fpath)

    #####################################################################
    ##  Compile the reference datasets into a dictionary by timepoint  ##
    #####################################################################

    REFERENCE_DATASETS = {}
    conditions = [CONDITIONS[cond_idx] for cond_idx in REFERENCE_CONDITIONS]
    # subset the metadata to include just the condition of interest.
    df_cond = DF_META[DF_META['condition'].isin(conditions)]
    timepoints = np.sort(df_cond['timepoint'].unique())
    # subset the datafile to include just cells in those conditions
    df_data = DF_MAIN[DF_MAIN['filename'].isin(df_cond['filename'].unique())]

    for tp in timepoints:
        df_data_timepoint = df_data[df_data['timepoint'] == tp]
        REFERENCE_DATASETS[tp] = df_data_timepoint

    # Perform clustering across a range of threshold values
    pstars = np.linspace(.4, .9, 11)
    pstar_cluster_results = []
    for pstar_idx, pstar in enumerate(pstars):    
        res = cluster_reference_data(
            REFERENCE_DATASETS, pstar, 
            genes=GENES, 
            timepoint_nclusters=NCLUSTERS, 
            means_inits=MEANS_INITS,
            precs_inits=PRECS_INITS,
            unclustered_idx=6,
        )
        pstar_cluster_results.append(res)

    timepoint_cluster_rates = {}
    for tp in timepoints:
        ncells_in_cluster = np.zeros([1+NCLUSTERS[tp], len(pstars)], dtype=int)
        timepoint_cluster_rates[tp] = ncells_in_cluster
        for pstar_idx, pstar in enumerate(pstars):
            res = pstar_cluster_results[pstar_idx]
            cluster_assignments, _, _, _, _ = (r[tp] for r in res)
            counts = {}
            freqs = {}
            for k in range(0, NCLUSTERS[tp]):
                counts[k] = np.count_nonzero(cluster_assignments == k)
                freqs[k] = counts[k] / len(cluster_assignments)
                ncells_in_cluster[k, pstar_idx] = counts[k]
            counts[k+1] = np.count_nonzero(cluster_assignments == 6)
            freqs[k+1] = counts[k+1] / len(cluster_assignments)
            ncells_in_cluster[k+1, pstar_idx] = counts[k+1]

    # Plot results of the sweep over threshold values
    for tp in timepoints:
        fig, ax = plt.subplots(1, 1)
        cluster_idxs = TIMEPOINT_CLUSTERS[tp]
        ncells_in_cluster = timepoint_cluster_rates[tp]
        cluster_colors = get_colors_at_timepoint(tp)
        cluster_names = get_names_at_timepoint(tp)
        for i in range(len(cluster_idxs)):
            label = cluster_names[i]
            color = cluster_colors[i]
            ax.plot(
                pstars, ncells_in_cluster[i], '.-', 
                label=label, 
                color=color,
                linewidth=3,
            )
        ax.set_title(f"Day {tp}")
        ax.set_xlabel("$p$")
        ax.set_ylabel("Cells in cluster")
        ax.legend()
        plt.savefig(f"{IMGDIR}/clusters_by_pstar_day_{tp}.pdf")
        plt.close()

    #~~~  Cluster the reference subset using p=0.65  ~~~#
    res = cluster_reference_data(
        REFERENCE_DATASETS, 0.65, 
        genes=GENES, 
        timepoint_nclusters=NCLUSTERS, 
        means_inits=MEANS_INITS,
        precs_inits=PRECS_INITS,
        unclustered_idx=6,
    )
    assignments, predictions, gmms, means, covs = res

    os.makedirs(f"{OUTDIR}/gmms", exist_ok=True)
    for tp in timepoints:
        np.save(f"{OUTDIR}/gmms/means_{tp}.npy", means[tp])
        np.save(f"{OUTDIR}/gmms/covs_{tp}.npy", covs[tp])
        np.save(f"{OUTDIR}/gmms/weights_{tp}.npy", gmms[tp].weights_)
        print(f"TIMEPOINT {tp}")
        print("COMPONENT MEANS:")
        for i, m in enumerate(means[tp]):
            print(f"Comp {i}: {m*NORMALIZATION_CONSTANT}")

    #~~~  Now assign labels to all cells at each timepoint  ~~~#
    DF_MAIN['cluster_assignment'] = 6
    DF_MAIN['cluster_assignment_corrected'] = np.nan
    cluster_assignments = {}
    cluster_assignments_corrected = {}
    gmm_component_mappings = {}
    
    for tp in timepoints:
        cluster_idxs = TIMEPOINT_CLUSTERS[tp]
        # Define a mapping from the index
        mapping = {
            i: j for i, j in zip(range(len(cluster_idxs)-1), cluster_idxs[0:-1])
        }
        mapping[6] = 6
        gmm_component_mappings[tp] = mapping
        data = DF_MAIN[DF_MAIN['timepoint'] == tp][GENES].to_numpy()
        
        assignments, probs = assign_cluster(
            data, gmms[tp], pstar=0.65, unclustered_idx=6
        )
        # Map the GMM component to the assumed associated cell type
        assignments = np.vectorize(mapping.get)(assignments)
        cluster_assignments[tp] = assignments
        DF_MAIN.loc[
            DF_MAIN['timepoint'] == tp,'cluster_assignment'
        ] = assignments.copy()
        DF_MAIN.loc[
            DF_MAIN['timepoint'] == tp,'cluster_assignment_corrected'
        ] = assignments.copy()

        
    # Store number of AN cells at D4 for each condition
    cond_idx_to_num_an_at_d4 = {}
    for cond_idx in CONDITIONS:
        d4_assignments = DF_MAIN[
            (DF_MAIN['timepoint'] == 4.0) & \
            (DF_MAIN['filename'].isin(CONDITION_FILES[cond_idx]))
        ]['cluster_assignment'].values
        cond_idx_to_num_an_at_d4[cond_idx] = np.count_nonzero(
            d4_assignments == CTYPE_TO_IDX['AN']
        )
    print(cond_idx_to_num_an_at_d4)

    # Perform correction for each condition
    for cond_idx, cond_name in CONDITIONS.items():
        cnt_an_d4 = cond_idx_to_num_an_at_d4[cond_idx]
        df_cond = DF_MAIN[DF_MAIN['filename'].isin(CONDITION_FILES[cond_idx])]

        for tp in [4.5, 5.0]:
            df_cond_tp = df_cond[df_cond['timepoint'] == tp]
            cluster_idxs = TIMEPOINT_CLUSTERS[tp]
            gmm = gmms[tp]
            mapping = gmm_component_mappings[tp]
            # Perform correction for AN and PN cells
            assignments = df_cond_tp['cluster_assignment'].to_numpy()
            probs = gmm.predict_proba(df_cond_tp[GENES].to_numpy())
            assignments_corrected = correct_clustering_an_vs_pn(
                assignments, cnt_an_d4, 
                probs[:,np.equal(cluster_idxs[0:-1], 
                                CTYPE_TO_IDX['AN'])].squeeze(), 
            )
            DF_MAIN.loc[
                (DF_MAIN['filename'].isin(CONDITION_FILES[cond_idx])) & \
                (DF_MAIN['timepoint'] == tp), 'cluster_assignment_corrected'
            ] = assignments_corrected
        
        
        
        # cluster_assignments_corrected[tp] = assignments_corrected
        # DF_MAIN.loc[
        #     DF_MAIN['timepoint'] == tp,'cluster_assignment_corrected'
        # ] = assignments_corrected.copy()


    DF_MAIN['cluster_assignment'].to_csv(
        f"{OUTDIR}/cluster_assignment.csv"
    )

    DF_MAIN['cluster_assignment_corrected'].to_csv(
        f"{OUTDIR}/cluster_assignment_corrected.csv"
    )

    make_plots(DF_MAIN, IMGDIR)


##########################
##  Plotting Functions  ##
##########################

def make_plots(df_main, imgdir):
    """Generate plots"""
    plot_combined_histogram(
        df_main, 
        imgdir=imgdir, 
        figsize=FIGSIZE_HISTOGRAMS,
        remove_legend=True,
    )
    plot_condition_histograms(
        df_main, 
        imgdir=imgdir, 
        figsize=FIGSIZE_HISTOGRAMS,
        use_corrected=False,
    )
    plot_condition_histograms(
        df_main, 
        imgdir=imgdir, 
        figsize=FIGSIZE_HISTOGRAMS,
        use_corrected=True,
    )
    plot_saez_histograms(
        "data/saez_dists",
        imgdir=imgdir,
        figsize=FIGSIZE_HISTOGRAMS,
    )


def plot_combined_histogram(df_main, imgdir, figsize, remove_legend=True):
    
    plot_order = np.flip(CTYPE_ORDER)  # flip order so EPI is on bottom
    ts = np.sort(df_main['timepoint'].unique())
    mindiff = np.diff(ts).min()
    bins = [t - mindiff/2 for t in ts] + [ts[-1] + mindiff/2]
    
    fig1, ax1 = plt.subplots(1, 1, figsize=figsize)
    fig2, ax2 = plt.subplots(1, 1, figsize=figsize)

    for ax in [ax1, ax2]:
        hplot = sns.histplot(
            data=df_main,
            x="timepoint",
            hue="cluster_assignment",
            hue_order=plot_order,
            palette=COLORMAP,
            multiple="stack" if ax is ax1 else 'fill',
            legend=True,
            bins=bins,
            shrink=0.9,
            stat='count' if ax is ax1 else 'proportion',
            ax=ax,
        )

        ax.set_xlabel("Time")
        handles = ax.get_legend().legend_handles
        legend_labels = [NAMEMAP[i] for i in plot_order]
        handles = [
            handles[i] for i in range(len(handles)) 
            if plot_order[i] in df_main['cluster_assignment'].unique()
        ]
        legend_labels = [
            legend_labels[i] for i in range(len(legend_labels)) 
            if plot_order[i] in df_main['cluster_assignment'].unique()
        ]
        ax.legend(
            handles, 
            legend_labels, 
            loc="upper left", 
            bbox_to_anchor=(1.01, 1), 
            title="Cell Type",
            scatterpoints=1,
            numpoints=1,
        )

    ax1.set_title(f"Pooled conditions")
    ax2.set_title(f"Pooled conditions")

    # Save legend separately
    figlegend, axlegend = plt.subplots(1, 1, figsize=LEGEND_SIZE)
    plt.axis('off')
    axlegend.legend(
        handles, 
        legend_labels, 
        title="Cell Type",
        scatterpoints=1,
        numpoints=1,
    )

    if remove_legend:
        for ax in [ax1, ax2]:
            ax.get_legend().remove()

    plt.figure(fig1)
    plt.tight_layout()
    plt.savefig(
        f"{imgdir}/clustering_counts_all_conditions.pdf", 
        bbox_inches='tight'
    )
    plt.close()

    plt.figure(fig2)
    plt.tight_layout()
    plt.savefig(
        f"{imgdir}/clustering_props_all_conditions.pdf", 
        bbox_inches='tight'
    )
    plt.close()

    plt.figure(figlegend)
    plt.tight_layout()
    plt.savefig(
        f"{imgdir}/legend.pdf", 
        bbox_inches='tight'
    )
    plt.close()


def plot_condition_histograms(df_main, imgdir, figsize, use_corrected=False):
    if use_corrected:
        assignment_key = "cluster_assignment_corrected"
        saveas_basename = "corrected_clustering_counts_condition"
    else:
        assignment_key = "cluster_assignment"
        saveas_basename = "clustering_counts_condition"

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
   

def plot_saez_histograms(datdir, imgdir, figsize):
    # Read the data
    cond_data = {
        i : np.zeros([7, 7], dtype=np.float64) for i in range(len(CONDITIONS))
    }
    for cond_idx, cond_name in CONDITIONS.items():
        fpath = f'{datdir}/{cond_name}.csv'
        with open(fpath, 'r') as f:
            csvreader = csv.reader(f, delimiter=',')
            header_types = next(csvreader)  # skip header
            for i, row in enumerate(csvreader):  # process each row
                cond_data[cond_idx][i,:] = row
    
    plot_order = np.flip(CTYPE_ORDER)  # flip order so EPI is on bottom
    ts = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    mindiff = np.diff(ts).min()
    bins = [t - mindiff/2 for t in ts] + [ts[-1] + mindiff/2]
    
    for cond_idx, cond_name in CONDITIONS.items():
        fig, ax = plt.subplots(1, 1, figsize=figsize)        
        
        print(cond_name)
        freqs_data = cond_data[cond_idx]
        data = []
        for i, tp in enumerate(ts):
            for j, ctype in enumerate(header_types):
                data.append({
                    'freq': freqs_data[i, j],
                    'timepoint': tp,
                    'cluster_assignment': CTYPE_TO_IDX[ctype],
                })
        df = pd.DataFrame(data)

        hplot = sns.histplot(
            data=df,
            x="timepoint",
            weights='freq',
            hue="cluster_assignment",
            hue_order=plot_order,  
            palette=COLORMAP,
            multiple="fill",
            legend=True,
            bins=bins,
            shrink=0.9,
            ax=ax,
            linewidth=0.5,
        )

        ax.set_xlabel("")
        ax.set_ylabel("")

        handles = ax.get_legend().legend_handles
        legend_labels = [NAMEMAP[i] for i in plot_order]
        handles = [
            handles[i] for i in range(len(handles)) 
            if plot_order[i] in df['cluster_assignment'].unique()
        ]
        legend_labels = [
            legend_labels[i] for i in range(len(legend_labels)) 
            if plot_order[i] in df['cluster_assignment'].unique()
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
            f"{imgdir}/saez_clustering_counts_condition_{cond_name}.pdf", 
            bbox_inches='tight'
        )
        plt.close()
        


#######################
##  Main Entrypoint  ##
#######################

if __name__ == "__main__":
    main()
