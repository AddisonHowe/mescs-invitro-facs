"""Script 1: Preprocessing.

Construct and save meta and main dataframes.
"""

import os
import warnings
warnings.filterwarnings(
    "ignore", category=UserWarning, 
    message="Ignoring `palette` because no `hue` variable has been assigned."
)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

NORMALIZE_VARIANCE = False

OUTDIR = "out/1_preprocessing"
IMGDIR = "out/1_preprocessing/images"

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(IMGDIR, exist_ok=True)

from constants import *

from helpers import load_facs_csv_file, get_meta_dataframe, load_all_files
from helpers import get_reference_dataset

subdir = f"{IMGDIR}/gexp_hists"
os.makedirs(subdir, exist_ok=True)
for fname in ALL_FILES:
    fig, axes = plt.subplots(1, len(GENES), figsize=(12, 3))
    df = load_facs_csv_file(
        f"{DATDIR}/{fname}", normalization=NORMALIZATION_CONSTANT
    )
    x = df[GENES].to_numpy()
    fig.suptitle(fname)
    for i, gene in enumerate(GENES):
        ax = axes[i]
        sns.histplot(x[:,i], ax=ax)
        ax.set_title(gene)
        ax.set_xlabel(f"Fluorescence")
        if i > 0:
            ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(f"{subdir}/gexp_histograms_{fname.removesuffix('.fcs .csv')}.png")
    plt.close()


DF_META = get_meta_dataframe(CONDITIONS)

DF_MAIN = load_all_files(
    ALL_FILES, datdir=DATDIR, normalization=NORMALIZATION_CONSTANT
)

DF_REF = get_reference_dataset(DF_MAIN, DF_META)

"""
For each experimental condition (in the reference set), plot the distribution 
of each gene over each day.
"""

for cond_idx in REFERENCE_CONDITIONS:
    condition = CONDITIONS[cond_idx]
    # subset the metadata to include just the condition of interest.
    df_cond = DF_META[DF_META['condition'] == condition]
    timepoints = np.sort(df_cond['timepoint'].unique())
    # subset the datafile to include just cells in those conditions
    df_data = DF_MAIN[DF_MAIN['filename'].isin(df_cond['filename'].unique())]
    
    nrows = len(GENES)
    ncols = len(timepoints)
    axw = 2
    axh = 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(axw*ncols, axh*nrows))
    fig.suptitle(CONDITIONS[cond_idx])

    for tidx, timepoint in enumerate(timepoints):
        df_data_timepoint = df_data[df_data['timepoint'] == timepoint]
        gexp = df_data_timepoint[GENES].to_numpy()
        for gidx, gene in enumerate(GENES):
            ax = axes[gidx, tidx]
            sns.histplot(gexp[:,gidx], ax=ax)
            if gidx == len(GENES) - 1:
                ax.set_xlabel(f"Day {timepoint:.1f}")
            if tidx == 0:
                ax.set_ylabel(gene)
            else:
                ax.set_ylabel("")
            ax.set_title(f"$n={len(gexp)}$", fontsize=6)
            ax.tick_params(axis='both', which='major', labelsize=6)

    plt.tight_layout()
    plt.savefig(f"{IMGDIR}/temporal_gexp_dist_{condition}", bbox_inches='tight')


"""
Create two arrays of plots, showing conditions by row, and genes by column. 
For the first, in each axis, plot for each time point the eCDF of the gene 
fluorescence. For the second, plot the median expression.
"""

use_same_axis = True
use_log_scale_x = False
palette = 'mako_r'   # RdPu Spectral
cmap = sns.color_palette(palette, 7)

nc = len(ALL_CONDITIONS)
ng = len(GENES)

fig1, axes1 = plt.subplots(nc, ng, figsize=(15, 15))
fig2, axes2 = plt.subplots(nc, ng, figsize=(12, 12))

gene_max_fluoro = {g: 0. for g in GENES}
for gidx, gene in enumerate(GENES):
    for cond_idx in ALL_CONDITIONS:
        condition = CONDITIONS[cond_idx]
        # subset the metadata to include just the condition of interest.
        df_cond = DF_META[DF_META['condition'] == condition]
        timepoints = np.sort(df_cond['timepoint'].unique())
        # subset the datafile to include just cells in those conditions
        df_data = DF_MAIN[
            DF_MAIN['filename'].isin(df_cond['filename'].unique())
        ]

        ax1 = axes1[cond_idx][gidx]  # axis in first plot
        ax2 = axes2[cond_idx][gidx]  # axis in second plot

        ecdf = sns.ecdfplot(
            data=df_data,
            x=gene,
            hue="timepoint",
            palette=cmap,
            log_scale=(10, None) if use_log_scale_x else None,
            ax=ax1,
        )            

        meds = df_data.groupby(df_data.timepoint)[[gene]].median()
        ax2.plot(timepoints, meds)
        
        if not use_log_scale_x:
            ax1.ticklabel_format(axis='x', style='sci', scilimits=(0,0))

        if gidx == 0:
            ax1.set_ylabel(f"{CONDITIONS[cond_idx]}", fontsize=6)
            ax2.set_ylabel(f"{CONDITIONS[cond_idx]}", fontsize=6)
        else:
            ax1.set_ylabel("")
            ax2.set_ylabel("")
        
        if cond_idx == 0:
            ax1.set_title(GENES[gidx])
            ax2.set_title(GENES[gidx])
        else:
            ax1.set_title("")
            ax2.set_title("")
        
        if cond_idx == nc - 1:
            ax1.set_xlabel("fluorescence")
            ax2.set_xlabel("time")
        else:
            ax1.set_xlabel("")
            ax2.set_xlabel("")

        if gidx == ng - 1 and cond_idx == 0:
            sns.move_legend(
                ax1, "upper left", 
                bbox_to_anchor=(1.05, 1), title_fontsize=8, fontsize=6
            )
        else:
            ax1.get_legend().remove()
        
        if ax1.get_xlim()[1] > gene_max_fluoro[gene]:
            gene_max_fluoro[gene] = ax1.get_xlim()[1]
    
if use_same_axis:
    for gidx, gene in enumerate(GENES):
        for i in range(nc):
            ax1 = axes1[i][gidx]
            ax1.set_xlim(0, gene_max_fluoro[gene])
        
fig1.suptitle("Gene expression eCDF temporal evolution")
fig2.suptitle("Median gene expression in time")

plt.figure(fig1)
fig1.tight_layout()
plt.savefig(f"{IMGDIR}/all_conditions_ecdfs", bbox_inches='tight')

plt.figure(fig2)
fig2.tight_layout()
plt.savefig(f"{IMGDIR}/all_conditions_median_gexp", bbox_inches='tight')


"""
Now merge the experimental conditions in the reference set and plot the 
distribution of each gene across time.
"""

REFERENCE_DATASETS = {}

conditions = [CONDITIONS[cond_idx] for cond_idx in REFERENCE_CONDITIONS]
# subset the metadata to include just the condition of interest.
df_cond = DF_META[DF_META['condition'].isin(conditions)]
timepoints = np.sort(df_cond['timepoint'].unique())
# subset the datafile to include just cells in those conditions
df_data = DF_MAIN[DF_MAIN['filename'].isin(df_cond['filename'].unique())]

nrows = len(GENES)
ncols = len(timepoints)
axw = 2
axh = 1
fig, axes = plt.subplots(nrows, ncols, figsize=(axw*ncols, axh*nrows))
fig.suptitle(f"Reference Set: {', '.join(conditions)}")

for tidx, timepoint in enumerate(timepoints):
    df_data_timepoint = df_data[df_data['timepoint'] == timepoint]
    REFERENCE_DATASETS[timepoint] = df_data_timepoint
    gexp = df_data_timepoint[GENES].to_numpy()
    for gidx, gene in enumerate(GENES):
        ax = axes[gidx, tidx]
        sns.histplot(gexp[:,gidx], ax=ax)
        if gidx == 0:
            ax.set_title(f"$n={len(gexp)}$", fontsize=6)
        if gidx == len(GENES) - 1:
            ax.set_xlabel(f"Day {timepoint:.1f}")
        if tidx == 0:
            ax.set_ylabel(gene)
        else:
            ax.set_ylabel("")
        ax.tick_params(axis='both', which='major', labelsize=6)

plt.tight_layout()
plt.savefig(f"{IMGDIR}/temporal_gexp_dists_ref_set")

#~~~  Save meta and main dataframes  ~~~#
DF_META.to_csv(f"{OUTDIR}/df_meta.csv")
DF_MAIN.to_csv(f"{OUTDIR}/df_main.csv")
DF_REF.to_csv(f"{OUTDIR}/df_ref.csv")
