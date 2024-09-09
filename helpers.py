"""Helper functions
"""

import numpy as np
import pandas as pd
from constants import DATDIR, NCELLS, FNAME_TO_SAMPLEIDX, CONDITION_SIGNALS
from constants import CONDITIONS, CONDITION_FILES, CONDITION_REFERENCE_FLAG

def load_facs_csv_file(fpath, normalization):
    df = pd.read_csv(
        fpath, index_col=0
    )
    df = df / normalization
    return df

def extract_time_from_filename(fname) -> (str, float):
    parts = fname.split("_")
    timestring = parts[0]
    timepoint = float(timestring[1:])
    return timestring, timepoint


def get_experiment_dataframe(condition_index):
    condition_str = CONDITIONS[condition_index]
    file_list = CONDITION_FILES[condition_index]
    is_reference = CONDITION_REFERENCE_FLAG[condition_index]
    cond_chir, cond_fgf, cond_pd = CONDITION_SIGNALS[condition_index]

    data = []
    for fname in file_list:
        fpath = f"{DATDIR}/{fname}"
        timestring, timepoint = extract_time_from_filename(fname)
        row = [
            condition_str, fname, fpath, FNAME_TO_SAMPLEIDX[fname],
            timestring, timepoint, 
            *cond_chir, *cond_fgf, *cond_pd,
            is_reference, NCELLS
        ]
        data.append(row)

    df = pd.DataFrame(
        data, 
        columns=[
            "condition", "filename", "path", "sample", 
            "timestring", "timepoint", 
            "chir_start", "chir_end",
            "fgf_start", "fgf_end",
            "pd_start","pd_end",
            "is_reference", "ncells",
        ]
    )   
    return df


def get_meta_dataframe(conditions):
    df_list = []
    for cond_idx in conditions:
        df_list.append(get_experiment_dataframe(cond_idx))

    return pd.concat(df_list, ignore_index=True)


def load_all_files(file_list, datdir, normalization):
    df_list = []
    for fname in file_list:
        fpath = f"{datdir}/{fname}"
        _, timepoint = extract_time_from_filename(fname)
        df_file = load_facs_csv_file(fpath, normalization)
        df_file['filename'] = fname
        df_file['timepoint'] = timepoint
        df_file['sample'] = FNAME_TO_SAMPLEIDX[fname]
        df_list.append(df_file)
    df = pd.concat(df_list, ignore_index=True)
    df['cellidx'] = pd.Series(range(len(df)))
    df = df.set_index('cellidx')
    return df


def get_reference_dataset(df_data, df_conditions):
    df_ref_conditions = df_conditions[df_conditions['is_reference']]
    fnames = df_ref_conditions['filename'].unique()
    df_ref_cells = df_data[df_data['filename'].isin(fnames)]
    return df_ref_cells


def get_signal_params_from_condition(
        cond_idx, r,
        t0=2.0,
    ):
    cond_sigs = CONDITION_SIGNALS[cond_idx]
    chir_sig, fgf_sig, pd_sig = cond_sigs
    sigparams = np.nan * np.ones([2, 4], dtype=np.float64)
    
    # First signal: CHIR
    chir_start, chir_end = chir_sig
    if chir_start == -1:
        sigparams[0,0:-1] = 0
    else:
        sigparams[0,0] = chir_end - t0  # signal changes at end time.
        sigparams[0,1] = 1.0  # CHIR goes from on...
        sigparams[0,2] = 0.0  # to off.
    # Second signal: FGF (Based on FGF and PD)
    fgf_start, fgf_end = fgf_sig
    pd_start, pd_end = pd_sig
    if pd_start == -1:
        # No PD, so FGF is either 1 or 0.9
        sigparams[1,0] = fgf_end - t0  # signal changes at end time.
        sigparams[1,1] = 1.0  # FGF goes from 1...
        sigparams[1,2] = 0.9  # to 0.9, since no PD.
    else:
        # PD present, so FGF is either 1 or 0
        sigparams[1,0] = fgf_end - t0  # signal changes at end time.
        sigparams[1,1] = 1.0  # FGF goes from 1...
        sigparams[1,2] = 0.0  # to 0., since there is PD.
    # Assume same rate of transition, as given, for both CHIR and FGF
    sigparams[:,-1] = r  
    return sigparams


def get_hist2d(data, edges_x, edges_y):
    x_bins = np.digitize(data[:, 0], edges_x)  # bin indices for x
    y_bins = np.digitize(data[:, 1], edges_y)  # bin indices for y
    hist2d = np.zeros([len(edges_y) - 1, len(edges_x) - 1])
    for xb, yb in zip(x_bins, y_bins):
        if xb == 0 or yb == 0 or xb == len(edges_x) or yb == len(edges_y):
            pass
        else:
            hist2d[yb-1, xb-1] += 1
    return hist2d
    