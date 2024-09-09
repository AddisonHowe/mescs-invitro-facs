# FACS Data Preprocessing Scripts

## Scripts

`script1.py` - Loads the raw FACS data and compiles and saves dataframes. These are saved in the specified output directory.
Saved files are:
* `df_meta.csv`
* `df_main.csv`
* `df_ref.csv`

The metadata dataframe contains a row for every sampled timepoint of every experiment in the loaded dataset. In this case, as there are 11 experiments comprising the initial experimental series, each with 7 sampling times, there are 77 rows in the saved metadata table.
The columns are:
* condition
* filename
* path
* sample
* timestring
* timepoint
* chir_start
* chir_end
* fgf_start
* fgf_end
* pd_start
* pd_end
* is_reference
* ncells

The main dataframe has a row for every cell sampled across all 11 experiments, and columns
* TBX6
* BRA
* CDX2
* SOX2
* SOX1
* filename
* timepoint
* sample

The gene columns contain the raw fluorescence data, scaled by $10^-4$, so that the readings are of unit order.
The filename column contains the name of the raw csv file containing that cell, and the timepoint column contains a float value corresponding to the time at which the cell was sampled. The sample column contains an integer value, uniquely mapping to the raw csv file.

The `df_ref.csv` file is of the same form as the main dataframe, but contains only those cells in the reference dataset. That is, those cells corresponding to the 3 experimental conditions **NO CHIR**, **CHIR 2-3**, and **CHIR 2-5**.